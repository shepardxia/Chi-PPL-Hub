"""
Classify each code block in cleaned data into one of:

  1. Complete         — parses and runs to a defined output
  2. Intentional fail — parses but times out / hits a runtime failure the author intended
  3. Incomplete       — parse error (e.g. `...` placeholders) OR undefined-name with
                        no matching runtime symbol; reader expected to complete
  4. Runtime-dep      — runs fine once a known-but-unwired runtime dep is present
  unclassified        — runtime error that doesn't fit any of the above; needs review

Also attaches quality_flags (currently: "viz_only") to complete blocks.
Category 3 is marked, not filled in — completions come later.

Pipeline:
  data/cleaned/*.jsonl --> classify.py --> data/classified/*.jsonl

Execution results are cached in data/classification_cache.json keyed on
a hash of the incremental assembly, so re-runs only re-check changed blocks.

Usage:
    python scripts/classify.py
    python scripts/classify.py --no-cache    # force re-run of every block
"""

import argparse
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Pull executor from the sibling eval/ directory.
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "eval"))
from executor import execute_webppl, assemble_incremental  # noqa: E402


CLEANED_DIR = _REPO_ROOT / "data" / "cleaned"
CLASSIFIED_DIR = _REPO_ROOT / "data" / "classified"
CACHE_FILE = _REPO_ROOT / "data" / "classification_cache.json"

# Seconds per block. Long enough for real inference (MCMC 10k samples etc.)
# but short enough to catch non-terminating teaching examples reasonably fast.
RUN_TIMEOUT = 20
# Seed every run, so the cache key is stable across invocations.
RUN_SEED = 42
# Parallelism: webppl runs as a subprocess, so threads (not processes)
# are the right unit — they release the GIL on subprocess.run.
DEFAULT_WORKERS = max(4, (os.cpu_count() or 4))


# Runtime symbols that are provided by dep packages the harness loads.
# An undefined-reference error mentioning one of these signals "category 4,
# dep available" — because we already load it. Any other undefined name is
# either category 3 (placeholder) or an unwired dep we don't yet handle.
KNOWN_RUNTIME_SYMBOLS = {
    # webppl-agents
    "makeGridWorldMDP", "GridWorld", "simulateMDP", "simulatePOMDP",
    # webppl-dp
    "DPmem", "CRP",
    # probmods-deps
    "MCMC_Callbacks", "euclideanDistance", "editor",
    # probmods-towdata
    "towData", "towMeans", "towConfigurations",
    # probmods-physics
    "physics", "worldWidth", "worldHeight", "Draw",
    # webppl-viz shim (provided by executor SHIM_HEADER)
    "viz", "vizPrint", "print",
    # webppl-timeit
    "timeit",
}


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def _hash_code(code):
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def classify_error(err):
    """Bucket a WebPPL error message into a kind.

    Returns one of: "parse", "undefined_ref:<name>", "timeout", "runtime".
    """
    if not err:
        return "runtime"

    # Timeout is signaled by execute_webppl with a specific string.
    if err.startswith("Timeout"):
        return "timeout"

    # Esprima parse errors come through as "Error: Line N: ..." and the
    # stack trace mentions esprima. The error_message we receive has already
    # been cleaned; check the message text.
    if re.search(r"Unexpected token|Unterminated|Invalid (regular expression|number|hex)", err):
        return "parse"
    # WebPPL-specific AST rejection — chained ternary, etc.
    if "isPrimitive doesn't handle" in err or "unknown AST node type" in err:
        return "parse"
    # WebPPL requires `var` for declarations (CPS transform constraint);
    # assignments without var are rejected at compile/parse time.
    if "Did you mean var" in err:
        return "parse"

    m = re.search(r"ReferenceError:\s*(\w+)\s+is not defined", err)
    if m:
        return f"undefined_ref:{m.group(1)}"

    return "runtime"


def has_inference_call(code):
    """Does the code invoke inference (Infer/Enumerate/MH/SMC/Rejection/...)?

    Used to flag a runnable block as "viz_only" if it's just rendering
    a pre-existing value without computation worth asking an LLM to reproduce.
    """
    inference_fns = [
        r"\bInfer\s*\(", r"\bEnumerate\s*\(", r"\bMH\s*\(", r"\bMCMC\s*\(",
        r"\bSMC\s*\(", r"\bRejection\s*\(", r"\bOptimize\s*\(",
        r"\bIncrementalMH\s*\(", r"\bParticleFilter\s*\(",
        r"\bHashMH\s*\(", r"\bHMC\s*\(",
    ]
    return any(re.search(p, code) for p in inference_fns)


# Pedagogical placeholder markers the book uses to signal "reader fills this in".
# A block containing any of these is an intentional template (cat 3), regardless
# of how it fails at runtime (could be ReferenceError, TypeError, cascade, etc.).
_PLACEHOLDER_PATTERNS = [
    re.compile(r"//\s*your\s+code\s+here", re.IGNORECASE),
    re.compile(r"//\s*edit\s+this\s+line", re.IGNORECASE),
    re.compile(r"//\s*fill\s+in", re.IGNORECASE),
    re.compile(r"//\s*TODO\b"),
]


def has_placeholder(code):
    return any(p.search(code) for p in _PLACEHOLDER_PATTERNS)


# ---------------------------------------------------------------------------
# Per-block classification
# ---------------------------------------------------------------------------

def _summarize_capture(c):
    """Fingerprint a single capture as a short type tag.

    Feeds into eval_mode selection at build_dataset time:
      - "distribution" → KL / cross-entropy / TV
      - "value_scalar" → (approx) exact match, or MC if stochastic
      - "value_array"  → empirical distribution comparison
      - "value_object" / "value_string" → exact match
      - "null"         → side-effect only
    """
    t = c.get("type")
    if t == "distribution" or t == "distribution_obj":
        return "distribution"
    if t == "null":
        return "null"
    v = c.get("value")
    if isinstance(v, bool):
        return "value_bool"
    if isinstance(v, (int, float)):
        return "value_scalar"
    if isinstance(v, list):
        return "value_array"
    if isinstance(v, dict):
        return "value_object"
    if isinstance(v, str):
        return "value_string"
    return "value_other"


def _run_block(code):
    """Execute one block and return the small dict we cache.

    Split out from classify_block so it can be called by a thread pool
    without touching the shared cache.
    """
    r = execute_webppl(code, timeout=RUN_TIMEOUT, random_seed=RUN_SEED)
    return {
        "success": r.success,
        "error_message": r.error_message,
        "num_captures": len(r.captures),
        "capture_types": [_summarize_capture(c) for c in r.captures],
    }


def classify_block(code, cache, use_cache=True):
    """Classify one block given its full incremental-assembled code string.

    Returns {category, dataset_fit, reason, signals}.
    """
    key = _hash_code(code)
    if use_cache and key in cache:
        result = cache[key]
    else:
        result = _run_block(code)
        cache[key] = result

    if result["success"]:
        return {
            "category": 1,
            "dataset_fit": True,
            "reason": "runs",
            "signals": {"error_kind": None, "num_captures": result["num_captures"]},
        }

    kind = classify_error(result["error_message"])

    if kind == "parse":
        return {
            "category": 3,
            "dataset_fit": False,
            "reason": "parse_error",
            "signals": {"error_kind": "parse", "error_message": result["error_message"][:200]},
        }

    if kind == "timeout":
        return {
            "category": 2,
            "dataset_fit": False,
            "reason": "timeout",
            "signals": {"error_kind": "timeout"},
        }

    if kind.startswith("undefined_ref:"):
        name = kind.split(":", 1)[1]
        if name in KNOWN_RUNTIME_SYMBOLS:
            # Runtime symbol we claim to provide, yet it's undefined — this
            # means a dep isn't actually wired or the dep's export failed.
            # Runnable-in-principle; flag for harness fix, not dataset drop.
            return {
                "category": 4,
                "dataset_fit": False,
                "reason": f"runtime_unwired:{name}",
                "signals": {"error_kind": "undefined_ref", "symbol": name},
            }
        return {
            "category": 3,
            "dataset_fit": False,
            "reason": f"undefined_ref:{name}",
            "signals": {"error_kind": "undefined_ref", "symbol": name},
        }

    # Plain runtime error. Could be category 2 (author-intended failure),
    # a real bug, or a dep issue we don't recognize. Leave unclassified
    # for manual review rather than force a label.
    return {
        "category": None,
        "dataset_fit": False,
        "reason": "unclassified_runtime_error",
        "signals": {
            "error_kind": "runtime",
            "error_message": result["error_message"][:200],
        },
    }


# ---------------------------------------------------------------------------
# Per-record pass
# ---------------------------------------------------------------------------

def _attach_quality_flags(cls, code):
    """Compute per-block quality flags that depend on block source, not runtime."""
    flags = []
    if cls["category"] == 1 and not has_inference_call(code):
        flags.append("viz_only")
    cls["quality_flags"] = flags
    return cls


def _run_with_cache(code, cache, use_cache):
    key = _hash_code(code)
    if use_cache and key in cache:
        return cache[key]
    result = _run_block(code)
    cache[key] = result
    return result


def _classification_from_result(result):
    """Turn a run result into a classification dict (no quality flags)."""
    if result["success"]:
        return {
            "category": 1,
            "dataset_fit": True,
            "reason": "runs",
            "signals": {
                "error_kind": None,
                "num_captures": result["num_captures"],
                "capture_types": result.get("capture_types", []),
            },
        }

    kind = classify_error(result["error_message"])
    if kind == "parse":
        return {
            "category": 3,
            "dataset_fit": False,
            "reason": "parse_error",
            "signals": {"error_kind": "parse", "error_message": result["error_message"][:200]},
        }
    if kind == "timeout":
        return {
            "category": 2,
            "dataset_fit": False,
            "reason": "timeout",
            "signals": {"error_kind": "timeout"},
        }
    if kind.startswith("undefined_ref:"):
        name = kind.split(":", 1)[1]
        if name in KNOWN_RUNTIME_SYMBOLS:
            return {
                "category": 4,
                "dataset_fit": False,
                "reason": f"runtime_unwired:{name}",
                "signals": {"error_kind": "undefined_ref", "symbol": name},
            }
        return {
            "category": 3,
            "dataset_fit": False,
            "reason": f"undefined_ref:{name}",
            "signals": {"error_kind": "undefined_ref", "symbol": name},
        }
    return {
        "category": None,
        "dataset_fit": False,
        "reason": "unclassified_runtime_error",
        "signals": {"error_kind": "runtime", "error_message": result["error_message"][:200]},
    }


def _classify_section_array(sections, cache, use_cache=True):
    """Classify the code blocks in one section array (independent incremental chain).

    Block N's incremental assembly runs blocks 0..N, so a single failing
    block poisons every downstream assembly. To avoid spurious cascades,
    we iterate: run the full assembly; on failure, walk incrementally
    through the *active* (not-yet-failed) blocks to find the first failure,
    record it, remove that block from the active set, and retry. When the
    active assembly passes, every remaining block is category 1.

    Returns a new sections list with `classification` attached to each
    code section. Cost: O((F+1) * N) worst case, where F = number of
    failing blocks.
    """
    code_indices = [i for i, s in enumerate(sections) if s["type"] == "code"]
    code_blocks = [sections[i]["content"] for i in code_indices]

    if not code_blocks:
        return list(sections)

    failed = {}  # original_block_idx -> classification dict
    active = list(range(len(code_blocks)))

    while active:
        active_code = [code_blocks[i] for i in active]
        full_code = assemble_incremental(active_code, len(active_code) - 1)
        full_result = _run_with_cache(full_code, cache, use_cache)

        if full_result["success"]:
            break

        # Locate the first failing block within `active`.
        for j in range(len(active)):
            sub = assemble_incremental(active_code, j)
            r = _run_with_cache(sub, cache, use_cache)
            if not r["success"]:
                orig_idx = active[j]
                failed[orig_idx] = _classification_from_result(r)
                active.pop(j)
                break
        else:
            # Full failed but no individual prefix failed — shouldn't happen
            # unless a non-deterministic error slipped through. Safety net:
            # fail the last block and move on.
            orig_idx = active[-1]
            failed[orig_idx] = _classification_from_result(full_result)
            active.pop()

    out = list(sections)
    for block_idx, section_idx in enumerate(code_indices):
        code = code_blocks[block_idx]
        if block_idx in failed:
            cls = failed[block_idx]
        else:
            cls = {
                "category": 1,
                "dataset_fit": True,
                "reason": "runs_in_active_assembly",
                "signals": {"error_kind": None},
            }
        # Pedagogical placeholder comments override: a block containing
        # `// Your code here` or similar is always cat-3 (incomplete) even
        # if WebPPL happens to accept it or it fails for a cascade-y reason.
        if has_placeholder(code):
            original_reason = cls.get("reason", "")
            cls = {
                "category": 3,
                "dataset_fit": False,
                "reason": "placeholder_comment",
                "signals": {
                    "placeholder": True,
                    "original_reason": original_reason,
                    "error_kind": cls.get("signals", {}).get("error_kind"),
                },
            }
        _attach_quality_flags(cls, code)
        out[section_idx] = dict(out[section_idx], classification=cls)
    return out


def classify_record(record, cache, use_cache=True):
    """Classify every code block in a record, including `solution_sections`
    when present (exercise records). Solutions run as an independent
    incremental chain — they are not concatenated with exercise-side blocks.
    """
    record = dict(record)
    record["sections"] = _classify_section_array(
        record.get("sections", []), cache, use_cache
    )
    if record.get("solution_sections"):
        record["solution_sections"] = _classify_section_array(
            record["solution_sections"], cache, use_cache
        )
    return record


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def load_cache():
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_cache(cache):
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignore cached results and re-run every block.")
    parser.add_argument("--source", type=str, default=None,
                        help="Only classify one source (stem of cleaned/*.jsonl).")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Parallel entries to classify (default: {DEFAULT_WORKERS}).")
    args = parser.parse_args()

    CLASSIFIED_DIR.mkdir(parents=True, exist_ok=True)
    cache = {} if args.no_cache else load_cache()

    files = sorted(CLEANED_DIR.glob("*.jsonl"))
    if args.source:
        files = [f for f in files if f.stem == args.source]

    for jsonl_file in files:
        print(f"Classifying {jsonl_file.name} (workers={args.workers})...")

        records = []
        with open(jsonl_file) as f:
            for line in f:
                records.append(json.loads(line))

        # Run entries in parallel — each entry's classify_record is independent.
        # The shared `cache` dict is mutated; CPython dict get/set is atomic
        # enough for our idempotent writes, but we save after each source.
        classified = [None] * len(records)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(classify_record, r, cache, not args.no_cache): idx
                for idx, r in enumerate(records)
            }
            done_count = 0
            for fut in as_completed(futures):
                idx = futures[fut]
                classified[idx] = fut.result()
                done_count += 1
                rec_id = records[idx].get("id", f"#{idx}")
                print(f"  [{done_count}/{len(records)}] {rec_id}")

        totals = {"1": 0, "2": 0, "3": 0, "4": 0, "unclassified": 0, "viz_only": 0}
        solution_totals = {"1": 0, "2": 0, "3": 0, "4": 0, "unclassified": 0, "viz_only": 0}

        def _tally(sec_array, bucket):
            for s in sec_array:
                if s["type"] != "code":
                    continue
                cls = s.get("classification", {})
                cat = cls.get("category")
                key = str(cat) if cat is not None else "unclassified"
                bucket[key] = bucket.get(key, 0) + 1
                if "viz_only" in cls.get("quality_flags", []):
                    bucket["viz_only"] += 1

        for r2 in classified:
            _tally(r2.get("sections") or [], totals)
            _tally(r2.get("solution_sections") or [], solution_totals)

        out_path = CLASSIFIED_DIR / jsonl_file.name
        with open(out_path, "w") as f:
            for record in classified:
                f.write(json.dumps(record) + "\n")

        # Checkpoint the cache after each source so an interrupt doesn't
        # throw away all the run results.
        save_cache(cache)

        print(f"  Records: {len(classified)}")
        print(f"  Blocks by category: {totals}")
        if any(v for v in solution_totals.values()):
            print(f"  Solution blocks by category: {solution_totals}")
        print(f"  Output: {out_path}")
        print()

    save_cache(cache)
    print(f"Cache: {len(cache)} entries at {CACHE_FILE}")


if __name__ == "__main__":
    main()
