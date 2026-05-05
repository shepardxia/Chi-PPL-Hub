"""Assemble curated atoms from agent emissions.

Agent emissions JSONL: each row has
  {id, source, source_block_indices, prompt, notes (optional)}

For each row, this script:
  1. Reads the source markdown file and splits into code blocks.
  2. Concatenates the listed blocks (in the order the agent listed them,
     deduped while preserving first-seen order).
  3. Wraps with `var ANSWER = ...` via `wrap_with_answer`.
  4. Runs via `execute_webppl(seed=42)`.
  5. On success: emits the full atom record to `--output`.
     On failure: emits the broken record to `--broken` with the error and
     the agent's notes for triage.

The agent does not write GT code or output; the pipeline does. This is the
v3 contract — agents own judgment (chunking, prompt, notes), pipeline owns
mechanics (concat, wrap, execute, classify).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.executor import execute_webppl
from eval.io import write_jsonl
from scripts.extract_atoms import classify_answer, split_blocks, wrap_with_answer


def _dedupe_keep_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _resolve_source(repo_root: Path, src: str) -> Path:
    """Accept either a repo-root-relative or absolute path."""
    p = Path(src)
    if p.is_absolute():
        return p
    candidate = repo_root / p
    if candidate.exists():
        return candidate
    # Allow "data/sources/..." or just "dippl/chapters/01-intro.md"
    candidate = repo_root / "data" / "sources" / p
    if candidate.exists():
        return candidate
    return repo_root / p  # let the caller see it doesn't exist


def assemble(emissions_path: Path, output_path: Path, broken_path: Path,
             *, timeout: int = 60) -> tuple[int, int]:
    repo_root = Path(__file__).resolve().parent.parent
    emissions = [json.loads(l) for l in emissions_path.read_text().splitlines() if l.strip()]

    atoms: list[dict] = []
    broken: list[dict] = []
    block_cache: dict[str, list[tuple[str, str]]] = {}

    t0 = time.time()
    for em in emissions:
        em_id = em.get("id", "<no-id>")
        src = em.get("source", "")
        src_path = _resolve_source(repo_root, src)
        if not src_path.exists():
            broken.append({**em, "error": f"source file not found: {src}"})
            continue

        key = str(src_path)
        if key not in block_cache:
            block_cache[key] = list(split_blocks(src_path.read_text()))
        blocks = block_cache[key]

        idxs_raw = em.get("source_block_indices", [])
        if not isinstance(idxs_raw, list) or not all(isinstance(i, int) for i in idxs_raw):
            broken.append({**em, "error": f"source_block_indices must be a list of ints, got {idxs_raw!r}"})
            continue
        idxs = _dedupe_keep_order(idxs_raw)
        try:
            picked_codes = [blocks[i][1] for i in idxs]
        except IndexError:
            broken.append({**em, "error": f"block index out of range (file has {len(blocks)} blocks, got {idxs})"})
            continue

        if not picked_codes and not em.get("synth_code"):
            broken.append({**em, "error": "no source blocks listed and no synth_code provided"})
            continue

        full_code = "\n\n".join(picked_codes).strip()
        if not full_code:
            broken.append({**em, "error": "all listed blocks are empty"})
            continue

        wrapped = wrap_with_answer(full_code)
        if wrapped is None:
            broken.append({**em, "error": "wrap_with_answer returned None (no clear last expression)",
                           "assembled_code": full_code})
            continue

        result = execute_webppl(wrapped, timeout=timeout, random_seed=42)
        if not (result.success and result.answer is not None):
            broken.append({
                **em,
                "error": result.error_message or "execution failed (no answer)",
                "stderr_tail": (result.stderr or "")[-500:],
                "wrapped_code": wrapped,
            })
            continue

        shape, mode = classify_answer(result.answer)
        atom = {
            "id": em_id,
            "source": src,
            "source_block_indices": idxs,
            "task_type": "write_from_scratch",
            "eval_mode": mode,
            "answer_shape": shape,
            "prompt": em.get("prompt", ""),
            "groundtruth_code": wrapped,
            "groundtruth_output": result.answer,
        }
        if em.get("notes"):
            atom["notes"] = em["notes"]
        atoms.append(atom)

        elapsed = time.time() - t0
        print(f"  [{len(atoms) + len(broken)}/{len(emissions)}] {em_id:50s} OK ({elapsed:.1f}s)", flush=True)

    write_jsonl(output_path, atoms)
    write_jsonl(broken_path, broken)
    return len(atoms), len(broken)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emissions", required=True, help="Agent JSONL: {id, source, source_block_indices, prompt, notes}")
    p.add_argument("--output", required=True, help="Output JSONL of fully-assembled atoms")
    p.add_argument("--broken", required=True, help="Output JSONL of emissions that failed assembly")
    p.add_argument("--timeout", type=int, default=60)
    args = p.parse_args()

    n_ok, n_broken = assemble(
        Path(args.emissions), Path(args.output), Path(args.broken), timeout=args.timeout,
    )
    print(f"\nDone: {n_ok} OK, {n_broken} broken")
    print(f"  → {args.output}")
    print(f"  → {args.broken}")


if __name__ == "__main__":
    main()
