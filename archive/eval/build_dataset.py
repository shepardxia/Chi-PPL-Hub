"""
Build evaluation dataset from classified WebPPL resources.

Reads `data/classified/*.jsonl` (not `data/cleaned/`) so we pick up
per-block `classification.dataset_fit` and `classification.quality_flags`.
Drops blocks where `dataset_fit` is false.

Chapter-style records (no `solution_sections`): the prose is the prompt
with code blocks replaced by `[BLOCK_N]`. Groundtruth is one code string
per block, wrapped in a single-element list so the schema is uniform
across sources.

Exercise records: prompt uses the exercise-side `sections` (clean questions
and starter templates, rendered as `[BLOCK_N]` placeholders). Groundtruth
for each block is the list of solution-side code blocks matching the same
`(exercise_num, sub_letter)` id parsed out of markdown headers. Unpaired
blocks on either side are written to `data/unpaired_blocks.jsonl` for
later investigation.

Usage:
    python eval/build_dataset.py
"""

import json
import re
from pathlib import Path

CLASSIFIED_DIR = Path("data/classified")
OUTPUT_FILE = Path("data/eval_dataset.jsonl")
UNPAIRED_FILE = Path("data/unpaired_blocks.jsonl")

# Scope: probmods2 textbook only. `webppl/examples/*.wppl` and `forestdb/`
# exist as classified files but are intentionally out of scope for the
# current dataset build.
_IN_SCOPE_SOURCES = {
    "probmods_chapters",
    "probmods_exercises",
    "probmods_teaching_extras",
}


# ---------------------------------------------------------------------------
# eval_mode: how a block should be scored
# ---------------------------------------------------------------------------
# Per-block evaluation mode, decided by static analysis of the groundtruth
# code. The harness reads eval_mode to pick the right metric:
#
#   distribution          KL / TV / cross-entropy on the captured distribution.
#                         Code calls Infer/Enumerate/MH/MCMC/etc.
#
#   stochastic_value      Monte Carlo: run N times with different seeds on
#                         both gen + gt, compare empirical distributions
#                         of the captured values. Code calls flip/gaussian/
#                         sample/etc. but does NOT wrap in Infer.
#
#   deterministic_value   Exact or approximate value match. Code has no
#                         stochastic primitives; a single run is enough.
#
#   side_effect           Only visual / drawing side-effects (canvas / viz
#                         on a pre-computed value). String match only.
#
#   unevaluable           Empty groundtruth, viz_only with no Infer,
#                         or code that can't be scored meaningfully.

_INFER_PATTERNS = [
    re.compile(p) for p in [
        r"\bInfer\s*\(",
        r"\bEnumerate\s*\(",
        r"\bMH\s*\(",
        r"\bMCMC\s*\(",
        r"\bSMC\s*\(",
        r"\bRejection\s*\(",
        r"\bIncrementalMH\s*\(",
        r"\bParticleFilter\s*\(",
        r"\bHashMH\s*\(",
        r"\bHMC\s*\(",
        r"\bOptimize\s*\(",
    ]
]

# Stochastic primitives: sampling operators whose value is random.
# `sample(` catches sample(Gaussian(...)), sample(Bernoulli(...)) etc.
_STOCHASTIC_PATTERNS = [
    re.compile(p) for p in [
        r"\bflip\s*\(",
        r"\bsample\s*\(",
        r"\buniform\s*\(",
        r"\buniformDraw\s*\(",
        r"\bRandomInteger\s*\(",
        r"\bgaussian\s*\(",
        r"\bdirichlet\s*\(",
        r"\bbeta\s*\(",
        r"\bgamma\s*\(",
        r"\bbinomial\s*\(",
        r"\bmultinomial\s*\(",
        r"\bpoisson\s*\(",
        r"\bcategorical\s*\(",
        r"\bdiscrete\s*\(",
        r"\bdirichletDrift\s*\(",
        r"\bdiagCovGaussian\s*\(",
    ]
]

# Side-effect / visualization-only primitives. Code dominated by these
# produces no inferrable output.
_SIDE_EFFECT_PATTERNS = [
    re.compile(p) for p in [
        r"\bcanvas\.\w+\s*\(",
        r"\bDraw\s*\(",
    ]
]


def compute_eval_mode(code, quality_flags=None):
    """Static-analysis classifier for how to score a block.

    Precedence (first match wins):
      1. Infer/Enumerate/MH/... anywhere        → distribution
      2. canvas.* or Draw(...) (side-effect)    → side_effect
      3. stochastic primitive (flip/sample/...) → stochastic_value (MC)
      4. else                                   → deterministic_value

    The `viz_only` flag is informational only here — a block can be tagged
    viz_only (no Infer call) yet still produce evaluable stochastic samples
    via raw `flip()` / `beta()` / `gaussian()` etc. The harness tolerates
    `deterministic_value` blocks that happen to produce no captures (pure
    definitions) — equivalent outputs = equivalent code on that metric.
    """
    if any(p.search(code) for p in _INFER_PATTERNS):
        return "distribution"
    if any(p.search(code) for p in _SIDE_EFFECT_PATTERNS):
        return "side_effect"
    if any(p.search(code) for p in _STOCHASTIC_PATTERNS):
        return "stochastic_value"
    return "deterministic_value"

# Records that are prerequisite JS content, not probabilistic programming.
# Tagged `javascript_prerequisite` so downstream can filter them out of
# probabilistic-programming eval splits while keeping them in the inventory.
_JS_PREREQUISITE_IDS = {
    "probmods2/appendix-js-basics",
    "probmods2-exercises/appendix-js-basics",
}


# ---------------------------------------------------------------------------
# Sub-exercise header parsing
# ---------------------------------------------------------------------------

# `## Exercise 1`, `## Exercise 1.1`, `## Exercise 1:`, `## Exercise 1: Title`
_EX_HEADER = re.compile(r"^\s*##\s+Exercise\s+(\d+(?:\.\d+)?)\b", re.MULTILINE)
# `### a)`, `### b)`, `### 1)`  (single letter or digit, closing paren)
_SUB_HEADER = re.compile(r"^\s*###\s+([A-Za-z0-9])\)", re.MULTILINE)
# `### Exercise 1.1`, `### Exercise 2.5` — decimal-numbered sub-exercises.
# Captures just the decimal part (e.g., "1" from "1.1") so it composes with
# the parent top-level exercise tag.
_SUB_HEADER_EX = re.compile(r"^\s*###\s+Exercise\s+\d+\.(\d+)\b", re.MULTILINE)


def _match_sub_header(line):
    """Return the sub-exercise marker (letter or digit) for this line, or None.

    Tries both the `### a)` style and the `### Exercise N.M` style. Returned
    value is lowercased and composes with the current exercise number into
    `<ex>.<sub>`.
    """
    m = _SUB_HEADER.match(line)
    if m:
        return m.group(1).lower()
    m = _SUB_HEADER_EX.match(line)
    if m:
        return m.group(1).lower()
    return None


def tag_blocks_by_subexercise(sections):
    """Walk sections in order; tag each code block with its current
    (exercise_num, sub_letter) derived from preceding prose.

    Returns list of (section_index, subexercise_id, code, classification)
    tuples for every code section. subexercise_id is a string like "1.a"
    or "1" or None when no ## Exercise header has been seen yet.
    """
    current_ex = None
    current_sub = None
    tagged = []

    for idx, section in enumerate(sections):
        if section["type"] == "prose":
            # Walk the prose line-by-line so we track header ordering.
            for line in section["content"].split("\n"):
                m_ex = _EX_HEADER.match(line)
                if m_ex:
                    current_ex = m_ex.group(1)
                    current_sub = None
                    continue
                sub = _match_sub_header(line)
                if sub:
                    current_sub = sub
        elif section["type"] == "code":
            # Synthetic slots inserted by _inject_synthetic_positions carry
            # their intended sub_id directly; honor it so they don't
            # inherit whatever header was last seen in the preceding prose.
            forced = section.get("synthetic_for_sub_id")
            if forced is not None:
                sub_id = forced
            else:
                sub_id = None
                if current_ex is not None:
                    sub_id = (
                        f"{current_ex}.{current_sub}" if current_sub else current_ex
                    )
            tagged.append((idx, sub_id, section["content"], section.get("classification", {})))
    return tagged


# ---------------------------------------------------------------------------
# Entry builders
# ---------------------------------------------------------------------------

def _prompt_with_placeholders(sections):
    """Join sections, replacing code blocks with [BLOCK_N] placeholders."""
    parts = []
    block_idx = 0
    for section in sections:
        if section["type"] == "code":
            parts.append(f"[BLOCK_{block_idx}]")
            block_idx += 1
        else:
            parts.append(section["content"])
    return "\n\n".join(parts)


def build_chapter_entry(record):
    """Chapter-style record: prose + code sections, no solutions.

    Keeps the prompt structurally intact: every code block gets a
    [BLOCK_N] placeholder in order. Only blocks with dataset_fit=True
    contribute to groundtruth; unfit blocks get an empty groundtruth
    list at that position (preserving block index alignment).
    """
    sections = record.get("sections") or []
    code_sections = [s for s in sections if s["type"] == "code"]
    if not code_sections:
        return None

    ground_truth = []
    quality_flags = []
    eval_modes = []
    for s in code_sections:
        cls = s.get("classification", {})
        if cls.get("dataset_fit"):
            code = s["content"]
            flags = list(cls.get("quality_flags", []))
            ground_truth.append([code])
            quality_flags.append(flags)
            eval_modes.append(compute_eval_mode(code, flags))
        else:
            ground_truth.append([])
            quality_flags.append([])
            eval_modes.append("unevaluable")

    return {
        "id": record["id"],
        "source": record["source"],
        "source_file": record.get("source_file", ""),
        "title": record.get("title", ""),
        "category": record.get("category", ""),
        "tags": record.get("tags", []),
        "prompt": _prompt_with_placeholders(sections),
        "ground_truth": ground_truth,
        "quality_flags": quality_flags,
        "eval_modes": eval_modes,
        "num_blocks": len(code_sections),
        "deps": record.get("deps", []),
    }


def _top_level(sub_id):
    """Return just the exercise number ('1.a' → '1', '2' → '2', None → None)."""
    if sub_id is None:
        return None
    return sub_id.split(".")[0]


def _scan_prose_subs(sections):
    """Return the set of sub_ids that appear as headers anywhere in the
    exercise prose (even if no code block follows them). Used to decide
    where a synthetic slot can be anchored vs when a solution sub-exercise
    has no exercise counterpart at all."""
    seen = set()
    current_ex = None
    current_sub = None
    for section in sections:
        if section["type"] != "prose":
            continue
        for line in section["content"].split("\n"):
            m_ex = _EX_HEADER.match(line)
            if m_ex:
                current_ex = m_ex.group(1)
                current_sub = None
                seen.add(current_ex)
                continue
            sub = _match_sub_header(line)
            if sub and current_ex is not None:
                current_sub = sub
                seen.add(f"{current_ex}.{current_sub}")
    return seen


def _inject_synthetic_positions(ex_sections, orphan_sub_ids):
    """Insert synthetic empty code sections into exercise sections so that
    solution-only sub-exercises still get a [BLOCK_N] slot in the prompt.

    Two-tier anchor lookup: first tries to insert after the prose section
    matching the orphan's exact sub_id; if no exact-sub prose exists, falls
    back to inserting after the last prose section under the same top-level
    exercise number.
    """
    if not orphan_sub_ids:
        return list(ex_sections)

    orphan_set = set(orphan_sub_ids)
    top_level_anchors = {_top_level(s) for s in orphan_set}

    current_ex = None
    current_sub = None
    last_exact = {}     # sub_id -> section idx
    last_top_level = {} # exercise_num -> section idx

    def _record(idx_):
        if current_ex is None:
            return
        sid = f"{current_ex}.{current_sub}" if current_sub else current_ex
        if sid in orphan_set:
            last_exact[sid] = idx_
        if current_ex in top_level_anchors:
            last_top_level[current_ex] = idx_

    for idx, section in enumerate(ex_sections):
        if section["type"] == "prose":
            for line in section["content"].split("\n"):
                m_ex = _EX_HEADER.match(line)
                if m_ex:
                    current_ex = m_ex.group(1)
                    current_sub = None
                    _record(idx)
                    continue
                sub = _match_sub_header(line)
                if sub and current_ex is not None:
                    current_sub = sub
                    _record(idx)
        _record(idx)

    insertions = []
    for sub_id in orphan_set:
        if sub_id in last_exact:
            insertions.append((last_exact[sub_id], sub_id))
        elif _top_level(sub_id) in last_top_level:
            insertions.append((last_top_level[_top_level(sub_id)], sub_id))
        # else: no anchor — caller will emit a bonus entry instead.

    out = list(ex_sections)
    for idx, sub_id in sorted(insertions, key=lambda t: -t[0]):
        out.insert(idx + 1, {
            "type": "code",
            "content": "",
            "classification": {
                "category": None,
                "dataset_fit": False,
                "reason": "synthetic_slot_for_solution_only_sub",
                "signals": {},
                "quality_flags": [],
            },
            "synthetic_for_sub_id": sub_id,
        })
    return out


def _extract_sub_prose(sol_sections, sub_id):
    """Pull the prose belonging to a single sub_id out of solution_sections,
    starting at its header and stopping at the next sub- or exercise-header.
    Used when emitting bonus entries for solution-only sub-exercises.
    """
    current_ex = None
    current_sub = None
    buf = []
    started = False
    for section in sol_sections:
        if section["type"] == "prose":
            for line in section["content"].split("\n"):
                m_ex = _EX_HEADER.match(line)
                sub_marker = _match_sub_header(line)
                if m_ex:
                    new_ex = m_ex.group(1)
                    new_sub = None
                    new_id = new_ex
                    if started and new_id != sub_id:
                        return "\n".join(buf).strip()
                    if new_id == sub_id:
                        started = True
                    current_ex = new_ex
                    current_sub = new_sub
                elif sub_marker and current_ex is not None:
                    new_sub = sub_marker
                    new_id = f"{current_ex}.{new_sub}"
                    if started and new_id != sub_id:
                        return "\n".join(buf).strip()
                    if new_id == sub_id:
                        started = True
                    current_sub = new_sub
                if started:
                    buf.append(line)
        elif section["type"] == "code" and started:
            # Break at first code block; prose-only for the prompt.
            return "\n".join(buf).strip()
    return "\n".join(buf).strip()


def _fuzzy_match_solutions(ex_sub_id, sol_tagged):
    """For an exercise sub_id, return solution tuples matching either
    exactly or via top-level fuzzy.

    Fuzzy expansion only fires when the exercise sub_id is top-level (no
    sub-letter). That covers "exercise groups sub-parts under one block,
    solution splits them". It stays additive to exact, so exercise `5`
    pairs with solution `5` AND `5.b` when both exist.

    Sub-letter exercise blocks (e.g. `1.a`) only match exact — they
    refer to a specific sub-part, not the whole exercise.
    """
    if ex_sub_id is None:
        return []
    exact = [t for t in sol_tagged if t[1] == ex_sub_id]
    if "." in ex_sub_id:
        return exact
    fuzzy = [
        t for t in sol_tagged
        if t[1] and t[1] != ex_sub_id and _top_level(t[1]) == ex_sub_id
    ]
    return exact + fuzzy


def build_exercise_entry(record, unpaired_sink, bonus_sink):
    """Pair exercise-side blocks with solution-side blocks.

    Pairing tiers:
      1. Exact sub_id match.
      2. Fuzzy: exercise `N` (no sub-letter) matches all solutions `N.*`.
      3. Synthetic-slot: solution has sub_id with no exercise code, but
         exercise prose has a matching header (exact or top-level).
      4. Solution-only bonus: solution sub_id has no exercise prose at all
         (e.g., solution file adds Exercise 3 that the exercise file skips).
         Emitted as a standalone bonus entry with flag `solution_only_review`.

    Remaining exercise orphans (block with no solution match) are still
    emitted in the main entry with empty groundtruth and logged.
    """
    ex_sections = record.get("sections") or []
    sol_sections = record.get("solution_sections") or []
    code_sections = [s for s in ex_sections if s["type"] == "code"]
    if not code_sections:
        return None

    ex_tagged_raw = tag_blocks_by_subexercise(ex_sections)
    sol_tagged = tag_blocks_by_subexercise(sol_sections)

    # sub_ids that exist somewhere on exercise side, either as a code
    # block (tagged) or as a prose header (scanned).
    ex_code_sub_ids = {t[1] for t in ex_tagged_raw if t[1] is not None}
    ex_prose_sub_ids = _scan_prose_subs(ex_sections)
    ex_all_sub_ids = ex_code_sub_ids | ex_prose_sub_ids
    ex_top_levels = {_top_level(s) for s in ex_all_sub_ids}

    sol_sub_ids_with_code = {t[1] for t in sol_tagged if t[1] is not None}

    # Classify solution sub_ids that aren't already covered by an exercise
    # code block (since those will pair directly via exact match).
    sol_orphan_sub_ids = sol_sub_ids_with_code - ex_code_sub_ids

    # Tier-3 candidates: prose anchor available (exact or top-level fallback).
    # Tier-4 candidates: no exercise prose anchor → bonus entries later.
    tier3 = set()
    tier4 = set()
    for sid in sol_orphan_sub_ids:
        if sid in ex_all_sub_ids or _top_level(sid) in ex_top_levels:
            tier3.add(sid)
        else:
            tier4.add(sid)

    # Strip tier3 when fuzzy match will already cover the case: solution
    # has a sub-letter id like `1.a` AND exercise has a top-level code block
    # at `1`. In that case ex `1` fuzzy-matches all `1.*`; injecting a
    # synthetic slot would create a redundant second position.
    tier3 = {
        sid for sid in tier3
        if not ("." in sid and _top_level(sid) in ex_code_sub_ids)
    }

    if tier3:
        ex_sections = _inject_synthetic_positions(ex_sections, tier3)
        code_sections = [s for s in ex_sections if s["type"] == "code"]

    ex_tagged = tag_blocks_by_subexercise(ex_sections)

    ground_truth = []
    quality_flags = []
    eval_modes = []
    matched_sol_sub_ids = set()
    pairing_notes = []

    for ex_tup in ex_tagged:
        _ex_idx, sub_id, _ex_code, _ex_cls = ex_tup
        matches = _fuzzy_match_solutions(sub_id, sol_tagged)
        fit_codes = []
        fit_flags = []
        for _sidx, _sid, scode, scls in matches:
            if scls.get("dataset_fit"):
                fit_codes.append(scode)
                fit_flags.extend(scls.get("quality_flags", []))
            matched_sol_sub_ids.add(_sid)
        ground_truth.append(fit_codes)
        position_flags = sorted(set(fit_flags))
        quality_flags.append(position_flags)
        # eval_mode from the first alternative's code (all alternatives for a
        # position should share intended behavior; if they diverge, the first
        # is the canonical expected answer).
        eval_modes.append(
            compute_eval_mode(fit_codes[0], position_flags) if fit_codes else "unevaluable"
        )
        if not matches and sub_id is not None:
            unpaired_sink.append({
                "source_file": record.get("source_file", ""),
                "record_id": record.get("id", ""),
                "side": "exercise",
                "section_index": _ex_idx,
                "subexercise_id": sub_id,
                "code": _ex_code,
                "reason": "no_matching_solution_id",
            })
            pairing_notes.append(f"exercise_orphan:{sub_id}")

    # Tier-4 bonus entries.
    bonus_ids = sorted(tier4)
    bonus_entries = []
    for sub_id in bonus_ids:
        sol_blocks = [t for t in sol_tagged if t[1] == sub_id]
        fit_codes = []
        fit_flags = []
        for _sidx, _sid, scode, scls in sol_blocks:
            if scls.get("dataset_fit"):
                fit_codes.append(scode)
                fit_flags.extend(scls.get("quality_flags", []))
            matched_sol_sub_ids.add(sub_id)
        if not fit_codes:
            continue
        prose = _extract_sub_prose(sol_sections, sub_id)
        bonus_flags = sorted(set(fit_flags + ["solution_only_review"]))
        bonus_entry = {
            "id": f"{record['id']}#bonus-{sub_id}",
            "source": record["source"],
            "source_file": record.get("solution_file", ""),
            "solution_file": record.get("solution_file", ""),
            "title": f"{record.get('title', '')} (solution-only: {sub_id})",
            "category": record.get("category", ""),
            "tags": record.get("tags", []) + ["solution_only_review"],
            "prompt": prose + "\n\n[BLOCK_0]",
            "ground_truth": [fit_codes],
            "quality_flags": [bonus_flags],
            "eval_modes": [compute_eval_mode(fit_codes[0], bonus_flags)],
            "num_blocks": 1,
            "deps": record.get("deps", []),
            "pairing_notes": [f"solution_only:{sub_id}"],
        }
        bonus_entries.append(bonus_entry)
        pairing_notes.append(f"solution_bonus:{sub_id}")

    # Log any remaining solution sub_ids that still didn't get matched
    # (shouldn't happen with the new logic, but keep as a safety net).
    for sol_tup in sol_tagged:
        _sidx, sub_id, scode, _scls = sol_tup
        if sub_id is not None and sub_id not in matched_sol_sub_ids:
            unpaired_sink.append({
                "source_file": record.get("solution_file", ""),
                "record_id": record.get("id", ""),
                "side": "solution",
                "section_index": _sidx,
                "subexercise_id": sub_id,
                "code": scode,
                "reason": "no_matching_exercise_id",
            })
            pairing_notes.append(f"solution_orphan:{sub_id}")

    main_entry = {
        "id": record["id"],
        "source": record["source"],
        "source_file": record.get("source_file", ""),
        "solution_file": record.get("solution_file", ""),
        "title": record.get("title", ""),
        "category": record.get("category", ""),
        "tags": record.get("tags", []),
        "prompt": _prompt_with_placeholders(ex_sections),
        "ground_truth": ground_truth,
        "quality_flags": quality_flags,
        "eval_modes": eval_modes,
        "num_blocks": len(code_sections),
        "deps": record.get("deps", []),
        "pairing_notes": pairing_notes,
    }
    bonus_sink.extend(bonus_entries)
    return main_entry


def build_entry(record, unpaired_sink, bonus_sink):
    if record.get("solution_sections"):
        entry = build_exercise_entry(record, unpaired_sink, bonus_sink)
    else:
        entry = build_chapter_entry(record)
    if entry and record.get("id") in _JS_PREREQUISITE_IDS:
        entry["tags"] = sorted(set((entry.get("tags") or []) + ["javascript_prerequisite"]))
    return entry


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    entries = []
    bonus_entries = []
    unpaired_sink = []

    for jsonl_file in sorted(CLASSIFIED_DIR.glob("*.jsonl")):
        if jsonl_file.stem not in _IN_SCOPE_SOURCES:
            print(f"Skipping {jsonl_file.name} (out of scope)")
            continue
        print(f"Processing {jsonl_file.name}...")
        source_count = 0
        with open(jsonl_file) as f:
            for line in f:
                record = json.loads(line)
                entry = build_entry(record, unpaired_sink, bonus_entries)
                if entry:
                    entries.append(entry)
                    source_count += 1
        print(f"  {source_count} entries")
    entries.extend(bonus_entries)

    with open(OUTPUT_FILE, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    with open(UNPAIRED_FILE, "w") as f:
        for item in unpaired_sink:
            f.write(json.dumps(item) + "\n")

    total_blocks = sum(e["num_blocks"] for e in entries)
    fit_blocks = sum(
        sum(1 for gt in e["ground_truth"] if gt)
        for e in entries
    )
    viz_only_positions = sum(
        sum(1 for flags in e["quality_flags"] if "viz_only" in flags)
        for e in entries
    )

    bonus_count = sum(1 for e in entries if "solution_only_review" in (e.get("tags") or []))

    from collections import Counter
    mode_counts = Counter()
    for e in entries:
        for mode in e.get("eval_modes") or []:
            mode_counts[mode] += 1

    print(f"\nDataset built:")
    print(f"  Entries:               {len(entries)}")
    print(f"    (main: {len(entries) - bonus_count}, solution-only bonus: {bonus_count})")
    print(f"  Total block positions: {total_blocks}")
    print(f"  Positions with fit groundtruth: {fit_blocks}")
    print(f"  viz_only positions:    {viz_only_positions}")
    print(f"  eval_mode distribution:")
    for mode, count in mode_counts.most_common():
        print(f"    {mode}: {count}")
    print(f"  Output:                {OUTPUT_FILE}")
    print(f"  Unpaired log ({len(unpaired_sink)} items): {UNPAIRED_FILE}")


if __name__ == "__main__":
    main()
