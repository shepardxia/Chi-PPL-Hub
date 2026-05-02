"""Re-extract atom prompts from source markdown, preserving GT and the
existing "given code" preamble.

Use this for atoms whose prose section is broken (e.g., entire prompt is
"[image]" because the original markdown had a single huge inline base64
image with no surrounding prose paragraphs in the local context window
of `truncate_prose`). The fix re-extracts prose from the source file
and substitutes only the prose portion of the prompt — the "given code"
section stays exactly as it was, since we can't reconstruct whether the
original used cumulative or standalone preamble without re-executing.

Default policy: only rebuild prompts whose prose section is shorter
than `--min-prose` chars after stripping `[image]` markers. This keeps
us from clobbering atoms whose prompts are intentionally brief but
still informative.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from scripts.extract_atoms import (
    SOURCES, split_blocks, truncate_prose, looks_like_full_program,
)

_PROSE_BOUNDARY = "\n\nThe following code is given"
_PROSE_FALLBACK_BOUNDARY = "\n\nWrite a WebPPL program"
_HEADER_RE = re.compile(r'^From the .+? "[^"]+":\n\n', re.DOTALL)


def split_prompt(prompt: str) -> tuple[str, str, str]:
    """Split an existing prompt into (header, prose, tail).

    `tail` includes the "given code" section (if any) and the trailing
    "Write a WebPPL program ..." instructions.
    """
    h = _HEADER_RE.match(prompt)
    header = h.group(0) if h else ""
    body = prompt[len(header):]
    if _PROSE_BOUNDARY in body:
        prose, tail = body.split(_PROSE_BOUNDARY, 1)
        return header, prose.strip(), _PROSE_BOUNDARY + tail
    if _PROSE_FALLBACK_BOUNDARY in body:
        prose, tail = body.split(_PROSE_FALLBACK_BOUNDARY, 1)
        return header, prose.strip(), _PROSE_FALLBACK_BOUNDARY + tail
    return header, body.strip(), ""


def parse_block_idx(atom_id: str, id_prefix: str) -> tuple[str, int] | None:
    pat = re.compile(rf"^{re.escape(id_prefix)}-(.+)/block-(\d+)$")
    m = pat.match(atom_id)
    if not m:
        return None
    return m.group(1), int(m.group(2))


def real_prose_len(prose: str) -> int:
    """Length of prose ignoring [image] markers and pure-whitespace lines."""
    cleaned = re.sub(r"\[image\]", "", prose)
    return len(cleaned.strip())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=list(SOURCES.keys()), required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--min-prose", type=int, default=80,
                   help="Only rebuild atoms whose existing prose section "
                        "(excluding [image] markers) is shorter than this.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cfg = SOURCES[args.source]
    src_dir = Path(cfg["dir"])

    rows = [json.loads(line) for line in open(args.dataset)]
    by_chapter: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        parsed = parse_block_idx(r["id"], cfg["id_prefix"])
        if parsed is None:
            continue
        stem, _ = parsed
        by_chapter[stem].append(r)

    n_changed = 0
    n_skipped_thick = 0
    n_atoms = 0
    samples = []

    for stem, atoms in by_chapter.items():
        path = src_dir / f"{stem}.md"
        if not path.exists():
            continue
        text = path.read_text()
        blocks = list(split_blocks(text))

        candidates = {}
        for i, (prose, code) in enumerate(blocks):
            if looks_like_full_program(code):
                candidates[i] = prose

        for atom in atoms:
            n_atoms += 1
            parsed = parse_block_idx(atom["id"], cfg["id_prefix"])
            idx = parsed[1]
            if idx not in candidates:
                continue

            header, current_prose, tail = split_prompt(atom["prompt"])
            if real_prose_len(current_prose) >= args.min_prose:
                n_skipped_thick += 1
                continue

            new_prose = truncate_prose(candidates[idx])
            if real_prose_len(new_prose) <= real_prose_len(current_prose):
                # No improvement available; don't churn.
                continue
            new_prompt = f"{header}{new_prose}{tail}"
            if len(samples) < 5:
                samples.append((atom["id"], real_prose_len(current_prose),
                                real_prose_len(new_prose)))
            atom["prompt"] = new_prompt
            n_changed += 1

    print(f"atoms scanned: {n_atoms}  thin-prose-rebuilt: {n_changed}  "
          f"skipped-thick: {n_skipped_thick}")
    for aid, ol, nl in samples:
        print(f"  {aid}: prose {ol} -> {nl} chars")

    if not args.dry_run:
        with open(args.dataset, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
