"""Shared helper: append atoms to data/atomized.jsonl.

The first script (conditioning, run via write_atomized.py) overwrites;
subsequent batch scripts append.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.io import write_jsonl


OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "atomized.jsonl"


def write_atoms(atoms, *, append=True):
    write_jsonl(OUT_PATH, atoms, append=append)
    print(f"wrote {len(atoms)} atoms to {OUT_PATH} "
          f"({'appended' if append else 'overwrote'})", file=sys.stderr)
