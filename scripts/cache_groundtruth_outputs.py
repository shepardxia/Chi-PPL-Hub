"""Run each atom's groundtruth_code, capture serialized answer, write back.

For atoms with `answer_shape == "samples"` (top-level samples), runs N
seeded reruns of the groundtruth and caches the list of N answers along
with `n_mc` and `seed` metadata. The harness reads the cache directly so
re-scoring doesn't re-run the groundtruth.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/cache_groundtruth_outputs.py \
        [--dataset data/atomized_v2.jsonl] [--output ...] \
        [--timeout 60] [--seed 42] [--n-mc 200] [--ids ...]
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from eval.config import DEFAULT_N_MC, DEFAULT_SEED, DEFAULT_TIMEOUT
from eval.executor import execute_webppl
from eval.harness import _run_mc
from eval.io import load_jsonl, write_jsonl
from eval.metrics import SHAPE_SAMPLES


def _cache_one(atom, *, timeout, seed, n_mc):
    t0 = time.time()
    if atom.get("answer_shape") == SHAPE_SAMPLES:
        answers, first_error = _run_mc(atom["groundtruth_code"], n_mc, timeout, base_seed=seed)
        non_null = [a for a in answers if a is not None]
        if not non_null:
            atom["groundtruth_output"] = None
            atom["groundtruth_error"] = first_error or "all reruns failed"
            return atom["id"], False, time.time() - t0, atom["groundtruth_error"]
        atom["groundtruth_output"] = non_null
        atom["groundtruth_meta"] = {"n_mc": n_mc, "seed": seed, "n_ok": len(non_null)}
        atom.pop("groundtruth_error", None)
        return atom["id"], True, time.time() - t0, None

    res = execute_webppl(atom["groundtruth_code"], timeout=timeout, random_seed=seed)
    if res.success:
        atom["groundtruth_output"] = res.answer
        atom.pop("groundtruth_error", None)
        return atom["id"], True, time.time() - t0, None
    atom["groundtruth_output"] = None
    atom["groundtruth_error"] = res.error_message
    return atom["id"], False, time.time() - t0, res.error_message


def cache_outputs(atoms, *, timeout=60, seed=42, n_mc=200, workers=4, verbose=True):
    n_ok = n_fail = 0
    failures = []
    t_total = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_cache_one, a, timeout=timeout, seed=seed, n_mc=n_mc): a for a in atoms}
        i = 0
        for fut in as_completed(futs):
            atom_id, ok, dt, err = fut.result()
            i += 1
            if ok:
                n_ok += 1
            else:
                n_fail += 1
                failures.append((atom_id, err))
            if verbose:
                tag = "OK  " if ok else "FAIL"
                print(f"[{i}/{len(atoms)}] {tag} {atom_id:55s} {dt:5.1f}s"
                      + (f"  {(err or '')[:80]}" if err else ""))
    print(f"\nDone in {time.time()-t_total:.1f}s. ok={n_ok} fail={n_fail}")
    for fid, err in failures:
        print(f"  {fid}: {err}")
    return atoms


def main():
    p = argparse.ArgumentParser(description="Cache groundtruth_output for each atom.")
    p.add_argument("--dataset", default="data/atomized_v2.jsonl")
    p.add_argument("--output", default=None,
                   help="Write to this path (default: in-place over --dataset)")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n-mc", type=int, default=DEFAULT_N_MC,
                   help="Reruns for samples-shape atoms")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--ids", nargs="+", default=None,
                   help="Only cache these atoms (others passed through unchanged)")
    args = p.parse_args()

    in_path = Path(args.dataset)
    out_path = Path(args.output) if args.output else in_path

    atoms = load_jsonl(in_path)
    target = [a for a in atoms if (args.ids is None or a["id"] in set(args.ids))]
    print(f"Caching outputs for {len(target)}/{len(atoms)} atoms "
          f"(timeout={args.timeout}s, seed={args.seed}, n_mc={args.n_mc}, workers={args.workers})")

    cache_outputs(target, timeout=args.timeout, seed=args.seed,
                  n_mc=args.n_mc, workers=args.workers, verbose=True)

    write_jsonl(out_path, atoms)
    print(f"\nWrote {len(atoms)} atoms to {out_path}")


if __name__ == "__main__":
    main()
