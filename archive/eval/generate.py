"""Stage 1: atom -> generation JSONL.

Generation costs LLM API calls; scoring (eval.score) is free to re-run.

Output: one record per atom with the generated code, raw response,
adapter metadata, and runtime. Plus a trailing summary line.

Usage:
    PYTHONPATH=. .venv/bin/python -m eval.generate \\
        --dataset data/atomized_v2.jsonl \\
        --adapter anthropic \\
        --model claude-haiku-4-5-20251001 \\
        --output data/eval_runs/<run-id>/generations.jsonl \\
        [--no-primer] [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

from eval.adapters import ADAPTER_REGISTRY
from eval.io import load_jsonl
from eval.prompt import PROMPT_VERSION


DEFAULT_WORKERS = 8


def _generate_one(adapter, atom):
    t0 = time.time()
    result = adapter.generate(atom)
    runtime = round(time.time() - t0, 3)
    return {
        "id": atom["id"],
        "prompt_version": PROMPT_VERSION,
        "adapter": {"name": adapter.name, **(result.api_metadata or {})},
        "generation": {
            "code": result.code,
            "raw_response": result.raw_response,
            "parse_warnings": result.parse_warnings,
        },
        "runtime_sec": runtime,
    }


def run_generation(
    dataset_path: Path,
    adapter,
    output_path: Path,
    *,
    max_atoms: int | None = None,
    ids: list[str] | None = None,
    workers: int = DEFAULT_WORKERS,
):
    atoms = load_jsonl(dataset_path)
    if ids:
        wanted = set(ids)
        atoms = [a for a in atoms if a["id"] in wanted]
    if max_atoms is not None:
        atoms = atoms[:max_atoms]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_lock = Lock()
    completed = [0]
    total_in = total_out = 0
    t_start = time.time()

    with open(output_path, "w") as out_f:
        def run_and_write(atom):
            nonlocal total_in, total_out
            rec = _generate_one(adapter, atom)
            meta = rec["adapter"]
            tin = meta.get("input_tokens") or 0
            tout = meta.get("output_tokens") or 0
            with write_lock:
                out_f.write(json.dumps(rec) + "\n")
                out_f.flush()
                completed[0] += 1
                warns = len(rec["generation"]["parse_warnings"])
                print(
                    f"[{completed[0]}/{len(atoms)}] {rec['id']:55s} "
                    f"warnings={warns}, in={tin}, out={tout}, "
                    f"{rec['runtime_sec']:.1f}s",
                    flush=True,
                )
            return tin, tout

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_and_write, a) for a in atoms]
            for fut in futures:
                try:
                    tin, tout = fut.result()
                    total_in += tin
                    total_out += tout
                except Exception as e:
                    print(f"  ATOM FAILED: {type(e).__name__}: {e}")

        total_runtime = round(time.time() - t_start, 2)
        summary = {
            "summary": True,
            "adapter": adapter.name,
            "prompt_version": PROMPT_VERSION,
            "n_atoms": len(atoms),
            "workers": workers,
            "total_runtime_sec": total_runtime,
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
        }
        out_f.write(json.dumps(summary) + "\n")

    return summary


def _resolve_adapter(name: str, **kwargs):
    if name not in ADAPTER_REGISTRY:
        raise SystemExit(f"Unknown adapter '{name}'. Registered: {sorted(ADAPTER_REGISTRY)}")
    return ADAPTER_REGISTRY[name](**kwargs)


def main():
    p = argparse.ArgumentParser(description="Stage 1: atom -> generation JSONL.")
    p.add_argument("--dataset", default="data/atomized_v2.jsonl")
    p.add_argument("--adapter", default="anthropic",
                   help=f"One of {sorted(ADAPTER_REGISTRY)}")
    p.add_argument("--output", required=True)
    p.add_argument("--max-atoms", type=int, default=None)
    p.add_argument("--ids", nargs="+", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--no-primer", action="store_true",
                   help="Disable WebPPL primer in system prompt")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = p.parse_args()

    kwargs = {}
    if args.model is not None:
        kwargs["model"] = args.model
    if args.no_primer:
        kwargs["with_primer"] = False

    adapter = _resolve_adapter(args.adapter, **kwargs)
    summary = run_generation(
        dataset_path=Path(args.dataset),
        adapter=adapter,
        output_path=Path(args.output),
        max_atoms=args.max_atoms,
        ids=args.ids,
        workers=args.workers,
    )

    print()
    print("=" * 60)
    print("GENERATION DONE")
    print("=" * 60)
    print(f"  Adapter:       {summary['adapter']}")
    print(f"  Prompt ver:    {summary['prompt_version']}")
    print(f"  Atoms:         {summary['n_atoms']}")
    print(f"  Wall clock:    {summary['total_runtime_sec']:.1f}s")
    print(f"  Input tokens:  {summary['total_input_tokens']:,}")
    print(f"  Output tokens: {summary['total_output_tokens']:,}")


if __name__ == "__main__":
    main()
