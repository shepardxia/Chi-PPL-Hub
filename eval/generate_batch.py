"""Submit atom generations as an Anthropic Message Batch (50% discount, async).

Each batch request has the atom's `id` as `custom_id`. After submission the
batch is polled until ended; results are streamed back and written to the
same JSONL format as eval.generate (one record per atom plus a summary
trailer), so eval.score reads them transparently.

Usage:
    PYTHONPATH=. .venv/bin/python -m eval.generate_batch \\
        --dataset data/atomized_v2.jsonl \\
        --model claude-sonnet-4-6 \\
        --output data/eval_runs/<run-id>/generations.jsonl \\
        [--no-primer]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from anthropic import Anthropic

from eval.io import load_jsonl
from eval.prompt import PROMPT_VERSION, format_messages, parse_response


DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.0
POLL_INTERVAL = 30
POLL_TIMEOUT = 3600  # 1 hour


def _atom_to_cid(atom_id: str) -> str:
    """Encode atom id to satisfy ^[a-zA-Z0-9_-]{1,64}$."""
    return atom_id.replace("/", "__").replace(".", "-")


def submit_batch(client: Anthropic, requests: list[dict]) -> str:
    batch = client.messages.batches.create(requests=requests)
    return batch.id


def wait_for_batch(client: Anthropic, batch_id: str, *,
                   poll_interval: int = POLL_INTERVAL,
                   timeout: int = POLL_TIMEOUT,
                   verbose: bool = True) -> object:
    t0 = time.time()
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        elapsed = time.time() - t0
        status = batch.processing_status
        counts = batch.request_counts
        if verbose:
            print(f"  [{elapsed:5.0f}s] status={status} "
                  f"processing={counts.processing} succeeded={counts.succeeded} "
                  f"errored={counts.errored} canceled={counts.canceled} "
                  f"expired={counts.expired}",
                  flush=True)
        if status == "ended":
            return batch
        if elapsed > timeout:
            raise TimeoutError(f"Batch {batch_id} not done after {timeout}s")
        time.sleep(poll_interval)


def collect_results(client: Anthropic, batch_id: str) -> dict:
    """Returns {custom_id: {ok, code, raw, warnings, meta}}."""
    out = {}
    for line in client.messages.batches.results(batch_id):
        cid = line.custom_id
        result = line.result
        if result.type == "succeeded":
            msg = result.message
            text_parts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
            text = "\n".join(text_parts)
            code, warnings = parse_response(text)
            out[cid] = {
                "ok": True,
                "code": code,
                "raw": text,
                "warnings": warnings,
                "meta": {
                    "stop_reason": msg.stop_reason,
                    "input_tokens": msg.usage.input_tokens,
                    "output_tokens": msg.usage.output_tokens,
                    "cache_creation_input_tokens": getattr(
                        msg.usage, "cache_creation_input_tokens", 0) or 0,
                    "cache_read_input_tokens": getattr(
                        msg.usage, "cache_read_input_tokens", 0) or 0,
                },
            }
        else:
            err = getattr(result, "error", None)
            err_msg = str(err) if err is not None else f"result_type={result.type}"
            out[cid] = {
                "ok": False,
                "code": "",
                "raw": "",
                "warnings": [f"batch error: {err_msg}"],
                "meta": {"error": err_msg, "result_type": result.type},
            }
    return out


def run_batch_generation(
    dataset_path: Path,
    output_path: Path,
    *,
    model: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    with_primer: bool = True,
    thinking_budget: int | None = None,
    max_atoms: int | None = None,
    ids: list[str] | None = None,
    poll_interval: int = POLL_INTERVAL,
    timeout: int = POLL_TIMEOUT,
):
    client = Anthropic()

    atoms = load_jsonl(dataset_path)
    if ids:
        atoms = [a for a in atoms if a["id"] in set(ids)]
    if max_atoms is not None:
        atoms = atoms[:max_atoms]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ""
    if not with_primer:
        suffix += "-noprimer"
    if thinking_budget is not None:
        suffix += f"-think{thinking_budget}"
    adapter_name = f"anthropic-batch:{model}{suffix}"

    requests = []
    for atom in atoms:
        msgs = format_messages(atom, with_primer=with_primer)
        system_msg = msgs[0]["content"]
        user_msgs = [{"role": m["role"], "content": m["content"]} for m in msgs[1:]]
        # System prompt is identical across all atoms in a batch — cache it.
        # Cache write happens on the first request; subsequent reads are
        # billed at ~10% of input rate.
        system_blocks = [{
            "type": "text",
            "text": system_msg,
            "cache_control": {"type": "ephemeral"},
        }]
        params = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_blocks,
            "messages": user_msgs,
        }
        if thinking_budget is not None:
            # Extended thinking requires temperature=1; ensure max_tokens
            # leaves room for both thinking and the final answer.
            params["temperature"] = 1.0
            params["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
        else:
            params["temperature"] = temperature
        requests.append({
            "custom_id": _atom_to_cid(atom["id"]),
            "params": params,
        })

    print(f"[batch] {adapter_name}: submitting {len(requests)} requests")
    t_start = time.time()
    batch_id = submit_batch(client, requests)
    print(f"[batch] id={batch_id}")

    print(f"[batch] polling every {poll_interval}s (timeout {timeout}s)")
    wait_for_batch(client, batch_id, poll_interval=poll_interval, timeout=timeout)

    print(f"[batch] streaming results")
    results = collect_results(client, batch_id)
    total_runtime = round(time.time() - t_start, 2)

    total_in = total_out = total_cache_create = total_cache_read = n_ok = 0
    with open(output_path, "w") as f:
        for atom in atoms:
            cid = _atom_to_cid(atom["id"])
            r = results.get(cid)
            if r is None:
                rec = {
                    "id": atom["id"],
                    "prompt_version": PROMPT_VERSION,
                    "adapter": {"name": adapter_name, "error": "missing from batch results"},
                    "generation": {"code": "", "raw_response": "",
                                   "parse_warnings": ["missing from batch"]},
                    "runtime_sec": 0.0,
                }
            else:
                meta = r["meta"]
                if r["ok"]:
                    n_ok += 1
                    total_in += meta.get("input_tokens") or 0
                    total_out += meta.get("output_tokens") or 0
                    total_cache_create += meta.get("cache_creation_input_tokens") or 0
                    total_cache_read += meta.get("cache_read_input_tokens") or 0
                rec = {
                    "id": atom["id"],
                    "prompt_version": PROMPT_VERSION,
                    "adapter": {
                        "name": adapter_name,
                        "model": model,
                        "with_primer": with_primer,
                        "batch_id": batch_id,
                        **meta,
                    },
                    "generation": {
                        "code": r["code"],
                        "raw_response": r["raw"],
                        "parse_warnings": r["warnings"],
                    },
                    "runtime_sec": 0.0,
                }
            f.write(json.dumps(rec) + "\n")

        summary = {
            "summary": True,
            "adapter": adapter_name,
            "prompt_version": PROMPT_VERSION,
            "n_atoms": len(atoms),
            "n_succeeded": n_ok,
            "batch_id": batch_id,
            "total_runtime_sec": total_runtime,
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "total_cache_creation_tokens": total_cache_create,
            "total_cache_read_tokens": total_cache_read,
        }
        f.write(json.dumps(summary) + "\n")

    return summary


def main():
    p = argparse.ArgumentParser(description="Anthropic batch generation (50% off).")
    p.add_argument("--dataset", default="data/atomized_v2.jsonl")
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--no-primer", action="store_true")
    p.add_argument("--thinking-budget", type=int, default=None,
                   help="Enable extended thinking with this token budget (e.g. 4000)")
    p.add_argument("--max-atoms", type=int, default=None)
    p.add_argument("--ids", nargs="+", default=None)
    p.add_argument("--poll-interval", type=int, default=POLL_INTERVAL)
    p.add_argument("--timeout", type=int, default=POLL_TIMEOUT)
    args = p.parse_args()

    # When thinking is enabled, bump max_tokens to leave room for both
    # the thinking trace and the answer (default args.max_tokens is the
    # answer budget; total = answer + thinking).
    max_tokens = args.max_tokens
    if args.thinking_budget is not None:
        max_tokens = args.max_tokens + args.thinking_budget

    summary = run_batch_generation(
        dataset_path=Path(args.dataset),
        output_path=Path(args.output),
        model=args.model,
        max_tokens=max_tokens,
        temperature=args.temperature,
        with_primer=not args.no_primer,
        thinking_budget=args.thinking_budget,
        max_atoms=args.max_atoms,
        ids=args.ids,
        poll_interval=args.poll_interval,
        timeout=args.timeout,
    )

    print()
    print("=" * 60)
    print("BATCH GENERATION DONE")
    print("=" * 60)
    print(f"  Adapter:       {summary['adapter']}")
    print(f"  Atoms:         {summary['n_atoms']}")
    print(f"  Succeeded:     {summary['n_succeeded']}")
    print(f"  Batch id:      {summary['batch_id']}")
    print(f"  Wall clock:      {summary['total_runtime_sec']:.1f}s")
    print(f"  Input tokens:    {summary['total_input_tokens']:,}")
    print(f"  Output tokens:   {summary['total_output_tokens']:,}")
    print(f"  Cache writes:    {summary['total_cache_creation_tokens']:,}")
    print(f"  Cache reads:     {summary['total_cache_read_tokens']:,}")


if __name__ == "__main__":
    main()
