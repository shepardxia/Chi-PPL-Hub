# Evaluation Methodology

How to score LLM-generated code against the probmods2 dataset.

## Task definition

Given an entry's `prompt` (prose with `[BLOCK_N]` placeholders), the LLM must produce a list of code strings, one per placeholder, such that the substituted document is a valid WebPPL program. The harness substitutes generated code into the prompt and runs blocks incrementally: block N is executed in the context of blocks 0..N concatenated.

**Input** to the harness: an entry + a list of generated code strings (length must equal `num_blocks`).
**Output**: per-block scores + aggregate metrics.

## Per-block scoring by `eval_mode`

### `distribution`

Both generated and groundtruth code produce a WebPPL distribution (via `Infer`, `Enumerate`, `MH`, `MCMC`, etc.) captured as a `{value → probability}` map. Compute:

- **KL(gen ‖ gt)** — forward KL divergence, smoothed with ε = 1e-10 for missing-support terms.
- **Cross-entropy** — H(gen, gt).
- **Total variation** — 0.5·∑|p − q|, bounded in [0, 1].

Authoritative tier. Lower = better (except cross-entropy which is directional).

### `stochastic_value`

Both sides produce stochastic scalar/array values (raw `flip()`, `sample()`, `gaussian()`). A single run is noisy; we compare empirical distributions via Monte Carlo:

1. Run generated code N times with seeds 42, 43, …, 42+N-1.
2. Run groundtruth N times with the same seeds.
3. Collect captured values from each run, stringified for hashability.
4. Build empirical frequency distributions and compute total variation.

Default N = 30. Increase for precision; decrease for speed. Scoring is by TV on the empirical distribution.

### `deterministic_value`

Pure computation: single run with seed 42 is sufficient. Compute:

- **Exact match** on all captured values (pairwise equal).
- **Approximate match** for numeric values (relative tolerance 5% by default).

Report the fraction of captures that match.

### `side_effect`

No capturable output. Fall back to string-level comparison only (see below).

### `unevaluable`

Empty groundtruth (cat-3 / unclassified blocks). Skipped entirely. Reported in the "coverage" section of aggregate output, not in metric averages.

## String-level metrics (always computed)

Regardless of `eval_mode`, every fit position is scored on:

- **Exact match** — after normalizing comments, whitespace, trailing semicolons.
- **Jaccard token similarity** — tokenize normalized code, compute |gen ∩ gt| / |gen ∪ gt|.

These are the universal baseline. Useful for reporting, weak as a primary metric (syntactically-equivalent programs can look very different).

## Multi-alternative groundtruth

An exercise position may have multiple valid answer codes in `ground_truth[i]` (from `### a)` having "here are three equivalent formulations" solutions). Scoring rule: compute the score against each alternative, take the best:

- Distribution: minimum KL / TV
- Value: any exact match → pass; else max approximate-match rate
- String: max similarity

## Incremental execution

Block N is always executed with blocks 0..N-1 prepended (using the GENERATED versions of those earlier blocks, not the groundtruth). This means an error in block 0 can poison downstream evaluation. The harness reports this cleanly: if block N's execution fails, record the error and continue without cascading a false "distribution comparison."

Groundtruth execution uses the groundtruth blocks for the same prefix — so groundtruth always runs in its own valid context.

## Filtering and aggregate metrics

Recommended splits:

- **Distribution-only** (strict): 275 positions. Only `eval_mode == "distribution"`. Clearest single-number metric (mean KL) but narrowest coverage.
- **All evaluable**: 434 fit positions. Report per-tier (distribution/stochastic/deterministic/side_effect). No unified "score" — each tier has its own scale.
- **All positions**: 484 positions. Includes `unevaluable` in coverage counts so "percent scored" reflects the honest denominator.

Additional flag-based filters:

- Exclude `quality_flags[i] == ["viz_only"]` unless you're evaluating rendering code specifically.
- Exclude entries with `javascript_prerequisite` in `tags` (pure JS basics, not probabilistic programming).
- Exclude entries with `solution_only_review` in `tags` unless those bonus entries have been manually validated.

## Typical report shape

```
Entries evaluated: 52
Block positions: 484
  Scored:       434 (89.7% of all)
  Unevaluable:   50 (10.3% — empty groundtruth)

By eval_mode:
  distribution:        275 scored | mean KL = 0.12 | TV = 0.06
  stochastic_value:     98 scored | mean TV = 0.18
  deterministic_value:  60 scored | exact 73% | approx 84%
  side_effect:           1 scored | string similarity = 0.82

String baseline (all 434):
  Exact match: 18%
  Mean Jaccard: 0.71

Execution health:
  gen blocks executed: 412 / 434 (94.9%)
  gt  blocks executed: 434 / 434 (100.0%)
```

## Running the harness

```
# Sanity check: score groundtruth against itself. Should yield ~perfect scores.
python eval/harness.py --dataset data/eval_dataset.jsonl --max-entries 5

# Future: plug in an LLM (not yet implemented):
python eval/harness.py --model <llm-name> --dataset data/eval_dataset.jsonl
```

## What's NOT scored

- Blocks with `eval_mode == "unevaluable"` — empty groundtruth.
- Prompts themselves — we never ask the model to reproduce prose.
- Chain-of-thought or reasoning output — only the extracted code per block.
- Execution artifacts beyond captures (stdout, timing, memory).

## Known limitations

**Non-seeded randomness** (fixed). WebPPL's `--random-seed` doesn't cover JS `Math.random`. The executor now preloads `eval/deps/probmods-seeded-random/preload.js` (via `node -r <path>`) which replaces `Math.random` with a Mulberry32 PRNG seeded from `WEBPPL_MATH_RANDOM_SEED` (set to the same seed as `--random-seed`). This is loaded BEFORE WebPPL's modules initialize, so every reference to `Math.random` (in book code or dep packages) gets the deterministic override.

**`webppl-timeit` / wall-clock timing** (not fixable here). `timeit()` measures real wall-clock time via `Date.now()` / `hrtime`. Its outputs are fundamentally non-deterministic across runs. Blocks whose captures include timeit results (e.g., `sequential-decisions` prefix chains) will show score < 100% on gt-vs-gt even though the logic is identical. If this matters for your eval, stub `timeit` to return a fixed value per call order, or exclude `timeit`-containing records.

**Cascade failures.** If block N fails to execute, downstream blocks N+1..M run with a broken prefix and may fail for unrelated reasons. The harness reports per-block execution success, but cascade-induced scores are noisy. For authoritative reporting, inspect per-block `gen_executed` / `gt_executed` flags.

**Single-alternative eval_mode.** `eval_modes` is computed from the first alternative's code. If alternatives use different primitives (one has Infer, another doesn't), the scoring tier may not fit all alternatives equally well. Currently this doesn't happen in the probmods2 corpus but could for user-added data.

**Sanity check expectations.** Running harness with `generated = groundtruth[first_alt]` should yield ≈ 0 divergence on clean entries. Deviations above zero flag blocks with non-seeded randomness or numerical nondeterminism.
