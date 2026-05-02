# Dataset Schema: `data/eval_dataset.jsonl`

One JSON object per line. One record per textbook unit (chapter, exercise, or teaching-extra). Sourced from probmods2 only.

## Top-level fields

| Field | Type | Description |
|---|---|---|
| `id` | string | Stable identifier, e.g. `probmods2/generative-models` or `probmods2-exercises/conditioning` |
| `source` | string | `probmods2` for chapters/teaching-extras, `probmods2-exercises` for exercises |
| `source_file` | string | Path to the markdown source under `data/sources/probmods2/` |
| `solution_file` | string\|null | Only on exercise records; path to the paired solution markdown |
| `title` | string | Human-readable title (from frontmatter) |
| `category` | string | Record kind: `tutorial` (chapter), exercise records leave this empty, `teaching_extra` |
| `tags` | list[string] | Includes `javascript_prerequisite` on JS-basics records, `solution_only_review` on bonus entries, and any user-added tags |
| `prompt` | string | The prose the LLM reads, with `[BLOCK_N]` placeholders where code is expected |
| `ground_truth` | list[list[string]] | Per-position list of valid answer alternatives. Outer index = block position (matches `[BLOCK_N]` in prompt). Inner list holds one or more valid code strings. Empty `[]` at a position means no valid groundtruth (block was cat-3/unclassified; not scored) |
| `quality_flags` | list[list[string]] | Per-position flags. Current tags: `viz_only` |
| `eval_modes` | list[string] | Per-position scoring mode (see below) |
| `num_blocks` | int | `len(ground_truth)` = number of `[BLOCK_N]` placeholders in prompt |
| `deps` | list[string] | Runtime dep packages this record needs: any of `probmods-deps`, `probmods-physics`, `probmods-draw`, `probmods-towdata` |
| `pairing_notes` | list[string] | Exercise records only. Notes about how blocks paired with solutions. Values include `exercise_orphan:<sub_id>`, `solution_bonus:<sub_id>` |

## `eval_modes` values

| Mode | Meaning | Metric tier |
|---|---|---|
| `distribution` | Block calls `Infer/Enumerate/MH/MCMC/...`; the viz capture is a full distribution object | KL, cross-entropy, total variation on the captured `getDist()` map |
| `stochastic_value` | Block calls raw stochastic primitives (`flip`, `sample`, `gaussian`, `beta`, ...) without wrapping in `Infer`; return value is a sample | Monte Carlo: run N times with different seeds on both generated and ground-truth, compare the empirical distribution of captures |
| `deterministic_value` | Pure computation, no stochastic primitives | Exact or approximate value match on a single run |
| `side_effect` | Block uses `Draw(...)` / `canvas.*` with no inferable return | String match only (execution produces no signal) |
| `unevaluable` | Empty groundtruth (`dataset_fit=false` at this position) | Skipped |

`eval_modes` is set via static analysis on the first groundtruth alternative's code. When a position has multiple alternatives (exercise records), all alternatives are presumed to share intended behavior; the harness can fall back to any of them during scoring.

## How to read `ground_truth`

```jsonc
"ground_truth": [
  ["var dist = Infer(...)"],                         // position 0: one canonical answer
  [],                                                // position 1: no groundtruth (cat-3)
  ["flip() ? flip(.7) : flip(.1)",                   // position 2: three valid alternatives
   "flip(flip() ? .7 : .1)",
   "flip(.4)"]
]
```

To run the groundtruth program for position N, pick any alternative from `ground_truth[N]` and assemble incrementally: blocks 0..N concatenated. Each block may reference definitions from earlier blocks.

## Example (partial)

```json
{
  "id": "probmods2-exercises/conditioning",
  "source": "probmods2-exercises",
  "prompt": "## Exercise 1: Fair coins and biased coins\n### a)\nI flip a fair coin...\n[BLOCK_0]\n### b)\n...",
  "ground_truth": [["var model = function() { ... }; Math.exp(Infer(...).score('H'))"]],
  "quality_flags": [[]],
  "eval_modes": ["distribution"],
  "num_blocks": 1,
  "deps": [],
  "tags": []
}
```

## Filtering suggestions

- **Strict split**: drop positions where any of: `eval_modes[i] == "unevaluable"`, `"viz_only" in quality_flags[i]`, `"javascript_prerequisite" in tags`.
- **Permissive split**: keep all fit positions; report metrics per-tier.
- **Distribution-only**: keep only positions with `eval_modes[i] == "distribution"`. 275 positions in the current build.

## Sizes (current build)

- 52 entries (21 chapters + 17 exercises + 14 teaching-extras)
- 484 block positions
- 434 with fit groundtruth (1.23 avg alternatives per fit position → 533 unique groundtruth strings)
- eval_mode counts: 275 distribution + 98 stochastic_value + 60 deterministic_value + 50 unevaluable + 1 side_effect
