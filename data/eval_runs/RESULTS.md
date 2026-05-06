# v3 LLM eval: 8-config sweep on `atomized_v2.jsonl` (76 atoms)
Each model + config × the full v2 dataset. Generation via Anthropic batch API; scoring via `eval/score.py` (seed=42, n_mc=20). Eval runs at `data/eval_runs/<config>-v3/`.

## Bucket counts per config (out of 76 atoms)

| config | TV=0 | TV<.05 | TV<.5 | TV<1 | TV=1 | val+ | val- | shape! | fail |
|---|---|---|---|---|---|---|---|---|---|
| `haiku-45-noprimer` | 29 | 6 | 5 | 6 | 2 | 4 | 2 | 1 | 21 |
| `haiku-45-primer` | 31 | 6 | 7 | 6 | 4 | 2 | 3 | 1 | 16 |
| `haiku-45-think-noprimer` | 25 | 7 | 5 | 5 | 3 | 3 | 1 | 1 | 26 |
| `haiku-45-think-primer` | 31 | 10 | 5 | 9 | 2 | 3 | 2 | 1 | 13 |
| `sonnet-46-noprimer` | 36 | 7 | 7 | 7 | 3 | 5 | 1 | 1 | 9 |
| `sonnet-46-primer` | 41 | 7 | 8 | 10 | 3 | 4 | 2 | 1 | 0 |
| `sonnet-46-think-noprimer` | 34 | 9 | 8 | 9 | 4 | 4 | 1 | 0 | 7 |
| `sonnet-46-think-primer` | 40 | 8 | 9 | 8 | 4 | 3 | 3 | 1 | 0 |

A "value" atom is one whose answer is a single deterministic number / string /
boolean or a small structured tuple — anything where exact equality (or
near-equality) is the right comparison metric, as opposed to a probability
distribution.

## Aggregate metrics per config

| config | exec | mean TV | mean KL | n_kl | n_tv | value-exact | n_value |
|---|---:|---:|---:|---:|---:|---:|---:|
| `haiku-45-noprimer` | 72% | 0.126 | 0.865 | 59 | 66 | 42.9% | 7 |
| `haiku-45-primer` | 79% | 0.139 | 1.286 | 68 | 74 | 40.0% | 5 |
| `haiku-45-think-noprimer` | 66% | 0.138 | 1.329 | 56 | 62 | 75.0% | 4 |
| `haiku-45-think-primer` | 83% | 0.134 | 0.859 | 74 | 80 | 60.0% | 5 |
| `sonnet-46-noprimer` | 88% | 0.130 | 1.529 | 74 | 81 | 57.1% | 7 |
| `sonnet-46-primer` | 100% | 0.140 | 1.614 | 86 | 93 | 42.9% | 7 |
| `sonnet-46-think-noprimer` | 91% | 0.148 | 2.086 | 80 | 88 | 66.7% | 6 |
| `sonnet-46-think-primer` | 100% | 0.135 | 1.595 | 86 | 93 | 42.9% | 7 |

- `exec`: fraction of atoms whose generated program ran without error
- `mean TV`: mean total-variation distance across all distribution-shape atoms (lower is better; record atoms contribute one TV per sub-distribution)
- `mean KL`: mean KL divergence (LM || GT) across same set; sensitive to outliers (a single "wrong family" atom can spike KL)
- `n_kl` / `n_tv`: how many atom-level metrics contributed to the means
- `value-exact`: fraction of value-shape atoms with exact-match output (over `n_value` atoms)
