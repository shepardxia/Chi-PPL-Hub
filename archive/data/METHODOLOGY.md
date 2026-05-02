# probmods2 WebPPL Dataset

52 textbook docs. LLM fills `[BLOCK_N]` placeholders, we execute both its code and groundtruth, compare distributions / values.

## Pipeline

```
scrape → clean → classify → build_dataset → eval_dataset.jsonl
                                                   │
                                 ┌─────────────────┴─────────────────┐
                                 ▼                                   ▼
                          generate (Stage 1)                    score (Stage 2)
                          LLM → generations.jsonl        generations + gt → scored.jsonl
                          (one call per block,           (executes both sides,
                           groundtruth as prior context)  compares by eval_mode)
```

Stages are decoupled: generation costs API money, scoring is free to re-run.

## Classifier (per block)

Every code block runs in headless WebPPL and gets bucketed:

| cat | meaning | dataset_fit |
|---|---|---|
| **1** | parses and runs to a defined output | ✅ |
| **2** | author-intended failure (timeout / runtime error by design) | ❌ |
| **3** | incomplete: parse error (`...` placeholders) **or** `// your code here` / `// edit this line` comment **or** undefined reference to an unknown symbol | ❌ |
| **4** | runnable once a known-but-unwired dep is loaded (0 in current build) | ❌ |
| **unclassified** | runtime error we don't recognize — kept as an investigation queue | ❌ |

Blocks run **incrementally** (block N assembled with blocks 0..N). If a block fails, splice it out and retest downstream so one broken block doesn't cascade a false-fail into every subsequent block.

## Scoring by eval_mode

Each block gets assigned a mode at build time (static analysis of the groundtruth code):

| eval_mode | Signal in code | Metric |
|---|---|---|
| `distribution` | `Infer / MCMC / Enumerate` | **KL**, TVD on captured distribution |
| `stochastic_value` | raw `flip / sample / gaussian` (no Infer) | Monte Carlo empirical TV (N seeded runs) |
| `deterministic_value` | pure computation | exact / approximate value match |
| `side_effect` | `canvas.* / Draw(...)` only | string match only |
| `unevaluable` | empty groundtruth (cat-3/unclassified) | skipped |

String exact-match + Jaccard similarity are always reported as a baseline floor.

## Design decisions

- **Per-block with groundtruth priors**: we ask the LLM for one block at a time, with blocks 0..N-1 already filled in using groundtruth. Isolates block N's score from cascade failures in its earlier outputs. Tests "can the model write block N given correct context".
- **Alternative groundtruths**: exercise positions can have several valid answers; score against each, take the best.
- **Math.random seeded** via a node `-r` preload so uuid helpers are deterministic across MC runs. WebPPL's own `--random-seed` doesn't cover JS Math.random.
- **Cascade-recovery in classifier**: blocks run incrementally; when a block fails, splice it out and retest downstream so one broken block doesn't poison the rest.

## Current numbers

| | entries | block positions | fit groundtruth |
|---|---|---|---|
| chapters | 21 | 298 | 273 |
| exercises | 17 | 123 (+ 140 solution) | 199 |
| teaching_extras | 14 | 39 | 28 |
| **total** | **52** | **484** | **432 (89%)** |

eval_modes over fit positions: **273 distribution · 98 stochastic · 60 deterministic · 1 side_effect · 52 unevaluable**.

## Example entry

`probmods2-exercises/conditioning` — 14 blocks, all `distribution` eval mode. First two positions:

### Prompt

> ## Exercise 1: Fair coins and biased coins
>
> **### a)**  I flip a fair coin. What is the probability that it lands heads?
>
> **\[BLOCK_0\]**
>
> **### b)**  I also have a biased coin, with P(heads)=0.9. I hand you one of the coins (either biased or fair) chosen uniformly randomly without telling you which. You flip it three times. Given that first two coin flips landed on heads, what is the posterior distribution for the next flip?
>
> **\[BLOCK_1\]**
>
> *(continues through BLOCK_13)*

### Groundtruths

```js
// ground_truth[0]
var model = function() { return flip() ? "H" : "T" }
var logProb = Infer({method:'enumerate'}, model).score('H');
Math.exp(logProb);
```

```js
// ground_truth[1]
var flipCoin = function(coinType) {
  return coinType == "fair" ? flip() : flip(0.9);
}
var model = function() {
  var coinType = flip() ? "fair" : "biased";
  var flip1 = flipCoin(coinType);
  var flip2 = flipCoin(coinType);
  var flip3 = flipCoin(coinType);
  condition(flip1 && flip2);
  return flip3;
}
viz.table(Infer({method:'enumerate'}, model));
```

Each prose sub-part ("### a)", "### b)", …) pairs with one `[BLOCK_N]` — specific inference question → specific WebPPL program. The LLM's block-N output is scored by plugging it into `ground_truth[0..N-1]`, running, and comparing the captured `Infer` distribution to groundtruth's (KL, TV).





