# Dataset Source Coverage

Tracks what's been scraped into the pipeline from every source directory, so we don't silently omit content.

## probmods2 (https://github.com/probmods/probmods2)

Clone at `data/sources/probmods2/`.

| Directory | Files | With code | Status | Output file |
|---|---|---|---|---|
| `chapters/` | 22 | 21 | scraped + cleaned + classified | `probmods_chapters.jsonl` |
| `exercises/` + `solutions/` (paired) | 17 ex / 16 sol | 17 | scraped + cleaned + classified | `probmods_exercises.jsonl` |
| `teaching_extras/` | 16 | 14 | scraped + cleaned + classified | `probmods_teaching_extras.jsonl` |
| `readings/` | 15 | **0** | skipped — pure reading-list prose (`Read this paper...`), no WebPPL | — |

### Known prompt-quality issues (entries to revisit)

Some entries have prose that assumes context the dataset entry doesn't include (references to "the chapter", multi-variant asks for a single block slot, etc). Flag here as we find them:

| entry_id | issue |
|---|---|
| `probmods2-exercises/05.1-sequential-decisions` | prose refers to "our line-world example from the chapter" (not in entry); Exercise 1 has three sub-parts (a/b/c) that each ask to modify `[BLOCK_0]` differently — single slot, ambiguous target |

### Classification summary

| Source | cat 1 (runs) | cat 2 (intentional fail) | cat 3 (incomplete) | cat 4 (unwired dep) | unclassified | viz_only |
|---|---|---|---|---|---|---|
| chapters | 273 | 3 | 16 | 0 | 6 | 114 |
| exercises (starter) | 68 | 0 | 55 | 0 | 0 | 33 |
| exercises (solutions) | 131 | 3 | 3 | 0 | 3 | 33 |
| teaching_extras | 28 | 0 | 10 | 0 | 1 | 4 |

*cat-1 / cat-2 counts on chapters have small run-to-run jitter (±2 blocks) due to the cascade-walk algorithm's splice-and-retry order under timing variation. The identity of the jitterable blocks doesn't change, just whether they're labeled cat-1 vs cat-2 on any given run.*

Cat-3 detection includes a pedagogical-placeholder rule: blocks containing `// Your code here`, `// edit this line`, `// fill in`, or `// TODO` are forced to cat-3 regardless of runtime outcome. This catches starter templates whose "meaningless" bodies happen to execute without error (e.g., an empty `Infer` callback — runs but produces nonsense).

**Unclassified-runtime diagnostic queue** (10 blocks, `category=null`, needs human review):

**Genuine "book code doesn't run in this WebPPL" (6 blocks)**:
- `145-non-parametric-models` [#13, #15] — `gaussian(0, 1)` positional call rejected by current WebPPL (expects `{mu, sigma}` object form)
- `145-non-parametric-models` [#21] — `gamma sample underflow` numerical edge case
- `bda-tow` [#19] — MCMC drift kernel proposes values outside Uniform prior support
- `learning-as-conditional-inference` solution [#7] — `Beta({a:10, b:10})` returns "Not implemented" from WebPPL

**Cascade-failures, would recover if upstream fixed (4 blocks)**:
- `bda-tow` [#21, #23] — depend on `editor.get('bda_bcm')` that #19 never set
- `inference-algorithms` solutions [#12, #14] — depend on `editor.get("posterior")` that upstream cat-3 MCMC block didn't set

**Shim limitation (1 block)**:
- `ToE` [#9] — Draw shim can't fully satisfy WebPPL CPS-transform (error `address.split is not a function`); this is not a missing-method issue but a deeper integration issue with headless Draw. Stays unclassified.

**Final dataset**: **52 entries**, **484 block positions**, **432 fit groundtruth positions** (89.3%), 166 viz_only, **0 unpaired**. Also carries per-position `eval_modes` (distribution / stochastic_value / deterministic_value / side_effect / unevaluable) — see `DATASET.md` and `EVAL.md`.

Scope restriction: `build_dataset.py` has `_IN_SCOPE_SOURCES` = {chapters, exercises, teaching_extras}. `forestdb.jsonl` and `probmods_examples.jsonl` (WebPPL-repo examples) exist in `data/classified/` but are filtered out at build time.

JS-prerequisite tag: 2 entries (`probmods2/appendix-js-basics`, `probmods2-exercises/appendix-js-basics`, 48 blocks combined) carry tag `javascript_prerequisite` so eval splits can filter them out. They're pure JS basics (`3 + 3`, string concat), not probabilistic programming content.

Scraper-level cleanups made this session:
- Triple-backtick fences with non-WebPPL lang tags (or JSON-looking content starting with `{`/`[`) now demote to prose. Previously JSON example-output blocks got captured as cat-3 parse-error pseudo-code.
- `### Exercise N.N` sub-exercise headers (used in inference-algorithms) now recognized by the sub_id walker. Composes correctly with parent `## Exercise N` tag.
- Exercise records now pass through the same HTML-comment stripping as chapters did all along. Previously `solution_sections` contained author-hidden `<!-- -->` content (e.g., the inference-algorithms "bonus exercise") which correctly vanishes now.

Pairing tiers used in `build_dataset.py`:
1. Exact sub_id match (exercise `1.a` ↔ solution `1.a`)
2. Fuzzy top-level (exercise `1` matches all `1.*` solutions, additively with any exact `1`)
3. Synthetic slot injection (solution has sub_id with no matching exercise code, but exercise prose has the header — exact or top-level fallback)
4. Bonus entries (solution sub_id has no exercise anchor at all)

Headless runtime shims available (via `eval/deps/`):
- `probmods-physics` — box2d + physics.js (real lib, DOM-less sandbox)
- `probmods-draw` — Draw(w,h,visible) canvas shim (no-op)
- `probmods-deps`, `probmods-towdata` — MCMC_Callbacks, tow data

**Skipped files (no code blocks after scrape):**
- `chapters/`: 1 file (the chapter index `index.md`)
- `teaching_extras/`: `IrrationalAgents.md`, `index.md`
- `exercises/`: `appendix-js-basics.md` has no paired solution file

## Other sources (currently out of scope)

| Source | Location | Status |
|---|---|---|
| WebPPL interpreter examples (`webppl/examples/*.wppl`) | `data/sources/webppl/examples/` | scraped as `probmods_examples.jsonl` (mis-named), not probmods2 textbook. Decision pending on inclusion. |
| ForestDB tutorial | `data/sources/forestdb/` | scraped as `forestdb.jsonl`, not classified. Deferred. |

## Pipeline stages

Each source flows through: `scrape_*.py` → `data/raw/*.jsonl` → `clean_scraped.py` → `data/cleaned/*.jsonl` → `classify.py` → `data/classified/*.jsonl` → `build_dataset.py` → `data/eval_dataset.jsonl`.

## How to extend

When adding a new source directory:
1. Write a scraper at `scripts/scrape_<source>.py` emitting to `data/raw/<source>.jsonl`. For fenced markdown, reuse `parse_sections` + `parse_frontmatter` from `scripts/scrape_probmods_chapters.py`.
2. If `source` label maps to an existing strategy in `scripts/clean_scraped.py:PARSE_STRATEGY`, no code change needed. Otherwise add a strategy entry.
3. Run `clean_scraped.py` (picks up anything in `data/raw/`) and `classify.py` (picks up anything in `data/cleaned/`).
4. Update this file.
