# Dataset Changes Log

A running record of every change to `data/atomized_v2.jsonl` (the live
eval dataset) and `data/atomized.jsonl` (the v1 untouched original).

Format per entry:
- **atom_id** — what changed, why, evidence.

---

## Drops (atoms removed from v1 → v2 in `scripts/build_atomized_v2.py`)

These 4 atoms execute but their groundtruth depends on `_.now()` (wall-clock
time), so the posterior is hardware/timing-dependent and not reproducible.
TV between two runs of the same code on the same machine = 1.0.

- `probmods2-process-models/ex1`
- `probmods2-process-models/ex3`
- `probmods2-process-models/ex4`
- `probmods2-process-models/ex5`

## Prompt rewrites (v1 → v2 in `scripts/build_atomized_v2.py`)

29 atoms had their `prompt` field rewritten to be self-contained
(removed cross-atom references like "Same setup as before", inlined
data and helper code that earlier subparts had defined).

Affected atoms (hand-rolled in the `REWRITES` dict):
- conditioning/ex5.b, ex5.c
- mixture-models/ex1.a, ex1.b
- occams-razor/ex1.2, ex1.3
- social-cognition/ex2.1, ex2.2, ex2.4, ex2.5
- hierarchical-models/ex2.2, ex2.3, ex2.4, ex3.2
- observing-sequences/ex1.b, ex1.c, ex2.a, ex2.d, ex3.a
- agents-as-programs/ex2.e, ex4.b
- inference-algorithms/ex1.1, ex1.2, ex1.3, ex2.1, ex2.2, ex2.3, ex2.4, ex2.5

`groundtruth_code` was left unchanged for all of these.

---

## Post-build edits (in-place modifications to `data/atomized_v2.jsonl`)

### Round 1 — example-formatting fixes

- **`probmods2-generative-models/ex2.b, ex2.c, ex7.a`** — added `;` to
  the prompt's example code so LLMs don't mirror ASI-ambiguous code
  (`var x = mem(...) \n [a, b]` parses as `var x = mem(...)[a, b]`,
  i.e. subscript, in JS).

- **`probmods2-hierarchical-models/ex3.1`** — prompt previously described
  the data schema but didn't include the actual `data` array; non-Sonnet
  models got `ReferenceError: data is not defined`. Now inlines the
  full 24-row `data` array and the simpler-baseline BDA model.

### Round 2 — return-type ambiguity / GT mismatch

These are dataset bugs where the LLMs' natural reading of the prompt
disagreed with the groundtruth's chosen representation. Fix targets the
prompt unless noted; GT_OUTPUT cache invalidated and re-run only when
groundtruth_code changed.

- **`probmods2-conditioning/ex6.d`** — *groundtruth_code changed*. GT
  returned `true`/`false` but the prompt asked for "vowel vs consonant".
  Updated GT to `return checkVowel(letter) ? 'vowel' : 'consonant'`.
  Re-cached `groundtruth_output`. Now KL=0 / TV=0 across all 9 runs.

- **`probmods2-conditioning/ex1.b, ex1.c, ex1.d`** — prompt now
  explicitly states the return type ("(as a boolean: true=heads,
  false=tails)" and "(return the string 'fair' or 'biased')") to match
  what the GT returns. GT unchanged.

- **`probmods2-agents-as-programs/ex4.a, ex4.b`** — first attempt was a
  no-op (string-replace mismatch on "World:" vs "The world has three
  objects:"). Fixed in Round 3.

- **`probmods2-mixture-models/ex2.a`** — appended a clarification that
  outputs should use 'group1' / 'group2' labels (matching GT) rather
  than 'bonafide' / 'malingerer' (LLM's natural reading from the prose).
  Did not fully address the underlying issue (see Pending Review).

### Round 3 — RSA object representation

- **`probmods2-agents-as-programs/ex4.a, ex4.b`** — prompts now
  explicitly describe objects as records `{shape, color}` matching the
  GT's `meaningPrior`. (Previous in-place attempt missed because the
  string-replace target didn't appear verbatim in the prompts; this
  round rewrote the visible "three objects: ..." clause.)

### Round 4 — disambiguate which variable to query

- **`probmods2-conditioning/ex2.b`** — prompt previously said "construct
  a case where intervening produces a different result from conditioning,
  return distributions over the queried variable, not necessarily cough"
  which is genuinely ambiguous. LLMs picked different variables (e.g.,
  `cold` instead of `lungCancer`), so even a correct probabilistic
  programmer scored TV~0.94. Tightened: explicitly says "intervene on
  `cough = true`, condition on `cough`, return distributions over
  `lungCancer`".

---

## Harness changes (eval/ side)

### Comparator: distribution-key normalization

`_normalize_dist` in `eval/metrics.py` previously keyed the comparison
by the WebPPL-serialized dict key (e.g., `"\"true\""` vs `"true"` for
the same boolean value). Two semantically-identical distributions could
spuriously mismatch when WebPPL formatted their keys differently.

Fix: derive the comparison key from `entry["val"]` via
`json.dumps(val, sort_keys=True)` so it canonicalizes regardless of
WebPPL's outer key formatting. Object values with the same fields in
different orders also now match (handled by `sort_keys=True`).

### Architectural rewrite — explicit ANSWER binding + canonical schema

Two coupled changes:

**Extraction.** The previous protocol was "end your program with the
answer expression"; the harness used a hand-rolled JS-text splitter to
find the last expression boundary. This produced ~5 distinct splitter
bugs (ASI rules around `}`, `)`, `[`-subscript continuation, comments-
in-strings, etc.) over the course of evaluation rounds.

New protocol: programs explicitly bind `var ANSWER = <expr>;` as their
last statement. The harness wraps with a fixed serializer header and
appends `JSON.stringify(__serialize(ANSWER))`. No JS parsing.

System prompt updated to require the binding. All 76 groundtruth_codes
mechanically wrapped. Splitter (`_split_last_expression`, ~80 lines)
deleted from `eval/executor.py`.

**Serialization.** Distributions are now serialized via WebPPL's built-in
`serializeDist(d)` (which exists — was previously hand-rolling an
equivalent). Output format is `{"__kind": "distribution", "probs": [...],
"support": [...]}` — parallel arrays, language-agnostic, suitable for
cross-PPL comparison. The `_normalize_dist` Python helper now reads
parallel arrays directly instead of reconstructing keys from a WebPPL-
flavored `{key: {val, prob}}` dict.

Continuous distributions (Beta, Gaussian, ...) still fall through to a
string-repr fallback (`{__kind: "distribution_continuous", repr}`)
since `serializeDist` throws "Not implemented" on them.

**Cache invalidation.** All 76 cached `groundtruth_output` values were
re-emitted with the canonical schema. Existing LLM generations (created
under the old "bare expression" protocol) won't run under the new
harness — they need to be re-generated.

### Executor: detect silent exits

`execute_webppl` previously marked `success=True` whenever WebPPL's
subprocess exited 0, regardless of stdout content. Some LLM-generated
programs exit 0 with no stdout (e.g. `__ANSWER` ended up undefined
because of a Categorical with too many vs/ps, or a CPS issue inside an
Infer body). The harness then reported `executed=True, answer=None` —
which the comparator handled with a benign "not a distribution" error,
silently classifying the failure as a successful but uncomparable run.

Fix: if exit==0 and stdout is empty, surface as `success=False` with a
clear error message ("program exited 0 but produced no output …").
This caught at least one prior false-positive (`occams-razor/ex1.2`
across 3 runs) where `executed` was being over-counted.

## Dataset expansion — round 1

Four additional datasets extracted from the available WebPPL textbook
ecosystem, using `scripts/extract_atoms.py`. Each prose-with-code block
becomes one atom; the extractor wraps the last expression in
`var ANSWER = ...;`, runs it, and keeps blocks that produce a
distribution / value / sample list.

| dataset | source | atoms | exec rate | KL median / q75 | TV median / q75 |
|---|---|---|---|---|---|
| exercises | probmods2/exercises | 76 | 76/76 | 0 / 0 | 0 / 0 |
| chapters | probmods2/chapters | 121 | 107/121 | 16.9 / 21.9 | 1.0 / 1.0 |
| dippl | dippl/chapters | 24 | 23/24 | 20.1 / 21.9 | 1.0 / 1.0 |
| forestdb | forestdb.org/models | 67 | 58/67 | 21.3 / 22.3 | 1.0 / 1.0 |
| problang | problang/chapters | 50 | 46/50 | 20.7 / 22.1 | 1.0 / 1.0 |

**Observation.** Sonnet 4.6 + primer (no thinking) executes 80-100% of
atoms across the board, but only the *exercises* dataset gives near-zero
TV/KL. Every other dataset has TV q75 = 1.0 — meaning the LLM produces
a runnable program whose output distribution shares no support with the
groundtruth.

**Diagnosis.** Chapter / model atoms are *demonstrations* of concepts,
not graded exercises. The extracted prose context (preceding paragraph)
typically says something like "Consider this binomial example" before
showing one specific parameter setting. The LLM faithfully writes a
binomial example (different parameters), and the comparator correctly
flags it as different. The atoms aren't ill-formed — they just aren't
*tests* in the way exercises are.

**Iteration paths** (none taken yet, awaiting direction):
- Tighten extractor prompts to embed the parameter values explicitly.
- Filter atom candidates to those where the prose clearly pins down a
  unique answer (heuristics: prose mentions "show that the result is
  X", or asks "what is the probability of Y" with a specific value).
- Switch from distribution comparison to *behavioral equivalence*:
  given the LLM's program and the GT program, sample N times from each
  and check that the *marginal distributions* match within tolerance,
  rather than that the support is identical. This handles cases where
  parameter choices differ but the PPL reasoning is correct.

## Pending review (not yet changed)

### Sparse-support / continuous-joint posteriors

When an atom's groundtruth is `Infer(MCMC, ...) → record/continuous`,
each MCMC sample is a unique high-dim point. Two chains (gen vs gt)
produce *disjoint* support sets by construction, so the discrete TV
between them collapses to ~1.0 even when both are correct. KL is at
the saturation cap (~22). My discrete-distribution comparator can't
distinguish "wrong" from "right but stochastically explored differently".

Atoms in this category (TV≥0.5 across most runs, even from the strongest
config):

- `probmods2-mixture-models/ex2.a` — 22 booleans + 2 continuous group rates
- `probmods2-mixture-models/ex1.b` — 2 mem'd prototypes × 3 props each
- `probmods2-occams-razor/ex2.2` — joint (relation, cp, b)
- `probmods2-occams-razor/ex2.3` — record of two arrays of expectations (less affected)
- `probmods2-learning-as-conditional-inference/ex2.1` — `post` is over continuous coinWeight
- `probmods2-hierarchical-models/ex2.3, ex2.4` — beta-on-beta continuous joint
- `probmods2-hierarchical-models/ex3.1, ex3.2` — gaussian joints
- `probmods2-inference-algorithms/ex4.a` — Dirichlet-distributed topic-model joint
- `probmods2-observing-sequences/ex2.c, ex3.a, ex3.b` — sentence-distribution (combinatorial space)
- `probmods2-agents-as-programs/ex2.d, ex2.e` — uniform alpha continuous

Possible fixes (each is a real engineering lift):
1. Reshape the atom to return marginals only (single scalar or finite-support dist).
2. Add a comparator for continuous Distributions that converts the support to samples and runs MMD / Wasserstein / KS.
3. Accept the limitation and mark these atoms "not scoreable by current harness".

**No change made yet; awaiting direction on (1) vs (2) vs (3).**
