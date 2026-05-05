# DIPPL Corpus: Overall Curation Notes

## Patterns Across the 7 Files

**Algorithm-walkthrough vs. problem pattern:** Most of the corpus is structured as a running tutorial — building up an implementation step by step, showing intermediate states of the code, and evolving it across many blocks. This makes chunking harder: most individual blocks are partial programs mid-construction, not standalone runnable problems. The practical atom-worthy blocks are almost always the final, complete version in a sequence.

**JavaScript vs. WebPPL split:** Files 03-enumeration, 05-particlefilter, and 06-mcmc contain substantial sections written in JavaScript (not WebPPL), marked `// language: javascript`. These implement the inference algorithms themselves (ExploreWeighted, LikelihoodWeighting, SimpleParticleFilter, MH). These blocks are tutorial content explaining the mechanics of inference, not model problems. They were skipped as atoms because: (a) they require mutation and `var` reassignment that is not idiomatic WebPPL; (b) an LM without source access cannot be asked to reproduce them in any well-defined way; (c) they are not "problems" but pedagogy.

**CPS transform sections:** Files 03-enumeration and 06-mcmc have blocks marked `// static` showing transformation rules as pseudocode patterns, not runnable programs. These are skipped.

**`///fold:` pattern:** Many blocks use a `///fold:` pragma to hide boilerplate in the DIPPL web UI. When assembling multi-block atoms, the fold sections in later blocks duplicate definitions from earlier blocks. The pipeline must strip or deduplicate these definitions; they create duplicate `var` declarations that will fail in most JS/WebPPL runtimes. This is flagged per atom.

**Old API in hmm.md:** The `hmm.md` file uses an older WebPPL API: `Enumerate(model, maxExecs)`, `ParticleFilter(model, numSamples)`, and `print(...)` wrapping the result. Modern WebPPL uses `Infer({method: 'enumerate', model: ...})` etc. The emitted atoms rewrite to modern syntax in their prompts; the GT code assembled from hmm.md blocks will use the old API. The pipeline should be aware that hmm.md GT may need adaptation.

**Canvas/Draw API:** Several blocks in 05-particlefilter use `Draw(...)`, `canvas.line(...)`, `loadImage(...)`. These require a browser-side rendering environment and cannot be run in a headless WebPPL evaluator. Atoms from that file focus on the pure probabilistic model (positions array, etc.) and exclude the canvas operations.

**`viz` and `viz.table` wrapping:** Many source blocks end with `viz(dist)` or `viz.table(Infer({...}))` rather than a bare expression. The wrap heuristic that binds the last expression to `ANSWER` will target the `viz(...)` call. Either the pipeline must strip `viz`/`viz.table` wrappers, or prompts must instruct the LM to return the distribution object directly (not pass it to viz). All prompts in this corpus take the latter approach: they instruct the LM to assign to `ANSWER` directly.

## Files Skipped Entirely

**01-introduction.md** — 26 lines of prose introduction with no code blocks whatsoever. Nothing to atomize.

## Judgment Calls

**02-webppl blocks 0, 1, 2 skipped:** Block 0 (`foo` function demo) is pure syntax illustration, too abstract. Blocks 1 and 2 are single-line `sample(Bernoulli(...))` and `viz(Bernoulli(...))` — too trivial (LM writes one line). Block 3 (geometric) is the smallest meaningful task.

**03-enumeration CPS tutorial skipped mostly:** The CPS sections (blocks 2–8) are tutorials building intuition. Only blocks 5 and 7 yield meaningful "translate to CPS" tasks because the CPS transformation has a concrete verifiable answer. Blocks 9–11 (JavaScript coroutine implementation) are algorithm walkthroughs, not problems. Block 15 (enumeration strategies) is the key practical atom from this file.

**04-factorseq blocks 4–6 skipped as standalone:** The binomial-with-factor-moved-up sequence (blocks 4, 5, 6) is a teaching illustration showing factor reordering. The models themselves are minor variants of each other. Instead, atom-5 (blocks 13+14) bundles the original and heuristic-factors versions as a single comparison task, which is more substantive.

**05-particlefilter JS implementation blocks skipped:** Blocks 7–14 are all JavaScript implementations of likelihood weighting and particle filtering. These are the algorithmic heart of the chapter but are not problems — they are reference implementations. Block 6 is the only standalone WebPPL HMM model in this file (pre-JS section). Blocks 15–16 (semi-Markov walk) could also be atoms but require `secondLast` which may not be a built-in, making them fragile.

**06-mcmc blocks 0, 3, 6 skipped:** These are the full JavaScript MH implementations. Block 2 (MHacceptProb) and blocks 4–5 (named _sample and named MHacceptProb) are also implementation internals. Only block 1 (the skewBinomial WebPPL model) and its two inference methods (exact and MCMC) yield clean atoms.

**Same source block used for two atoms (06-mcmc):** Block 1 generates both `mcmc/atom-1` (exact enumeration) and `mcmc/atom-2` (MCMC sampling). This is intentional — the same model under two different inference methods gives materially different outputs, so they are distinct tasks. `source_block_indices` overlap is allowed per the contract.

**hmm.md atom-2 duplicate-var risk:** Assembling blocks 0+1+5+8 creates duplicate declarations of `hmm_recur` and `hmm` (defined in both blocks 5 and 8). The pipeline must use only block 8's definitions. Recommended: if the pipeline concatenates blocks naively, this will break. The atom notes document this. An alternative would be to use only blocks [0, 1, 8] and rely on the prompt describing the `trueobs` definition inline, since block 8 references it from its fold.

**Canvas atoms from 05-particlefilter:** Atoms 2–4 (Gaussian random walk, semi-Markov walk, Gaussian mixture) are purely generative models where the "output" is a list of positions/points. The source blocks have canvas draw calls that are irrelevant to the model itself. Prompts focus only on the array-returning computation.

## Contract Suggestions

**Wrapping `viz(...)` calls:** The pipeline wrap heuristic needs a specific rule: if the last top-level expression is a `viz(...)` or `viz.table(...)` call, strip the `viz`/`viz.table` wrapper and bind the inner argument to `ANSWER`. Currently, prompts work around this by telling the LM to write `var ANSWER = dist;` explicitly — but the GT assembled from source will still have `viz(dist)` as the last expression.

**`///fold:` stripping:** The pipeline should strip or skip everything between `///fold:` and `///` when assembling blocks, to avoid duplicate var declarations across adjacent blocks.

**Old WebPPL API (hmm.md):** `hmm.md` uses `Enumerate(model, N)` and `ParticleFilter(model, N)` which may not be valid in the evaluation runtime. The pipeline should either shim these or flag hmm.md GT as needing a compatibility layer.

**`// language: javascript` blocks:** The pipeline should detect `// language: javascript` at the start of a block and either skip it or treat it as a JavaScript fragment (not WebPPL). Many blocks in 03-enumeration, 05-particlefilter, and 06-mcmc have this tag.

**`secondLast` built-in status:** The `semiMarkovWalk` function in 05-particlefilter uses `secondLast(prevStates)` but this function does not appear to be defined in the block or its fold. The pipeline should flag undefined-variable errors; the `secondLast` function may be a global provided by the DIPPL runtime but not standard WebPPL.
