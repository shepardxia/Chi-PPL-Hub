# Atom dataset review process

When changes are made to atom prompts, GT code, or extraction logic, run a
review pass before declaring the round complete. This file documents that
process so it's not reinvented each iteration.

## Why automated cleanup isn't enough

The auto-extracted datasets (chapters / dippl / forestdb / problang) pull
prose verbatim from source markdown. Pattern-based sanitizers fix specific
known artifacts (Liquid templates, Pandoc citations, image data, etc.) but
miss issues that require *judgment*:

- prose that asks for "the result shown above" but the figure isn't visible
- shape spec says "distribution" but the prose actually describes a single sample
- the GT's ANSWER expression doesn't match what the prose asks for
- prose only makes sense in the context of an earlier chapter not in our preamble
- GT computes one specific tuple but prose is general ("show how the speaker behaves")

These need a reader, not a regex.

## Review rounds

Each modification round (prompt change, GT change, harness change) followed by:

1. **Re-render** the atoms HTML so the modified state is browsable.
2. **Subagent review pass** (manual `Agent` calls, one per dataset, run
   in parallel). Reviews **every atom** in the dataset — no sampling.
   Pin `model: "sonnet"` to keep cost reasonable; reviewers don't need
   Opus-grade reasoning to read prompts and flag issues.
3. **Findings file** at `data/review/<round>-<dataset>.jsonl` with one row
   per flagged atom: `{id, category, severity, finding, suggested_action}`.
4. **Triage**: I read findings, decide which are real issues, which are
   false positives, which to defer. Decisions logged in `DATASET_CHANGES.md`.
5. **Apply fixes** that are clear enough to be safe; defer judgment calls
   to the user.

## Finding categories

Reviewers should classify each issue:

- `prose-broken` — prose references content not in the prompt (figures,
  values, prior context the LLM can't see)
- `prose-vague` — prose is genuinely underspecified about what to compute
- `shape-mismatch-spec` — declared `answer_shape` doesn't match the GT's
  actual return type or the prose's intent
- `gt-mismatch` — GT code's `ANSWER` doesn't match what the prose asks for
- `template-leak` — markdown / template / citation syntax leaked through
- `dead-reference` — prose mentions a function name (e.g. `CRPmem`) that
  isn't defined in our WebPPL distribution
- `other` — anything else worth flagging

Severity:
- `block` — atom is unscoreable in current state
- `warn` — atom has a quality issue but might still produce a useful score
- `info` — minor, FYI

## Anti-patterns to avoid

- **Don't sample.** "Review 25 of 119" is the lazy default we're trying
  to escape. The whole point is full coverage; if it's too slow,
  parallelize across more agents instead of cutting the count.
- **Don't inherit Opus.** Subagents inherit the parent model unless
  pinned. For routine reading-and-flagging, set `model: "sonnet"`.
- **Don't act on subagent findings without reading them.** They are
  recommendations, not commands. Each finding is one judgment call by one
  reader; some will be wrong.
- **Don't fix everything at once.** Fix one category at a time and
  re-review; otherwise it's hard to attribute changes.
- **Don't regenerate every prompt.** Track which atoms actually changed
  and re-run gen only on those (`scripts/rebuild_prompts.py` pattern).
