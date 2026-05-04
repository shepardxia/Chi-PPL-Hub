"""Aggregate all atomized datasets into a single static HTML page.

Reads every dataset listed in `DATASETS`, loads its corresponding run
from `data/eval_runs/<run_name>/scored.jsonl` (if present), computes a
per-atom score bucket, and renders an interactive single-page browser:

- Sticky header with summary stats (total / scoreable / per-bucket)
- Filter pills: by dataset, by score bucket
- Live text search (id / source / shape / prompt fragment)
- Compact row per atom; click to expand prompt + GT + model response

Open the resulting file in any modern browser; no server, no deps.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.io import iter_scored, load_jsonl


# Default registry: which datasets to aggregate, with their run names.
DATASETS: list[tuple[str, str, str]] = [
    # (display label, dataset jsonl, scored.jsonl run name)
    ("exercises", "data/atomized_v2.jsonl",                 "sonnet-46-primer-v3"),
    ("chapters",  "data/atomized_probmods_chapters.jsonl",  "sonnet-46-primer-chapters"),
    ("dippl",     "data/atomized_dippl.jsonl",              "sonnet-46-primer-dippl"),
    ("forestdb",  "data/atomized_forestdb.jsonl",           "sonnet-46-primer-forestdb"),
    ("problang",  "data/atomized_problang.jsonl",           "sonnet-46-primer-problang"),
]


# ---------------------------------------------------------------------------
# Bucketing — the same logic the audit log uses
# ---------------------------------------------------------------------------

BUCKETS = [
    ("TV=0",     "tv=0",   "good",  "TV exactly 0 (matched)"),
    ("TV<.05",   "tv05",   "good",  "TV < 0.05"),
    ("TV<.5",    "tv5",    "warn",  "TV < 0.5"),
    ("TV<1",     "tv1",    "warn",  "0.5 ≤ TV < 1"),
    ("TV=1",     "tveq1",  "bad",   "TV exactly 1 (full disagreement)"),
    ("val+",     "valok",  "good",  "value match (approx)"),
    ("val-",     "valno",  "bad",   "value mismatch"),
    ("shape!",   "shapem", "bad",   "shape mismatch"),
    ("fail",     "fail",   "bad",   "execution failure"),
    ("no-run",   "norun",  "muted", "no run available"),
]


def bucket_for(scored_rec: dict | None) -> str:
    if scored_rec is None:
        return "no-run"
    ev = scored_rec.get("evaluation", {}) or {}
    gen = ev.get("gen", {}) or {}
    if not gen.get("executed"):
        return "fail"
    cmp_node = ev.get("comparison") or {}
    err = cmp_node.get("error", "") or ""
    if cmp_node.get("ok") is False and err.startswith(("not a", "samples must")):
        return "shape!"
    metrics = ev.get("metrics") or {}
    tvs = [v for k, v in metrics.items() if k.endswith("tv")]
    if tvs:
        worst = max(tvs)
        if worst == 0: return "TV=0"
        if worst < 0.05: return "TV<.05"
        if worst < 0.5: return "TV<.5"
        if worst < 1: return "TV<1"
        return "TV=1"
    exacts = [v for k, v in metrics.items() if k.endswith("approx")]
    if exacts:
        return "val+" if all(v == 1.0 for v in exacts) else "val-"
    return "shape!"


def bucket_class(bucket: str) -> str:
    for label, _slug, cls, _desc in BUCKETS:
        if label == bucket:
            return cls
    return ""


def bucket_slug(bucket: str) -> str:
    for label, slug, _cls, _desc in BUCKETS:
        if label == bucket:
            return slug
    return "other"


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

HEAD = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>WebPPL eval atoms</title>
<style>
:root {
  --fg: #1c1c1c; --muted: #7a7a7a; --bg: #f7f7f6; --card: #fff;
  --border: #e3e3e0; --code-bg: #f4f4f2; --accent: #0066cc;
  --good: #18794e; --good-bg: #e6f5ec;
  --warn: #92400e; --warn-bg: #fef3c7;
  --bad: #b91c1c;  --bad-bg: #fde8e8;
  --muted-bg: #ececea;
  --pill-bg: #fff; --pill-active: #0a3d6f; --pill-active-bg: #e3eef9;
}
* { box-sizing: border-box; }
body {
  font: 13.5px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
  color: var(--fg); background: var(--bg); margin: 0;
}
header {
  position: sticky; top: 0; z-index: 20;
  background: var(--card); border-bottom: 1px solid var(--border);
  padding: 10px 20px;
}
header h1 { font-size: 15px; margin: 0 0 6px; font-weight: 600; }
header .summary { color: var(--muted); font-size: 12.5px; margin-bottom: 8px; }
header .summary b { color: var(--fg); font-variant-numeric: tabular-nums; }
.filterbar { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; }
.filterbar .label {
  font-size: 11px; color: var(--muted); text-transform: uppercase;
  letter-spacing: 0.5px; margin-right: 4px;
}
.pill {
  display: inline-flex; align-items: center; gap: 4px;
  background: var(--pill-bg); border: 1px solid var(--border);
  border-radius: 999px; padding: 3px 10px;
  font-size: 12px; cursor: pointer; user-select: none;
  font-variant-numeric: tabular-nums;
}
.pill.active {
  background: var(--pill-active-bg); border-color: var(--pill-active);
  color: var(--pill-active);
}
.pill .count { color: var(--muted); font-size: 11px; }
.pill.active .count { color: var(--pill-active); }
.pill.bad-tone { color: var(--bad); }
.pill.warn-tone { color: var(--warn); }
.pill.good-tone { color: var(--good); }
.pill.bad-tone.active { background: var(--bad-bg); border-color: var(--bad); color: var(--bad); }
.pill.warn-tone.active { background: var(--warn-bg); border-color: var(--warn); color: var(--warn); }
.pill.good-tone.active { background: var(--good-bg); border-color: var(--good); color: var(--good); }
header input[type=search] {
  flex: 1; min-width: 200px;
  padding: 5px 10px; border: 1px solid var(--border); border-radius: 4px;
  font-size: 12.5px; margin-left: 8px;
}
main { padding: 14px 20px 60px; max-width: 1400px; margin: 0 auto; }
.atom {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 5px; margin-bottom: 4px;
}
.atom > summary {
  cursor: pointer; list-style: none; user-select: none;
  display: flex; align-items: center; gap: 8px;
  padding: 7px 12px; font-size: 12.5px;
}
.atom > summary::-webkit-details-marker { display: none; }
.atom > summary::before {
  content: "▸"; color: var(--muted); font-size: 10px; flex-shrink: 0;
}
.atom[open] > summary::before { content: "▾"; }
.atom > summary > .aid {
  font-family: ui-monospace, Menlo, monospace;
  font-weight: 500; color: var(--accent); flex: 1;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.atom > summary > .meta { display: flex; gap: 4px; flex-shrink: 0; }
.badge {
  display: inline-block; background: var(--muted-bg); color: var(--muted);
  font-size: 10.5px; padding: 1px 6px; border-radius: 3px;
  font-variant-numeric: tabular-nums;
}
.badge.ds { background: #eef4fb; color: #134c80; font-weight: 500; }
.badge.shape { background: #fff7df; color: #6b4a00; }
.badge.bucket.good { background: var(--good-bg); color: var(--good); font-weight: 500; }
.badge.bucket.warn { background: var(--warn-bg); color: var(--warn); font-weight: 500; }
.badge.bucket.bad  { background: var(--bad-bg);  color: var(--bad);  font-weight: 500; }
.badge.bucket.muted { background: var(--muted-bg); color: var(--muted); }
.badge.tv { background: #f0f0ee; color: #444; }
.body { padding: 4px 14px 14px 30px; border-top: 1px solid var(--border); }
.section { margin-top: 12px; }
.section-title {
  font-size: 10.5px; font-weight: 600; color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 4px;
}
pre {
  background: var(--code-bg); border: 1px solid var(--border);
  border-radius: 4px; padding: 8px 10px; overflow-x: auto;
  margin: 0; font: 12px/1.5 ui-monospace, Menlo, monospace;
  white-space: pre-wrap; word-wrap: break-word;
}
.prompt-md {
  white-space: pre-wrap;
  font-size: 13px; line-height: 1.5;
}
.prompt-md code {
  background: var(--code-bg); padding: 1px 4px; border-radius: 3px;
  font-family: ui-monospace, Menlo, monospace; font-size: 12px;
}
.gt-output { max-height: 240px; overflow: auto; }
.run-row {
  margin-top: 6px; padding: 6px 8px;
  background: #fcfcfb; border: 1px solid var(--border); border-radius: 4px;
}
.run-row .run-name { font-family: ui-monospace, Menlo, monospace; font-weight: 500; font-size: 11.5px; }
.err { color: var(--bad); font-family: ui-monospace, Menlo, monospace; font-size: 11.5px; white-space: pre-wrap; }
.empty {
  text-align: center; color: var(--muted); padding: 40px 20px;
  font-size: 13px;
}
</style>
</head>
<body>
"""

TAIL = """\
<script>
const atoms = Array.from(document.querySelectorAll('.atom'));
const search = document.getElementById('search');
const dsPills = Array.from(document.querySelectorAll('.pill[data-ds]'));
const bkPills = Array.from(document.querySelectorAll('.pill[data-bk]'));
const visibleCount = document.getElementById('visible-count');
const emptyMsg = document.getElementById('empty-msg');

function activeSet(pills, attr) {
  const set = new Set();
  for (const p of pills) if (p.classList.contains('active')) set.add(p.dataset[attr]);
  return set;
}
function applyFilter() {
  const q = search.value.trim().toLowerCase();
  const ds = activeSet(dsPills, 'ds');
  const bk = activeSet(bkPills, 'bk');
  let visible = 0;
  for (const a of atoms) {
    const okDs = ds.size === 0 || ds.has(a.dataset.ds);
    const okBk = bk.size === 0 || bk.has(a.dataset.bk);
    const okQ = !q || a.dataset.search.indexOf(q) !== -1;
    const show = okDs && okBk && okQ;
    a.style.display = show ? '' : 'none';
    if (show) visible++;
  }
  visibleCount.textContent = visible;
  emptyMsg.style.display = visible === 0 ? '' : 'none';
}

function togglePill(pill) {
  pill.classList.toggle('active');
  applyFilter();
}
for (const p of [...dsPills, ...bkPills]) p.addEventListener('click', () => togglePill(p));
search.addEventListener('input', applyFilter);
search.addEventListener('keydown', e => {
  if (e.key === 'Escape') { search.value = ''; applyFilter(); }
});
</script>
</body>
</html>
"""


def _render_prompt(prompt: str) -> str:
    out = []
    in_fence = False
    fence_buf = []
    for line in prompt.split("\n"):
        if line.startswith("```"):
            if in_fence:
                code = "\n".join(fence_buf)
                out.append('<pre>' + html.escape(code) + '</pre>')
                fence_buf = []
                in_fence = False
            else:
                in_fence = True
            continue
        if in_fence:
            fence_buf.append(line)
        else:
            esc = _inline_code(html.escape(line))
            out.append(esc)
    if in_fence and fence_buf:
        out.append('<pre>' + html.escape("\n".join(fence_buf)) + '</pre>')
    return '<div class="prompt-md">' + "\n".join(out) + "</div>"


def _inline_code(text: str) -> str:
    parts = text.split("`")
    if len(parts) < 3:
        return text
    out = []
    for i, p in enumerate(parts):
        if i % 2 == 0:
            out.append(p)
        else:
            out.append("<code>" + p + "</code>")
    return "".join(out)


def _truncate(s: str, limit: int = 4000) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n\n... ({len(s) - limit} more chars truncated)"


def _shape_label(shape) -> str:
    if isinstance(shape, dict) and "record" in shape:
        keys = ", ".join(shape["record"].keys())
        return f"record({keys})"
    return str(shape)


def _short_metric(scored_rec: dict | None) -> str:
    """One short string summarizing the metric for the row badge."""
    if scored_rec is None:
        return ""
    ev = scored_rec.get("evaluation", {}) or {}
    metrics = ev.get("metrics") or {}
    tvs = [v for k, v in metrics.items() if k.endswith("tv")]
    if tvs:
        worst = max(tvs)
        return f"TV={worst:.2f}"
    exacts = [v for k, v in metrics.items() if k.endswith("approx")]
    if exacts:
        return "✓" if all(v == 1.0 for v in exacts) else "✗"
    return ""


def _render_run(scored_rec: dict | None) -> str:
    if scored_rec is None:
        return ('<div class="section"><div class="section-title">model response</div>'
                '<div class="empty" style="padding:12px">no run available for this atom</div></div>')
    ev = scored_rec.get("evaluation", {}) or {}
    gen = ev.get("gen", {}) or {}
    code = scored_rec.get("generation", {}).get("code", "") or ""
    error = gen.get("error")
    out = ['<div class="section"><div class="section-title">model response</div>']
    if error:
        out.append(f'<div class="run-row"><div class="section-title">error</div>'
                   f'<pre class="err">{html.escape(str(error)[:600])}</pre></div>')
    out.append(f'<div class="run-row">'
               f'<pre>{html.escape(code)}</pre></div>')
    out.append('</div>')
    return "".join(out)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_run(runs_dir: Path, run_name: str) -> dict[str, dict]:
    scored = runs_dir / run_name / "scored.jsonl"
    if not scored.exists():
        return {}
    return {rec["id"]: rec for rec in iter_scored(scored)}


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render(grouped: list[tuple[str, list[dict], dict[str, dict]]]) -> str:
    """grouped: list of (dataset_label, atoms, scored_by_id)."""
    parts = [HEAD]

    # Compute per-dataset and per-bucket counts.
    ds_counts = {}
    bucket_counts = {label: 0 for label, *_ in BUCKETS}
    total = 0
    for ds_label, atoms, scored_by_id in grouped:
        ds_counts[ds_label] = len(atoms)
        total += len(atoms)
        for atom in atoms:
            b = bucket_for(scored_by_id.get(atom["id"]))
            bucket_counts[b] = bucket_counts.get(b, 0) + 1

    scoreable = sum(c for b, c in bucket_counts.items()
                    if b not in ("shape!", "fail", "no-run"))

    # Header
    summary_html = (
        f'<b>{total}</b> atoms · '
        f'<b>{scoreable}</b> scoreable ({scoreable/total:.0%}) · '
        f'showing <b id="visible-count">{total}</b>'
    )
    ds_pills = " ".join(
        f'<span class="pill" data-ds="{html.escape(label)}">'
        f'{html.escape(label)} <span class="count">{ds_counts[label]}</span>'
        f'</span>'
        for label, _ds, _run in DATASETS
        if label in ds_counts
    )
    bk_pills = " ".join(
        f'<span class="pill {cls}-tone" data-bk="{html.escape(label)}" title="{html.escape(desc)}">'
        f'{html.escape(label)} <span class="count">{bucket_counts.get(label, 0)}</span>'
        f'</span>'
        for label, _slug, cls, desc in BUCKETS
        if bucket_counts.get(label, 0) > 0
    )

    parts.append(
        '<header>'
        '<h1>WebPPL eval atoms</h1>'
        f'<div class="summary">{summary_html}</div>'
        '<div class="filterbar">'
        '<span class="label">dataset</span>'
        f'{ds_pills}'
        '<span class="label" style="margin-left:12px">bucket</span>'
        f'{bk_pills}'
        '<input id="search" type="search" placeholder="search id / source / shape / prompt...">'
        '</div>'
        '</header>'
    )

    parts.append('<main>')
    parts.append('<div id="empty-msg" class="empty" style="display:none">No atoms match the current filter.</div>')

    for ds_label, atoms, scored_by_id in grouped:
        for atom in atoms:
            aid = atom["id"]
            shape_str = _shape_label(atom.get("answer_shape", ""))
            src = atom.get("source", "")
            scored_rec = scored_by_id.get(aid)
            bucket = bucket_for(scored_rec)
            metric_short = _short_metric(scored_rec)

            prompt_html = _render_prompt(atom.get("prompt", ""))
            gt_code = html.escape(atom.get("groundtruth_code", ""))
            gt_output = atom.get("groundtruth_output")
            gt_output_str = json.dumps(gt_output, indent=2) if gt_output is not None else "(not cached)"
            gt_output_str = _truncate(gt_output_str, 4000)
            gt_output_html = html.escape(gt_output_str)

            search_haystack = " ".join([
                aid, src, ds_label, shape_str, str(atom.get("eval_mode", "")),
                atom.get("prompt", "")[:1000],
            ]).lower()

            metric_badge = (f'<span class="badge tv">{html.escape(metric_short)}</span>'
                            if metric_short else "")
            bucket_badge = (
                f'<span class="badge bucket {bucket_class(bucket)}">{html.escape(bucket)}</span>'
            )

            parts.append(
                f'<details class="atom" '
                f'data-ds="{html.escape(ds_label)}" '
                f'data-bk="{html.escape(bucket)}" '
                f'data-search="{html.escape(search_haystack, quote=True)}">'
                f'<summary>'
                f'<span class="aid">{html.escape(aid)}</span>'
                f'<span class="meta">'
                f'<span class="badge ds">{html.escape(ds_label)}</span>'
                f'<span class="badge shape">{html.escape(shape_str)}</span>'
                f'{metric_badge}{bucket_badge}'
                f'</span>'
                f'</summary>'
                f'<div class="body">'
                f'<div class="section"><div class="section-title">prompt</div>{prompt_html}</div>'
                f'<div class="section"><div class="section-title">groundtruth code</div>'
                f'<pre>{gt_code}</pre></div>'
                f'<div class="section"><div class="section-title">groundtruth output</div>'
                f'<pre class="gt-output">{gt_output_html}</pre></div>'
                f'{_render_run(scored_rec)}'
                f'</div>'
                f'</details>'
            )

    parts.append('</main>')
    parts.append(TAIL)
    return "".join(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="data/eval_runs")
    p.add_argument("--output", default="data/atoms.html")
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    grouped: list[tuple[str, list[dict], dict[str, dict]]] = []
    total_atoms = 0
    for label, ds_path, run_name in DATASETS:
        ds = Path(ds_path)
        if not ds.exists():
            continue
        atoms = load_jsonl(ds)
        scored_by_id = _load_run(runs_dir, run_name)
        grouped.append((label, atoms, scored_by_id))
        total_atoms += len(atoms)

    out = Path(args.output)
    out.write_text(render(grouped))
    print(f"Wrote {total_atoms} atoms across {len(grouped)} datasets -> {out} ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
