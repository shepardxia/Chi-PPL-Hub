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
.code-pair { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.code-pair > div > .section-title { margin-bottom: 4px; }
@media (max-width: 900px) { .code-pair { grid-template-columns: 1fr; } }
pre {
  background: var(--code-bg); border: 1px solid var(--border);
  border-radius: 4px; padding: 8px 10px; overflow-x: auto;
  margin: 0; font: 12px/1.5 ui-monospace, Menlo, monospace;
  white-space: pre-wrap; word-wrap: break-word;
}
.dist-viz {
  background: var(--code-bg); border: 1px solid var(--border);
  border-radius: 4px; padding: 6px 8px;
  font: 11px ui-monospace, Menlo, monospace;
}
.dist-viz svg { display: block; }
.dist-row { display: flex; align-items: center; gap: 6px; margin: 1px 0; }
.dist-row .lab {
  flex: 0 0 38%; max-width: 38%;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  color: #444;
}
.dist-row .bar-track {
  flex: 1; height: 9px; background: #e8e8e6; border-radius: 2px; overflow: hidden;
}
.dist-row .bar-fill { height: 100%; background: #6a8caf; }
.dist-row .pv {
  flex: 0 0 50px; text-align: right; color: var(--muted);
  font-variant-numeric: tabular-nums;
}
.atom.is-target { box-shadow: 0 0 0 2px var(--accent); }
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
const main = document.querySelector('main');
const search = document.getElementById('search');
const sortSel = document.getElementById('sort');
const dsPills = Array.from(document.querySelectorAll('.pill[data-ds]'));
const bkPills = Array.from(document.querySelectorAll('.pill[data-bk]'));
const visibleCount = document.getElementById('visible-count');
const emptyMsg = document.getElementById('empty-msg');

const BUCKET_ORDER = {
  'fail': 0, 'shape!': 1, 'TV=1': 2, 'val-': 3,
  'TV<1': 4, 'TV<.5': 5, 'TV<.05': 6, 'TV=0': 7, 'val+': 8, 'no-run': 9,
};

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

function applySort() {
  const mode = sortSel.value;
  const sorted = [...atoms];
  if (mode === 'bucket') {
    sorted.sort((a, b) => (BUCKET_ORDER[a.dataset.bk] ?? 99) - (BUCKET_ORDER[b.dataset.bk] ?? 99)
                          || a.dataset.aid.localeCompare(b.dataset.aid));
  } else if (mode === 'tv-desc' || mode === 'tv-asc') {
    const dir = mode === 'tv-desc' ? -1 : 1;
    sorted.sort((a, b) => {
      const av = a.dataset.tv === '' ? null : parseFloat(a.dataset.tv);
      const bv = b.dataset.tv === '' ? null : parseFloat(b.dataset.tv);
      if (av === null && bv === null) return a.dataset.aid.localeCompare(b.dataset.aid);
      if (av === null) return 1;
      if (bv === null) return -1;
      return dir * (av - bv) || a.dataset.aid.localeCompare(b.dataset.aid);
    });
  } else {
    sorted.sort((a, b) => a.dataset.aid.localeCompare(b.dataset.aid));
  }
  for (const a of sorted) main.appendChild(a);
}

function togglePill(pill) {
  pill.classList.toggle('active');
  applyFilter();
}
for (const p of [...dsPills, ...bkPills]) p.addEventListener('click', () => togglePill(p));
search.addEventListener('input', applyFilter);
search.addEventListener('keydown', e => {
  if (e.key === 'Escape') { e.target.blur(); search.value = ''; applyFilter(); }
});
sortSel.addEventListener('change', applySort);

// URL hash deep linking: open + scroll to atom on load and on hashchange.
function focusFromHash() {
  const h = location.hash.slice(1);
  if (!h) return;
  const t = document.getElementById(h);
  if (!t) return;
  for (const a of atoms) a.classList.remove('is-target');
  t.classList.add('is-target');
  t.open = true;
  t.scrollIntoView({block: 'start'});
}
window.addEventListener('hashchange', focusFromHash);

// Update hash when an atom is opened (so URL is shareable).
for (const a of atoms) {
  a.addEventListener('toggle', () => {
    if (a.open) history.replaceState(null, '', '#' + a.id);
  });
}

// Keyboard shortcuts: / to focus search, j/k to navigate visible atoms, esc to clear.
function visibleAtoms() { return atoms.filter(a => a.style.display !== 'none'); }
function currentIndex(list) {
  for (let i = 0; i < list.length; i++) if (list[i].classList.contains('is-target')) return i;
  return -1;
}
function focusAtom(a) {
  for (const x of atoms) x.classList.remove('is-target');
  a.classList.add('is-target');
  a.scrollIntoView({block: 'center', behavior: 'smooth'});
  history.replaceState(null, '', '#' + a.id);
}
document.addEventListener('keydown', e => {
  if (e.target === search || e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') {
    if (e.key === 'Escape') e.target.blur();
    return;
  }
  if (e.key === '/') { e.preventDefault(); search.focus(); search.select(); return; }
  if (e.key === 'Escape') {
    document.querySelectorAll('.atom.is-target').forEach(a => a.classList.remove('is-target'));
    return;
  }
  const list = visibleAtoms();
  if (list.length === 0) return;
  if (e.key === 'j' || e.key === 'ArrowDown') {
    e.preventDefault();
    const i = currentIndex(list);
    focusAtom(list[Math.min(i + 1, list.length - 1)] || list[0]);
  } else if (e.key === 'k' || e.key === 'ArrowUp') {
    e.preventDefault();
    const i = currentIndex(list);
    focusAtom(list[Math.max(i - 1, 0)] || list[0]);
  } else if (e.key === 'Enter') {
    const i = currentIndex(list);
    if (i >= 0) list[i].open = !list[i].open;
  }
});

focusFromHash();
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


def _safe_id_for_hash(aid: str) -> str:
    """DOM ids — strip characters that break URL hash + querySelector."""
    return aid.replace("/", "--").replace(" ", "_")


def _render_distribution_viz(d: dict, max_rows: int = 12) -> str | None:
    """Render a {__kind:distribution, support, probs} as a horizontal bar chart.

    Returns None if `d` isn't a recognizable distribution.
    """
    if not isinstance(d, dict) or d.get("__kind") != "distribution":
        return None
    support = d.get("support") or []
    probs = d.get("probs") or []
    if not support or len(support) != len(probs):
        return None
    pairs = sorted(zip(support, probs), key=lambda kv: -kv[1])
    truncated = len(pairs) > max_rows
    pairs = pairs[:max_rows]
    pmax = max((p for _, p in pairs), default=1.0) or 1.0
    rows = []
    for v, p in pairs:
        if isinstance(v, (dict, list)):
            label = json.dumps(v, sort_keys=True)
        else:
            label = str(v)
        if len(label) > 40:
            label = label[:37] + "…"
        bar_pct = max(1.0, 100.0 * p / pmax) if p > 0 else 0
        rows.append(
            f'<div class="dist-row">'
            f'<span class="lab" title="{html.escape(str(v))}">{html.escape(label)}</span>'
            f'<span class="bar-track"><span class="bar-fill" style="width:{bar_pct:.1f}%"></span></span>'
            f'<span class="pv">{p:.4f}</span>'
            f'</div>'
        )
    suffix = (f'<div class="dist-row"><span class="lab" style="color:var(--muted)">'
              f'… {len(d["support"]) - max_rows} more</span></div>') if truncated else ""
    return '<div class="dist-viz">' + "".join(rows) + suffix + '</div>'


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


def _gen_code(scored_rec: dict | None) -> str | None:
    if scored_rec is None:
        return None
    return scored_rec.get("generation", {}).get("code", "") or None


def _gen_error(scored_rec: dict | None) -> str | None:
    if scored_rec is None:
        return None
    return (scored_rec.get("evaluation", {}).get("gen", {}) or {}).get("error")


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
        '<h1>WebPPL eval atoms <span class="small" style="color:var(--muted);font-weight:400;font-size:11px">'
        '· press <code>/</code> to search, <code>j</code>/<code>k</code> to navigate, <code>esc</code> to clear</span></h1>'
        f'<div class="summary">{summary_html}</div>'
        '<div class="filterbar">'
        '<span class="label">dataset</span>'
        f'{ds_pills}'
        '<span class="label" style="margin-left:12px">bucket</span>'
        f'{bk_pills}'
        '<select id="sort" style="margin-left:8px;padding:4px 8px;border:1px solid var(--border);'
        'border-radius:4px;font-size:12px;background:#fff">'
        '<option value="aid">sort: id</option>'
        '<option value="bucket">sort: bucket (worst→best)</option>'
        '<option value="tv-desc">sort: TV high→low</option>'
        '<option value="tv-asc">sort: TV low→high</option>'
        '</select>'
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
            gt_code = atom.get("groundtruth_code", "")
            gt_output = atom.get("groundtruth_output")
            gt_output_str = json.dumps(gt_output, indent=2) if gt_output is not None else "(not cached)"
            gt_output_str = _truncate(gt_output_str, 4000)

            gen_code = _gen_code(scored_rec)
            gen_err = _gen_error(scored_rec)

            search_haystack = " ".join([
                aid, src, ds_label, shape_str, str(atom.get("eval_mode", "")),
                atom.get("prompt", "")[:1000],
            ]).lower()

            metric_badge = (f'<span class="badge tv">{html.escape(metric_short)}</span>'
                            if metric_short else "")
            bucket_badge = (
                f'<span class="badge bucket {bucket_class(bucket)}">{html.escape(bucket)}</span>'
            )

            # Sort key: TV value (or sentinel) for sort-by-tv option
            tv_for_sort = ""
            if scored_rec is not None:
                metrics = (scored_rec.get("evaluation", {}) or {}).get("metrics") or {}
                tvs = [v for k, v in metrics.items() if k.endswith("tv")]
                if tvs:
                    tv_for_sort = f"{max(tvs):.6f}"

            # Side-by-side GT vs gen code
            if gen_code is not None:
                code_section = (
                    f'<div class="section"><div class="code-pair">'
                    f'<div><div class="section-title">groundtruth code</div>'
                    f'<pre>{html.escape(gt_code)}</pre></div>'
                    f'<div><div class="section-title">generated code</div>'
                    + (f'<pre class="err">{html.escape(str(gen_err)[:600])}</pre>' if gen_err else '')
                    + f'<pre>{html.escape(gen_code)}</pre></div>'
                    f'</div></div>'
                )
            else:
                code_section = (
                    f'<div class="section"><div class="section-title">groundtruth code</div>'
                    f'<pre>{html.escape(gt_code)}</pre></div>'
                )

            # GT output: distribution-shape gets the bar chart, plus collapsible JSON
            gt_viz = _render_distribution_viz(gt_output) if isinstance(gt_output, dict) else None
            if gt_viz:
                output_section = (
                    f'<div class="section"><div class="section-title">groundtruth output</div>'
                    f'{gt_viz}'
                    f'<details style="margin-top:6px"><summary class="small" '
                    f'style="cursor:pointer;color:var(--muted);font-size:11px">raw JSON</summary>'
                    f'<pre class="gt-output">{html.escape(gt_output_str)}</pre></details>'
                    f'</div>'
                )
            else:
                output_section = (
                    f'<div class="section"><div class="section-title">groundtruth output</div>'
                    f'<pre class="gt-output">{html.escape(gt_output_str)}</pre></div>'
                )

            atom_dom_id = "atom-" + _safe_id_for_hash(aid)
            parts.append(
                f'<details class="atom" id="{html.escape(atom_dom_id)}" '
                f'data-ds="{html.escape(ds_label)}" '
                f'data-bk="{html.escape(bucket)}" '
                f'data-tv="{tv_for_sort}" '
                f'data-aid="{html.escape(aid)}" '
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
                f'{code_section}'
                f'{output_section}'
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
