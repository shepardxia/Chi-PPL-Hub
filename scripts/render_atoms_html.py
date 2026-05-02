"""Render atomized dataset (+ optional run results) into a static HTML page.

Self-contained — no server, no JS dependencies (one tiny inline JS for the
filter box). Open the resulting file in a browser; click an atom to expand,
type in the search box to filter live.

If `data/eval_runs/<run>/scored.jsonl` exists, each atom gets per-run
sub-cards showing the generated code and metrics.

Usage:
    .venv/bin/python scripts/render_atoms_html.py \
        [--dataset data/atomized_v2.jsonl] \
        [--runs-dir data/eval_runs] \
        [--output data/atoms.html]
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.io import iter_scored, load_jsonl


HEAD = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>WebPPL atoms — v2</title>
<style>
  :root {
    --fg: #1c1c1c;
    --muted: #666;
    --bg: #fafafa;
    --card: #fff;
    --border: #e0e0e0;
    --code-bg: #f5f5f4;
    --accent: #0066cc;
    --badge: #efefef;
    --good: #18794e;
    --good-bg: #e6f5ec;
    --bad: #b91c1c;
    --bad-bg: #fde8e8;
    --warn: #92400e;
    --warn-bg: #fef3c7;
  }
  * { box-sizing: border-box; }
  body {
    font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    color: var(--fg);
    background: var(--bg);
    margin: 0;
    padding: 0;
  }
  header {
    position: sticky;
    top: 0;
    background: var(--card);
    border-bottom: 1px solid var(--border);
    padding: 12px 24px;
    z-index: 10;
  }
  header h1 {
    font-size: 16px;
    margin: 0 0 8px;
    display: inline-block;
    margin-right: 16px;
  }
  header .stats {
    display: inline-block;
    color: var(--muted);
    margin-right: 16px;
  }
  header input {
    padding: 6px 10px;
    border: 1px solid var(--border);
    border-radius: 4px;
    font-size: 13px;
    width: 320px;
  }
  main { padding: 24px; max-width: 1300px; margin: 0 auto; }
  .atom {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 6px;
    margin-bottom: 12px;
    padding: 12px 16px;
  }
  .atom > summary,
  .run > summary {
    cursor: pointer;
    list-style: none;
    user-select: none;
  }
  .atom > summary::-webkit-details-marker,
  .run > summary::-webkit-details-marker { display: none; }
  .atom > summary::before,
  .run > summary::before {
    content: "▶";
    color: var(--muted);
    font-size: 10px;
    display: inline-block;
    margin-right: 6px;
    transition: transform 0.15s;
  }
  .atom[open] > summary::before,
  .run[open] > summary::before {
    transform: rotate(90deg);
  }
  .atom-id {
    font-family: ui-monospace, "SF Mono", "Cascadia Code", Menlo, monospace;
    font-weight: 600;
    color: var(--accent);
  }
  .badges { display: inline; margin-left: 8px; }
  .badge {
    display: inline-block;
    background: var(--badge);
    color: var(--muted);
    font-size: 11px;
    padding: 2px 6px;
    border-radius: 3px;
    margin-right: 4px;
  }
  .badge.shape { background: #fff8e1; color: #6d4c00; }
  .badge.task { background: #e8f4ff; color: #004080; }
  .badge.good { background: var(--good-bg); color: var(--good); }
  .badge.bad { background: var(--bad-bg); color: var(--bad); }
  .badge.warn { background: var(--warn-bg); color: var(--warn); }
  .section {
    margin-top: 12px;
  }
  .section-title {
    font-size: 11px;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
  }
  pre {
    background: var(--code-bg);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 10px 12px;
    overflow-x: auto;
    margin: 0;
    font: 12.5px/1.55 ui-monospace, "SF Mono", Menlo, monospace;
    white-space: pre-wrap;
    word-wrap: break-word;
  }
  .prompt-md {
    white-space: pre-wrap;
    font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
    font-size: 13.5px;
    line-height: 1.55;
  }
  .prompt-md code {
    background: var(--code-bg);
    padding: 1px 4px;
    border-radius: 3px;
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
    font-size: 12.5px;
  }
  .small { font-size: 12px; color: var(--muted); }
  .gt-output { max-height: 280px; overflow: auto; }
  .runs {
    margin-top: 12px;
  }
  .run {
    background: #fdfdfd;
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 8px 10px;
    margin-bottom: 6px;
  }
  .run-name {
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
    font-weight: 600;
    font-size: 12.5px;
  }
  .err {
    color: var(--bad);
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
    font-size: 12px;
  }
</style>
</head>
<body>
"""

TAIL = """\
<script>
const search = document.getElementById('search');
const atoms = Array.from(document.querySelectorAll('.atom'));
function applyFilter() {
  const q = search.value.trim().toLowerCase();
  let visible = 0;
  for (const a of atoms) {
    const hay = a.dataset.search;
    const show = !q || hay.indexOf(q) !== -1;
    a.style.display = show ? '' : 'none';
    if (show) visible++;
  }
  document.getElementById('visible-count').textContent = visible;
}
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
            esc = html.escape(line)
            esc = _inline_code(esc)
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


def _load_runs(runs_dir: Path) -> dict[str, dict[str, dict]]:
    """Return {run_name: {atom_id: scored_record}}, skipping runs without scored.jsonl."""
    out: dict[str, dict[str, dict]] = {}
    if not runs_dir.exists():
        return out
    for sub in sorted(runs_dir.iterdir()):
        scored = sub / "scored.jsonl"
        if not sub.is_dir() or not scored.exists():
            continue
        by_atom = {rec["id"]: rec for rec in iter_scored(scored)}
        if by_atom:
            out[sub.name] = by_atom
    return out


def _fmt_metric(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 100 else f"{v:.1f}"
    return str(v)


def _badge_for_metric(name: str, value) -> str:
    """Color-code a metric badge by how good it is."""
    if value is None:
        cls = ""
    elif name.endswith("kl"):
        v = float(value)
        if v < 0.05: cls = "good"
        elif v < 1.0: cls = "warn"
        else: cls = "bad"
    elif name.endswith("tv"):
        v = float(value)
        if v < 0.05: cls = "good"
        elif v < 0.3: cls = "warn"
        else: cls = "bad"
    elif name.endswith("exact"):
        cls = "good" if value else "bad"
    else:
        cls = ""
    return f'<span class="badge {cls}">{html.escape(name)}={_fmt_metric(value)}</span>'


def _render_run(run_name: str, rec: dict | None) -> str:
    if rec is None:
        return ""
    ev = rec.get("evaluation", {}) or {}
    gen = ev.get("gen", {}) or {}
    executed = gen.get("executed", False)
    error = gen.get("error")
    metrics = ev.get("metrics") or {}
    code = rec.get("generation", {}).get("code", "") or ""

    # Header badges
    badges = []
    badges.append(
        f'<span class="badge {"good" if executed else "bad"}">'
        f'{"executed" if executed else "failed"}</span>'
    )
    for k, v in metrics.items():
        badges.append(_badge_for_metric(k, v))
    badges_html = " ".join(badges)

    parts = [
        f'<details class="run">',
        f'<summary><span class="run-name">{html.escape(run_name)}</span> {badges_html}</summary>',
    ]
    if error:
        parts.append(f'<div class="section"><div class="section-title">error</div>'
                     f'<pre class="err">{html.escape(str(error)[:500])}</pre></div>')
    parts.append(
        f'<div class="section"><div class="section-title">generated code</div>'
        f'<pre>{html.escape(code)}</pre></div>'
    )
    parts.append('</details>')
    return "".join(parts)


def render(atoms: list[dict], runs: dict[str, dict[str, dict]]) -> str:
    parts = [HEAD]
    n_runs = len(runs)
    parts.append(
        '<header>'
        '<h1>WebPPL atoms — v2</h1>'
        f'<span class="stats">'
        f'<span id="visible-count">{len(atoms)}</span>/{len(atoms)} atoms'
        f' · {n_runs} run{"s" if n_runs != 1 else ""}'
        '</span>'
        '<input id="search" type="search" placeholder="filter by id / source / task type / shape...">'
        '</header>'
    )
    parts.append('<main>')

    for atom in atoms:
        aid = atom["id"]
        src = atom.get("source", "")
        ttype = atom.get("task_type", "")
        eval_mode = atom.get("eval_mode", "")
        shape = atom.get("answer_shape", "")
        shape_str = _shape_label(shape)

        prompt_html = _render_prompt(atom.get("prompt", ""))
        gt_code = html.escape(atom.get("groundtruth_code", ""))
        gt_output = atom.get("groundtruth_output")
        gt_output_str = json.dumps(gt_output, indent=2) if gt_output is not None else "(not cached)"
        gt_output_str = _truncate(gt_output_str, 4000)
        gt_output_html = html.escape(gt_output_str)

        searchable = " ".join([aid, src, ttype, shape_str, str(eval_mode)]).lower()

        # per-atom run summary on the header (small dots: green/red)
        run_dots = []
        for run_name in sorted(runs.keys()):
            rec = runs[run_name].get(aid)
            if rec is None:
                continue
            ev = (rec.get("evaluation", {}) or {})
            executed = (ev.get("gen", {}) or {}).get("executed", False)
            run_dots.append(
                f'<span class="badge {"good" if executed else "bad"}" title="{html.escape(run_name)}">'
                f'{html.escape(run_name)}'
                f'</span>'
            )

        parts.append(
            f'<details class="atom" data-search="{html.escape(searchable, quote=True)}">'
            f'<summary>'
            f'<span class="atom-id">{html.escape(aid)}</span>'
            f'<span class="badges">'
            f'<span class="badge task">{html.escape(ttype)}</span>'
            f'<span class="badge shape">{html.escape(shape_str)}</span>'
            f'<span class="badge">{html.escape(src)}</span>'
            f'</span>'
            f'</summary>'

            f'<div class="section">'
            f'<div class="section-title">prompt</div>'
            f'{prompt_html}'
            f'</div>'

            f'<div class="section">'
            f'<div class="section-title">groundtruth_code</div>'
            f'<pre>{gt_code}</pre>'
            f'</div>'

            f'<div class="section">'
            f'<div class="section-title">groundtruth_output</div>'
            f'<pre class="gt-output">{gt_output_html}</pre>'
            f'</div>'
        )

        if runs:
            parts.append('<div class="section"><div class="section-title">model responses</div>'
                         '<div class="runs">')
            for run_name in sorted(runs.keys()):
                rec = runs[run_name].get(aid)
                parts.append(_render_run(run_name, rec))
            parts.append('</div></div>')

        parts.append('</details>')

    parts.append('</main>')
    parts.append(TAIL)
    return "".join(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/atomized_v2.jsonl")
    p.add_argument("--runs-dir", default="data/eval_runs")
    p.add_argument("--output", default="data/atoms.html")
    args = p.parse_args()

    atoms = load_jsonl(Path(args.dataset))
    runs = _load_runs(Path(args.runs_dir))
    html_str = render(atoms, runs)
    out = Path(args.output)
    out.write_text(html_str)
    print(f"Wrote {len(atoms)} atoms × {len(runs)} runs -> {out} ({len(html_str):,} bytes)")


if __name__ == "__main__":
    main()
