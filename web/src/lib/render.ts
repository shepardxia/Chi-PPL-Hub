// Build-time HTML helpers — ported from scripts/render_atoms_html.py.

import type { AnswerShape } from './types';

export function escapeHtml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/** Render a prompt with ```fenced``` blocks → <pre>, and `inline` → <code>. */
export function renderPrompt(prompt: string): string {
  const out: string[] = [];
  let inFence = false;
  let buf: string[] = [];
  for (const line of prompt.split('\n')) {
    if (line.startsWith('```')) {
      if (inFence) {
        out.push('<pre>' + escapeHtml(buf.join('\n')) + '</pre>');
        buf = [];
        inFence = false;
      } else {
        inFence = true;
      }
      continue;
    }
    if (inFence) {
      buf.push(line);
    } else {
      out.push(inlineCode(escapeHtml(line)));
    }
  }
  if (inFence && buf.length) out.push('<pre>' + escapeHtml(buf.join('\n')) + '</pre>');
  return '<div class="prompt-md">' + out.join('\n') + '</div>';
}

function inlineCode(text: string): string {
  const parts = text.split('`');
  if (parts.length < 3) return text;
  return parts
    .map((p, i) => (i % 2 === 0 ? p : '<code>' + p + '</code>'))
    .join('');
}

export function shapeLabel(shape: AnswerShape): string {
  if (shape && typeof shape === 'object' && 'record' in shape) {
    return `record(${Object.keys(shape.record).join(', ')})`;
  }
  return String(shape);
}

export function truncate(s: string, limit = 4000): string {
  return s.length <= limit ? s : s.slice(0, limit) + `\n\n... (${s.length - limit} more chars truncated)`;
}

/** Render a {__kind:distribution, support, probs} object as a horizontal bar chart. */
export function renderDistViz(d: unknown, maxRows = 12): string | null {
  if (!d || typeof d !== 'object') return null;
  const dd = d as Record<string, unknown>;
  if (dd.__kind !== 'distribution') return null;
  const support = (dd.support ?? []) as unknown[];
  const probs = (dd.probs ?? []) as number[];
  if (support.length === 0 || support.length !== probs.length) return null;
  const pairs: { v: unknown; p: number }[] = support.map((v, i) => ({ v, p: probs[i] }));
  pairs.sort((a, b) => b.p - a.p);
  const truncated = pairs.length > maxRows;
  const head = pairs.slice(0, maxRows);
  const pmax = head.length ? Math.max(...head.map((p) => p.p)) : 1.0;
  const safePmax = pmax || 1.0;
  const rows: string[] = [];
  for (const { v, p } of head) {
    let label = typeof v === 'string' ? v : JSON.stringify(v);
    if (label.length > 40) label = label.slice(0, 37) + '…';
    const barPct = p > 0 ? Math.max(1.0, (100.0 * p) / safePmax) : 0;
    rows.push(
      `<div class="dist-row">` +
      `<span class="lab" title="${escapeHtml(String(v))}">${escapeHtml(label)}</span>` +
      `<span class="bar-track"><span class="bar-fill" style="width:${barPct.toFixed(1)}%"></span></span>` +
      `<span class="pv">${p.toFixed(4)}</span>` +
      `</div>`
    );
  }
  const suffix = truncated
    ? `<div class="dist-row"><span class="lab" style="color:var(--muted)">… ${pairs.length - maxRows} more</span></div>`
    : '';
  return '<div class="dist-viz">' + rows.join('') + suffix + '</div>';
}

export function safeIdForHash(aid: string): string {
  return aid.replace(/\//g, '--').replace(/ /g, '_');
}
