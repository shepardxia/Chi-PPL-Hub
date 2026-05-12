import type { AnswerShape, RenderOutput } from './types';

export function escapeHtml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// ─── Markdown ───────────────────────────────────────────────────────────────
// Tiny markdown handler: ``` fenced blocks, `inline code`, **bold**, paragraphs.

interface MdBlock { type: 'p' | 'code'; content: string; lang?: string; }

function parseMarkdown(src: string): MdBlock[] {
  const parts: MdBlock[] = [];
  const lines = src.split('\n');
  let buf: string[] = [];
  let inFence = false;
  let fenceLang = '';
  let fenceBuf: string[] = [];
  const flushPara = () => {
    if (!buf.length) return;
    parts.push({ type: 'p', content: buf.join('\n') });
    buf = [];
  };
  for (const line of lines) {
    if (inFence) {
      if (line.startsWith('```')) {
        parts.push({ type: 'code', lang: fenceLang, content: fenceBuf.join('\n') });
        inFence = false; fenceBuf = [];
      } else {
        fenceBuf.push(line);
      }
      continue;
    }
    if (line.startsWith('```')) {
      flushPara();
      inFence = true;
      fenceLang = line.slice(3).trim();
      continue;
    }
    if (line.trim() === '') { flushPara(); continue; }
    buf.push(line);
  }
  flushPara();
  return parts;
}

function inlineMd(text: string): string {
  const escaped = escapeHtml(text);
  const out: string[] = [];
  let i = 0;
  while (i < escaped.length) {
    if (escaped[i] === '`') {
      const j = escaped.indexOf('`', i + 1);
      if (j > 0) {
        out.push(`<code class="md-inline-code">${escaped.slice(i + 1, j)}</code>`);
        i = j + 1;
        continue;
      }
    }
    if (escaped[i] === '*' && escaped[i + 1] === '*') {
      const j = escaped.indexOf('**', i + 2);
      if (j > 0) {
        out.push(`<strong>${escaped.slice(i + 2, j)}</strong>`);
        i = j + 2;
        continue;
      }
    }
    const next = escaped.indexOf('`', i);
    const nextB = escaped.indexOf('**', i);
    let end = escaped.length;
    if (next > -1) end = Math.min(end, next);
    if (nextB > -1) end = Math.min(end, nextB);
    out.push(escaped.slice(i, end));
    i = end;
  }
  return out.join('');
}

export function renderMarkdown(src: string): string {
  const blocks = parseMarkdown(src);
  return '<div class="md">' + blocks.map((b) =>
    b.type === 'code'
      ? renderCode(b.content, b.lang || 'webppl')
      : `<p>${inlineMd(b.content)}</p>`
  ).join('') + '</div>';
}

// ─── WebPPL syntax highlighter ──────────────────────────────────────────────

const WEBPPL_KW = new Set([
  'var','function','return','if','else','for','while','true','false','null','undefined','new',
]);
const WEBPPL_INFER = new Set([
  'Infer','Enumerate','MCMC','SMC','rejection','sample','factor','observe','condition',
  'expectation','flip','uniform','uniformDraw','gaussian','beta','dirichlet','categorical',
  'discrete','mem','mapData','map','map2','reduce','Categorical','Bernoulli','Binomial',
  'Gaussian','Beta','Dirichlet','Vector','Math','repeat',
]);

interface Tok { t: string; v: string; }

function tokenize(src: string): Tok[] {
  const tokens: Tok[] = [];
  let i = 0;
  while (i < src.length) {
    const c = src[i];
    if (c === '/' && src[i + 1] === '/') {
      const j = src.indexOf('\n', i);
      const end = j === -1 ? src.length : j;
      tokens.push({ t: 'cm', v: src.slice(i, end) });
      i = end;
      continue;
    }
    if (c === '/' && src[i + 1] === '*') {
      const j = src.indexOf('*/', i + 2);
      const end = j === -1 ? src.length : j + 2;
      tokens.push({ t: 'cm', v: src.slice(i, end) });
      i = end;
      continue;
    }
    if (c === "'" || c === '"' || c === '`') {
      const q = c;
      let j = i + 1;
      while (j < src.length && src[j] !== q) {
        if (src[j] === '\\') j += 2;
        else j++;
      }
      tokens.push({ t: 's', v: src.slice(i, j + 1) });
      i = j + 1;
      continue;
    }
    if (/[0-9]/.test(c) || (c === '.' && /[0-9]/.test(src[i + 1] ?? ''))) {
      let j = i;
      while (j < src.length && /[0-9.eE+-]/.test(src[j])) j++;
      tokens.push({ t: 'n', v: src.slice(i, j) });
      i = j;
      continue;
    }
    if (/[A-Za-z_$]/.test(c)) {
      let j = i;
      while (j < src.length && /[A-Za-z0-9_$]/.test(src[j])) j++;
      const v = src.slice(i, j);
      let t = 'i';
      if (WEBPPL_KW.has(v)) t = 'k';
      else if (WEBPPL_INFER.has(v)) t = 'b';
      else if (src[j] === '(') t = 'f';
      tokens.push({ t, v });
      i = j;
      continue;
    }
    if (/[{}()\[\],;:]/.test(c)) {
      tokens.push({ t: 'p', v: c });
      i++;
      continue;
    }
    if (/[+\-*/<>=!&|?]/.test(c)) {
      let j = i;
      while (j < src.length && /[+\-*/<>=!&|?]/.test(src[j])) j++;
      tokens.push({ t: 'o', v: src.slice(i, j) });
      i = j;
      continue;
    }
    tokens.push({ t: 'w', v: c });
    i++;
  }
  return tokens;
}

function highlightLines(src: string): Tok[][] {
  const tokens = tokenize(src);
  const lines: Tok[][] = [[]];
  for (const tok of tokens) {
    const segs = tok.v.split('\n');
    segs.forEach((seg, i) => {
      if (i > 0) lines.push([]);
      if (seg) lines[lines.length - 1].push({ t: tok.t, v: seg });
    });
  }
  return lines;
}

export function renderCode(code: string, lang = 'webppl'): string {
  const lines = highlightLines(code || '');
  const body = lines.map((toks, i) => {
    const content = toks.length === 0
      ? '​'
      : toks.map((tk) => `<span class="tok-${tk.t}">${escapeHtml(tk.v)}</span>`).join('');
    return `<div class="code-line"><span class="code-ln">${i + 1}</span><span class="code-content">${content}</span></div>`;
  }).join('');
  return (
    `<div class="code">` +
    `<div class="code-lang">${escapeHtml(lang)}</div>` +
    `<pre class="code-body">${body}</pre>` +
    `</div>`
  );
}

// ─── Output normalization ───────────────────────────────────────────────────
// Take the executor's raw answer (the JSON stored as `groundtruth_output` or
// `evaluation.gen.answer`) and convert it to RenderOutput.

export function normalizeOutput(answer: unknown, shape: AnswerShape): RenderOutput {
  if (answer == null) return null;
  if (shape === 'distribution') {
    if (typeof answer === 'object' && answer !== null) {
      const a = answer as Record<string, unknown>;
      if (a.__kind === 'distribution' && Array.isArray(a.support) && Array.isArray(a.probs)) {
        return {
          kind: 'distribution',
          support: (a.support as unknown[]).map(stringifyKey),
          probs: a.probs as number[],
        };
      }
    }
    return { kind: 'value', value: answer };
  }
  if (shape === 'samples') {
    if (Array.isArray(answer)) {
      // Aggregate samples → {support, counts}
      const counts = new Map<string, number>();
      for (const item of answer) {
        const key = stringifyKey(item);
        counts.set(key, (counts.get(key) ?? 0) + 1);
      }
      const support = Array.from(counts.keys());
      const cs = support.map((s) => counts.get(s)!);
      return { kind: 'samples', support, counts: cs };
    }
    return { kind: 'value', value: answer };
  }
  if (shape === 'value') {
    return { kind: 'value', value: answer };
  }
  if (typeof shape === 'object' && shape !== null && 'record' in shape) {
    if (typeof answer === 'object' && answer !== null && !Array.isArray(answer)) {
      const fields: Record<string, RenderOutput> = {};
      const subShape = (shape as { record: Record<string, AnswerShape> }).record;
      for (const [k, sub] of Object.entries(subShape)) {
        const inner = normalizeOutput((answer as Record<string, unknown>)[k], sub);
        if (inner) fields[k] = inner;
      }
      return { kind: 'record', fields };
    }
    return { kind: 'value', value: answer };
  }
  return { kind: 'value', value: answer };
}

function stringifyKey(v: unknown): string {
  if (typeof v === 'string') return v;
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : v.toFixed(4);
  return JSON.stringify(v);
}

// ─── Mirrored bar chart (SVG) ───────────────────────────────────────────────
// Renders two distributions overlaid: A goes up from mid-axis, B goes down.

interface ChartSeries { kind: 'distribution' | 'samples'; support: string[]; probs?: number[]; counts?: number[]; }

function seriesProbs(s: ChartSeries | null | undefined, support: string[]): number[] {
  if (!s) return support.map(() => 0);
  return support.map((label) => {
    const idx = s.support.indexOf(label);
    if (idx === -1) return 0;
    if (s.kind === 'samples') {
      const total = (s.counts ?? []).reduce((a, b) => a + b, 0) || 1;
      return ((s.counts ?? [])[idx] ?? 0) / total;
    }
    return (s.probs ?? [])[idx] ?? 0;
  });
}

export function renderChart(opts: {
  a: ChartSeries | null;
  b: ChartSeries | null;
  labelA: string;
  labelB: string;
  maxBars?: number;
}): string {
  const { a, b, labelA, labelB, maxBars = 20 } = opts;
  const supportSet = new Set<string>();
  for (const s of a?.support ?? []) supportSet.add(s);
  for (const s of b?.support ?? []) supportSet.add(s);
  let support = Array.from(supportSet);
  if (support.length === 0) {
    return '<div class="out-empty">(no distribution)</div>';
  }

  // Rank support by max(a, b) probability and keep top N for readability.
  let aProbs = seriesProbs(a, support);
  let bProbs = seriesProbs(b, support);
  let truncated = 0;
  if (support.length > maxBars) {
    const ranked = support.map((s, i) => ({ s, p: Math.max(aProbs[i], bProbs[i]) }))
      .sort((x, y) => y.p - x.p)
      .slice(0, maxBars)
      .map(({ s }) => s);
    truncated = support.length - ranked.length;
    support = ranked;
    aProbs = seriesProbs(a, support);
    bProbs = seriesProbs(b, support);
  }
  const maxP = Math.max(0.01, ...aProbs, ...bProbs);

  const w = 640, h = 240;
  const padL = 40, padR = 16, padT = 18, padB = 28;
  const innerW = w - padL - padR;
  const innerH = h - padT - padB;
  const midY = padT + innerH / 2;
  const halfH = innerH / 2 - 4;
  const colW = innerW / Math.max(1, support.length);
  const barW = Math.max(8, Math.min(40, colW * 0.7));
  const ticks = [0, maxP / 2, maxP];
  const fmtY = (v: number) => v === 0 ? '0' : maxP >= 0.1 ? v.toFixed(2) : maxP >= 0.01 ? v.toFixed(3) : v.toExponential(1);

  const gridLines: string[] = [];
  for (let i = 0; i < ticks.length; i++) {
    const t = ticks[i];
    const yA = midY - (t / maxP) * halfH;
    const yB = midY + (t / maxP) * halfH;
    gridLines.push(
      `<line x1="${padL}" y1="${yA}" x2="${w - padR}" y2="${yA}" class="chart-grid"/>` +
      `<text x="${padL - 6}" y="${yA}" class="chart-yt" text-anchor="end" dominant-baseline="middle">${fmtY(t)}</text>`
    );
    if (i > 0) {
      gridLines.push(
        `<line x1="${padL}" y1="${yB}" x2="${w - padR}" y2="${yB}" class="chart-grid"/>` +
        `<text x="${padL - 6}" y="${yB}" class="chart-yt" text-anchor="end" dominant-baseline="middle">${fmtY(t)}</text>`
      );
    }
  }
  const midAxis = `<line x1="${padL}" y1="${midY}" x2="${w - padR}" y2="${midY}" class="chart-axis"/>`;

  // Decide x-label step so we don't overprint when there are many bars.
  // Min char-width ~6.5px for 10px mono; allow up to 8 chars per label.
  const maxChars = Math.max(2, Math.min(8, Math.floor(colW / 7)));
  const xLabelStep = support.length > 24 ? 4 : support.length > 16 ? 2 : 1;
  const truncLabel = (s: string) => s.length > maxChars ? s.slice(0, Math.max(1, maxChars - 1)) + '…' : s;

  const fmtBar = (p: number) => {
    if (p < 0.005) return '';
    if (maxP >= 0.1) return p.toFixed(2);
    if (maxP >= 0.01) return p.toFixed(3);
    return p.toExponential(1);
  };

  const bars = support.map((s, i) => {
    const x = padL + i * colW + (colW - barW) / 2;
    const ha = (aProbs[i] / maxP) * halfH;
    const hb = (bProbs[i] / maxP) * halfH;
    const aLabel = fmtBar(aProbs[i]);
    const bLabel = fmtBar(bProbs[i]);
    const sEsc = escapeHtml(truncLabel(s));
    const showLabel = (i % xLabelStep === 0);
    return (
      `<g>` +
      `<rect x="${x.toFixed(2)}" y="${(midY - ha).toFixed(2)}" width="${barW.toFixed(2)}" height="${ha.toFixed(2)}" class="chart-bar chart-bar-a"><title>${escapeHtml(s)}: A=${aProbs[i].toFixed(3)}, B=${bProbs[i].toFixed(3)}</title></rect>` +
      `<rect x="${x.toFixed(2)}" y="${midY.toFixed(2)}" width="${barW.toFixed(2)}" height="${hb.toFixed(2)}" class="chart-bar chart-bar-b"><title>${escapeHtml(s)}: A=${aProbs[i].toFixed(3)}, B=${bProbs[i].toFixed(3)}</title></rect>` +
      (aLabel ? `<text x="${(x + barW / 2).toFixed(2)}" y="${(midY - ha - 4).toFixed(2)}" class="chart-val chart-val-a" text-anchor="middle">${aLabel}</text>` : '') +
      (bLabel ? `<text x="${(x + barW / 2).toFixed(2)}" y="${(midY + hb + 12).toFixed(2)}" class="chart-val chart-val-b" text-anchor="middle">${bLabel}</text>` : '') +
      (showLabel ? `<text x="${(x + barW / 2).toFixed(2)}" y="${(h - 8).toFixed(2)}" class="chart-xt" text-anchor="middle">${sEsc}</text>` : '') +
      `</g>`
    );
  }).join('');

  const truncatedNote = truncated > 0
    ? ` <span class="chart-trunc">· top ${support.length} of ${support.length + truncated}</span>`
    : '';

  return (
    `<div class="chart">` +
    `<div class="chart-legend">` +
    `<span class="chart-legend-item chart-legend-a"><span class="chart-legend-swatch"></span> ${escapeHtml(labelA)}</span>` +
    `<span class="chart-legend-item chart-legend-b"><span class="chart-legend-swatch"></span> ${escapeHtml(labelB)}</span>` +
    truncatedNote +
    `</div>` +
    `<svg viewBox="0 0 ${w} ${h}" class="chart-svg" role="img" aria-label="distribution overlay">` +
    gridLines.join('') + midAxis + bars +
    `</svg>` +
    `</div>`
  );
}

// ─── Value / record renderers ───────────────────────────────────────────────

export function renderValueOutput(output: RenderOutput, fallback = '(no output)'): string {
  if (!output) return `<div class="out-empty">${escapeHtml(fallback)}</div>`;
  if (output.kind === 'value') {
    const v = output.value;
    let txt: string;
    if (Array.isArray(v)) {
      txt = '[' + v.map((x) =>
        typeof x === 'number' ? x.toFixed(4) : JSON.stringify(x),
      ).join(', ') + ']';
    } else if (typeof v === 'number') {
      txt = v.toFixed(4);
    } else {
      txt = JSON.stringify(v, null, 2);
    }
    return `<div class="out-value"><pre class="out-value-pre">${escapeHtml(txt)}</pre></div>`;
  }
  if (output.kind === 'record') {
    const rows = Object.entries(output.fields).map(([k, v]) => {
      let val: string;
      if (v && v.kind === 'value') {
        val = typeof v.value === 'number' ? v.value.toFixed(4) : JSON.stringify(v.value);
      } else if (v && v.kind === 'distribution') {
        val = `dist(${v.support.length})`;
      } else {
        val = '...';
      }
      return (
        `<div class="out-record-row">` +
        `<span class="out-record-key">${escapeHtml(k)}</span>` +
        `<span class="out-record-eq">=</span>` +
        `<span class="out-record-val">${escapeHtml(val)}</span>` +
        `</div>`
      );
    }).join('');
    return `<div class="out-record">${rows}</div>`;
  }
  return `<div class="out-empty">${escapeHtml(fallback)}</div>`;
}

export function shapeLabel(shape: AnswerShape): string {
  if (shape && typeof shape === 'object' && 'record' in shape) {
    return `record(${Object.keys((shape as { record: Record<string, AnswerShape> }).record).join(', ')})`;
  }
  return String(shape);
}

export function atomDomId(atomId: string): string {
  return 'atom-' + atomId.replace(/\//g, '--').replace(/ /g, '_');
}

export function isDistLike(o: RenderOutput): o is { kind: 'distribution'; support: string[]; probs: number[] } | { kind: 'samples'; support: string[]; counts: number[] } {
  return !!o && (o.kind === 'distribution' || o.kind === 'samples');
}

/** Cap a distribution / samples output to its top-N most probable values.
 *  LM-generated distributions sometimes carry tens of thousands of support items
 *  (e.g. when the model produced a degenerate distribution); we never need to
 *  ship more than a few dozen to render the chart. */
export function truncateOutput(o: RenderOutput, n = 48): RenderOutput {
  if (!o) return o;
  if (o.kind === 'distribution') {
    if (o.support.length <= n) return o;
    const idx = o.support.map((_, i) => i).sort((a, b) => o.probs[b] - o.probs[a]).slice(0, n);
    return { kind: 'distribution', support: idx.map((i) => o.support[i]), probs: idx.map((i) => o.probs[i]) };
  }
  if (o.kind === 'samples') {
    if (o.support.length <= n) return o;
    const idx = o.support.map((_, i) => i).sort((a, b) => o.counts[b] - o.counts[a]).slice(0, n);
    return { kind: 'samples', support: idx.map((i) => o.support[i]), counts: idx.map((i) => o.counts[i]) };
  }
  if (o.kind === 'record') {
    return { kind: 'record', fields: Object.fromEntries(Object.entries(o.fields).map(([k, v]) => [k, truncateOutput(v, n)])) };
  }
  return o;
}

/** Heuristic: does this output look like a continuous distribution that
 *  needs binning? (Many distinct numeric support values, low per-bucket mass.) */
function looksContinuous(o: RenderOutput): boolean {
  if (!o || (o.kind !== 'distribution' && o.kind !== 'samples')) return false;
  if (o.support.length < 16) return false;
  let numeric = 0;
  let nonInt = 0;
  const sample = o.support.length > 100 ? o.support.slice(0, 100) : o.support;
  for (const s of sample) {
    const x = Number(s);
    if (!Number.isFinite(x)) continue;
    numeric++;
    if (!Number.isInteger(x)) nonInt++;
  }
  if (numeric / sample.length < 0.9) return false;
  // Mostly non-integer values, OR a *lot* of distinct integers → continuous.
  return nonInt / numeric > 0.5 || o.support.length > 64;
}

/** Bin a set of likely-continuous distributions/samples into shared bins.
 *  Keeps charts comparable: all series get the same bin edges from the union
 *  range. nBins ≈ 24 fits the 640px chart cleanly. */
function binShared(outputs: RenderOutput[], nBins = 24): RenderOutput[] {
  let min = Infinity, max = -Infinity;
  for (const o of outputs) {
    if (!o || (o.kind !== 'distribution' && o.kind !== 'samples')) continue;
    for (const s of o.support) {
      const x = Number(s);
      if (!Number.isFinite(x)) continue;
      if (x < min) min = x;
      if (x > max) max = x;
    }
  }
  if (!Number.isFinite(min) || min === max) return outputs;
  const step = (max - min) / nBins;
  const labels: string[] = [];
  const decimals = pickDecimals(step);
  for (let i = 0; i < nBins; i++) {
    const mid = min + step * (i + 0.5);
    labels.push(mid.toFixed(decimals));
  }
  return outputs.map((o) => {
    if (!o) return o;
    if (o.kind !== 'distribution' && o.kind !== 'samples') return o;
    const bins = new Array(nBins).fill(0);
    const weights: number[] = o.kind === 'distribution'
      ? o.probs
      : (() => {
          const total = o.counts.reduce((a, b) => a + b, 0) || 1;
          return o.counts.map((c) => c / total);
        })();
    for (let i = 0; i < o.support.length; i++) {
      const x = Number(o.support[i]);
      if (!Number.isFinite(x)) continue;
      let b = Math.floor((x - min) / step);
      if (b >= nBins) b = nBins - 1;
      if (b < 0) b = 0;
      bins[b] += weights[i];
    }
    return { kind: 'distribution', support: labels, probs: bins };
  });
}

function pickDecimals(step: number): number {
  if (step >= 1) return 1;
  if (step >= 0.1) return 2;
  if (step >= 0.01) return 3;
  return 4;
}

/** Process a per-atom set of outputs: bin if continuous, truncate if huge. */
export function prepareAtomOutputs(outputs: RenderOutput[]): RenderOutput[] {
  const someContinuous = outputs.some(looksContinuous);
  if (someContinuous) return binShared(outputs);
  return outputs.map((o) => truncateOutput(o));
}
