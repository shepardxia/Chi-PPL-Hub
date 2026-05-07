// Ported from scripts/render_atoms_html.py:bucket_for / bucket_class.
// Bucket-naming convention is intentionally stable across this and the
// legacy Python renderer so that audit-log diff comparisons keep working.

import type { Bucket, ScoredRun } from './types';

export const BUCKETS: { label: Bucket; tone: 'good' | 'warn' | 'bad' | 'muted'; desc: string }[] = [
  { label: 'TV=0',   tone: 'good',  desc: 'TV exactly 0 (matched)' },
  { label: 'TV<.05', tone: 'good',  desc: 'TV < 0.05' },
  { label: 'TV<.5',  tone: 'warn',  desc: 'TV < 0.5' },
  { label: 'TV<1',   tone: 'warn',  desc: '0.5 ≤ TV < 1' },
  { label: 'TV=1',   tone: 'bad',   desc: 'TV exactly 1 (full disagreement)' },
  { label: 'val+',   tone: 'good',  desc: 'value match (approx)' },
  { label: 'val-',   tone: 'bad',   desc: 'value mismatch' },
  { label: 'shape!', tone: 'bad',   desc: 'shape mismatch' },
  { label: 'fail',   tone: 'bad',   desc: 'execution failure' },
  { label: 'no-run', tone: 'muted', desc: 'no run available' },
];

export function bucketFor(rec: ScoredRun | undefined | null): Bucket {
  if (!rec) return 'no-run';
  const ev = rec.evaluation ?? {};
  const gen = ev.gen ?? {};
  if (!gen.executed) return 'fail';
  const cmp = ev.comparison ?? {};
  const err = cmp.error ?? '';
  if (cmp.ok === false && (err.startsWith('not a') || err.startsWith('samples must'))) {
    return 'shape!';
  }
  const metrics = ev.metrics ?? {};
  const tvs = Object.entries(metrics).filter(([k]) => k.endsWith('tv')).map(([_, v]) => v);
  if (tvs.length > 0) {
    const worst = Math.max(...tvs);
    if (worst === 0) return 'TV=0';
    if (worst < 0.05) return 'TV<.05';
    if (worst < 0.5) return 'TV<.5';
    if (worst < 1) return 'TV<1';
    return 'TV=1';
  }
  const exacts = Object.entries(metrics).filter(([k]) => k.endsWith('approx')).map(([_, v]) => v);
  if (exacts.length > 0) return exacts.every((v) => v === 1.0) ? 'val+' : 'val-';
  return 'shape!';
}

export function shortMetric(rec: ScoredRun | undefined | null): string {
  if (!rec) return '';
  const metrics = rec.evaluation?.metrics ?? {};
  const tvs = Object.entries(metrics).filter(([k]) => k.endsWith('tv')).map(([_, v]) => v);
  if (tvs.length > 0) return `TV=${Math.max(...tvs).toFixed(2)}`;
  const exacts = Object.entries(metrics).filter(([k]) => k.endsWith('approx')).map(([_, v]) => v);
  if (exacts.length > 0) return exacts.every((v) => v === 1.0) ? '✓' : '✗';
  return '';
}

export function toneFor(bucket: Bucket): 'good' | 'warn' | 'bad' | 'muted' {
  return BUCKETS.find((b) => b.label === bucket)?.tone ?? 'muted';
}
