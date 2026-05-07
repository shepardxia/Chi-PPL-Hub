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

export function tvValues(rec: ScoredRun | undefined | null): number[] {
  const m = rec?.evaluation?.metrics ?? {};
  return Object.entries(m).filter(([k]) => k.endsWith('tv')).map(([, v]) => v);
}

export function approxValues(rec: ScoredRun | undefined | null): number[] {
  const m = rec?.evaluation?.metrics ?? {};
  return Object.entries(m).filter(([k]) => k.endsWith('approx')).map(([, v]) => v);
}

export function maxTV(rec: ScoredRun | undefined | null): number | null {
  const tvs = tvValues(rec);
  return tvs.length ? Math.max(...tvs) : null;
}

export function bucketFor(rec: ScoredRun | undefined | null): Bucket {
  if (!rec) return 'no-run';
  const ev = rec.evaluation ?? {};
  if (!ev.gen?.executed) return 'fail';
  const cmp = ev.comparison ?? {};
  const err = cmp.error ?? '';
  if (cmp.ok === false && (err.startsWith('not a') || err.startsWith('samples must'))) {
    return 'shape!';
  }
  const worst = maxTV(rec);
  if (worst !== null) {
    if (worst === 0) return 'TV=0';
    if (worst < 0.05) return 'TV<.05';
    if (worst < 0.5) return 'TV<.5';
    if (worst < 1) return 'TV<1';
    return 'TV=1';
  }
  const exacts = approxValues(rec);
  if (exacts.length > 0) return exacts.every((v) => v === 1.0) ? 'val+' : 'val-';
  return 'shape!';
}

export function shortMetric(rec: ScoredRun | undefined | null): string {
  const worst = maxTV(rec);
  if (worst !== null) return `TV=${worst.toFixed(2)}`;
  const exacts = approxValues(rec);
  if (exacts.length > 0) return exacts.every((v) => v === 1.0) ? '✓' : '✗';
  return '';
}

export function toneFor(bucket: Bucket): 'good' | 'warn' | 'bad' | 'muted' {
  return BUCKETS.find((b) => b.label === bucket)?.tone ?? 'muted';
}

export function barClassFor(bucket: Bucket): string {
  return 'b-' + toneFor(bucket);
}
