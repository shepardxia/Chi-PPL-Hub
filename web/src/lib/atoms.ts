// Build-time loader for atom JSONL files and their associated scored runs.
//
// Reads from ../data (relative to web/), so the dataset stays the source
// of truth in git and the site rebuilds when data changes.

import { readFile, readdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { join, resolve } from 'node:path';
import type { Atom, Collection, ScoredRun } from './types';

// Astro's build/prerender step runs with CWD = the Astro project root (web/).
// The dataset lives one level up. import.meta.url-based resolution would point
// into the bundled output dir during prerender and miss the data files entirely.
const REPO_ROOT = resolve(process.cwd(), '..');
const DATA_DIR = join(REPO_ROOT, 'data');
const RUNS_DIR = join(DATA_DIR, 'eval_runs');

// Registry of collections to surface in the site. Adding one is a
// one-line edit: drop a JSONL into data/, append an entry here.
export const COLLECTIONS: Collection[] = [
  {
    slug: 'atomized-v2',
    label: 'exercises (v2)',
    jsonlPath: 'data/atomized_v2.jsonl',
    primaryRun: 'sonnet-46-primer-v3',
    description: 'Hand-curated WebPPL exercises (76 atoms) — the canonical v2 dataset.',
  },
  {
    slug: 'curated-v3-dippl',
    label: 'dippl (v3)',
    jsonlPath: 'data/curated_v3/dippl.jsonl',
    description: 'Curated v3 — dippl pilot (M-to-N chunking).',
  },
];

async function readJsonl<T>(absPath: string): Promise<T[]> {
  if (!existsSync(absPath)) return [];
  const text = await readFile(absPath, 'utf8');
  const out: T[] = [];
  for (const line of text.split('\n')) {
    const t = line.trim();
    if (!t) continue;
    out.push(JSON.parse(t) as T);
  }
  return out;
}

async function loadAllRuns(): Promise<Record<string, Record<string, ScoredRun>>> {
  if (!existsSync(RUNS_DIR)) return {};
  const out: Record<string, Record<string, ScoredRun>> = {};
  const entries = await readdir(RUNS_DIR, { withFileTypes: true });
  for (const e of entries) {
    if (!e.isDirectory()) continue;
    const scoredPath = join(RUNS_DIR, e.name, 'scored.jsonl');
    const recs = await readJsonl<ScoredRun>(scoredPath);
    if (recs.length === 0) continue;
    const byId: Record<string, ScoredRun> = {};
    for (const r of recs) byId[r.id] = r;
    out[e.name] = byId;
  }
  return out;
}

export interface CollectionData {
  collection: Collection;
  atoms: Atom[];
  /** runs that scored at least one atom in this collection */
  runs: Record<string, Record<string, ScoredRun>>;
}

let _cache: Map<string, CollectionData> | null = null;
let _allRunsCache: Record<string, Record<string, ScoredRun>> | null = null;

export async function loadAllRunsCached() {
  if (_allRunsCache) return _allRunsCache;
  _allRunsCache = await loadAllRuns();
  return _allRunsCache;
}

export async function loadCollection(slug: string): Promise<CollectionData | null> {
  const collection = COLLECTIONS.find((c) => c.slug === slug);
  if (!collection) return null;
  if (!_cache) _cache = new Map();
  const cached = _cache.get(slug);
  if (cached) return cached;
  const atoms = await readJsonl<Atom>(join(REPO_ROOT, collection.jsonlPath));
  const allRuns = await loadAllRunsCached();
  const ids = new Set(atoms.map((a) => a.id));
  const runs: Record<string, Record<string, ScoredRun>> = {};
  for (const [runName, byId] of Object.entries(allRuns)) {
    const matched: Record<string, ScoredRun> = {};
    for (const [aid, rec] of Object.entries(byId)) {
      if (ids.has(aid)) matched[aid] = rec;
    }
    if (Object.keys(matched).length > 0) runs[runName] = matched;
  }
  const data: CollectionData = { collection, atoms, runs };
  _cache.set(slug, data);
  return data;
}

export async function loadAllCollections(): Promise<CollectionData[]> {
  const out: CollectionData[] = [];
  for (const c of COLLECTIONS) {
    const d = await loadCollection(c.slug);
    if (d) out.push(d);
  }
  return out;
}

export function primaryRunFor(d: CollectionData): string | null {
  const pref = d.collection.primaryRun;
  if (pref && d.runs[pref]) return pref;
  const first = Object.keys(d.runs)[0];
  return first ?? null;
}

/** Group atoms by source-file basename — same key as the legacy renderer. */
export function groupKey(atom: Atom): string {
  const src = atom.source ?? '';
  if (src) {
    const stem = src.split('/').pop() ?? src;
    return stem.endsWith('.md') ? stem.slice(0, -3) : stem;
  }
  const aid = atom.id;
  return aid.includes('/') ? aid.split('/')[0] : aid;
}

/** Stable atom-id-derived URL fragment. */
export function atomFragment(atomId: string): string {
  return atomId.replace(/\//g, '--').replace(/ /g, '_');
}
