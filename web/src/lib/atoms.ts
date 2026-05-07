import { readFile, readdir } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import type { Atom, Collection, ScoredRun } from './types';

// Astro's build/prerender step runs with CWD = the Astro project root (web/).
// import.meta.url-based resolution would point into the bundled output dir
// during prerender and miss the data files entirely.
const REPO_ROOT = resolve(process.cwd(), '..');
const DATA_DIR = join(REPO_ROOT, 'data');
const RUNS_DIR = join(DATA_DIR, 'eval_runs');

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
  let text: string;
  try {
    text = await readFile(absPath, 'utf8');
  } catch (e: any) {
    if (e?.code === 'ENOENT') return [];
    throw e;
  }
  const out: T[] = [];
  for (const line of text.split('\n')) {
    const t = line.trim();
    if (!t) continue;
    out.push(JSON.parse(t) as T);
  }
  return out;
}

async function loadAllRuns(): Promise<Record<string, Record<string, ScoredRun>>> {
  let entries;
  try {
    entries = await readdir(RUNS_DIR, { withFileTypes: true });
  } catch (e: any) {
    if (e?.code === 'ENOENT') return {};
    throw e;
  }
  const dirs = entries.filter((e) => e.isDirectory());
  const loaded = await Promise.all(
    dirs.map(async (e) => {
      const recs = await readJsonl<ScoredRun>(join(RUNS_DIR, e.name, 'scored.jsonl'));
      return [e.name, recs] as const;
    }),
  );
  const out: Record<string, Record<string, ScoredRun>> = {};
  for (const [name, recs] of loaded) {
    if (recs.length === 0) continue;
    const byId: Record<string, ScoredRun> = {};
    for (const r of recs) byId[r.id] = r;
    out[name] = byId;
  }
  return out;
}

export interface CollectionData {
  collection: Collection;
  atoms: Atom[];
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
  return Object.keys(d.runs)[0] ?? null;
}

export function groupKey(atom: Atom): string {
  const src = atom.source ?? '';
  if (src) {
    const stem = src.split('/').pop() ?? src;
    return stem.endsWith('.md') ? stem.slice(0, -3) : stem;
  }
  const aid = atom.id;
  return aid.includes('/') ? aid.split('/')[0] : aid;
}

export function atomDomId(atomId: string): string {
  return 'atom-' + atomId.replace(/\//g, '--').replace(/ /g, '_');
}
