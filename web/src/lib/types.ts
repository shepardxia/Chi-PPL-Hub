// Atom + run record types — mirrors what scripts/render_atoms_html.py renders.
// Loaded from JSONL files in ../data at build time.

export type AnswerShape =
  | 'value'
  | 'samples'
  | 'distribution'
  | { record: Record<string, AnswerShape> }
  | string; // fallback for unknown shapes

export interface Atom {
  id: string;
  source: string;
  source_block_indices?: number[];
  task_type: string;
  eval_mode: string;
  answer_shape: AnswerShape;
  prompt: string;
  groundtruth_code: string;
  groundtruth_output: unknown;
  wrap_target?: string;
  notes?: string;
}

export interface ScoredRun {
  id: string;
  generation?: { code?: string };
  evaluation?: {
    gen?: { executed?: boolean; error?: string };
    comparison?: { ok?: boolean; error?: string };
    metrics?: Record<string, number>;
  };
}

export interface Collection {
  /** URL slug, e.g. "atomized-v2" */
  slug: string;
  /** Human label, e.g. "exercises" */
  label: string;
  /** Path relative to repo root */
  jsonlPath: string;
  /** Default run name to drive bucket badges + active code panel */
  primaryRun?: string;
  /** Free-form description shown on the collection landing page */
  description?: string;
}

export type Bucket =
  | 'TV=0' | 'TV<.05' | 'TV<.5' | 'TV<1' | 'TV=1'
  | 'val+' | 'val-' | 'shape!' | 'fail' | 'no-run';
