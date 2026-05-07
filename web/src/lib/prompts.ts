// Source of truth lives at data/prompts/{system_base,webppl_primer}.txt;
// eval/prompt.py reads the same files. Do not inline the strings here.

import { readFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';

export const PROMPT_VERSION = 'v2-atom';

const PROMPTS_DIR = resolve(process.cwd(), '..', 'data', 'prompts');

async function load(name: string): Promise<string> {
  const text = await readFile(join(PROMPTS_DIR, name), 'utf8');
  return text.replace(/\n+$/, '');
}

export const SYSTEM_PROMPT_BASE = await load('system_base.txt');
export const WEBPPL_PRIMER = await load('webppl_primer.txt');

export function systemPrompt(withPrimer: boolean): string {
  return withPrimer ? `${SYSTEM_PROMPT_BASE}\n\n${WEBPPL_PRIMER}` : SYSTEM_PROMPT_BASE;
}

/** Run names follow `<model>-<primer-flag>-...` convention; "noprimer" → false. */
export function runHasPrimer(runName: string): boolean {
  return !runName.toLowerCase().includes('noprimer');
}
