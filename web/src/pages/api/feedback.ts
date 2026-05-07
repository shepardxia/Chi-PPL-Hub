import type { APIRoute } from 'astro';

type Vote = 'up' | 'down' | 'neutral';

interface FeedbackBody {
  atom_id: string;
  collection: string;
  dataset_version: string;
  rater_id: string;
  rater_name: string;
  vote: Vote;
  comment?: string;
}

function bad(msg: string, status = 400) {
  return new Response(JSON.stringify({ error: msg }), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

function isValid(b: any): b is FeedbackBody {
  return (
    b && typeof b === 'object'
    && typeof b.atom_id === 'string' && b.atom_id.length > 0 && b.atom_id.length < 256
    && typeof b.collection === 'string' && b.collection.length > 0 && b.collection.length < 64
    && typeof b.dataset_version === 'string' && b.dataset_version.length > 0 && b.dataset_version.length < 64
    && typeof b.rater_id === 'string' && b.rater_id.length > 0 && b.rater_id.length < 64
    && typeof b.rater_name === 'string' && b.rater_name.length > 0 && b.rater_name.length < 80
    && (b.vote === 'up' || b.vote === 'down' || b.vote === 'neutral')
    && (b.comment === undefined || (typeof b.comment === 'string' && b.comment.length < 4000))
  );
}

export const POST: APIRoute = async ({ request, locals }) => {
  let body: any;
  try {
    body = await request.json();
  } catch {
    return bad('invalid JSON');
  }
  if (!isValid(body)) return bad('invalid feedback payload');

  const env = (locals as any).runtime?.env as { DB?: D1Database } | undefined;
  const db = env?.DB;
  if (!db) return bad('database not configured', 500);

  const stmt = db.prepare(
    `INSERT INTO feedback
     (atom_id, collection, dataset_version, rater_id, rater_name, vote, comment, visibility)
     VALUES (?, ?, ?, ?, ?, ?, ?, 'private')`
  );
  await stmt.bind(
    body.atom_id, body.collection, body.dataset_version,
    body.rater_id, body.rater_name, body.vote, body.comment ?? '',
  ).run();
  return new Response(JSON.stringify({ ok: true }), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  });
};

// Public GET is gated to "no" until visibility logic is decided.
export const GET: APIRoute = async () => {
  return new Response(JSON.stringify({ error: 'feedback is private' }), {
    status: 403,
    headers: { 'content-type': 'application/json' },
  });
};
