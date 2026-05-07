-- ppl-gym feedback log.
--
-- One row per feedback action. (atom_id, rater_id) is NOT unique:
-- raters can revise — the latest row by created_at represents their
-- current opinion. Older rows are kept for audit.

CREATE TABLE IF NOT EXISTS feedback (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  atom_id         TEXT    NOT NULL,
  collection      TEXT    NOT NULL,
  dataset_version TEXT    NOT NULL,
  rater_id        TEXT    NOT NULL,
  rater_name      TEXT    NOT NULL,
  vote            TEXT    CHECK (vote IN ('up', 'down', 'neutral')) NOT NULL,
  comment         TEXT    NOT NULL DEFAULT '',
  visibility      TEXT    CHECK (visibility IN ('private', 'public')) NOT NULL DEFAULT 'private',
  created_at      INTEGER NOT NULL DEFAULT (unixepoch())
);

CREATE INDEX IF NOT EXISTS feedback_atom_id_idx ON feedback (atom_id, created_at DESC);
CREATE INDEX IF NOT EXISTS feedback_rater_idx   ON feedback (rater_id, created_at DESC);
CREATE INDEX IF NOT EXISTS feedback_created_idx ON feedback (created_at DESC);
