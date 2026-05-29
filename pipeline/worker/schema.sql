-- Run once after `wrangler d1 create vov-corrections`:
--   wrangler d1 execute vov-corrections --file=./schema.sql

CREATE TABLE IF NOT EXISTS corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    license TEXT,
    site TEXT,
    plugin TEXT,
    field TEXT,
    was TEXT,
    now_val TEXT,
    photo_hash TEXT,
    photo_url TEXT,
    region TEXT,
    customer_created_at TEXT,
    received_at TEXT,
    verified TEXT DEFAULT 'pending'
);
CREATE INDEX IF NOT EXISTS idx_corr_license  ON corrections(license);
CREATE INDEX IF NOT EXISTS idx_corr_verified ON corrections(verified);
CREATE INDEX IF NOT EXISTS idx_corr_received ON corrections(received_at);
