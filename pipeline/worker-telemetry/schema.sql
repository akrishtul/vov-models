-- Run once after `wrangler d1 create vov-telemetry`:
--   wrangler d1 execute vov-telemetry --file=./schema.sql

-- One row per customer site. Updated on every heartbeat.
CREATE TABLE IF NOT EXISTS sites (
    license             TEXT PRIMARY KEY,
    url                 TEXT,
    name                TEXT,
    plugin_version      TEXT,
    model_version       TEXT,
    license_tier        TEXT,
    capture_mode        TEXT,
    ai_slot             TEXT,
    cloud_optin         INTEGER DEFAULT 0,
    latest_status       TEXT DEFAULT 'healthy',
    latest_scans_hour   INTEGER DEFAULT 0,
    latest_success_rate REAL,
    latest_avg_latency_ms INTEGER DEFAULT 0,
    latest_seen         TEXT,
    first_seen          TEXT
);
CREATE INDEX IF NOT EXISTS idx_sites_status ON sites(latest_status);
CREATE INDEX IF NOT EXISTS idx_sites_seen   ON sites(latest_seen);

-- Append-only history. Last 30 days kept by a cron purge.
CREATE TABLE IF NOT EXISTS heartbeats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    license TEXT NOT NULL,
    received_at TEXT NOT NULL,
    scans INTEGER DEFAULT 0,
    success INTEGER DEFAULT 0,
    fallback_used INTEGER DEFAULT 0,
    avg_latency_ms INTEGER DEFAULT 0,
    by_provider TEXT,
    by_region TEXT,
    top_errors TEXT,
    model_version TEXT,
    plugin_version TEXT,
    status TEXT
);
CREATE INDEX IF NOT EXISTS idx_hb_license ON heartbeats(license, received_at DESC);
CREATE INDEX IF NOT EXISTS idx_hb_received ON heartbeats(received_at);

-- Open + resolved alerts.
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    license TEXT NOT NULL,
    site_name TEXT,
    status TEXT NOT NULL,
    reason TEXT,
    created_at TEXT NOT NULL,
    resolved_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_alerts_open ON alerts(resolved_at, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_lic  ON alerts(license);
