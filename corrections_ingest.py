"""
Corrections ingest endpoint.

Customer plugins POST batches of anonymized correction records to this endpoint.
Deploy as a Cloudflare Worker, AWS Lambda, or run as a Flask app on the same VPS.

Wire format:
  POST /v1/submit
  Content-Type: application/json
  {
    "license": "VOV-XXXX-XXXX",
    "site":    "https://acme.valetops.com",
    "plugin":  "1.2.0",
    "rows":    [ { field, was, now, photo_hash, photo_url, region, created_at }, ... ]
  }

Each row goes into the training dataset (after a second LLM auto-verify pass
to guard against malicious submissions).
"""

import json
import os
import sqlite3
import sys
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

DB_PATH = os.environ.get("VOV_PIPELINE_DB", "/var/lib/vov/pipeline.sqlite3")


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS corrections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            license TEXT,
            site TEXT,
            plugin_version TEXT,
            field TEXT,
            was TEXT,
            now_val TEXT,
            photo_hash TEXT,
            photo_url TEXT,
            region TEXT,
            customer_created_at TEXT,
            received_at TEXT NOT NULL,
            verified TEXT          -- 'pending', 'verified', 'rejected'
        );
        CREATE INDEX IF NOT EXISTS idx_corr_verified ON corrections(verified);
        CREATE INDEX IF NOT EXISTS idx_corr_license  ON corrections(license);
    """)
    conn.commit()


def store_batch(payload: dict) -> int:
    conn = sqlite3.connect(DB_PATH)
    ensure_table(conn)
    license_key = payload.get("license", "")
    site        = payload.get("site", "")
    plugin_v    = payload.get("plugin", "")
    now_iso     = datetime.now(timezone.utc).isoformat()
    count = 0
    for row in payload.get("rows", []):
        conn.execute("""
            INSERT INTO corrections
              (license, site, plugin_version, field, was, now_val,
               photo_hash, photo_url, region, customer_created_at, received_at, verified)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
        """, (
            license_key, site, plugin_v,
            row.get("field", ""), row.get("was", ""), row.get("now", ""),
            row.get("photo_hash", ""), row.get("photo_url", ""),
            row.get("region", ""), row.get("created_at", ""),
            now_iso,
        ))
        count += 1
    conn.commit()
    return count


class IngestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/v1/submit":
            self.send_response(404); self.end_headers(); return

        length = int(self.headers.get("Content-Length", "0") or 0)
        if length > 5 * 1024 * 1024:
            self.send_response(413); self.end_headers(); return

        try:
            body = self.rfile.read(length)
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            self.send_response(400); self.end_headers(); return

        if not isinstance(payload.get("rows"), list):
            self.send_response(400); self.end_headers(); return

        try:
            n = store_batch(payload)
        except Exception as e:
            sys.stderr.write(f"ingest error: {e}\n")
            self.send_response(500); self.end_headers(); return

        resp = json.dumps({"ok": True, "stored": n}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, fmt, *args):
        sys.stderr.write("[ingest] " + (fmt % args) + "\n")


def serve(port: int = 8002):
    HTTPServer(("0.0.0.0", port), IngestHandler).serve_forever()


if __name__ == "__main__":
    serve(int(os.environ.get("PORT", 8002)))
