"""
Corrections pull — training-side fetcher.

Walks the corrections Cloudflare Worker's GET /v1/export endpoint and
downloads any verified driver-corrected photos into the training pool.
These are the highest-value samples we have because every label was
literally typed by a human standing next to the actual car.

Usage:
    VOV_CORRECTIONS_URL=https://vov-corrections.valetops.workers.dev/v1/export \
    VOV_CORRECTIONS_TOKEN=<bearer> \
    python3 corrections_pull.py

Idempotent — already-pulled photo hashes are tracked in
data/corrections_pulled.json so re-runs only fetch new rows.
"""

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

DATA_DIR    = Path(os.environ.get("VOV_PIPELINE_DATA", "pipeline/data"))
PHOTOS_DIR  = Path(os.environ.get("VOV_PIPELINE_PHOTOS", DATA_DIR / "photos"))
LEDGER      = DATA_DIR / "corrections_pulled.json"
ENDPOINT    = os.environ.get("VOV_CORRECTIONS_URL",
                             "https://vov-corrections.valetops.workers.dev/v1/export")
TOKEN       = os.environ.get("VOV_CORRECTIONS_TOKEN", "")
PAGE_SIZE   = 500
MAX_PAGES   = 50
USER_AGENT  = "vov-pipeline-corrections/1.0"


def fetch(url: str) -> dict:
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Authorization": f"Bearer {TOKEN}" if TOKEN else "",
    })
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.loads(r.read().decode("utf-8"))


def download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def load_ledger() -> dict:
    if LEDGER.exists():
        try:
            return json.loads(LEDGER.read_text())
        except Exception:
            pass
    return {"hashes": [], "last_id": 0}


def save_ledger(state: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(state, indent=2, sort_keys=True))


def run():
    if not TOKEN:
        print("[skip] VOV_CORRECTIONS_TOKEN not set — corrections pull skipped.")
        return
    state    = load_ledger()
    seen     = set(state.get("hashes") or [])
    last_id  = int(state.get("last_id") or 0)
    new_count = 0
    for page in range(MAX_PAGES):
        url = f"{ENDPOINT}?after={last_id}&limit={PAGE_SIZE}"
        try:
            data = fetch(url)
        except Exception as e:
            print(f"[warn] export fetch failed: {e}", file=sys.stderr)
            return
        rows = (data or {}).get("rows") or []
        if not rows:
            break
        # Process this page
        for r in rows:
            try:
                photo_hash = r.get("photo_hash") or ""
                photo_url  = r.get("photo_url")  or ""
                label_make = r.get("now_make")   or ""
                label_model= r.get("now_model")  or ""
                if not photo_url or not photo_hash:
                    continue
                if photo_hash in seen:
                    continue
                if not label_make or not label_model:
                    continue
                blob = download(photo_url)
                key  = f"{label_make}_{label_model}".replace(" ", "_")
                sub  = PHOTOS_DIR / "corrections" / key
                sub.mkdir(parents=True, exist_ok=True)
                (sub / f"{photo_hash}.jpg").write_bytes(blob)
                (sub / f"{photo_hash}.meta.json").write_text(json.dumps({
                    "source": "venue_correction",
                    "make":   label_make,
                    "model":  label_model,
                    "license":  r.get("region") or "",
                    "submitted_at": r.get("received_at") or "",
                }, indent=2))
                seen.add(photo_hash)
                new_count += 1
                time.sleep(0.1)
            except Exception as e:
                print(f"[warn] row {r.get('id')}: {e}", file=sys.stderr)
        last_id = max(last_id, max((int(r.get("id") or 0)) for r in rows))
        if len(rows) < PAGE_SIZE:
            break
    state["hashes"]  = sorted(seen)
    state["last_id"] = last_id
    save_ledger(state)
    print(f"corrections: +{new_count} new photos, total tracked: {len(seen)}, last_id={last_id}")


if __name__ == "__main__":
    run()
