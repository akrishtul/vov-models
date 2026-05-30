"""
Mapillary pull — street-view photos with vehicle labels.

Mapillary has ~1 billion street-view images. Their Graph API exposes
search by class ('car', 'truck', etc.) and bounding-box detection.
Free CC-BY-SA-4.0 photos with a CLIENT_ID token.

We pull a configurable budget of photos per class, prioritizing the
classes our existing models.json says we cover. Stored at:

    pipeline/data/photos/mapillary/{Make}_{Model}/{photo_id}.jpg
    {photo_id}.meta.json siblings

Token setup:
  1. Sign up at mapillary.com (free)
  2. Go to Developer → Apps → create new app → grab "Client Token"
  3. Add as GitHub secret VOV_MAPILLARY_TOKEN, expose to the daily-pull workflow.

We do NOT have a free make/model classifier hooked into Mapillary's
bounding boxes today — their detection model labels classes like 'car'
not 'Honda Civic'. So this pull goes into a pre-label staging folder
(pipeline/data/photos/mapillary_unlabeled/) for the local make/model
classifier to LATER auto-label once we have a strong enough model.
Initially these photos won't enter training; they become useful in
iteration 2+ as semi-supervised data.
"""

import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

DATA_DIR  = Path(os.environ.get("VOV_PIPELINE_DATA", "pipeline/data"))
STAGING   = Path(os.environ.get("VOV_PIPELINE_PHOTOS", DATA_DIR / "photos")) / "mapillary_unlabeled"
LEDGER    = DATA_DIR / "mapillary_pulled.json"
TOKEN     = os.environ.get("VOV_MAPILLARY_TOKEN", "").strip()
MAX_IMGS  = int(os.environ.get("VOV_MAPILLARY_BUDGET", 500))
PER_PAGE  = 100
USER_AGENT = "vov-pipeline-mapillary/1.0"
GRAPH     = "https://graph.mapillary.com/images"


def fetch(url: str) -> dict:
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Authorization": f"OAuth {TOKEN}",
    })
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.loads(r.read().decode("utf-8"))


def download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def run():
    if not TOKEN:
        print("[skip] VOV_MAPILLARY_TOKEN not set — Mapillary pull skipped.")
        return
    state = {}
    if LEDGER.exists():
        try:
            state = json.loads(LEDGER.read_text())
        except Exception:
            state = {}
    seen = set(state.get("ids", []))
    pulled = 0
    STAGING.mkdir(parents=True, exist_ok=True)

    # Query the Graph API for images that contain 'car' detections.
    # Field 'detections.value' filters server-side. 'thumb_1024_url' is the
    # downsampled JPEG link we actually need (we don't need 6K originals).
    params = {
        "fields": "id,thumb_1024_url,detections.value,computed_geometry,captured_at",
        "limit":  PER_PAGE,
    }
    # Pagination via 'after' cursor.
    cursor = state.get("after", "")
    while pulled < MAX_IMGS:
        url = GRAPH + "?" + urllib.parse.urlencode(params) + (f"&after={cursor}" if cursor else "")
        try:
            data = fetch(url)
        except Exception as e:
            print(f"[warn] graph fetch: {e}", file=sys.stderr)
            break
        items = data.get("data") or []
        if not items:
            break
        for it in items:
            iid = it.get("id")
            if not iid or iid in seen:
                continue
            # Defensive: keep only images that include a 'car' or 'truck' detection.
            dets = it.get("detections") or {}
            det_vals = []
            if isinstance(dets, dict) and "data" in dets:
                det_vals = [d.get("value") for d in (dets.get("data") or [])]
            elif isinstance(dets, list):
                det_vals = [d.get("value") for d in dets]
            if not any(v in ("car", "truck", "SUV", "van") for v in det_vals):
                continue
            thumb = it.get("thumb_1024_url") or ""
            if not thumb:
                continue
            try:
                blob = download(thumb)
                if len(blob) < 8 * 1024:
                    continue
                h = hashlib.sha256(blob).hexdigest()[:16]
                fp = STAGING / f"{h}.jpg"
                fp.write_bytes(blob)
                (STAGING / f"{h}.meta.json").write_text(json.dumps({
                    "source":      "mapillary",
                    "mapillary_id": iid,
                    "detections": det_vals,
                    "captured_at": it.get("captured_at"),
                    "license":    "CC-BY-SA-4.0",
                }, indent=2))
                seen.add(iid)
                pulled += 1
                time.sleep(0.1)
            except Exception as e:
                print(f"[warn] dl {thumb}: {e}", file=sys.stderr)
            if pulled >= MAX_IMGS:
                break
        cursor = ((data.get("paging") or {}).get("cursors") or {}).get("after") or ""
        if not cursor:
            break
    state["ids"]   = sorted(seen)
    state["after"] = cursor
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(state, indent=2))
    print(f"Mapillary: +{pulled} new images, total tracked: {len(seen)}")


if __name__ == "__main__":
    run()
