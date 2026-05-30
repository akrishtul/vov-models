"""
OpenImages V6 pull — Google's 9M-image, CC-BY-4.0 dataset.

We filter to the "Car" subclass (~50K images) and pull thumbnails into
the unlabeled staging folder. Like Mapillary, these are unlabeled at
the make/model level — they enter training only after our local
classifier can auto-label them with high confidence (iteration 2+).

OpenImages publishes per-class image-id lists at
https://storage.googleapis.com/openimages/v6/oidv6-train-images-with-labels-with-rotation.csv
filtered by 'Car' label (/m/0k4j). We grab the IDs, then fetch
thumbs from the public images bucket.

Bandwidth-budget guarded: defaults to 1000 photos per run.
"""

import csv
import hashlib
import io
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

DATA_DIR  = Path(os.environ.get("VOV_PIPELINE_DATA", "pipeline/data"))
STAGING   = Path(os.environ.get("VOV_PIPELINE_PHOTOS", DATA_DIR / "photos")) / "openimages_unlabeled"
LEDGER    = DATA_DIR / "openimages_pulled.json"
BUDGET    = int(os.environ.get("VOV_OPENIMAGES_BUDGET", 1000))
USER_AGENT = "vov-pipeline-openimages/1.0"

# Class IDs (machine ids) for relevant car/truck/SUV taxonomy in OpenImages V6
CLASS_IDS = ["/m/0k4j", "/m/07r04", "/m/01bjv", "/m/0pg52"]  # Car, Truck, Van, Bus
# The class-image-lists are published per-class with simple URLs.
INDEX_URL = "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-human-imagelabels.csv"
IMG_URL_BASE = "https://storage.googleapis.com/openimages/2018_04/train/"
# Note: V6 train set images live at urls like https://storage.googleapis.com/openimages/2018_04/train/<image_id>.jpg


def http_text(url: str, timeout: int = 60) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def http_bytes(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def run():
    state = {}
    if LEDGER.exists():
        try:
            state = json.loads(LEDGER.read_text())
        except Exception:
            state = {}
    seen = set(state.get("ids") or [])
    STAGING.mkdir(parents=True, exist_ok=True)

    # OpenImages publishes its label index as a several-hundred-MB CSV — too
    # large to load whole in CI. Instead we stream + filter line-by-line.
    print(f"Streaming OpenImages label index → filtering for {CLASS_IDS}")
    try:
        req = urllib.request.Request(INDEX_URL, headers={"User-Agent": USER_AGENT})
        wanted = set(CLASS_IDS)
        candidates = []
        with urllib.request.urlopen(req, timeout=120) as r:
            # Read in 1MB chunks, scan for CSV rows matching our classes.
            buf = b""
            count_scanned = 0
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                buf += chunk
                # Process complete lines only.
                lines, _, buf = buf.rpartition(b"\n")
                for ln in lines.split(b"\n"):
                    if not ln:
                        continue
                    count_scanned += 1
                    # Format: ImageID,Source,LabelName,Confidence
                    try:
                        s = ln.decode("utf-8", errors="ignore")
                        parts = s.split(",")
                        if len(parts) < 4:
                            continue
                        if parts[2] in wanted and parts[3] == "1":
                            candidates.append(parts[0])
                            if len(candidates) >= BUDGET * 3:
                                # plenty of dedup headroom
                                raise StopIteration
                    except Exception:
                        continue
    except StopIteration:
        pass
    except Exception as e:
        print(f"[warn] index stream: {e}", file=sys.stderr)
        return

    # Dedupe + skip already-pulled.
    candidates = [c for c in dict.fromkeys(candidates) if c not in seen]
    print(f"Candidate pool: {len(candidates)} new image IDs.")
    pulled = 0
    for img_id in candidates:
        if pulled >= BUDGET:
            break
        url = f"{IMG_URL_BASE}{img_id}.jpg"
        try:
            blob = http_bytes(url, timeout=30)
            if len(blob) < 8 * 1024:
                continue
            h = hashlib.sha256(blob).hexdigest()[:16]
            fp = STAGING / f"{h}.jpg"
            if fp.exists():
                continue
            fp.write_bytes(blob)
            (STAGING / f"{h}.meta.json").write_text(json.dumps({
                "source":   "openimages_v6",
                "openimages_id": img_id,
                "license":  "CC-BY-4.0",
            }, indent=2))
            seen.add(img_id)
            pulled += 1
            time.sleep(0.05)
        except Exception as e:
            print(f"[warn] dl {url}: {e}", file=sys.stderr)
            continue

    state["ids"] = sorted(seen)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(state, indent=2))
    print(f"OpenImages: +{pulled} new photos, total tracked: {len(seen)}")


if __name__ == "__main__":
    run()
