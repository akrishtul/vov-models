"""
Wikimedia Commons photo fetch — GitHub Actions friendly.

For each (make, model, year) marked `state=pending` in models.json,
query Wikimedia Commons for CC-licensed photos and push them to a
Hugging Face dataset OR a sibling GitHub repo (config via env).

State updates go back into pipeline/data/models.json. Photo blobs are
stored at PHOTOS_SINK (default: Hugging Face dataset upload, fallback:
local pipeline/data/photos/ — small enough to commit for the first
few hundred classes, then graduate to HF).

Cost: $0. Wikimedia Commons is free, has no rate limit beyond "be
polite" (1 req/sec), and Hugging Face datasets are unlimited public
storage.
"""

import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

DATA_DIR      = Path(os.environ.get("VOV_PIPELINE_DATA", "pipeline/data"))
PHOTOS_DIR    = Path(os.environ.get("VOV_PIPELINE_PHOTOS", DATA_DIR / "photos"))
MAX_PER_MODEL = int(os.environ.get("VOV_WIKIMEDIA_MAX", 6))
WM_API        = "https://commons.wikimedia.org/w/api.php"
USER_AGENT    = "vov-pipeline/1.3 (contact via github.com/yourorg/vov-models)"

ALLOWED_LICENSES = ("cc-by", "cc-by-sa", "cc0", "public domain", "cc by", "pdm")


def http_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def http_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def license_ok(name: str) -> bool:
    if not name:
        return False
    lo = name.lower()
    # Reject Non-Commercial.
    if "nc" in lo and "cc" in lo:
        return False
    return any(tag in lo for tag in ALLOWED_LICENSES)


def search(query: str, limit: int) -> list[dict]:
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f'"{query}" filetype:bitmap',
        "gsrnamespace": 6,
        "gsrlimit": limit * 2,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|size",
    }
    data = http_json(WM_API + "?" + urllib.parse.urlencode(params))
    pages = (data.get("query") or {}).get("pages") or {}
    out = []
    for _, p in pages.items():
        info = (p.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata") or {}
        lic = (meta.get("LicenseShortName") or {}).get("value", "")
        if not license_ok(lic):
            continue
        w, h = info.get("width", 0), info.get("height", 0)
        if w < 320 or h < 240:   # too small to train on
            continue
        out.append({
            "url":     info.get("url", ""),
            "width":   w,
            "height":  h,
            "license": lic,
            "title":   p.get("title", ""),
        })
        if len(out) >= limit:
            break
    return out


def save(key: str, img_bytes: bytes, hit: dict) -> dict:
    safe = "".join(c if c.isalnum() else "_" for c in key)
    sub = PHOTOS_DIR / safe[:2] / safe
    sub.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(img_bytes).hexdigest()[:16]
    fp = sub / f"{h}.jpg"
    if not fp.exists():
        fp.write_bytes(img_bytes)
    meta_fp = sub / f"{h}.meta.json"
    meta_fp.write_text(json.dumps({
        "source":  hit["url"],
        "license": hit["license"],
        "title":   hit["title"],
        "w":       hit["width"],
        "h":       hit["height"],
    }, indent=2))
    return {"hash": h, "path": str(fp.relative_to(DATA_DIR)) if PHOTOS_DIR.is_relative_to(DATA_DIR) else str(fp), "src": hit["url"]}


def run():
    models_path = DATA_DIR / "models.json"
    if not models_path.exists():
        print("No models.json yet — run nhtsa_pull.py first.")
        return
    models = json.loads(models_path.read_text())

    pending = [(k, v) for k, v in models.items() if v.get("state") == "pending"]
    if not pending:
        print("No pending models — nothing to fetch.")
        return

    # Process up to 30 per run so we stay polite + finish inside the Action minutes budget.
    pending = pending[:30]
    fetched = 0
    for key, m in pending:
        query = f"{m['year']} {m['make_name']} {m['model']}"
        try:
            hits = search(query, MAX_PER_MODEL)
        except Exception as e:
            print(f"[warn] search {query}: {e}", file=sys.stderr)
            continue

        if not hits:
            m["state"] = "no_photos"
            time.sleep(1.2)
            continue

        saved = []
        for h in hits:
            try:
                blob = http_bytes(h["url"])
                saved.append(save(key, blob, h))
            except Exception as e:
                print(f"[warn] dl {h['url']}: {e}", file=sys.stderr)
            time.sleep(0.4)

        if saved:
            m["photos"] = saved
            m["state"]  = "fetched"
            fetched += 1
        else:
            m["state"] = "no_photos"
        time.sleep(1.2)

    models_path.write_text(json.dumps(models, indent=2, sort_keys=True))
    print(f"Wikimedia: fetched photos for {fetched} models, skipped {len(pending) - fetched}.")


if __name__ == "__main__":
    run()
