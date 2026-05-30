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
# v2026.05.30 — bumped 25 → 100 per Alex's "maximize" directive.
MAX_PER_MODEL = int(os.environ.get("VOV_WIKIMEDIA_MAX", 100))
# v2026.05.30 — process up to 200 models per run (was 30). Wikimedia has no
# hard rate limit so the practical ceiling is the GH Action's wall-clock budget;
# at ~3 photos/sec we still finish 200 models × 100 photos in well under 90 min.
MAX_MODELS_PER_RUN = int(os.environ.get("VOV_WIKIMEDIA_MODELS_PER_RUN", 200))
# v2026.05.30 — supplement mode: also revisit already-fetched models that have
# fewer than this many photos, re-fetching with the new wider query set.
SUPPLEMENT_TARGET  = int(os.environ.get("VOV_WIKIMEDIA_SUPPLEMENT_TARGET", 80))
SUPPLEMENT_BUDGET  = int(os.environ.get("VOV_WIKIMEDIA_SUPPLEMENT_BUDGET", 100))
WM_API        = "https://commons.wikimedia.org/w/api.php"
USER_AGENT    = "vov-pipeline/2.0 (contact via github.com/akrishtul/vov-models)"

ALLOWED_LICENSES = ("cc-by", "cc-by-sa", "cc0", "public domain", "cc by", "pdm")


def query_variants(make: str, model: str, year):
    """
    v2026.05.30 — Multi-query expansion. The 25-photo cap from the single
    '{year} {make} {model}' query was a ceiling more than a desire. Wikimedia
    serves different photo sets per phrasing, so issuing 5 variants and
    deduping by URL multiplies usable hits without retraining the LLM filter.
    """
    yr = str(year or "").strip()
    base = f"{make} {model}".strip()
    out = []
    if yr:
        out.append(f"{yr} {make} {model}")
    out.extend([
        f"{make} {model}",
        f"{base} car",
        f"{base} sedan",
        f"{base} exterior",
        f"{base} parked",
        f"{base} side view",
    ])
    # Dedupe while preserving order.
    seen, uniq = set(), []
    for q in out:
        if q not in seen:
            seen.add(q); uniq.append(q)
    return uniq


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


def fetch_for_model(key: str, m: dict, target: int, already_have_urls: set) -> list:
    """
    v2026.05.30 — Fan out multiple search queries per model, dedupe results
    by source URL, and stop early once we have `target` saved photos. Returns
    the list of new photo dicts ({hash, path, src}) added in this run.
    """
    new_saved = []
    seen_urls = set(already_have_urls)
    for query in query_variants(m.get("make_name", ""), m.get("model", ""), m.get("year", "")):
        if len(new_saved) >= target:
            break
        try:
            hits = search(query, max(1, target - len(new_saved)) + 5)
        except Exception as e:
            print(f"[warn] search {query}: {e}", file=sys.stderr)
            continue
        for h in hits:
            if h["url"] in seen_urls:
                continue
            seen_urls.add(h["url"])
            if len(new_saved) >= target:
                break
            try:
                blob = http_bytes(h["url"])
                new_saved.append(save(key, blob, h))
            except Exception as e:
                print(f"[warn] dl {h['url']}: {e}", file=sys.stderr)
            time.sleep(0.3)
        time.sleep(0.5)  # be polite between query variants
    return new_saved


def run():
    models_path = DATA_DIR / "models.json"
    if not models_path.exists():
        print("No models.json yet — run nhtsa_pull.py first.")
        return
    models = json.loads(models_path.read_text())

    pending = [(k, v) for k, v in models.items() if v.get("state") == "pending"]
    # v2026.05.30 — supplement queue: already-fetched models with < SUPPLEMENT_TARGET photos.
    needs_more = [
        (k, v) for k, v in models.items()
        if v.get("state") == "fetched" and len(v.get("photos") or []) < SUPPLEMENT_TARGET
    ]
    print(f"queue: {len(pending)} pending  /  {len(needs_more)} fetched-but-undersize")

    # Process pending first (highest priority — those classes have ZERO photos).
    pending = pending[:MAX_MODELS_PER_RUN]
    new_for_pending = 0
    for key, m in pending:
        saved = fetch_for_model(key, m, MAX_PER_MODEL, set())
        if saved:
            m["photos"] = saved
            m["state"]  = "fetched"
            new_for_pending += 1
        else:
            m["state"] = "no_photos"
        time.sleep(0.6)

    # Then top up under-fed classes from the supplement queue (budget-limited).
    needs_more = needs_more[:SUPPLEMENT_BUDGET]
    new_for_supp = 0
    for key, m in needs_more:
        already_urls = { (p.get("src") or "") for p in (m.get("photos") or []) }
        have_now = len(m.get("photos") or [])
        target_new = max(0, SUPPLEMENT_TARGET - have_now)
        if target_new == 0:
            continue
        added = fetch_for_model(key, m, target_new, already_urls)
        if added:
            m["photos"] = (m.get("photos") or []) + added
            new_for_supp += len(added)
        time.sleep(0.6)

    models_path.write_text(json.dumps(models, indent=2, sort_keys=True))
    print(f"Wikimedia: pending → fetched: {new_for_pending}/{len(pending)} models. "
          f"Supplement: +{new_for_supp} photos across {len(needs_more)} models.")


if __name__ == "__main__":
    run()
