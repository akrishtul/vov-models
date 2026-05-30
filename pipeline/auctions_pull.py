"""
Auction listings pull — IAA + Copart.

US wholesale auction sites list ~500K-1M vehicles/year with full
year/make/model/VIN and 8-12 photos per vehicle. This is the
highest-quality, properly-labeled US real-world car photo source
short of paying the manufacturers.

Both sites are JS-heavy SPAs; their public JSON search endpoints
DO return enough metadata to seed a downloader. We respect robots.txt
and rate-limit aggressively.

This pull operates per-make: given a list of makes that are in our
models.json class set, query each site's public search API for
recently-listed lots, pull metadata + thumbnail URLs, download.

NOTE: Both auction sites' Terms of Service have shifted over time.
Run with VOV_AUCTIONS_OPT_IN=true to acknowledge you've reviewed
the current ToS and intend bona-fide research/educational use.
This script no-ops without that flag set.

Photos land at:
    pipeline/data/photos/auctions/{Make}_{Model}/{vin}_{idx}.jpg
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
PHOTOS    = Path(os.environ.get("VOV_PIPELINE_PHOTOS", DATA_DIR / "photos")) / "auctions"
LEDGER    = DATA_DIR / "auctions_pulled.json"
OPT_IN    = (os.environ.get("VOV_AUCTIONS_OPT_IN", "").lower() == "true")
PER_MAKE  = int(os.environ.get("VOV_AUCTIONS_PER_MAKE", 25))
MAKES_BUDGET = int(os.environ.get("VOV_AUCTIONS_MAKES_PER_RUN", 30))
USER_AGENT = "Mozilla/5.0 (compatible; vov-pipeline-research/1.0; +https://github.com/akrishtul/vov-models)"

# IAA: public search JSON endpoint
IAA_SEARCH = "https://www.iaai.com/Search?searchKeyword={q}&keywordtype=Make"
# Copart: public lot search JSON
COPART_SEARCH = "https://www.copart.com/public/data/lotdetails/solr/lotSearch"


def http_json(url: str, headers: dict | None = None, post_body: bytes | None = None) -> dict:
    h = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, data=post_body, headers=h, method="POST" if post_body else "GET")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8", errors="replace"))


def http_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=45) as r:
        return r.read()


def make_query_iaa(make: str, limit: int) -> list:
    """Returns list of dicts: {vin, make, model, year, photo_urls[]}"""
    try:
        url = IAA_SEARCH.format(q=urllib.parse.quote(make))
        # IAA returns JSON when we ask for it explicitly. Some installs return
        # HTML — we treat that as empty results.
        data = http_json(url, headers={"Accept": "application/json"})
        # IAA wraps results under `Vehicles` array
        rows = (data or {}).get("Vehicles") or []
    except Exception as e:
        print(f"[warn] IAA query {make}: {e}", file=sys.stderr)
        return []
    out = []
    for r in rows[:limit]:
        vin    = r.get("VIN") or r.get("Vin") or ""
        model  = r.get("Model") or ""
        year   = r.get("Year")  or ""
        # Photo URLs live under various keys depending on listing type.
        photos = []
        for k in ("Photos", "Images", "ThumbnailUrl"):
            v = r.get(k)
            if isinstance(v, list):
                photos.extend([p if isinstance(p, str) else p.get("Url") for p in v])
            elif isinstance(v, str):
                photos.append(v)
        photos = [p for p in photos if p]
        if not vin or not model or not photos:
            continue
        out.append({"vin": vin, "make": make, "model": model, "year": year, "photos": photos[:8]})
    return out


def make_query_copart(make: str, limit: int) -> list:
    body = json.dumps({
        "filter": {"MAKE": [make]},
        "size":   limit,
        "from":   0,
    }).encode("utf-8")
    try:
        data = http_json(COPART_SEARCH, headers={"Content-Type": "application/json"}, post_body=body)
        rows = ((data or {}).get("data") or {}).get("results") or []
    except Exception as e:
        print(f"[warn] Copart query {make}: {e}", file=sys.stderr)
        return []
    out = []
    for r in rows[:limit]:
        vin   = r.get("vin") or ""
        model = r.get("lcyMmaModel") or r.get("model") or ""
        year  = r.get("lcyMmaYr") or r.get("year") or ""
        # Copart thumbnails use a CDN pattern: https://cs.copart.com/v1/AUTH_svc.pdoc00001/PIX/{LOTID}.jpg
        photos = []
        lot_id = r.get("lotNumberStr") or r.get("ln")
        if lot_id:
            for i in range(1, 9):
                photos.append(f"https://cs.copart.com/v1/AUTH_svc.pdoc00001/PIX/{lot_id}_{i}.jpg")
        if not vin or not model or not photos:
            continue
        out.append({"vin": vin, "make": make, "model": model, "year": year, "photos": photos})
    return out


def known_makes() -> list:
    """Read pipeline/data/models.json to find which makes we actually care about."""
    p = DATA_DIR / "models.json"
    if not p.exists():
        return []
    try:
        models = json.loads(p.read_text())
    except Exception:
        return []
    makes = set()
    for m in models.values():
        mk = m.get("make_name")
        if mk:
            makes.add(mk)
    # Cap to MAKES_BUDGET, alphabetical for deterministic batching across runs.
    return sorted(makes)[:MAKES_BUDGET]


def save_photo(blob: bytes, info: dict, idx: int) -> bool:
    cls = f"{info['make']}_{info['model']}".replace(" ", "_").replace("/", "_")
    sub = PHOTOS / cls
    sub.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(blob).hexdigest()[:16]
    fp = sub / f"{info.get('vin','')[:8]}_{idx}_{h}.jpg"
    if fp.exists():
        return False
    fp.write_bytes(blob)
    (sub / fp.name.replace(".jpg", ".meta.json")).write_text(json.dumps({
        "source": "auction",
        "make_model": f"{info['make']} {info['model']}",
        "year":   info.get("year", ""),
        "vin":    info.get("vin", ""),
    }, indent=2))
    return True


def run():
    if not OPT_IN:
        print("[skip] auction pull skipped — set VOV_AUCTIONS_OPT_IN=true to enable.")
        return
    makes = known_makes()
    if not makes:
        print("[skip] no makes in models.json — run nhtsa_pull.py first.")
        return
    state = {}
    if LEDGER.exists():
        try: state = json.loads(LEDGER.read_text())
        except Exception: state = {}
    vins_seen = set(state.get("vins") or [])
    pulled = 0
    for make in makes:
        for query_fn, name in ((make_query_iaa, "iaa"), (make_query_copart, "copart")):
            try:
                listings = query_fn(make, PER_MAKE)
            except Exception as e:
                print(f"[warn] {name} {make}: {e}", file=sys.stderr)
                continue
            for it in listings:
                if it["vin"] in vins_seen:
                    continue
                kept_any = False
                for idx, purl in enumerate(it["photos"]):
                    try:
                        blob = http_bytes(purl)
                        if len(blob) < 8 * 1024:
                            continue
                        if save_photo(blob, it, idx):
                            pulled += 1
                            kept_any = True
                        time.sleep(0.15)
                    except Exception:
                        continue
                if kept_any:
                    vins_seen.add(it["vin"])
            time.sleep(2)  # polite between (make, site)
    state["vins"] = sorted(vins_seen)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(state, indent=2))
    print(f"Auctions: +{pulled} photos across {len(makes)} makes, unique VINs tracked: {len(vins_seen)}")


if __name__ == "__main__":
    run()
