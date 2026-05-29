"""
Daily NHTSA VPIC pull — GitHub Actions friendly.

State is stored as JSON files in pipeline/data/ (committed back to the repo
by the workflow). No database server needed. No cost.

Files written:
  pipeline/data/makes.json      — { make_id: { name, first_seen } }
  pipeline/data/models.json     — { "<make_id>:<model>:<year>": { state, first_seen } }
  pipeline/data/pull_history.json — append-only log

NHTSA VPIC API: https://vpic.nhtsa.dot.gov/api/ — free, public, no key.

v1.3.2 — valet-realistic mode:
  - Filters NHTSA's 12,000 makes down to ~120 brands valet operations actually
    see (luxury + premium + mainstream + EV). NHTSA includes trailers, RVs,
    motorcycles, kit-cars, defunct brands — none useful for restaurant valet.
  - First bootstrap pull does ONE model year (THIS_YEAR); subsequent runs
    extend year range as the catalog grows. Avoids 13-hour first run.
  - Incremental save every BATCH_SAVE_EVERY requests so a timeout doesn't
    discard hours of work.
  - VPIC API takes ~0.5-1s per call. 120 makes × 1 year = ~120 requests, ~2 min.
"""

import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR  = Path(os.environ.get("VOV_PIPELINE_DATA", "pipeline/data"))
VPIC_BASE = "https://vpic.nhtsa.dot.gov/api/vehicles"

THIS_YEAR  = datetime.now(timezone.utc).year

# Year range scales with how much of the catalog we've already pulled.
# Bootstrap (no prior models) = 1 year. Subsequent runs = 4 years (this-1 → this+3).
BOOTSTRAP_YEAR_RANGE = range(THIS_YEAR, THIS_YEAR + 1)         # 1 year
STEADY_YEAR_RANGE    = range(THIS_YEAR - 1, THIS_YEAR + 3)     # 4 years

# Whitelist of car brands valet operations regularly see — covers ~99% of
# US restaurant/hotel valet traffic. We filter NHTSA's GetAllMakes by this
# instead of trying to pull every motorcycle + trailer + obscure brand.
# Names match NHTSA's MAKE_NAME column exactly (uppercase).
VALET_BRAND_WHITELIST = {
    # Luxury / premium European
    "ACURA", "ALFA ROMEO", "ASTON MARTIN", "AUDI", "BENTLEY",
    "BMW", "BUGATTI", "FERRARI", "JAGUAR", "LAMBORGHINI",
    "LAND ROVER", "LEXUS", "LOTUS", "MASERATI", "MAYBACH",
    "MCLAREN", "MERCEDES-BENZ", "MINI", "PAGANI", "PORSCHE",
    "ROLLS-ROYCE", "SMART", "VOLVO",
    # Premium / mainstream American
    "BUICK", "CADILLAC", "CHEVROLET", "CHRYSLER", "DODGE",
    "FORD", "GMC", "JEEP", "LINCOLN", "RAM", "TESLA",
    # Premium / mainstream Asian
    "GENESIS", "HONDA", "HYUNDAI", "INFINITI", "ISUZU",
    "KIA", "MAZDA", "MITSUBISHI", "NISSAN", "SCION",
    "SUBARU", "SUZUKI", "TOYOTA",
    # Mainstream European
    "FIAT", "MERCURY", "MINI COOPER", "PEUGEOT", "RENAULT",
    "SAAB", "SEAT", "SKODA", "VOLKSWAGEN",
    # Modern EVs / startups
    "FISKER", "KARMA", "LUCID", "LUCID MOTORS", "POLESTAR",
    "RIVIAN", "VINFAST",
    # Trucks / SUVs / commercial we DO see at hotels
    "FREIGHTLINER", "FUSO", "HUMMER", "INTERNATIONAL", "MACK",
    "PETERBILT", "STERLING", "VOLVO TRUCK", "WESTERN STAR",
    # Niche / collector / specialty (occasional)
    "BENTLEY MOTORS", "MORGAN", "NOBLE", "PANOZ", "SALEEN",
    "SHELBY", "TVR",
    # Classic / older that still hit the curb
    "AMERICAN MOTORS", "DATSUN", "EAGLE", "GEO", "OLDSMOBILE",
    "PLYMOUTH", "PONTIAC", "SATURN",
}

# Pause between VPIC requests so we don't hammer the API.
SLEEP_BETWEEN_REQS_SEC = 0.4

# Save partial progress every N model lookups so a timeout doesn't lose
# everything. Each save is a fast disk write to pipeline/data/.
BATCH_SAVE_EVERY = 20


def http_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "vov-pipeline/1.3.2 (github.com/akrishtul/vov-models)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def load(name: str, default):
    p = DATA_DIR / name
    if p.exists():
        return json.loads(p.read_text())
    return default


def save(name: str, data) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    p = DATA_DIR / name
    p.write_text(json.dumps(data, indent=2, sort_keys=True))


def is_whitelisted(name: str) -> bool:
    n = (name or "").strip().upper()
    return n in VALET_BRAND_WHITELIST


def pull_makes(existing: dict) -> int:
    """Fetch NHTSA's full make list, but only ADD whitelisted brands to our
    existing catalog. Anything outside the whitelist is silently skipped.
    Idempotent — re-running adds zero new makes once the whitelist is covered."""
    data = http_json(f"{VPIC_BASE}/GetAllMakes?format=json")
    now = datetime.now(timezone.utc).isoformat()
    new_count = 0
    for row in data.get("Results", []):
        mid = str(row["Make_ID"])
        name = (row.get("Make_Name") or "").strip()
        if not name or mid in existing:
            continue
        if not is_whitelisted(name):
            continue
        existing[mid] = {"name": name, "first_seen": now}
        new_count += 1
    return new_count


def pull_models(makes: dict, models: dict) -> int:
    """For every make in our local catalog (already whitelist-filtered) and
    every year in the active YEAR_RANGE, fetch model list. Incremental save
    every BATCH_SAVE_EVERY requests so a timeout doesn't discard work."""

    # First run (no models cached): pull only THIS_YEAR to finish in minutes,
    # not hours. Subsequent runs extend to the steady-state range.
    year_range = BOOTSTRAP_YEAR_RANGE if not models else STEADY_YEAR_RANGE

    now = datetime.now(timezone.utc).isoformat()
    new_count = 0
    reqs_since_save = 0
    total_reqs = 0

    for mid, mk in makes.items():
        for year in year_range:
            total_reqs += 1
            url = f"{VPIC_BASE}/GetModelsForMakeIdYear/makeId/{mid}/modelyear/{year}?format=json"
            try:
                data = http_json(url)
            except Exception as e:
                print(f"[warn] {mk['name']} {year}: {e}", file=sys.stderr)
                time.sleep(SLEEP_BETWEEN_REQS_SEC)
                continue
            for row in data.get("Results", []):
                model = (row.get("Model_Name") or "").strip()
                if not model:
                    continue
                key = f"{mid}:{model}:{year}"
                if key in models:
                    continue
                models[key] = {
                    "make_id":      int(mid),
                    "make_name":    mk["name"],
                    "model":        model,
                    "year":         year,
                    "state":        "pending",
                    "photos":       [],
                    "first_seen":   now,
                }
                new_count += 1
            reqs_since_save += 1
            time.sleep(SLEEP_BETWEEN_REQS_SEC)
            if reqs_since_save >= BATCH_SAVE_EVERY:
                save("models.json", models)
                reqs_since_save = 0
                print(f"[batch] saved progress after {total_reqs} requests "
                      f"(+{new_count} models so far)", file=sys.stderr)
    return new_count


def run():
    makes  = load("makes.json", {})
    models = load("models.json", {})
    history = load("pull_history.json", [])

    started = datetime.now(timezone.utc).isoformat()
    try:
        new_makes  = pull_makes(makes)
        # Persist makes before the long model-loop in case THAT step times out.
        save("makes.json", makes)
        new_models = pull_models(makes, models)
    except Exception as e:
        print(f"[fatal] {e}", file=sys.stderr)
        # Save whatever we have so we don't lose progress.
        save("makes.json", makes)
        save("models.json", models)
        sys.exit(1)

    finished = datetime.now(timezone.utc).isoformat()
    history.append({
        "started":  started,
        "finished": finished,
        "new_makes":  new_makes,
        "new_models": new_models,
    })
    if len(history) > 365:
        history = history[-365:]

    save("makes.json",        makes)
    save("models.json",       models)
    save("pull_history.json", history)

    print(f"NHTSA pull: +{new_makes} makes, +{new_models} models "
          f"(total {len(makes)} makes / {len(models)} model-years)")


if __name__ == "__main__":
    run()
