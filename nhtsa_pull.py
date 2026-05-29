"""
Daily NHTSA VPIC pull — GitHub Actions friendly.

State is stored as JSON files in pipeline/data/ (committed back to the repo
by the workflow). No database server needed. No cost.

Files written:
  pipeline/data/makes.json      — { make_id: { name, first_seen } }
  pipeline/data/models.json     — { "<make_id>:<model>:<year>": { state, first_seen } }
  pipeline/data/pull_history.json — append-only log

NHTSA VPIC API: https://vpic.nhtsa.dot.gov/api/ — free, public, no key.
"""

import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR  = Path(os.environ.get("VOV_PIPELINE_DATA", "pipeline/data"))
VPIC_BASE = "https://vpic.nhtsa.dot.gov/api/vehicles"

THIS_YEAR  = datetime.now(timezone.utc).year
YEAR_RANGE = range(THIS_YEAR - 1, THIS_YEAR + 3)


def http_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "vov-pipeline/1.3 (github.com/.../vov-models)"})
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


def pull_makes(existing: dict) -> int:
    data = http_json(f"{VPIC_BASE}/GetAllMakes?format=json")
    now = datetime.now(timezone.utc).isoformat()
    new_count = 0
    for row in data.get("Results", []):
        mid = str(row["Make_ID"])
        name = (row.get("Make_Name") or "").strip()
        if not name or mid in existing:
            continue
        existing[mid] = {"name": name, "first_seen": now}
        new_count += 1
    return new_count


def pull_models(makes: dict, models: dict) -> int:
    now = datetime.now(timezone.utc).isoformat()
    new_count = 0
    for mid, mk in makes.items():
        for year in YEAR_RANGE:
            url = f"{VPIC_BASE}/GetModelsForMakeIdYear/makeId/{mid}/modelyear/{year}?format=json"
            try:
                data = http_json(url)
            except Exception as e:
                print(f"[warn] {mk['name']} {year}: {e}", file=sys.stderr)
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
    return new_count


def run():
    makes  = load("makes.json", {})
    models = load("models.json", {})
    history = load("pull_history.json", [])

    started = datetime.now(timezone.utc).isoformat()
    try:
        new_makes  = pull_makes(makes)
        new_models = pull_models(makes, models)
    except Exception as e:
        print(f"[fatal] {e}", file=sys.stderr)
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
