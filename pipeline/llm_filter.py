"""
LLM auto-filter — Gemini 1.5 Flash (free 1500 req/day).

For each fetched photo, asks Gemini whether the image clearly shows the
labeled (year, make, model) and whether it's training-usable. Updates
each model's photos array with a `verdict` field.

GitHub Actions friendly — reads/writes pipeline/data/models.json.

Why Gemini Flash:
  - 1,500 requests/day free, no card required
  - Multimodal (accepts images)
  - Good enough for binary "is this a Civic" judgment
  - Easy to swap for Cloudflare Workers AI Llama Vision (also free)

Skips silently if no API key is set in the environment.
"""

import base64
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

DATA_DIR = Path(os.environ.get("VOV_PIPELINE_DATA", "pipeline/data"))
API_KEY  = os.environ.get("GEMINI_API_KEY", "")
MODEL    = os.environ.get("VOV_LLM_MODEL", "gemini-1.5-flash-latest")
ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

PROMPT = (
    "You are a dataset curator. Look at this photo and answer two yes/no questions "
    "about a {year} {make} {model}. "
    "1) Does this photo clearly show that vehicle (or any year of the same generation)? "
    "2) Is the photo usable for ML training (vehicle is the main subject, "
    "fully visible, no heavy watermark, reasonable lighting)? "
    'Respond ONLY with JSON: {{"shows_vehicle":true|false,"trainable":true|false,'
    '"confidence":0.0-1.0,"reason":"short"}}'
)


def ask_gemini(image_bytes: bytes, prompt: str) -> dict:
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg",
                                 "data": base64.b64encode(image_bytes).decode()}},
            ]
        }],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
        },
    }
    req = urllib.request.Request(
        f"{ENDPOINT}?key={API_KEY}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        body = json.loads(r.read().decode("utf-8"))
    try:
        text = body["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(text)
    except (KeyError, IndexError, json.JSONDecodeError):
        return {"shows_vehicle": None, "trainable": None, "confidence": 0, "reason": "unparseable"}


def classify(v: dict) -> str:
    if v.get("shows_vehicle") is True and v.get("trainable") is True and v.get("confidence", 0) >= 0.85:
        return "accept"
    if v.get("shows_vehicle") is False or v.get("trainable") is False:
        return "reject"
    return "review"


def run():
    if not API_KEY:
        print("GEMINI_API_KEY not set — skipping LLM filter. Photos will go straight to human review queue.")
        return

    path = DATA_DIR / "models.json"
    if not path.exists():
        print("No models.json — nothing to filter.")
        return
    models = json.loads(path.read_text())

    accept = reject = review = 0
    daily_budget = 1400   # leave headroom under the 1500/day Gemini free cap

    for key, m in models.items():
        if m.get("state") != "fetched":
            continue
        for photo in m.get("photos", []):
            if "verdict" in photo:
                continue
            if daily_budget <= 0:
                break
            photo_path = DATA_DIR / photo["path"] if not Path(photo["path"]).is_absolute() else Path(photo["path"])
            if not photo_path.exists():
                continue
            try:
                v = ask_gemini(photo_path.read_bytes(), PROMPT.format(year=m["year"], make=m["make_name"], model=m["model"]))
            except Exception as e:
                print(f"[warn] gemini call: {e}", file=sys.stderr)
                time.sleep(2)
                continue
            verdict = classify(v)
            photo["verdict"]    = verdict
            photo["confidence"] = v.get("confidence", 0)
            photo["reason"]     = v.get("reason", "")
            if verdict == "accept": accept += 1
            elif verdict == "reject": reject += 1
            else: review += 1
            daily_budget -= 1
            time.sleep(0.5)

        # Move to next state if all photos have verdicts.
        if m["photos"] and all("verdict" in p for p in m["photos"]):
            m["state"] = "triaged"

        if daily_budget <= 0:
            break

    path.write_text(json.dumps(models, indent=2, sort_keys=True))
    print(f"Gemini filter: accept={accept}, reject={reject}, review={review} "
          f"({1400 - daily_budget} API calls used)")


if __name__ == "__main__":
    run()
