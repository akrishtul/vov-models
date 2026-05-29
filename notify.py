"""
Slack webhook notifier.

Free — create a Slack Incoming Webhook in your workspace, paste the URL
into the SLACK_WEBHOOK GitHub secret. Workflows call this script for:

  python notify.py success     — monthly training completed
  python notify.py failure     — monthly training failed
  python notify.py health_ok   — daily health check passed (silent unless verbose)
  python notify.py rollback    — daily health check rolled back the manifest

Silent no-op if SLACK_WEBHOOK is not set, so all workflows can call it
unconditionally without breaking when no webhook is configured yet.
"""

import json
import os
import sys
import urllib.request
from pathlib import Path

WEBHOOK = os.environ.get("SLACK_WEBHOOK", "")
VERSION = os.environ.get("VERSION", "")
RUN_URL = os.environ.get("RUN_URL", "")
SOURCE  = os.environ.get("SOURCE", "")

MANIFEST = Path("pipeline/manifest.json")


def read_classifier_meta() -> dict:
    if not MANIFEST.exists():
        return {}
    try:
        return json.loads(MANIFEST.read_text()).get("vehicle_classifier", {})
    except Exception:
        return {}


def post(payload: dict) -> None:
    if not WEBHOOK:
        print("SLACK_WEBHOOK not set — skipping notification.")
        return
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        WEBHOOK, data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            print(f"Slack response: {r.status}")
    except Exception as e:
        print(f"Slack notify failed: {e}")


def success() -> None:
    vc = read_classifier_meta()
    top1 = (vc.get("test_accuracy") or {}).get("top1", "n/a")
    classes = vc.get("classes", "n/a")
    msg = {
        "text": (
            f":white_check_mark: *vov vehicle-classifier v{VERSION or vc.get('version', '?')}* published.\n"
            f"• Classes: {classes}\n"
            f"• Top-1 test accuracy: {top1}\n"
            f"• <{RUN_URL}|GitHub Action run>\n"
            f"Customers auto-update within 24h via jsDelivr."
        ),
    }
    post(msg)


def failure() -> None:
    msg = {
        "text": (
            f":x: *vov pipeline monthly training failed.*\n"
            f"<{RUN_URL}|Open the run logs>\n"
            "Service is not disrupted — customer plugins continue running the prior model. "
            "Investigate when you can; the next scheduled retrain will retry."
        ),
    }
    post(msg)


def health_ok() -> None:
    # Silent unless verbose flag. Just log.
    print(f"Health check OK (source={SOURCE})")


def rollback() -> None:
    msg = {
        "text": (
            ":rotating_light: *vov pipeline auto-rollback triggered.*\n"
            "Daily health check could not verify the published model. "
            f"Manifest reverted to previous version.\n<{RUN_URL}|Open the rollback run>"
        ),
    }
    post(msg)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "success"
    handlers = {
        "success":   success,
        "failure":   failure,
        "health_ok": health_ok,
        "rollback":  rollback,
    }
    h = handlers.get(mode)
    if not h:
        print(f"Unknown notify mode: {mode}")
        sys.exit(0)
    h()
