"""
Build / bump the manifest after a Kaggle training run.

Reads pipeline/artifacts/metadata.json (written by the Kaggle notebook),
computes sha256 of the ONNX, and writes pipeline/manifest.json. The
GitHub Actions workflow handles the Release creation; jsDelivr CDN
auto-picks-up the new manifest from the repo within minutes.

No external services, no API keys, no upload SDKs.
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
MANIFEST = ROOT / "manifest.json"

# Allow override for monthly-train.yml (runs from repo root, not pipeline/).
if not ARTIFACTS.exists() and Path("pipeline/artifacts").exists():
    ARTIFACTS = Path("pipeline/artifacts")
if not MANIFEST.parent.exists() and Path("pipeline/manifest.json").parent.exists():
    MANIFEST = Path("pipeline/manifest.json")

# GitHub repo where models are released. Set as env in the workflow.
GH_REPO = os.environ.get("GH_REPO", "yourorg/vov-models")  # override in workflow
VERSION = os.environ.get("MODEL_VERSION", "")
NOTES   = os.environ.get("RELEASE_NOTES", "Monthly retrain")


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def gh_release_url(version: str, filename: str) -> str:
    return f"https://github.com/{GH_REPO}/releases/download/v{version}/{filename}"


def jsdelivr_url(version: str, filename: str) -> str:
    # Pinned-version jsDelivr URL (mutable @latest also works once tag exists).
    return f"https://cdn.jsdelivr.net/gh/{GH_REPO}@v{version}/{filename}"


def run():
    if not VERSION:
        sys.exit("MODEL_VERSION env var required")

    meta_path = ARTIFACTS / "metadata.json"
    if not meta_path.exists():
        sys.exit(f"missing {meta_path}")
    meta = json.loads(meta_path.read_text())

    onnx = ARTIFACTS / f"vehicle-classifier-v{VERSION}.onnx"
    labels = ARTIFACTS / f"vehicle-classifier-v{VERSION}.labels.json"
    if not onnx.exists():
        sys.exit(f"missing {onnx}")
    if not labels.exists():
        sys.exit(f"missing {labels}")

    sha = sha256_of(onnx)
    size_mb = round(onnx.stat().st_size / (1024 * 1024), 2)

    # Regression gate vs. previous manifest.
    previous = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    prev_acc = ((previous.get("vehicle_classifier") or {}).get("test_accuracy") or {}).get("top1")
    new_acc  = (meta.get("test_accuracy") or {}).get("top1")
    if new_acc is not None and prev_acc is not None and new_acc < prev_acc - 0.005:
        if os.environ.get("VOV_FORCE_PUBLISH") == "1":
            print(f"[force] top-1 regressed {prev_acc:.4f} → {new_acc:.4f} — publishing anyway per VOV_FORCE_PUBLISH=1")
        else:
            sys.exit(f"ABORT — top-1 regressed {prev_acc:.4f} → {new_acc:.4f} (set VOV_FORCE_PUBLISH=1 to override)")

    manifest = {
        "schema_version": 1,
        "generated":      datetime.now(timezone.utc).isoformat(),
        "vehicle_classifier": {
            "version":            VERSION,
            "released":           datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "size_mb":            size_mb,
            "classes":            meta.get("classes"),
            "url":                gh_release_url(VERSION, onnx.name),
            "url_cdn":            jsdelivr_url(VERSION, onnx.name),
            "sha256":             sha,
            "min_plugin_version": "1.2.0",
            "input_shape":        meta.get("input_shape", [1, 3, 224, 224]),
            "label_map_url":      gh_release_url(VERSION, labels.name),
            "label_map_url_cdn":  jsdelivr_url(VERSION, labels.name),
            "changelog":          [ln.strip() for ln in NOTES.splitlines() if ln.strip()],
            "test_accuracy":      meta.get("test_accuracy", {}),
        },
        # v2026-05-31 — both plate models swapped from GitHub release URLs
        # (CORS-blocked: release-assets.githubusercontent.com refuses our
        # browser origin) to jsDelivr-served paths in our own repo's
        # /models/ folder. Originals mirrored verbatim into vov-models/models/.
        "plate_detector": previous.get("plate_detector") or {
            "version": "2024.10.0",
            "url": "https://cdn.jsdelivr.net/gh/akrishtul/vov-models@main/models/yolo-v9-t-384-license-plates-end2end.onnx",
            "size_mb": 7.4,
            "input_size": 384,
        },
        "plate_ocr": previous.get("plate_ocr") or {
            "version": "2024.10.0",
            "url": "https://cdn.jsdelivr.net/gh/akrishtul/vov-models@main/models/cct_s_v2_global.onnx",
            "size_mb": 5.0,
            "charset": "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_",
            "max_len": 9,
        },
    }

    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote manifest for vehicle-classifier v{VERSION} (sha {sha[:12]}, {size_mb} MB)")
    print(f"Once the GitHub Release tag exists, jsDelivr will serve:")
    print(f"  manifest: https://cdn.jsdelivr.net/gh/{GH_REPO}@main/pipeline/manifest.json")
    print(f"  model:    {jsdelivr_url(VERSION, onnx.name)}")


if __name__ == "__main__":
    run()
