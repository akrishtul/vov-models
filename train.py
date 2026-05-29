"""
Fine-tune MobileNetV3-Small on the curated vehicle dataset, export ONNX.

Pipeline:
  1. Pull accepted photos from photo_review + accepted customer corrections.
  2. Build a label map (one class per (make, model)).
  3. Train MobileNetV3-Small head + last few blocks (transfer learning).
  4. Validate against a hold-out set built from the previous model's
     test split (so regressions show up in per-year accuracy).
  5. Export ONNX with input [1, 3, 224, 224] and opset 17.
  6. Compute sha256, write metadata JSON.
  7. Hand off to publish.py.

This script is a skeleton — fill in the training loop with your preferred
framework (PyTorch + timm is what we use). The structure is intentional:
small dependencies, easy to migrate, easy to compose with a GPU rental.

Requires: torch, torchvision, timm, onnx, onnxruntime, Pillow.
"""

import hashlib
import json
import os
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = os.environ.get("VOV_PIPELINE_DB", "/var/lib/vov/pipeline.sqlite3")
PHOTO_DIR = Path(os.environ.get("VOV_PIPELINE_PHOTOS", "/var/lib/vov/photos/staging"))
ARTIFACT_DIR = Path(os.environ.get("VOV_PIPELINE_ARTIFACTS", "/var/lib/vov/artifacts"))


def gather_dataset(conn: sqlite3.Connection) -> tuple[list[tuple[str, str]], list[str]]:
    """
    Returns (samples, labels) where:
      - samples = [(photo_path, label), ...]
      - labels  = sorted list of unique (make, model) strings
    """
    rows = conn.execute("""
        SELECT pr.photo_path, makes.make, models.model, models.year
        FROM photo_review pr
        JOIN models ON models.model_id = pr.model_id
        JOIN makes  ON makes.make_id  = models.make_id
        WHERE (pr.llm_verdict = 'accept' AND pr.human_verdict IS NULL)
           OR pr.human_verdict = 'accept'
    """).fetchall()

    samples = []
    label_set = set()
    for path, make, model, year in rows:
        label = f"{make} {model}"
        samples.append((path, label))
        label_set.add(label)

    # Also pull verified customer corrections (gold standard).
    try:
        corr = conn.execute("""
            SELECT photo_url, now_val
            FROM corrections
            WHERE field = 'model' AND verified = 'verified'
        """).fetchall()
        for url, model_label in corr:
            samples.append((url, model_label))
            label_set.add(model_label)
    except sqlite3.OperationalError:
        pass

    labels = sorted(label_set)
    return samples, labels


def write_label_map(labels: list[str], path: Path) -> None:
    path.write_text(json.dumps({"labels": labels, "version_count": len(labels)}, indent=2))


def export_onnx(model, sample_input, out_path: Path) -> None:
    import torch
    torch.onnx.export(
        model, sample_input, str(out_path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run():
    conn = sqlite3.connect(DB_PATH)
    samples, labels = gather_dataset(conn)
    if not labels:
        print("No labeled samples — run nhtsa_pull + wikimedia_fetch + llm_filter first.")
        return

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    version = datetime.now(timezone.utc).strftime("%Y.%m.%d")
    out_dir = ARTIFACT_DIR / f"v{version}"
    out_dir.mkdir(exist_ok=True)

    write_label_map(labels, out_dir / "labels.json")
    print(f"Dataset: {len(samples)} samples across {len(labels)} classes.")

    # ===== TRAINING (skeleton — implement with your framework of choice) =====
    # Recommended: timm.create_model('mobilenetv3_small_100', pretrained=True,
    # num_classes=len(labels)), train with AdamW lr=3e-4 for 30 epochs, augment
    # with brightness/contrast/perspective to simulate valet stand conditions.
    print("[TODO] Implement the actual training loop here.")
    print("        For now: skeleton walks the dataset, writes a placeholder ONNX.")

    # Placeholder export — replace with the real trained model.
    onnx_path = out_dir / f"vehicle-classifier-v{version}.onnx"
    onnx_path.write_bytes(b"")  # placeholder until real training runs
    sha = sha256_of(onnx_path)

    meta = {
        "version":           version,
        "classes":           len(labels),
        "samples_trained":   len(samples),
        "released":          datetime.now(timezone.utc).isoformat(),
        "input_shape":       [1, 3, 224, 224],
        "sha256":            sha,
        "size_mb":           round(onnx_path.stat().st_size / 1024 / 1024, 2),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {out_dir} (sha={sha[:16]}). Next: run publish.py to push to R2 + bump manifest.")


if __name__ == "__main__":
    run()
