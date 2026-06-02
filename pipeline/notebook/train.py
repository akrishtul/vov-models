"""
VOV vehicle classifier — Kaggle training script.

v2 — Seed-with-Stanford-Cars edition.

Runs on a Kaggle P100. Loads two datasets and trains a single classifier:
  1. Stanford Cars (Hugging Face: tanganke/stanford_cars)
       16,185 photos, 196 classes (Make/Model/Year, mostly 2009-2012).
       This is the SEED dataset — guarantees we have enough data to train
       on day 1 instead of waiting weeks for Wikimedia accumulation.
  2. Our own Wikimedia-fetched + verified customer corrections data
       Accumulates over time via daily-pull workflow + customer fixes.
       Adds newer (2018+) makes/models that Stanford Cars doesn't cover.

Fine-tunes MobileNetV3-Small from ImageNet pretraining, exports ONNX
for on-device browser inference.

Outputs to /kaggle/working/:
  vehicle-classifier-v{YYYY.MM.DD}.onnx
  vehicle-classifier-v{YYYY.MM.DD}.labels.json
  metadata.json   (test accuracy, sample count, etc.)

These artifacts are picked up by the monthly-train.yml GitHub Action,
which creates a Release + bumps manifest.json.

The script is intentionally self-contained — Kaggle gives us a fresh
container each run. We pip install everything inline.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# -------- Bootstrap deps --------
# Kaggle's free GPU is a Tesla P100 (sm_60 / Pascal architecture). PyTorch 2.5+
# dropped support for sm_60 — Kaggle's preinstalled torch errors at first
# .to('cuda') call with "Tesla P100 is not compatible with this PyTorch install".
# Pin to torch 2.4.1 + torchvision 0.19.1 (last release supporting Pascal).
# Force-reinstall with --upgrade so we override Kaggle's preinstalled newer torch.
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade",
                       "torch==2.4.1", "torchvision==0.19.1"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "timm>=1.0", "onnx>=1.15", "Pillow>=10.0", "tqdm",
                       "datasets>=2.18"])

import torch
import torch.nn as nn
import timm
from PIL import Image
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
from torchvision import transforms

# -------- Config --------
REPO          = os.environ.get("VOV_REPO",          "akrishtul/vov-models")
BRANCH        = os.environ.get("VOV_BRANCH",        "main")
OUT_DIR       = Path("/kaggle/working")
WORK_DIR      = Path("/kaggle/working/repo")
VERSION       = os.environ.get("MODEL_VERSION") or datetime.now(timezone.utc).strftime("%Y.%m.%d")
INPUT_SIZE    = 224
BATCH         = 64
# v2026.05.30 — bumped 4+12 → 6+24 to extract more from the bigger pool.
# P100 wall-clock budget: 16 epochs × ~45s = ~12 min; 30 epochs × ~45s = ~22 min,
# well under the 4.5h GH timeout (and well under Kaggle's 12h kernel cap).
EPOCHS_HEAD   = 6
EPOCHS_FULL   = 24
LR_HEAD       = 1e-3
LR_FULL       = 3e-4
TEST_FRAC     = 0.10
VAL_FRAC      = 0.10
SEED          = 42

# Hugging Face seed datasets to pull. Each entry is tried independently —
# missing/private/renamed datasets log a [warn] and the run continues with
# whatever else loaded.
#
# v2026.05.30 LESSON (run #11): adding 'Multimodal-Fatima/StanfordCars_train'
# as a second HF mirror created 393 fake classes (vs the canonical 196)
# because its label-string convention differs from tanganke's. Same car,
# different label string → model treats them as different classes → 9.95%
# test top-1 collapse. Reverted to a single canonical seed. To add MORE
# real-world classes safely, drop them under pipeline/data/photos/<source>/
# <Make_Model>/ and the labeled-source walker below picks them up under
# a single shared label namespace.
SEED_DATASETS = [
    {
        "hf_name": "tanganke/stanford_cars",
        "train_split": "train",
        "test_split":  "test",
        "image_col":   "image",
        "label_col":   "label",
    },
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
print(f"Device: {DEVICE}")

# -------- Fetch dataset --------
# Clone (or update) the models repo so we have models.json + data/photos/.
if not WORK_DIR.exists():
    subprocess.check_call(["git", "clone", "--depth", "1",
                           "--branch", BRANCH,
                           f"https://github.com/{REPO}.git", str(WORK_DIR)])

# v2026.06.01 — models.json (Wikimedia label/photo registry) is optional. On a
# fresh repo it doesn't exist yet; the first model trains on the Stanford Cars
# seed alone. Previously this open() crashed the run before training (task #49).
models_meta = {}
_models_json = WORK_DIR / "pipeline/data/models.json"
if _models_json.exists():
    with open(_models_json) as f:
        models_meta = json.load(f)
else:
    print("[info] no pipeline/data/models.json yet — training on Stanford Cars seed only (expected on first run).")

# -------- Build labels --------
# Strategy: take Stanford Cars' label set as the canonical class taxonomy
# (it's 196 well-curated US-relevant car classes), then add any new
# Wikimedia-fetched classes that aren't already there. The model has to
# learn one class per output.
print("Loading Stanford Cars dataset from Hugging Face…")
from datasets import load_dataset, Image as HFImage

seed_train_items: list[tuple] = []   # (PIL-loader, label_str)
seed_test_items:  list[tuple] = []
seed_label_set:   set[str]    = set()

for spec in SEED_DATASETS:
    name = spec["hf_name"]
    print(f"  → {name}")
    try:
        hf_train = load_dataset(name, split=spec["train_split"])
    except Exception as e:
        print(f"    [warn] failed to load train split for {name}: {e} — skipping")
        continue
    # v2026.05.30 — gracefully handle single-split mirrors. If test_split fails
    # to load (e.g. name has no test split), reuse the train and we'll handle
    # the eval split downstream by carving TEST_FRAC out of train.
    hf_test = None
    if spec.get("test_split") and spec["test_split"] != spec["train_split"]:
        try:
            hf_test = load_dataset(name, split=spec["test_split"])
        except Exception as e:
            print(f"    [info] no separate test split for {name}: {e}")
    if hf_test is None:
        hf_test = hf_train   # placeholder; we won't double-count by skipping convert below

    # Resolve label int → name from the feature spec.
    try:
        names = hf_train.features[spec["label_col"]].names
    except Exception:
        names = []
    if not names:
        print(f"    [warn] {name} has no label names; skipping")
        continue

    def _convert(ds, dest, split):
        # v2026.06.01 FIX — stamp the split ("train"/"test") so __getitem__ maps
        # each row back to the CORRECT dataset. The old code's split check
        # (`"train" in tuple`) was always False, so every TRAIN item was paired
        # with a TEST-set image → labels mismatched → ~20% accuracy. Core bug.
        # Read the label column directly (no per-row image decode).
        lbl_ints = ds[spec["label_col"]]
        for i, lbl_int in enumerate(lbl_ints):
            lbl = names[lbl_int] if 0 <= lbl_int < len(names) else f"unknown_{lbl_int}"
            dest.append(("hf", name, spec["image_col"], i, lbl, split))
            seed_label_set.add(lbl)

    # The HF datasets keep images lazy-cast already, so the rows above are
    # actually pointers into the HF datasets we now reference by closure.
    # We stash the dataset objects below for the Dataset class to access.
    spec["_train_ds"] = hf_train
    spec["_test_ds"]  = hf_test
    _convert(hf_train, seed_train_items, "train")
    # Only count as test items if it's a genuinely distinct split. Single-split
    # mirrors get folded into train+val via TEST_FRAC carve downstream.
    if hf_test is not hf_train:
        _convert(hf_test,  seed_test_items, "test")
        print(f"    {len(hf_train)} train + {len(hf_test)} test photos, {len(names)} classes")
    else:
        print(f"    {len(hf_train)} train photos (single split — test carved downstream), {len(names)} classes")

print(f"\nSeed dataset total: {len(seed_train_items)} train + {len(seed_test_items)} test, "
      f"{len(seed_label_set)} classes")

# -------- Wikimedia samples on top --------
wiki_items: list[tuple] = []
wiki_label_set: set[str] = set()
photos_root = WORK_DIR / "pipeline/data/photos"

for key, m in models_meta.items():
    if m.get("state") not in ("triaged", "fetched"):
        continue
    label = f"{m['make_name']} {m['model']}"
    for photo in m.get("photos", []):
        v = photo.get("verdict", "accept")
        if v == "reject":
            continue
        p = WORK_DIR / "pipeline/data" / photo["path"] if not Path(photo["path"]).is_absolute() else Path(photo["path"])
        if not p.exists():
            continue
        wiki_items.append(("path", str(p), label))
        wiki_label_set.add(label)

print(f"Wikimedia samples: {len(wiki_items)} photos, {len(wiki_label_set)} classes "
      f"(of which {len(wiki_label_set - seed_label_set)} new vs seed)")

# v2026.05.30 — Walk every additional labeled photo source dropped under
# pipeline/data/photos/<source>/<Make_Model>/*.jpg. Each per-source pull
# script lays out files in this canonical shape, so train.py is source-agnostic.
# Skips sources whose folders look "unlabeled" (e.g. mapillary_unlabeled/,
# openimages_unlabeled/) — those wait for semi-supervised labeling later.
LABELED_SOURCES = ("vmmrdb", "auctions", "corrections")
extra_source_items: list[tuple] = []
extra_source_label_set: set[str] = set()
for src_name in LABELED_SOURCES:
    src_root = WORK_DIR / "pipeline/data/photos" / src_name
    if not src_root.exists():
        continue
    n_src = 0
    n_cls = 0
    for cls_dir in src_root.iterdir():
        if not cls_dir.is_dir():
            continue
        label = cls_dir.name.replace("_", " ").strip()
        if not label:
            continue
        kept = 0
        for img in cls_dir.glob("*.jpg"):
            extra_source_items.append(("path", str(img), label))
            kept += 1
        for img in cls_dir.glob("*.jpeg"):
            extra_source_items.append(("path", str(img), label))
            kept += 1
        for img in cls_dir.glob("*.png"):
            extra_source_items.append(("path", str(img), label))
            kept += 1
        if kept:
            n_src += kept
            n_cls += 1
            extra_source_label_set.add(label)
    print(f"  [{src_name}] {n_src} photos across {n_cls} classes")

wiki_items.extend(extra_source_items)
wiki_label_set.update(extra_source_label_set)
print(f"Combined Wikimedia + labeled sources: {len(wiki_items)} photos, "
      f"{len(wiki_label_set)} classes")

# Final label space = union of both sources.
all_labels = sorted(seed_label_set | wiki_label_set)
label_to_idx = {lbl: i for i, lbl in enumerate(all_labels)}
print(f"\nFinal classifier: {len(all_labels)} classes total "
      f"({len(seed_train_items)} seed train + {len(wiki_items)} wiki samples)")

if len(seed_train_items) == 0 and len(wiki_items) == 0:
    print("FATAL: no samples found anywhere. Check Hugging Face download + Wikimedia paths.")
    sys.exit(2)

# -------- Dataset classes --------
train_tx = transforms.Compose([
    transforms.Resize((INPUT_SIZE + 24, INPUT_SIZE + 24)),
    transforms.RandomCrop(INPUT_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomPerspective(distortion_scale=0.15, p=0.4),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
eval_tx = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class UnifiedDataset(Dataset):
    """One Dataset that pulls from either Hugging Face rows or local file paths,
    depending on the item type stamped at build time."""

    def __init__(self, items, tx, seed_specs):
        self.items = items
        self.tx = tx
        # build lookup so we can resolve ("hf", name, col, i, lbl) rows back
        # to the right HF dataset
        self.hf_lookup = {}  # (unused; kept for compat)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        item = self.items[i]
        try:
            if item[0] == "hf":
                _, name, col, idx, lbl, split = item
                ds = None
                for s in SEED_DATASETS:
                    if s["hf_name"] == name:
                        ds = s["_train_ds"] if split == "train" else s["_test_ds"]
                        break
                img = ds[idx][col] if (ds is not None and idx < len(ds)) else Image.new("RGB", (INPUT_SIZE, INPUT_SIZE))
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(img)
                img = img.convert("RGB")
                label = lbl
            else:
                _, path, label = item
                img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"  [item load failed] {item}: {e}", file=sys.stderr)
            img = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE))
            label = item[-1] if isinstance(item, tuple) else all_labels[0]
        return self.tx(img), label_to_idx[label]


# Simpler approach — build per-split lists and pass them to UnifiedDataset.
# The HF rows already encode which split they came from (train vs test).
# We keep all Wikimedia items in the train pool so they augment without
# poisoning the held-out test set.

# Test set: pull only from HF's test split (Stanford Cars holdout).
test_items = seed_test_items

# Train + val: HF train + all Wikimedia.
train_pool = seed_train_items + wiki_items

# Stratify-by-frequency wouldn't help here (some classes have 1 sample),
# so use a plain random val split of the pooled train set.
import random
random.seed(SEED)
random.shuffle(train_pool)
n_val   = max(1, int(len(train_pool) * VAL_FRAC))
val_items   = train_pool[:n_val]
train_items = train_pool[n_val:]

# Per-DS helper that ignores the test-vs-train guess and just trusts the items.
class SimpleDataset(Dataset):
    def __init__(self, items, tx):
        self.items = items
        self.tx = tx

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        item = self.items[i]
        try:
            if item[0] == "hf":
                # v2026.06.01 FIX — items are 6-tuples carrying their split; pick the
                # matching HF dataset by name+split so train images keep their labels.
                _, name, col, idx, lbl, split = item
                ds = None
                for s in SEED_DATASETS:
                    if s["hf_name"] == name:
                        ds = s["_train_ds"] if split == "train" else s["_test_ds"]
                        break
                img = ds[idx][col] if (ds is not None and idx < len(ds)) else Image.new("RGB", (INPUT_SIZE, INPUT_SIZE))
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(img)
                img = img.convert("RGB")
                label = lbl
            else:
                _, path, label = item
                img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"  [item load failed] {item}: {e}", file=sys.stderr)
            img = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE))
            # Pull the label from the correct position, never the split tag.
            if isinstance(item, tuple) and item and item[0] == "hf" and len(item) >= 6:
                label = item[4]
            elif isinstance(item, tuple) and len(item) >= 3:
                label = item[2]
            else:
                label = all_labels[0]
        return self.tx(img), label_to_idx[label]


train_ds = SimpleDataset(train_items, train_tx)
val_ds   = SimpleDataset(val_items,   eval_tx)
test_ds  = SimpleDataset(test_items,  eval_tx)

print(f"\nSplit sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=2, pin_memory=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)
test_dl  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)

# -------- Model --------
# v2026.06.01 — backbone via torchvision (weights from download.pytorch.org).
# timm's pretrained download (HF hub model.safetensors) KeyError'd on Kaggle's
# current image; torchvision's MobileNetV3-Small is the same architecture/shape
# (input [1,3,224,224] -> logits [1,N]) so the on-device ONNX/CoreML stays compatible.
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
_in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(_in_features, len(all_labels))
model.to(DEVICE)
crit = nn.CrossEntropyLoss(label_smoothing=0.05)


def run_epoch(dl, train_mode: bool, opt=None) -> tuple[float, float]:
    model.train(train_mode)
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in dl:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        if train_mode:
            opt.zero_grad()
        with torch.set_grad_enabled(train_mode):
            logits = model(x)
            loss = crit(logits, y)
            if train_mode:
                loss.backward()
                opt.step()
        loss_sum += loss.item() * x.size(0)
        correct  += (logits.argmax(1) == y).sum().item()
        total    += x.size(0)
    return loss_sum / total, correct / total


# -------- Head-only phase --------
for p in model.parameters(): p.requires_grad = False
for p in model.classifier.parameters(): p.requires_grad = True
opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD)
for ep in range(EPOCHS_HEAD):
    tl, ta = run_epoch(train_dl, True, opt)
    vl, va = run_epoch(val_dl,   False)
    print(f"[head ep {ep+1}/{EPOCHS_HEAD}] train_loss={tl:.4f} train_acc={ta:.4f} val_loss={vl:.4f} val_acc={va:.4f}")

# -------- Full fine-tune --------
for p in model.parameters(): p.requires_grad = True
opt = torch.optim.AdamW(model.parameters(), lr=LR_FULL, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_FULL)
best_val = 0.0
for ep in range(EPOCHS_FULL):
    tl, ta = run_epoch(train_dl, True, opt)
    vl, va = run_epoch(val_dl,   False)
    sched.step()
    if va > best_val:
        best_val = va
        torch.save(model.state_dict(), OUT_DIR / "best.pt")
    print(f"[full ep {ep+1}/{EPOCHS_FULL}] train_loss={tl:.4f} train_acc={ta:.4f} val_loss={vl:.4f} val_acc={va:.4f}  best={best_val:.4f}")

# -------- Test --------
model.load_state_dict(torch.load(OUT_DIR / "best.pt"))
tl, ta = run_epoch(test_dl, False)
print(f"\nTest accuracy (top-1): {ta:.4f}")

# top-5
model.eval()
top5_correct = 0
total = 0
with torch.no_grad():
    for x, y in test_dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        top5 = logits.topk(min(5, len(all_labels)), dim=1).indices
        top5_correct += (top5 == y.unsqueeze(1)).any(dim=1).sum().item()
        total += y.size(0)
top5 = top5_correct / max(total, 1)
print(f"Test accuracy (top-5): {top5:.4f}")

# -------- Export ONNX --------
onnx_path   = OUT_DIR / f"vehicle-classifier-v{VERSION}.onnx"
labels_path = OUT_DIR / f"vehicle-classifier-v{VERSION}.labels.json"
meta_path   = OUT_DIR / "metadata.json"

dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device=DEVICE)
torch.onnx.export(
    model, dummy, str(onnx_path),
    input_names=["input"], output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17,
)
print(f"Exported ONNX: {onnx_path} ({onnx_path.stat().st_size / (1024*1024):.2f} MB)")

with open(labels_path, "w") as f:
    json.dump({"labels": all_labels, "version_count": len(all_labels)}, f)

import hashlib
sha = hashlib.sha256(onnx_path.read_bytes()).hexdigest()

with open(meta_path, "w") as f:
    json.dump({
        "version":          VERSION,
        "classes":          len(all_labels),
        "samples_trained":  len(train_items) + len(val_items),
        "samples_test":     len(test_items),
        "released":         datetime.now(timezone.utc).isoformat(),
        "input_shape":      [1, 3, INPUT_SIZE, INPUT_SIZE],
        "sha256":           sha,
        "size_mb":          round(onnx_path.stat().st_size / (1024 * 1024), 2),
        "test_accuracy":    {"top1": round(ta, 4), "top5": round(top5, 4)},
        "training_config":  {
            "epochs_head":   EPOCHS_HEAD,
            "epochs_full":   EPOCHS_FULL,
            "lr_head":       LR_HEAD,
            "lr_full":       LR_FULL,
            "batch_size":    BATCH,
            "input_size":    INPUT_SIZE,
            "backbone":      "torchvision_mobilenet_v3_small",
            "seed_datasets": [s["hf_name"] for s in SEED_DATASETS],
            "wikimedia_samples": len(wiki_items),
        },
    }, f, indent=2)

print(f"\n✓ Done. Artifacts in {OUT_DIR}:")
for p in OUT_DIR.glob("vehicle-classifier-v*"):
    print(f"  {p.name} ({p.stat().st_size:,} bytes)")
print(f"  metadata.json (test top-1: {ta:.4f}, top-5: {top5:.4f})")
