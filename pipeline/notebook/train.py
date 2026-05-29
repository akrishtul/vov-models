"""
VOV vehicle classifier — Kaggle training script.

Runs on a Kaggle P100. Reads the labeled dataset (Wikimedia-fetched +
verified customer corrections), fine-tunes MobileNetV3-Small from
ImageNet pretraining, exports ONNX for on-device browser inference.

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
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "torch>=2.2", "torchvision>=0.17", "timm>=1.0",
                       "onnx>=1.15", "Pillow>=10.0", "tqdm"])

import torch
import torch.nn as nn
import timm
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

# -------- Config --------
REPO          = os.environ.get("VOV_REPO",          "akrishtul/vov-models")
BRANCH        = os.environ.get("VOV_BRANCH",        "main")
OUT_DIR       = Path("/kaggle/working")
WORK_DIR      = Path("/kaggle/working/repo")
VERSION       = os.environ.get("MODEL_VERSION") or datetime.now(timezone.utc).strftime("%Y.%m.%d")
INPUT_SIZE    = 224
BATCH         = 64
EPOCHS_HEAD   = 6
EPOCHS_FULL   = 20
LR_HEAD       = 1e-3
LR_FULL       = 3e-4
TEST_FRAC     = 0.10
VAL_FRAC      = 0.10
SEED          = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)

# -------- Fetch dataset --------
# Clone (or update) the models repo so we have models.json + data/photos/.
if not WORK_DIR.exists():
    subprocess.check_call(["git", "clone", "--depth", "1",
                           "--branch", BRANCH,
                           f"https://github.com/{REPO}.git", str(WORK_DIR)])

with open(WORK_DIR / "pipeline/data/models.json") as f:
    models_meta = json.load(f)

# Build (path, label) pairs from accepted Wikimedia photos.
samples: list[tuple[Path, str]] = []
label_set: set[str] = set()
photos_root = WORK_DIR / "pipeline/data/photos"

for key, m in models_meta.items():
    if m.get("state") not in ("triaged", "fetched"):
        continue
    label = f"{m['make_name']} {m['model']}"
    for photo in m.get("photos", []):
        v = photo.get("verdict", "accept")    # default-accept when no LLM filter ran
        if v == "reject":
            continue
        p = WORK_DIR / "pipeline/data" / photo["path"] if not Path(photo["path"]).is_absolute() else Path(photo["path"])
        if not p.exists():
            continue
        samples.append((p, label))
        label_set.add(label)

if len(samples) < 100:
    print(f"Too few samples ({len(samples)}) — refusing to train a model that will look broken.")
    print("Add more data via the daily-pull workflow or wait for customer corrections to accumulate.")
    sys.exit(2)

labels = sorted(label_set)
label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
print(f"Dataset: {len(samples)} samples across {len(labels)} classes.")

# -------- Dataset class --------
train_tx = transforms.Compose([
    transforms.Resize((INPUT_SIZE + 24, INPUT_SIZE + 24)),
    transforms.RandomCrop(INPUT_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomPerspective(distortion_scale=0.15, p=0.4),  # mimic phone angle
    transforms.RandomGrayscale(p=0.05),                            # bad lighting
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
eval_tx = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class VehDataset(Dataset):
    def __init__(self, items, tx):
        self.items = items
        self.tx = tx

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # Replace broken decode with a black image so DataLoader doesn't crash.
            img = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE))
        return self.tx(img), label_to_idx[label]


full = VehDataset(samples, train_tx)
n_test = int(len(samples) * TEST_FRAC)
n_val  = int(len(samples) * VAL_FRAC)
n_train = len(samples) - n_test - n_val
train_ds, val_ds, test_ds = random_split(
    full, [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(SEED),
)
# eval splits use eval_tx
val_ds.dataset  = VehDataset(samples, eval_tx)
test_ds.dataset = VehDataset(samples, eval_tx)

train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=2, pin_memory=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)
test_dl  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)

# -------- Model --------
model = timm.create_model("mobilenetv3_small_100", pretrained=True, num_classes=len(labels))
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
        top5 = logits.topk(min(5, len(labels)), dim=1).indices
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
    json.dump({"labels": labels, "version_count": len(labels)}, f)

import hashlib
sha = hashlib.sha256(onnx_path.read_bytes()).hexdigest()

with open(meta_path, "w") as f:
    json.dump({
        "version":          VERSION,
        "classes":          len(labels),
        "samples_trained":  len(samples),
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
            "backbone":      "mobilenetv3_small_100",
        },
    }, f, indent=2)

print(f"\n✓ Done. Artifacts in {OUT_DIR}:")
for p in OUT_DIR.glob("vehicle-classifier-v*"):
    print(f"  {p.name} ({p.stat().st_size:,} bytes)")
print(f"  metadata.json (test top-1: {ta:.4f}, top-5: {top5:.4f})")
