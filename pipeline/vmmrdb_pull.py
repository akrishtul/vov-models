"""
VMMRdb pull — Vehicle Make Model Recognition database (UMD / FAU).

The "VMMRdb" academic dataset contains ~291,000 real-world car photos
across ~9,170 make/model/year classes. The maintainers publish it on
multiple mirrors. This script downloads from a public mirror, unpacks
the dataset, and lays out the photos under

    pipeline/data/photos/vmmrdb/{Make}_{Model}/{hash}.jpg

with per-file `.meta.json` siblings tagging source = 'vmmrdb'.

Idempotent — the bulk download is keyed by a sentinel file in
pipeline/data/vmmrdb_state.json, so re-running this script with the
sentinel present is a no-op.

Set VOV_VMMRDB_URL to override the download location. If neither the
environment variable nor any of the default mirrors resolve, this
script no-ops with a friendly warning so the rest of the daily-pull
workflow continues.
"""

import hashlib
import json
import os
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

DATA_DIR    = Path(os.environ.get("VOV_PIPELINE_DATA", "pipeline/data"))
PHOTOS_DIR  = Path(os.environ.get("VOV_PIPELINE_PHOTOS", DATA_DIR / "photos")) / "vmmrdb"
STATE_PATH  = DATA_DIR / "vmmrdb_state.json"
SCRATCH     = Path(os.environ.get("VOV_PIPELINE_SCRATCH", "/tmp/vov-vmmrdb"))
USER_AGENT  = "vov-pipeline-vmmrdb/1.0"

# v1.0 — Tried mirrors, in order. We probe each with a HEAD; first 200 wins.
# Override with VOV_VMMRDB_URL pointing at a tar.gz / zip with a top-level
# {Make}_{Model}_{Year}/ folder structure.
DEFAULT_MIRRORS = [
    os.environ.get("VOV_VMMRDB_URL", "").strip(),
    "https://vmmrdb.cecsresearch.org/vmmrdb_v2.tar.gz",      # university-hosted, may rotate
    "https://archive.org/download/vmmrdb-v2/vmmrdb_v2.tar.gz",  # Internet Archive mirror (community uploads sometimes here)
]


def http_head(url: str) -> int:
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status
    except Exception:
        return 0


def http_download(url: str, dest: Path, log_every_mb: int = 50) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=180) as r:
        total = int(r.headers.get("Content-Length", 0))
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            done = 0
            next_log_at = log_every_mb * 1024 * 1024
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if done >= next_log_at:
                    pct = (100 * done / total) if total else 0
                    print(f"  ... {done/1e6:.0f} MB ({pct:.0f}%)")
                    next_log_at += log_every_mb * 1024 * 1024


def safe_extract(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip" or archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(target)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            # Defensive: refuse any member with absolute / traversal paths.
            members = [m for m in t.getmembers() if not (m.name.startswith("/") or ".." in Path(m.name).parts)]
            t.extractall(target, members=members)
    else:
        raise RuntimeError(f"Unknown archive format: {archive}")


def parse_class_dir(name: str) -> tuple:
    """
    VMMRdb folder names look like 'acura_tl_2003' or 'ford_f-150_2010'.
    Returns (Make_Model, year) where year is best-effort or empty.
    """
    parts = name.split("_")
    # Strip trailing 4-digit year if present
    year = ""
    if parts and parts[-1].isdigit() and len(parts[-1]) == 4:
        year = parts[-1]
        parts = parts[:-1]
    if not parts:
        return ("", year)
    # First token = make. Rest joined w/ space = model.
    make = parts[0].replace("-", " ").title()
    model_tok = " ".join(parts[1:]).replace("-", " ").title()
    return (f"{make} {model_tok}".strip(), year)


def import_folder_layout(root: Path) -> dict:
    """
    Walk the extracted dataset folder, copy/move every image into our canonical
    path layout. Returns counts {classes, images}.
    """
    classes, images = 0, 0
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        cls, year = parse_class_dir(entry.name)
        if not cls:
            continue
        dest_dir = PHOTOS_DIR / cls.replace(" ", "_")
        dest_dir.mkdir(parents=True, exist_ok=True)
        kept = 0
        for f in entry.glob("**/*"):
            if not f.is_file():
                continue
            if f.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                continue
            try:
                data = f.read_bytes()
                if len(data) < 4 * 1024:  # tiny / corrupt
                    continue
                h = hashlib.sha256(data).hexdigest()[:16]
                out = dest_dir / f"{h}.jpg"
                if out.exists():
                    continue
                out.write_bytes(data)
                meta = dest_dir / f"{h}.meta.json"
                meta.write_text(json.dumps({
                    "source": "vmmrdb",
                    "make_model": cls,
                    "year":   year,
                    "origin": str(f.relative_to(root)),
                }, indent=2))
                kept += 1
            except Exception as e:
                print(f"[warn] {f}: {e}", file=sys.stderr)
        if kept:
            classes += 1
            images += kept
    return {"classes": classes, "images": images}


def run():
    if STATE_PATH.exists():
        st = json.loads(STATE_PATH.read_text())
        print(f"VMMRdb already imported: {st.get('classes', 0)} classes, {st.get('images', 0)} photos. Skipping.")
        return
    chosen_url = None
    for u in DEFAULT_MIRRORS:
        if not u:
            continue
        code = http_head(u)
        print(f"  probe HEAD {u} → {code}")
        if 200 <= code < 400:
            chosen_url = u
            break
    if not chosen_url:
        print("[skip] no VMMRdb mirror responded — set VOV_VMMRDB_URL to a working archive URL.")
        return
    SCRATCH.mkdir(parents=True, exist_ok=True)
    archive_path = SCRATCH / Path(chosen_url).name
    print(f"Downloading {chosen_url} → {archive_path}")
    http_download(chosen_url, archive_path)
    print(f"Extracting…")
    extract_to = SCRATCH / "extract"
    if extract_to.exists():
        shutil.rmtree(extract_to)
    safe_extract(archive_path, extract_to)

    # The archive often has one top-level directory ('vmmrdb_v2') wrapping the
    # class folders. Detect + descend.
    children = [c for c in extract_to.iterdir() if c.is_dir()]
    root = children[0] if len(children) == 1 else extract_to

    stats = import_folder_layout(root)
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps({
        "source": "vmmrdb",
        "url":    chosen_url,
        "classes":  stats["classes"],
        "images":   stats["images"],
    }, indent=2))
    print(f"VMMRdb import done: +{stats['classes']} classes, +{stats['images']} photos.")
    # Free the scratch tarball — 30GB will fill the cache.
    try:
        shutil.rmtree(SCRATCH)
    except Exception:
        pass


if __name__ == "__main__":
    run()
