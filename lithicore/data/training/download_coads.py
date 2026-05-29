"""download_coads.py — Batch download COADS collection from Zenodo.

COADS (Central Ohio Archaeological Digitization Survey) has ~2,400
individual 3D models in GLB format, each as a separate Zenodo record
tagged "bsu_aal". Downloads the GLB files, converts to PLY, and saves
metadata to a CSV for training label assignment.

Usage:
    python3 lithicore/data/training/download_coads.py \
        --dest /data/dibble-training/raw/COADS \
        --max 100
"""

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

import trimesh


def get_records(page: int = 1, size: int = 50) -> dict:
    """Query Zenodo API for COADS records."""
    url = (
        f"https://zenodo.org/api/records?"
        f"q=bsu_aal+COADS&page={page}&size={size}&sort=newest"
    )
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "Dibble/1.0")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def extract_type(title: str, description: str) -> str:
    """Extract artefact type from title/description."""
    title_lower = title.lower()
    desc_lower = description.lower()

    if "biface" in title_lower or "biface" in desc_lower:
        return "Biface"
    elif "projectile" in title_lower or "point" in title_lower:
        return "Projectile Point"
    elif "scraper" in title_lower or "scraper" in desc_lower:
        return "Scraper"
    elif "gorget" in title_lower:
        return "Gorget"
    elif "drill" in title_lower:
        return "Drill"
    elif "core" in title_lower:
        return "Core"
    elif "flake" in title_lower:
        return "Flake"
    elif "knife" in title_lower:
        return "Knife"
    elif "tool" in title_lower:
        return "Tool"
    else:
        return "Unknown"


def extract_location(description: str) -> str:
    """Extract county/state from Zenodo description."""
    # Common pattern: "collected in <Town>, <County>, <State>"
    match = re.search(
        r"(?:collected|found)\s+in\s+([^,]+(?:,\s*[^,]+)?(?:,\s*[^,]+)?)",
        description, re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return ""


def download_glb(record: dict, dest: Path) -> tuple[Path | None, dict]:
    """Download GLB file from a Zenodo record. Returns (path, metadata)."""
    meta_info = record.get("metadata", {})
    files = record.get("files", [])
    glb = next((f for f in files if f["key"].endswith(".glb")), None)
    if not glb:
        return None, {}

    filename = glb["key"]
    out_path = dest / filename
    title = meta_info.get("title", "")
    description = meta_info.get("description", "") or ""

    # Extract plaintext from HTML description
    plain_desc = re.sub(r"<[^>]+>", "", description).strip()

    metadata = {
        "file_hash": out_path.stem,
        "title": title,
        "type": extract_type(title, plain_desc),
        "location": extract_location(plain_desc),
        "doi": record.get("doi", ""),
        "publication_date": meta_info.get("publication_date", ""),
    }

    if out_path.exists():
        print(f"    Already exists: {filename}")
        return out_path, metadata

    url = glb["links"]["self"]
    print(f"    Downloading {filename} ({glb['size']/1e6:.0f} MB) — {title[:50]}")

    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "Dibble/1.0")
        with urllib.request.urlopen(req, timeout=120) as resp:
            with open(out_path, "wb") as f:
                f.write(resp.read())
        return out_path, metadata
    except Exception as e:
        print(f"    Failed: {e}")
        return None, {}


def convert_to_ply(glb_path: Path, ply_dir: Path) -> Path | None:
    """Convert GLB to PLY via trimesh."""
    ply_path = ply_dir / f"{glb_path.stem}.ply"
    if ply_path.exists():
        return ply_path
    try:
        mesh = trimesh.load(str(glb_path))
        mesh.export(str(ply_path))
        return ply_path
    except Exception as e:
        print(f"    Conversion failed: {glb_path.name}: {e}")
        return None


def save_metadata(meta_path: Path, metadata: dict) -> None:
    """Append one row to the metadata CSV."""
    fieldnames = ["file_hash", "title", "type", "location", "doi", "publication_date"]
    is_new = not meta_path.exists() or meta_path.stat().st_size == 0
    with open(meta_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow(metadata)


def load_downloaded(glb_dir: Path) -> set:
    """Return set of downloaded file stems."""
    if not glb_dir.exists():
        return set()
    return {f.stem for f in glb_dir.glob("*.glb")}


def main():
    parser = argparse.ArgumentParser(description="Download COADS 3D models")
    parser.add_argument("--dest", default="/data/dibble-training/raw/COADS",
                        help="Destination directory")
    parser.add_argument("--max", type=int, default=100,
                        help="Maximum models to download (0 = all)")
    parser.add_argument("--convert", action="store_true",
                        help="Convert GLB to PLY after download")
    args = parser.parse_args()

    dest = Path(args.dest)
    glb_dir = dest / "glb"
    ply_dir = dest / "ply"
    meta_path = dest / "metadata.csv"
    glb_dir.mkdir(parents=True, exist_ok=True)
    ply_dir.mkdir(parents=True, exist_ok=True)

    # Check existing downloads
    existing = load_downloaded(glb_dir)
    if existing:
        print(f"Resuming: {len(existing)} models already downloaded")

    # Query first page to get total count
    data = get_records(page=1, size=1)
    total = data["hits"]["total"]
    max_dl = args.max if args.max > 0 else total
    print(f"COADS: {total} records available, target {max_dl}")

    downloaded = len(existing)
    page = 1
    while downloaded < max_dl:
        data = get_records(page=page, size=25)
        hits = data["hits"]["hits"]
        if not hits:
            break

        for record in hits:
            if downloaded >= max_dl:
                break

            path, meta = download_glb(record, glb_dir)
            if path:
                if args.convert:
                    convert_to_ply(path, ply_dir)
                if meta:
                    save_metadata(meta_path, meta)
                downloaded += 1
            elif meta.get("file_hash") in existing:
                downloaded += 1  # Count already-downloaded

            if downloaded % 10 == 0:
                print(f"  Progress: {downloaded}/{max_dl}")

            time.sleep(0.5)

        page += 1

    print(f"\nDone: {downloaded} models downloaded")
    n_glb = len(list(glb_dir.glob("*.glb"))) if glb_dir.exists() else 0
    n_ply = len(list(ply_dir.glob("*.ply"))) if ply_dir.exists() else 0
    n_meta = sum(1 for _ in open(meta_path)) - 1 if meta_path.exists() else 0
    print(f"GLB: {n_glb}   PLY: {n_ply}   Metadata: {n_meta}")


if __name__ == "__main__":
    main()
