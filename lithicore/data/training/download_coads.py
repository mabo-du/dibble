"""download_coads.py — Batch download COADS collection from Zenodo.

COADS (Central Ohio Archaeological Digitization Survey) has ~2,400
individual 3D models in GLB format, each as a separate Zenodo record
tagged "bsu_aal". This script queries the Zenodo API in pages,
downloads the GLB files, and optionally converts them to PLY.

Usage:
    python3 lithicore/data/training/download_coads.py \
        --dest /data/dibble-training/raw/COADS \
        --max 100
"""

import argparse
import json
import os
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


def download_glb(record: dict, dest: Path) -> Path | None:
    """Download GLB file from a Zenodo record. Returns path or None."""
    files = record.get("files", [])
    glb = next((f for f in files if f["key"].endswith(".glb")), None)
    if not glb:
        return None

    filename = glb["key"]
    out_path = dest / filename

    if out_path.exists():
        print(f"    Already exists: {filename}")
        return out_path

    url = glb["links"]["self"]
    doi = record.get("doi", "?")
    title = record.get("metadata", {}).get("title", "?")[:50]

    print(f"    Downloading {filename} ({glb['size']/1e6:.0f} MB) — {title}")
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "Dibble/1.0")
        with urllib.request.urlopen(req, timeout=120) as resp:
            with open(out_path, "wb") as f:
                f.write(resp.read())
        return out_path
    except Exception as e:
        print(f"    Failed: {e}")
        return None


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
    glb_dir.mkdir(parents=True, exist_ok=True)
    ply_dir.mkdir(parents=True, exist_ok=True)

    # Query first page to get total count
    data = get_records(page=1, size=1)
    total = data["hits"]["total"]
    max_dl = args.max if args.max > 0 else total
    print(f"COADS: {total} records available, downloading max {max_dl}")

    downloaded = 0
    page = 1
    while downloaded < max_dl:
        data = get_records(page=page, size=25)
        hits = data["hits"]["hits"]
        if not hits:
            break

        for record in hits:
            if downloaded >= max_dl:
                break

            path = download_glb(record, glb_dir)
            if path and args.convert:
                convert_to_ply(path, ply_dir)

            downloaded += 1
            if downloaded % 10 == 0:
                print(f"  Progress: {downloaded}/{max_dl}")

            # Rate limit: be nice to Zenodo
            time.sleep(0.5)

        page += 1

    print(f"\nDone: {downloaded} models downloaded")
    if glb_dir.exists():
        print(f"GLB:  {sum(1 for _ in glb_dir.glob('*.glb'))} files")
    if ply_dir.exists():
        print(f"PLY:  {sum(1 for _ in ply_dir.glob('*.ply'))} files")


if __name__ == "__main__":
    main()
