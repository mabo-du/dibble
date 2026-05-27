"""download_and_process.py — Download and process Open Aurignacian datasets.

Downloads the 3D meshes from Zenodo, extracts morphometric features,
and builds a real-data training matrix for the classifier.

Usage:
    python -m lithicore.data.training.download_and_process

Requires: tqdm for progress bars (pip install tqdm)
Output: lithicore/data/training/processed/training_matrix.csv
"""

import csv
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import trimesh

# Add project source to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "lithicore" / "src"))

from lithicore._classification import extract_features

RAW_DIR = Path(__file__).resolve().parent / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent / "processed"

# Dataset definitions
DATASETS = [
    {
        "name": "Fumane Cave (Vol 1)",
        "csv_file": RAW_DIR / "RF_3D_Dataset.csv",
        "zip_url": "https://zenodo.org/api/records/15131708/files/RF_3D_Meshes.zip/content",
        "zip_file": RAW_DIR / "RF_3D_Meshes.zip",
        "mesh_prefix": "RF_3D_Meshes",
        "id_prefix": "RF",
        "n_meshes": 948,
    },
    {
        "name": "Castelcivita (Vol 2)",
        "csv_file": RAW_DIR / "CTC_3D_Dataset.csv",
        "zip_url": "https://zenodo.org/api/records/10631390/files/CTC_3D_Meshes.zip/content",
        "zip_file": RAW_DIR / "CTC_3D_Meshes.zip",
        "mesh_prefix": "CTC_3D_Meshes",
        "id_prefix": "CTC",
        "n_meshes": 538,
    },
]


def is_valid_zip(path: Path) -> bool:
    """Check if a file is a valid zip archive."""
    if not path.exists():
        return False
    try:
        with zipfile.ZipFile(path, "r") as zf:
            return zf.testzip() is None
    except (zipfile.BadZipFile, Exception):
        return False


def download_file(url: str, dest: Path, desc: str = "") -> None:
    """Download a file with progress bar and resume support."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Check if we have a valid zip already
    if is_valid_zip(dest):
        print(f"  Already downloaded: {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")
        return

    # Remove corrupted partial download
    if dest.exists():
        print(f"  Removing corrupted partial download ({dest.stat().st_size / 1e6:.0f} MB)...")
        dest.unlink()

    print(f"  Downloading {desc} ({dest.name})...")

    import time
    import urllib.request
    from urllib.error import HTTPError, URLError

    # Retry loop for transient server errors
    max_retries = 5
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url)
            response = urllib.request.urlopen(req, timeout=60)
            break
        except (HTTPError, URLError) as e:
            code = getattr(e, "code", 0)
            if code in (502, 503, 504) and attempt < max_retries - 1:
                wait = (attempt + 1) * 10
                print(f"\n  Server error ({code}). Retrying in {wait}s (attempt {attempt + 2}/{max_retries})...")
                time.sleep(wait)
            else:
                print(f"\n  Download failed: {e}")
                print("  Zenodo may be under load. Try again later.")
                return
        except Exception as e:
            print(f"\n  Download error: {e}")
            return

    total = int(response.headers.get("content-length", 0))
    chunk_size = 1024 * 1024  # 1 MB chunks

    downloaded = 0
    with open(dest, "wb") as f:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            pct = downloaded / total * 100 if total > 0 else 0
            mb = downloaded / 1e6
            print(f"\r    {mb:.0f} / {total/1e6:.0f} MB ({pct:.0f}%)", end="")
    print()

    # Verify the downloaded zip
    if not is_valid_zip(dest):
        print(f"  ERROR: Download corrupted. Please try again.")
        dest.unlink()
        return


def get_typology(row: dict) -> str:
    """Map Zenodo metadata to broad typology category."""
    cls = row.get("Class", "")
    blank = row.get("Blank", "")

    if cls in ("Core", "Core-Tool"):
        return "Core"
    elif cls == "Tool":
        return "Tool"
    elif blank == "Blade":
        return "Blade"
    elif blank == "Bladelet":
        return "Bladelet"
    elif blank == "Flake":
        return "Flake"
    else:
        return "Other"


def process_dataset(ds: dict) -> list[dict]:
    """Download, extract, and process one dataset.

    Returns list of feature dicts for the training matrix.
    """
    print(f"\n{'='*60}")
    print(f"  Processing: {ds['name']}")
    print(f"  CSV: {ds['csv_file']}")
    print(f"{'='*60}")

    # Check CSV exists
    if not ds["csv_file"].exists():
        print(f"  ERROR: CSV not found at {ds['csv_file']}")
        print(f"  Download CSVs first, then run this script.")
        return []

    # Read metadata
    with open(ds["csv_file"]) as f:
        reader = csv.DictReader(f)
        metadata = list(reader)
    print(f"  Loaded {len(metadata)} metadata records")

    # Download mesh archive if needed
    if is_valid_zip(ds["zip_file"]):
        print(f"  Mesh archive ready: {ds['zip_file'].stat().st_size / 1e6:.0f} MB")
    else:
        if ds["zip_file"].exists():
            print(f"  Removing corrupted archive ({ds['zip_file'].stat().st_size / 1e6:.0f} MB)...")
            ds["zip_file"].unlink()
        print(f"  Mesh archive not found. Downloading {ds['n_meshes']} meshes...")
        print(f"  This is a large download (~1.8 GB per volume).")
        resp = input("  Download now? (y/n): ")
        if resp.lower() != "y":
            print("  Skipping download.")
            return []

        download_file(ds["zip_url"], ds["zip_file"], desc=ds["name"])
        if not is_valid_zip(ds["zip_file"]):
            print("  Download failed. Try again.")
            return []

    # Extract zip
    extract_dir = RAW_DIR / ds["mesh_prefix"]
    if not extract_dir.exists():
        print(f"  Extracting {ds['zip_file'].name}...")
        with zipfile.ZipFile(ds["zip_file"], "r") as zf:
            zf.extractall(path=extract_dir)
        print(f"  Extracted to {extract_dir}")
    else:
        n_extracted = len(list(extract_dir.rglob("*.ply")))
        print(f"  Already extracted: {extract_dir} ({n_extracted} PLY files)")

    # Build mesh path lookup
    mesh_files = {}
    for fpath in extract_dir.rglob("*.ply"):
        mesh_files[fpath.stem] = fpath
    print(f"  Found {len(mesh_files)} mesh files on disk")

    # Process each artefact
    results = []
    errors = []
    n = len(metadata)

    for i, row in enumerate(metadata):
        artefact_id = row.get("ID", "").strip()
        typology = get_typology(row)

        if typology == "Other":
            continue  # Skip unclassifiable

        # Find mesh file
        mesh_path = mesh_files.get(artefact_id)
        if mesh_path is None:
            errors.append(f"  Mesh not found: {artefact_id}")
            continue

        # Load mesh and extract features
        try:
            mesh = trimesh.load(str(mesh_path), force="mesh")
            fv = extract_features(mesh)

            # Build result row
            result = {
                "artefact_id": artefact_id,
                "typology": typology,
                "dataset": ds["name"],
                "source_csv": ds["csv_file"].name,
            }
            # Add all feature vector fields
            for name in fv.FEATURE_NAMES:
                result[name] = getattr(fv, name)
            results.append(result)

        except Exception as exc:
            errors.append(f"  Error processing {artefact_id}: {exc}")

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{n} ({len(results)} succeeded, {len(errors)} errors)")

    print(f"\n  Done: {len(results)} artefacts processed, {len(errors)} errors")
    if errors:
        for e in errors[:5]:
            print(f"    {e}")
        if len(errors) > 5:
            print(f"    ... and {len(errors) - 5} more")

    return results


def main() -> None:
    """Main entry point."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    for ds in DATASETS:
        results = process_dataset(ds)
        all_results.extend(results)

    if not all_results:
        print("\nNo data processed. Exiting.")
        return

    # Build typology summary
    from collections import Counter
    type_counts = Counter(r["typology"] for r in all_results)
    print(f"\n{'='*60}")
    print(f"  TOTAL: {len(all_results)} artefacts processed")
    print(f"  Typology distribution:")
    for typ, count in sorted(type_counts.items()):
        print(f"    {typ}: {count}")
    print(f"{'='*60}")

    # Save training matrix
    import pandas as pd
    df = pd.DataFrame(all_results)
    output_path = PROCESSED_DIR / "training_matrix.csv"
    df.to_csv(output_path, index=False)
    print(f"\n  Training matrix saved: {output_path}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Feature dimensions: {len([c for c in df.columns if c in df.columns[:22]])}")

    # Save summary
    summary_path = PROCESSED_DIR / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Total artefacts: {len(all_results)}\n")
        f.write(f"Typology distribution:\n")
        for typ, count in sorted(type_counts.items()):
            f.write(f"  {typ}: {count}\n")
    print(f"  Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
