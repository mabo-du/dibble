"""patch_features.py — Re-extract edge-angle features for available meshes.

The training_matrix.csv was generated with a version of _classification.py
where edge_angle_std_deg was always 0.0 (column name mismatch). This script
reads every mesh that's on disk, re-extracts features with the current fixed
code, and patches only the edge-angle columns in the existing CSV.

OOM-safe: processes one mesh at a time in a fresh subprocess worker.

Usage:
    python3 lithicore/data/training/patch_features.py
"""

from __future__ import annotations

import csv
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR = PROJECT_ROOT / "lithicore" / "src"
sys.path.insert(0, str(SRC_DIR))

DATA_DIR = Path("/data/dibble-training/raw")
MATRIX_PATH = PROJECT_ROOT / "lithicore" / "data" / "training" / "processed" / "training_matrix.csv"

# Edge-angle columns that need patching (the ones that were broken by name mismatch)
PATCH_COLUMNS = [
    "edge_angle_mean_deg",
    "edge_angle_std_deg",
    "edge_angle_skewness",
    "edge_angle_kurtosis",
]


def _index_meshes() -> dict[str, Path]:
    """Build a {stem: path} lookup for all mesh files on disk.

    Scans known directories with controlled depth (3 levels max) to avoid
    the 175 GB raw data's recursive explosion.
    """
    SUFFIXES = (".ply", ".stl", ".obj", ".wrl", ".vrml")
    index: dict[str, Path] = {}
    mesh_dirs = [
        *DATA_DIR.glob("*_Meshes"),
        DATA_DIR / "COADS",
        DATA_DIR / "lombao_cores",
        DATA_DIR / "morales_retouch",
        DATA_DIR / "levantine_handaxes",
    ]
    for d in mesh_dirs:
        if not d.is_dir():
            continue
        # Level 1: files at root
        for f in d.iterdir():
            if f.suffix in SUFFIXES:
                index[f.stem] = f
        # Level 2: one subdirectory deep
        for sub in d.iterdir():
            if not sub.is_dir():
                continue
            for f in sub.iterdir():
                if f.suffix in SUFFIXES:
                    index[f.stem] = f
                # Level 3: two subdirs deep (e.g. *_Meshes/*_Meshes/*.ply)
                if f.is_dir():
                    for f3 in f.iterdir():
                        if f3.suffix in SUFFIXES:
                            index[f3.stem] = f3
    return index


def main() -> None:
    print("=" * 60)
    print("  Feature re-extraction: edge-angle columns")
    print("=" * 60)

    if not MATRIX_PATH.exists():
        print(f"  ERROR: Training matrix not found at {MATRIX_PATH}")
        sys.exit(1)

    # Load existing matrix
    rows: list[dict] = []
    with open(MATRIX_PATH, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            rows.append(row)

    print(f"  Loaded {len(rows)} rows from training matrix")
    print(f"  Patch columns: {PATCH_COLUMNS}")

    # Build mesh index once (fast: non-recursive, targeted walks)
    print("  Indexing available meshes...")
    t0 = time.time()
    mesh_index = _index_meshes()
    print(f"  Found {len(mesh_index)} meshes on disk ({time.time()-t0:.1f}s)")

    # Find and process each mesh
    patched = 0
    not_found = 0
    failed = 0
    skipped_already = 0
    start_time = time.time()
    # Count total to patch for progress
    total_to_patch = sum(
        1 for row in rows
        if float(row.get("edge_angle_std_deg", "0")) == 0.0
        and mesh_index.get(row["artefact_id"]) is not None
    )

    for i, row in enumerate(rows):
        aid = row["artefact_id"]
        ds = row.get("dataset", "")

        # Skip if edge_angle_std_deg is already non-zero (some rows may have been patched)
        current_std = float(row.get("edge_angle_std_deg", "0"))
        if current_std != 0.0:
            skipped_already += 1
            continue

        # Skip datasets without available meshes
        if "Levantine" in ds:
            not_found += 1
            continue

        mesh_path = mesh_index.get(aid)
        if mesh_path is None:
            not_found += 1
            continue

        # Run lightweight edge-angle worker (~0.5s vs ~20s for full extract_features)
        worker = PROJECT_ROOT / "lithicore" / "data" / "training" / "_edge_worker.py"
        try:
            result = subprocess.run(
                [sys.executable, str(worker), str(mesh_path), aid],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0 or not result.stdout.strip():
                failed += 1
                if result.stderr:
                    print(f"  Error [{aid}]: {result.stderr.strip()[:80]}")
                continue

            # Parse: artefact_id,mean,std,skew,kurt
            parts = result.stdout.strip().split(",")
            if len(parts) >= 5 and parts[0] == aid:
                rows[i]["edge_angle_mean_deg"] = parts[1]
                rows[i]["edge_angle_std_deg"] = parts[2]
                rows[i]["edge_angle_skewness"] = parts[3]
                rows[i]["edge_angle_kurtosis"] = parts[4]

            patched += 1
            if patched % 200 == 0:
                elapsed = time.time() - start_time
                rate = patched / elapsed if elapsed > 0 else 0
                eta = (total_to_patch - patched) / rate if rate > 0 else 0
                print(f"  ... {patched}/{total_to_patch} ({rate:.1f}/s, ETA {eta:.0f}s)")

        except subprocess.TimeoutExpired:
            print(f"  Timeout [{aid}]")
            failed += 1
        except Exception as e:
            print(f"  Exception [{aid}]: {e}")
            failed += 1

    # Write updated matrix
    with open(MATRIX_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Done in {elapsed:.0f}s")
    print(f"  Patched:   {patched}")
    print(f"  Not found: {not_found}")
    print(f"  Failed:    {failed}")
    print(f"  Skipped:   {skipped_already}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
