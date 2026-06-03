"""patch_edge_angles.py — Fix edge_angle_std_deg in existing training matrix.

The training_matrix.csv was generated with a version of process_safe.py where
CSV column names used 'edge_angle_mean' and 'edge_angle_std' (missing '_deg'
suffix). This meant the values were stored under the wrong keys, and the actual
LithicFeatureVector fields remained at default 0.0.

This script reads each mesh that is available on disk, computes ONLY the four
edge-angle features (fast — ~0.02-7s per mesh depending on size), and patches
the corresponding rows in the training matrix.

Strategy: process the ~2,800 available meshes sequentially. Estimated time:
~1-2 hours depending on mix of small/large meshes.

Usage:
    python3 lithicore/data/training/patch_edge_angles.py
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
EDGE_WORKER = PROJECT_ROOT / "lithicore" / "data" / "training" / "_edge_worker.py"

PATCH_COLUMNS = [
    "edge_angle_mean_deg",
    "edge_angle_std_deg",
    "edge_angle_skewness",
    "edge_angle_kurtosis",
]


def _index_meshes() -> dict[str, Path]:
    """Build {artefact_id: mesh_path} lookup for all meshes on disk."""
    SUFFIXES = (".ply", ".stl", ".obj", ".wrl", ".vrml")
    index: dict[str, Path] = {}

    mesh_dirs = [
        *DATA_DIR.glob("*_Meshes"),
        DATA_DIR / "COADS" / "ply",
        DATA_DIR / "lombao_cores",
        DATA_DIR / "morales_retouch",
        DATA_DIR / "levantine_handaxes",
        DATA_DIR / "edgeangle_validation",
    ]

    for d in mesh_dirs:
        if not d.is_dir():
            continue
        for f in d.iterdir():
            if f.suffix in SUFFIXES:
                index[f.stem] = f
        for sub in d.iterdir():
            if not sub.is_dir():
                continue
            for f in sub.iterdir():
                if f.suffix in SUFFIXES:
                    index[f.stem] = f
                if f.is_dir():
                    for f3 in f.iterdir():
                        if f3.suffix in SUFFIXES:
                            index[f3.stem] = f3

    return index


def main() -> None:
    print("=" * 60)
    print("  Edge-angle feature patching")
    print("=" * 60)

    if not MATRIX_PATH.exists():
        print(f"  ERROR: Training matrix not found")
        sys.exit(1)

    # Load existing matrix
    with open(MATRIX_PATH, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    print(f"  Loaded {len(rows)} rows from training matrix")

    # Count rows that need patching (edge_angle_std_deg is 0.0)
    to_patch = [
        (i, row) for i, row in enumerate(rows)
        if float(row.get("edge_angle_std_deg", "0")) == 0.0
    ]
    print(f"  Rows needing patch: {len(to_patch)}")

    # Build mesh index
    print("  Indexing available meshes...")
    t0 = time.time()
    # Also index with COADS_ prefix for COADS files
    mesh_index = _index_meshes()
    # Add COADS-prefixed entries
    coads_ply = list((DATA_DIR / "COADS" / "ply").glob("*.ply"))
    for f in coads_ply:
        mesh_index[f"COADS_{f.stem}"] = f
    print(f"  Found {len(mesh_index)} meshes on disk ({time.time()-t0:.1f}s)")

    # Patch each row
    patched = 0
    skipped = 0
    errors = 0
    t0 = time.time()
    last_save = 0

    for idx, row in to_patch:
        aid = row["artefact_id"]
        mesh_path = mesh_index.get(aid)

        if mesh_path is None:
            skipped += 1
            continue

        try:
            result = subprocess.run(
                [sys.executable, str(EDGE_WORKER), str(mesh_path), aid],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0 or not result.stdout.strip():
                errors += 1
                continue

            # Parse: artefact_id,mean,std,skew,kurt
            parts = result.stdout.strip().split(",")
            if len(parts) >= 5 and parts[0] == aid:
                rows[idx]["edge_angle_mean_deg"] = parts[1]
                rows[idx]["edge_angle_std_deg"] = parts[2]
                rows[idx]["edge_angle_skewness"] = parts[3]
                rows[idx]["edge_angle_kurtosis"] = parts[4]
                patched += 1
            else:
                errors += 1

        except (subprocess.TimeoutExpired, Exception):
            errors += 1

        # Periodic save every 500 patches (crash-safe)
        if patched - last_save >= 500:
            with open(MATRIX_PATH, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            last_save = patched

        if patched % 200 == 0 and patched > 0:
            elapsed = time.time() - t0
            rate = patched / elapsed if elapsed > 0 else 0
            eta = (len(to_patch) - patched) / rate if rate > 0 else 0
            print(f"  {patched}/{len(to_patch)} patched ({rate:.1f}/s, ETA {eta:.0f}s, "
                  f"{skipped} no mesh, {errors} errors)")

    # Final write
    with open(MATRIX_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Done in {elapsed:.0f}s")
    print(f"  Patched:   {patched}")
    print(f"  No mesh:   {skipped}")
    print(f"  Errors:    {errors}")

    # Verify
    with open(MATRIX_PATH, newline="") as f:
        reader = csv.DictReader(f)
        still_zero = sum(
            1 for row in reader
            if float(row.get("edge_angle_std_deg", "0")) == 0.0
        )
    print(f"  Still zero after patch: {still_zero}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
