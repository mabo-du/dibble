"""process_new_data.py — Process unintegrated meshes and retrain classifiers.

Scans all mesh directories on disk, finds meshes not yet in the training
matrix, extracts features via the lightweight edge_worker, determines labels
from metadata/dataset rules, appends to the training matrix, and retrains.

OOM-safe design:
  - Each mesh processed in a fresh subprocess (_worker.py)
  - One mesh at a time, no batching
  - gc.collect() between major phases
  - Model training is single-threaded (n_jobs=1)

Usage:
    # After COADS download completes:
    python3 lithicore/data/training/process_new_data.py

    # Or specify an additional PLY directory:
    python3 lithicore/data/training/process_new_data.py --ply-dir /data/dibble-training/raw/COADS/ply
"""

from __future__ import annotations

import argparse
import csv
import gc
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR = PROJECT_ROOT / "lithicore" / "src"
sys.path.insert(0, str(SRC_DIR))

from lithicore._classification import ClassifierModel  # noqa: E402

DATA_DIR = Path("/data/dibble-training/raw")
MODELS_DIR = PROJECT_ROOT / "lithicore" / "data" / "models"
MATRIX_PATH = PROJECT_ROOT / "lithicore" / "data" / "training" / "processed" / "training_matrix.csv"
WORKER = PROJECT_ROOT / "lithicore" / "data" / "training" / "_worker.py"

# Known mesh directories — maps data_dir -> dataset_name, typology_label
# typology_label=None means "resolve from metadata"
MESH_SOURCES: list[dict] = [
    {"dir": DATA_DIR / "RF_3D_Meshes",      "dataset": "Fumane Cave (Vol 1)",       "label": None},
    {"dir": DATA_DIR / "CTC_3D_Meshes",      "dataset": "Castelcivita (Vol 2)",      "label": None},
    {"dir": DATA_DIR / "Ca_3D_Meshes",       "dataset": "Cala (Vol 3)",               "label": None},
    {"dir": DATA_DIR / "RB_3D_Meshes",       "dataset": "Bombrini (Vol 4)",           "label": None},
    # COADS (from downloaded PLY, not from the structured Meshes dir)
    {"dir": DATA_DIR / "COADS" / "ply",      "dataset": "COADS (Ohio)",               "label": "Biface"},
    # Levantine handaxes — PLY files only (WRL would need conversion)
    {"dir": DATA_DIR / "levantine_handaxes",  "dataset": "Levantine Acheulean Handaxes", "label": "Biface"},
    # Lombao cores — STL files
    {"dir": DATA_DIR / "lombao_cores",       "dataset": "Lombao Experimental Cores (Spain)", "label": "Experimental Core"},
    # Morales retouch — STL files
    {"dir": DATA_DIR / "morales_retouch",    "dataset": "Morales Retouch (Spain)",     "label": "Retouched Flake"},
]

# Typology labels for technological system
TECH_LABELS: dict[str, str] = {
    "Biface": "Handaxe",
    "Experimental Core": "Initialization",
    "Retouched Flake": "Maintenance",
}


def index_existing_matrix() -> set[str]:
    """Return set of artefact IDs already in the training matrix."""
    if not MATRIX_PATH.exists():
        return set()
    existing: set[str] = set()
    with open(MATRIX_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing.add(row.get("artefact_id", ""))
    return existing


def find_all_meshes(source: dict) -> dict[str, Path]:
    """Find all mesh files in a source directory, keyed by stem."""
    meshes: dict[str, Path] = {}
    d = source["dir"]
    if not d.is_dir():
        return meshes
    for suffix in (".ply", ".stl", ".obj"):
        for f in d.rglob(f"*{suffix}"):
            meshes[f.stem] = f
    return meshes


def get_source_csv(source: dict) -> str:
    """Get the source_csv value to write in the matrix."""
    d = source["dir"]
    if "COADS" in str(d):
        return "coads_metadata"
    if "levantine" in str(d):
        return "levantine_handaxes"
    if "lombao" in str(d):
        return "lombao_metadata"
    if "morales" in str(d):
        return "morales_metadata"
    # For Fumane volumes, use the first metadata CSV found
    csv_files = list(d.parent.glob(f"{d.name[:2]}*_metadata.csv"))
    if csv_files:
        return csv_files[0].name
    return d.name


def process_mesh(mesh_path: Path, artefact_id: str, source: dict) -> dict | None:
    """Run _worker.py on one mesh and return the row dict, or None on failure.

    Each mesh runs in a fresh subprocess — on completion, all memory is freed.
    """
    src_csv = get_source_csv(source)
    try:
        result = subprocess.run(
            [sys.executable, str(WORKER), str(mesh_path), artefact_id,
             source.get("label", ""), source["dataset"], src_csv],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0 or not result.stdout.strip():
            if result.stderr:
                print(f"    Error [{artefact_id}]: {result.stderr.strip()[:100]}")
            return None

        # Parse the CSV output from _worker.py
        output_line = result.stdout.strip()
        reader = csv.DictReader(output_line.splitlines())
        for row in reader:
            row["artefact_id"] = artefact_id
            row["dataset"] = source["dataset"]
            row["typology"] = source.get("label", "")
            row["source_csv"] = src_csv
            return row
        return None
    except subprocess.TimeoutExpired:
        print(f"    Timeout [{artefact_id}] (>120s)")
        return None
    except Exception as e:
        print(f"    Exception [{artefact_id}]: {e}")
        return None


def append_to_matrix(new_rows: list[dict]) -> None:
    """Append new rows to the training matrix CSV."""
    if not new_rows:
        return

    fieldnames = list(new_rows[0].keys())
    is_new = not MATRIX_PATH.exists() or MATRIX_PATH.stat().st_size == 0

    with open(MATRIX_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerows(new_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process new meshes and expand the training matrix"
    )
    parser.add_argument("--ply-dir", type=str, default=None,
                        help="Additional directory of PLY files to process")
    parser.add_argument("--retrain", action="store_true", default=True,
                        help="Run retrain.py after processing")
    parser.add_argument("--no-retrain", action="store_false", dest="retrain",
                        help="Skip retraining")
    args = parser.parse_args()

    # Add extra directory if specified
    if args.ply_dir:
        extra_path = Path(args.ply_dir)
        if extra_path.is_dir():
            MESH_SOURCES.insert(0, {
                "dir": extra_path,
                "dataset": f"COADS (Ohio)",
                "label": "Biface",
            })
            print(f"  Added extra directory: {extra_path}")

    # Index existing matrix
    print("Indexing existing training matrix...")
    existing_ids = index_existing_matrix()
    print(f"  {len(existing_ids)} artefacts already in matrix")

    # Scan all source directories for new meshes
    all_new: list[dict] = []
    total_new = 0

    for source in MESH_SOURCES:
        print(f"\n{'─'*60}")
        print(f"  Source: {source['dataset']}")
        print(f"  Dir:    {source['dir']}")
        print(f"{'─'*60}")

        if not source["dir"].is_dir():
            print(f"  Directory not found, skipping")
            continue

        meshes = find_all_meshes(source)
        print(f"  Found {len(meshes)} mesh files on disk")

        # Filter to only new meshes
        new_meshes = {k: v for k, v in meshes.items() if k not in existing_ids}
        print(f"  New (not in matrix): {len(new_meshes)}")

        if not new_meshes:
            continue

        # Process each new mesh
        processed = 0
        errors = 0
        t0 = time.time()

        for artefact_id, mesh_path in sorted(new_meshes.items()):
            row = process_mesh(mesh_path, artefact_id, source)
            if row:
                all_new.append(row)
                processed += 1
            else:
                errors += 1

            if processed % 20 == 0 and processed > 0:
                elapsed = time.time() - t0
                rate = processed / elapsed if elapsed > 0 else 0
                print(f"  ... {processed}/{len(new_meshes)} processed ({rate:.1f}/s)")

        elapsed = time.time() - t0
        print(f"  Done: {processed} processed, {errors} errors in {elapsed:.0f}s")

    # Append new rows to matrix
    if all_new:
        print(f"\n{'='*60}")
        print(f"  Appending {len(all_new)} new rows to training matrix...")
        append_to_matrix(all_new)
        total_new = len(all_new)

        # Verify new count
        new_count = len(index_existing_matrix())
        print(f"  Matrix now has {new_count} artefacts (+{total_new})")
    else:
        print(f"\n  No new artefacts to add.")

    # Free memory before retrain
    del all_new
    gc.collect()

    # Retrain
    if args.retrain and total_new > 0:
        print(f"\n{'='*60}")
        print(f"  Retraining classifiers with {total_new} new artefacts...")
        print(f"{'='*60}")
        retrain_script = PROJECT_ROOT / "lithicore" / "data" / "training" / "retrain.py"
        result = subprocess.run(
            [sys.executable, str(retrain_script)],
            capture_output=False,
        )
        if result.returncode != 0:
            print(f"  Retraining failed (exit code {result.returncode})")
    elif args.retrain:
        print(f"\n  No new data — skipping retrain")

    print(f"\n{'='*60}")
    print(f"  Summary: {total_new} new artefacts added to training matrix")
    print(f"  Run `python3 lithicore/data/training/retrain.py` to retrain classifiers")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
