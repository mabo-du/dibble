"""process_safe.py — Memory-safe training data processing.

Each mesh is processed in a fresh Python subprocess to prevent
C-level memory accumulation (trimesh/numpy) from causing OOM crashes.

Usage:
    python3 lithicore/data/training/process_safe.py
"""

import csv
import gc
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "lithicore" / "src"))

DATA_DIR = Path("/data/dibble-training/raw")
OUTPUT_DIR = Path(__file__).resolve().parent / "processed"
WORKER = Path(__file__).resolve().parent / "_worker.py"
BATCH_SIZE = 10  # Process this many meshes per subprocess

VOLUMES = [
    {
        "name": "Fumane Cave (Vol 1)",
        "mesh_dir": DATA_DIR / "RF_3D_Meshes",
        "csv": DATA_DIR / "Fumane_3D_metadata.csv",
        "id_prefix": "RF",
    },
    {
        "name": "Castelcivita (Vol 2)",
        "mesh_dir": DATA_DIR / "CTC_3D_Meshes",
        "csv": DATA_DIR / "Castelcivita_3D_metadata.csv",
        "id_prefix": "CTC",
    },
    {
        "name": "Cala (Vol 3)",
        "mesh_dir": DATA_DIR / "Ca_3D_Meshes",
        "csv": DATA_DIR / "Cala_3D_metadata.csv",
        "id_prefix": "Ca",
    },
    {
        "name": "Bombrini (Vol 4)",
        "mesh_dir": DATA_DIR / "RB_3D_Meshes",
        "csv": DATA_DIR / "Bombrini_3D_metadata.csv",
        "id_prefix": "RB",
    },
]

FEATURE_NAMES = [
    "length_mm", "width_mm", "thickness_mm", "volume_mm3",
    "surface_area_mm2", "elongation", "flatness", "compactness",
    "circularity", "rectangularity", "diameter_mm", "curvature",
    "edge_length_mean", "edge_length_std", "edge_length_skewness",
    "edge_length_kurtosis", "edge_angle_mean_deg", "edge_angle_std_deg",
    "edge_angle_skewness", "edge_angle_kurtosis",
    "convexity", "solidity",
]
FIELD_NAMES = ["artefact_id", "typology", "dataset", "source_csv"] + FEATURE_NAMES


def get_typology(row: dict) -> str:
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


def build_mesh_lookup(mesh_dir: Path) -> dict[str, Path]:
    lookup = {}
    for fpath in mesh_dir.rglob("*.ply"):
        lookup[fpath.stem] = fpath
    return lookup


def process_volume(vol: dict, csv_path: Path) -> int:
    """Process one volume via subprocess workers. Returns count of processed artifacts."""
    name = vol["name"]
    mesh_dir = vol["mesh_dir"]
    meta_path = vol["csv"]

    print(f"\n{'='*60}")
    print(f"  Processing: {name}")
    print(f"{'='*60}")

    # Load metadata
    if not meta_path.exists():
        print(f"  WARNING: CSV not found at {meta_path}, skipping")
        return 0

    with open(meta_path, newline="") as f:
        reader = csv.DictReader(f)
        metadata = list(reader)
    print(f"  Loaded {len(metadata)} metadata records")

    # Build mesh lookup
    mesh_lookup = build_mesh_lookup(mesh_dir)
    n_meshes = len(mesh_lookup)
    print(f"  Found {n_meshes} PLY files on disk")
    if n_meshes == 0:
        return 0

    # Build task list: (mesh_path, artefact_id, typology, dataset, csv_name)
    tasks = []
    seen_ids = set()
    for row in metadata:
        artefact_id = row.get("ID", "").strip()
        if not artefact_id or artefact_id in seen_ids:
            continue
        seen_ids.add(artefact_id)
        typology = get_typology(row)
        if typology == "Other":
            continue
        mesh_path = mesh_lookup.get(artefact_id)
        if mesh_path is None:
            continue
        tasks.append((str(mesh_path), artefact_id, typology, name, meta_path.name))

    print(f"  Queued {len(tasks)} artifacts to process (after filtering)")

    # Process in batches via subprocess
    succeeded = 0
    total = len(tasks)
    start = time.time()

    # Open CSV once and write header if new
    is_new = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELD_NAMES)
        if is_new:
            writer.writeheader()

    for batch_start in range(0, total, BATCH_SIZE):
        batch = tasks[batch_start:batch_start + BATCH_SIZE]

        # Build subprocess args: run worker for each task
        for task in batch:
            mesh_path, artefact_id, typology, dataset, csv_name = task
            try:
                result = subprocess.run(
                    [sys.executable, str(WORKER), mesh_path, artefact_id, typology, dataset, csv_name],
                    capture_output=True, text=True, timeout=120,
                )
                if result.returncode == 0 and result.stdout.strip():
                    line = result.stdout.strip()
                    with open(csv_path, "a", newline="") as f:
                        f.write(line + "\n")
                    succeeded += 1
                else:
                    if result.stderr:
                        print(f"    Error [{artefact_id}]: {result.stderr.strip()[:100]}")
            except subprocess.TimeoutExpired:
                print(f"    Timeout [{artefact_id}]")
            except Exception as e:
                print(f"    Exception [{artefact_id}]: {e}")

        # Progress report
        done = min(batch_start + BATCH_SIZE, total)
        elapsed = time.time() - start
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        print(f"  {done}/{total} ({succeeded} ok) — {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

    elapsed = time.time() - start
    print(f"\n  Volume done: {succeeded}/{total} artifacts in {elapsed:.0f}s ({total/elapsed:.1f}/s)")
    return succeeded


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "training_matrix.csv"

    # Remove partial output from previous run
    if output_path.exists():
        print(f"Removing previous training matrix ({output_path})")
        output_path.unlink()

    total = 0
    for vol in VOLUMES:
        count = process_volume(vol, output_path)
        total += count
        gc.collect()

    # Final summary
    if total == 0:
        print("\nNo artifacts processed.")
        return

    types = Counter()
    with open(output_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            types[row["typology"]] += 1

    print(f"\n{'='*60}")
    print(f"  Training matrix: {output_path.resolve()}")
    print(f"  Total artifacts: {total}")
    print(f"\n  Typology distribution:")
    for typ, count in sorted(types.items()):
        print(f"    {typ}: {count}")

    summary_path = OUTPUT_DIR / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Total artifacts: {total}\n")
        for typ, count in sorted(types.items()):
            f.write(f"  {typ}: {count}\n")
    print(f"\n  Summary saved: {summary_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
