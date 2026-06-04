#!/usr/bin/env python3
"""batch_ph.py — Batch compute Persistent Homology features for all training meshes.

This script processes all available 3D mesh files and caches the PH feature vectors.
It is designed to be interrupt-safe: each artefact is cached individually, so
partial progress is preserved across runs.

Usage:
    python3 lithicore/data/training/batch_ph.py

    # Or with custom parameters:
    python3 lithicore/data/training/batch_ph.py --n-points 2000 --batch-size 100
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

# Ensure lithicore is importable
_script_dir = Path(__file__).resolve().parent
_src_dir = _script_dir.parent.parent / "lithicore" / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from lithicore._ph_features import batch_compute_ph, load_ph_matrix, CACHE_DIR


def find_mesh(aid: str, raw_dir: Path) -> Path | None:
    """Find the mesh file for an artefact ID."""
    # Morales STL files
    if aid.startswith("Morales_"):
        suffix = aid.replace("Morales_", "")
        p = raw_dir / "morales_retouch" / f"{suffix}.stl"
        if p.exists(): return p

    # COADS PLY files (stored as {hash}.ply without COADS_ prefix)
    if aid.startswith("COADS_"):
        suffix = aid.replace("COADS_", "")
        for sub in ["ply", "glb"]:
            p = raw_dir / "COADS" / sub / f"{suffix}.{sub}"
            if p.exists(): return p

    # EdgeAngle corpus
    if aid.startswith("EAP") or aid.startswith("BU-") or aid.startswith("WEM-"):
        for fmt in ["stl", "ply"]:
            p = raw_dir / "edgeangle" / f"{aid}.{fmt}"
            if p.exists(): return p

    # Standard PLY search
    for p in raw_dir.rglob(f"{aid}.ply"):
        return p
    for p in raw_dir.rglob(f"{aid}.stl"):
        return p
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch compute PH features")
    parser.add_argument("--n-points", type=int, default=2000, help="Vertex subsample count")
    parser.add_argument("--cache-dir", type=str, default=str(CACHE_DIR), help="Cache directory")
    parser.add_argument("--limit", type=int, default=0, help="Limit artefacts to process (0=all)")
    args = parser.parse_args()

    # Project root is 4 levels up from lithicore/data/training/batch_ph.py
    PROJECT = Path(__file__).resolve().parent.parent.parent.parent
    MATRIX = PROJECT / "lithicore" / "data" / "training" / "processed" / "training_matrix.csv"
    raw_dirs = [
        Path("/data/dibble-training/raw"),
        PROJECT / "lithicore" / "data" / "training" / "raw",
    ]

    # Find the valid raw directory
    raw_dir: Path | None = None
    for candidate in raw_dirs:
        d = candidate.resolve() if candidate.is_symlink() else candidate
        if d.is_dir() and d.exists():
            raw_dir = d
            break

    if raw_dir is None:
        print("ERROR: No training data directory found.")
        sys.exit(1)

    # Read matrix
    rows = list(csv.DictReader(open(MATRIX)))

    # Build mesh list
    meshes: list[tuple[str, str]] = []
    for r in rows:
        aid = r["artefact_id"]
        mesh_path = find_mesh(aid, raw_dir)
        if mesh_path:
            meshes.append((aid, str(mesh_path)))

    print(f"Training matrix: {len(rows)} artefacts")
    print(f"Meshes available: {len(meshes)}")

    if args.limit > 0:
        meshes = meshes[:args.limit]
        print(f"Processing limit: {args.limit}")

    # Batch compute
    t0 = time.time()
    n = batch_compute_ph(
        meshes,
        cache_dir=args.cache_dir,
        n_points=args.n_points,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"\nTime: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"Rate: {n / max(elapsed, 1):.1f} artefacts/s")

    # Generate PCA-reduced matrix
    print("\nComputing PCA-reduced PH matrix...")
    all_ids = [m[0] for m in meshes]
    X_ph, valid_idx = load_ph_matrix(all_ids, cache_dir=args.cache_dir)
    if X_ph is not None:
        print(f"PH matrix: {X_ph.shape[0]} artefacts, {X_ph.shape[1]} components, {len(valid_idx)} valid")
        # Save alongside the training matrix
        output_path = Path(args.cache_dir) / "ph_pca_matrix.npy"
        np.save(str(output_path), X_ph)
        # Also save valid indices
        import json
        (Path(args.cache_dir) / "ph_valid_indices.json").write_text(json.dumps(valid_idx))
        print(f"Saved: {output_path}")
    else:
        print("WARNING: Could not generate PH matrix — insufficient cached data.")

    print("\nDone!")


if __name__ == "__main__":
    import numpy as np  # noqa: F401 — needed for np.save
    # Fix PROJECT for raw_dir fallback
    import sys as _sys
    _script_dir = Path(__file__).resolve().parent
    _src_dir = _script_dir.parent.parent / "lithicore" / "src"
    if str(_src_dir) not in _sys.path:
        _sys.path.insert(0, str(_src_dir))
    main()
