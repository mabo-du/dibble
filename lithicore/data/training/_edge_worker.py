"""Lightweight worker — extracts ONLY edge-angle features from one mesh.

Runs in ~0.5s per mesh (vs ~20s for full extract_features) by skipping
scar detection, platform angles, landmarks, etc.

Usage (called by patch_features.py):
    python3 _edge_worker.py <mesh_path> <artefact_id>
    # Output: artefact_id,edge_angle_mean_deg,edge_angle_std_deg,edge_angle_skewness,edge_angle_kurtosis
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "lithicore" / "src"))

from lithicore._classification import _compute_edge_angles
from scipy.stats import skew as scipy_skew, kurtosis as scipy_kurtosis


if __name__ == "__main__":
    mesh_path = sys.argv[1]
    artefact_id = sys.argv[2] if len(sys.argv) > 2 else ""

    mesh = trimesh.load(mesh_path, force="mesh")

    angles = _compute_edge_angles(mesh)
    n = len(angles)

    mean_val = float(np.mean(angles)) if n > 0 else 0.0
    std_val = float(np.std(angles)) if n > 1 else 0.0
    skew_val = float(scipy_skew(angles)) if n > 2 else 0.0
    kurt_val = float(scipy_kurtosis(angles)) if n > 2 else 0.0

    # Output as CSV line matching the edge-angle columns
    print(f"{artefact_id},{mean_val:.4f},{std_val:.4f},{skew_val:.4f},{kurt_val:.4f}")
