"""_comparison.py — Mesh comparison metrics for overlay analysis.

exports: ComparisonResult, compare_meshes(mesh_a, mesh_b) -> ComparisonResult
used_by: GUI comparison mode, CLI
rules:   Both meshes must be oriented in the same coordinate space.
         Hausdorff distance uses trimesh's built-in proximity query.
         Volume comparison requires watertight meshes.
agent:   deepseek-v4-flash | 2026-05-26 | Initial implementation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import trimesh


@dataclass
class ComparisonResult:
    """Result of comparing two meshes."""
    hausdorff_distance_mm: float
    volume_difference_mm3: float
    volume_a_mm3: float
    volume_b_mm3: float
    surface_area_difference_mm2: float
    centroid_distance_mm: float
    length_diff_mm: float
    width_diff_mm: float
    thickness_diff_mm: float
    metrics: List[str]  # names of available metrics


def _hausdorff_distance(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh) -> float:
    """Compute one-sided Hausdorff distance from A to B."""
    # Sample points on mesh A surface
    points_a = trimesh.sample.sample_surface(mesh_a, 5000)[0]
    # Compute closest distances to mesh B
    _, distances, _ = mesh_b.nearest.on_surface(points_a)
    return float(np.max(distances))


def _safe_volume(mesh: trimesh.Trimesh) -> float:
    """Get mesh volume, filling holes if needed."""
    if mesh.is_watertight:
        return mesh.volume
    filled = mesh.copy()
    trimesh.repair.fill_holes(filled)
    return filled.volume if filled.is_watertight else 0.0


def compare_meshes(
    mesh_a: trimesh.Trimesh,
    mesh_b: trimesh.Trimesh,
) -> ComparisonResult:
    """Compare two oriented meshes and return difference metrics.

    Both meshes should be oriented in the same coordinate space
    (e.g., both auto-oriented via orient_auto).
    """
    # Hausdorff distance
    h_a_to_b = _hausdorff_distance(mesh_a, mesh_b)
    h_b_to_a = _hausdorff_distance(mesh_b, mesh_a)
    hausdorff = max(h_a_to_b, h_b_to_a)

    # Volume
    vol_a = _safe_volume(mesh_a)
    vol_b = _safe_volume(mesh_b)

    # Surface area
    area_a = mesh_a.area
    area_b = mesh_b.area

    # Centroid
    centroid_a = np.mean(mesh_a.vertices, axis=0)
    centroid_b = np.mean(mesh_b.vertices, axis=0)

    # Bounding box extents
    ext_a = mesh_a.bounding_box_oriented.extents
    ext_b = mesh_b.bounding_box_oriented.extents
    sorted_a = sorted(ext_a, reverse=True)
    sorted_b = sorted(ext_b, reverse=True)

    return ComparisonResult(
        hausdorff_distance_mm=round(hausdorff, 3),
        volume_difference_mm3=round(abs(vol_a - vol_b), 2),
        volume_a_mm3=round(vol_a, 2),
        volume_b_mm3=round(vol_b, 2),
        surface_area_difference_mm2=round(abs(area_a - area_b), 2),
        centroid_distance_mm=round(float(np.linalg.norm(centroid_a - centroid_b)), 3),
        length_diff_mm=round(abs(sorted_a[0] - sorted_b[0]), 2),
        width_diff_mm=round(abs(sorted_a[1] - sorted_b[1]), 2),
        thickness_diff_mm=round(abs(sorted_a[2] - sorted_b[2]), 2),
        metrics=[
            "hausdorff_distance_mm", "volume_difference_mm3",
            "volume_a_mm3", "volume_b_mm3",
            "surface_area_difference_mm2", "centroid_distance_mm",
            "length_diff_mm", "width_diff_mm", "thickness_diff_mm",
        ],
    )
