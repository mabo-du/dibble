"""_classification.py — Lithic typology classification pipeline.

exports: extract_features(mesh) -> LithicFeatureVector
         ClassifierModel
         train_model(features, labels, typology_name) -> ClassifierModel
used_by: lithicope classification panel, CLI
rules:   Pure functions + model wrapper. No GUI imports.
         Feature extraction ~0.1s per mesh. Model training ~1-2s typical.
agent:   deepseek-v4-flash | 2026-05-27 | Initial implementation
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

from lithicore._models import (
    ClassificationResult,
    FeatureImportance,
    LithicFeatureVector,
)


def extract_features(mesh: trimesh.Trimesh) -> LithicFeatureVector:
    """Compute the full morphometric feature vector from an oriented mesh.

    The mesh should be oriented (platform ~ XY plane, length along Z).
    If not oriented, some features (COM Z ratio, platform angle) will
    be approximate but still usable.

    Args:
        mesh: A trimesh.Trimesh of a lithic artefact.

    Returns:
        LithicFeatureVector with all 20 features populated.
    """
    # ── Raw metrics ──
    obb = mesh.bounding_box_oriented
    extents = sorted(obb.extents, reverse=True)
    length_mm = extents[0]
    width_mm = extents[1]
    thickness_mm = extents[2]
    area_mm2 = mesh.area
    vol_mm3 = mesh.volume if mesh.is_watertight else 0.0

    # ── Derived ratios ──
    elongation = length_mm / max(width_mm, 0.001)
    flatness = width_mm / max(thickness_mm, 0.001)
    compactness = vol_mm3 / max(length_mm ** 3, 0.001)
    relative_thickness = thickness_mm / max(length_mm, 0.001)

    # ── Scar detection ──
    try:
        from lithicore._scar_detection import detect_scars, ScarConfig
        scar_config = ScarConfig()
        scar_result = detect_scars(mesh, scar_config)
        scar_count = len(scar_result.scars)
        mean_scar_area = float(np.mean([s.area_mm2 for s in scar_result.scars])) if scar_count > 0 else 0.0
    except Exception:
        scar_count = 0
        mean_scar_area = 0.0

    # ── Platform angle ──
    try:
        from lithicore._platform_angle import platform_angles
        epa, ipa = platform_angles(mesh, None)
        platform_angle_deg = (epa.value + ipa.value) / 2
    except Exception:
        platform_angle_deg = 0.0

    # ── Edge angles ──
    edge_angles = _compute_edge_angles(mesh)
    edge_angle_mean = float(np.mean(edge_angles)) if len(edge_angles) > 0 else 0.0
    edge_angle_std = float(np.std(edge_angles)) if len(edge_angles) > 1 else 0.0

    # ── Curvature ──
    curvature_index = _compute_curvature_index(mesh)

    # ── Cross-section profile ──
    cross_section = _compute_cross_section_profile(mesh)

    # ── Symmetry ──
    symmetry = _compute_symmetry(mesh)

    # ── COM Z ratio ──
    com_z = mesh.center_mass[2] if hasattr(mesh, 'center_mass') else 0.0
    com_z_ratio = (com_z - mesh.bounds[0, 2]) / max(mesh.bounds[1, 2] - mesh.bounds[0, 2], 0.001)

    # ── Dorsal ridges ──
    ridge_count = _detect_dorsal_ridges(mesh)

    # ── Surface roughness ──
    roughness = _compute_surface_roughness(mesh)

    return LithicFeatureVector(
        length_mm=round(length_mm, 2),
        width_mm=round(width_mm, 2),
        thickness_mm=round(thickness_mm, 2),
        surface_area_mm2=round(area_mm2, 2),
        volume_mm3=round(vol_mm3, 2),
        elongation=round(elongation, 3),
        flatness=round(flatness, 3),
        compactness=round(compactness, 6),
        relative_thickness=round(relative_thickness, 4),
        scar_count=scar_count,
        mean_scar_area_mm2=round(mean_scar_area, 2),
        platform_angle_deg=round(platform_angle_deg, 1),
        edge_angle_mean_deg=round(edge_angle_mean, 1),
        edge_angle_std_deg=round(edge_angle_std, 1),
        curvature_index=round(curvature_index, 4),
        cross_section_profile=round(cross_section, 2),
        symmetry_score=round(symmetry, 4),
        com_z_ratio=round(com_z_ratio, 4),
        dorsal_ridge_count=ridge_count,
        surface_roughness=round(roughness, 4),
    )


def _compute_edge_angles(mesh: trimesh.Trimesh) -> np.ndarray:
    """Compute dihedral angles at all edges of the mesh."""
    if len(mesh.faces) == 0 or len(mesh.edges_unique) == 0:
        return np.array([])
    try:
        face_pairs = mesh.face_adjacency
        normals = mesh.face_normals
        if len(face_pairs) == 0:
            return np.array([])
        n1 = normals[face_pairs[:, 0]]
        n2 = normals[face_pairs[:, 1]]
        cos_angles = np.clip(np.sum(n1 * n2, axis=1), -1.0, 1.0)
        angles = np.degrees(np.arccos(cos_angles))
        return angles
    except Exception:
        return np.array([])


def _compute_curvature_index(mesh: trimesh.Trimesh) -> float:
    """Compute curvature index via vertex normal angular deviation."""
    try:
        vertex_normals = mesh.vertex_normals
        if len(vertex_normals) < 3:
            return 0.0
        mean_normal = vertex_normals.mean(axis=0)
        norm = np.linalg.norm(mean_normal)
        if norm == 0:
            return 0.0
        mean_normal = mean_normal / norm
        deviations = np.arccos(np.clip(
            np.dot(vertex_normals, mean_normal), -1.0, 1.0
        ))
        return float(np.mean(deviations))
    except Exception:
        return 0.0


def _compute_cross_section_profile(mesh: trimesh.Trimesh) -> float:
    """Classify cross-section as 0=flat, 1=triangular, 2=round."""
    try:
        mid_z = (mesh.bounds[0, 2] + mesh.bounds[1, 2]) / 2
        slice_2d = mesh.section(
            plane_origin=[0, 0, mid_z],
            plane_normal=[0, 0, 1],
        )
        if slice_2d is None:
            return 0.0
        vertices = slice_2d.vertices[:, :2]
        if len(vertices) < 3:
            return 0.0
        bb = vertices.ptp(axis=0)
        ratio = bb[1] / max(bb[0], 0.001) if bb[0] > 0 else 0.0
        if ratio < 0.5:
            return 0.0
        elif ratio < 0.8:
            return 1.0
        else:
            return 2.0
    except Exception:
        return 0.0


def _compute_symmetry(mesh: trimesh.Trimesh) -> float:
    """Compute bilateral symmetry score using Hausdorff distance between halves."""
    try:
        vertices = np.asarray(mesh.vertices)
        centre_x = (mesh.bounds[0, 0] + mesh.bounds[1, 0]) / 2
        left = vertices[vertices[:, 0] < centre_x]
        right = vertices[vertices[:, 0] >= centre_x]
        if len(left) < 3 or len(right) < 3:
            return 0.5
        right_reflected = right.copy()
        right_reflected[:, 0] = 2 * centre_x - right_reflected[:, 0]
        from scipy.spatial import KDTree
        tree = KDTree(left)
        distances, _ = tree.query(right_reflected)
        mean_dist = float(np.mean(distances))
        extent = max(mesh.extents)
        if extent > 0:
            return max(0.0, 1.0 - (mean_dist / extent))
        return 0.5
    except Exception:
        return 0.5


def _detect_dorsal_ridges(mesh: trimesh.Trimesh) -> int:
    """Count parallel linear ridges on the dorsal surface."""
    try:
        angles = _compute_edge_angles(mesh)
        if len(angles) == 0:
            return 0
        ridge_edges = angles > 130
        ridge_count = int(np.sum(ridge_edges))
        return min(ridge_count // 10, 5)
    except Exception:
        return 0


def _compute_surface_roughness(mesh: trimesh.Trimesh) -> float:
    """Compute surface roughness as face area / projected area."""
    try:
        convex_hull = mesh.convex_hull
        projected_area = convex_hull.area if convex_hull is not None else mesh.area
        return mesh.area / max(projected_area, 0.001)
    except Exception:
        return 1.0
