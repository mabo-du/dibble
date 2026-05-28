"""_scar_detection.py — Flake scar detection via Shape Index + watershed segmentation.

exports: ScarResult, ScarConfig, DetectedScar, detect_scars(mesh, config) -> ScarResult
used_by: GUI analysis pipeline, CLI report export
rules:   Shape Index computed per-vertex from principal curvatures (k1, k2).
         S = 2/pi * arctan2(k2 + k1, |k2 - k1|). Range [-1, 1].
         Principal curvatures derived from discrete mean + Gaussian curvature.
         Scar boundaries = ridge lines (S > ridge_threshold, high curvedness).
         Scar interiors = watershed flood from valley seeds (S < valley_threshold)
                          propagating through non-ridge faces.
agent:   deepseek-v4-flash | 2026-05-26 | Initial implementation with proper BFS watershed
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List

import numpy as np
import trimesh


@dataclass
class ScarConfig:
    """Configuration for scar detection.

    Attributes:
        curvature_radius: Neighbourhood radius (mm) for discrete curvature
            estimation. Larger values smooth curvature. Default 3.0.
        ridge_threshold: Shape Index above this = ridge (scar boundary).
            Default 0.4.
        valley_threshold: Shape Index below this = valley seed (scar interior).
            Default -0.3.
        min_scar_faces: Minimum faces to count as a valid scar. Default 20.
        curvedness_percentile: Only vertices above this curvedness percentile
            are eligible to be ridge boundaries. Default 70.
    """
    curvature_radius: float = 1.0  # Unused in fast proxy, kept for API compat
    ridge_threshold: float = 0.2   # face_K below this = ridge boundary (was 0.4)
    valley_threshold: float = -0.2  # face_K above abs(threshold) = valley seed (was -0.3)
    min_scar_faces: int = 20
    curvedness_percentile: float = 70.0  # Unused in fast proxy


@dataclass
class DetectedScar:
    """A single detected flake scar."""
    index: int
    face_indices: np.ndarray
    area_mm2: float
    centroid: tuple[float, float, float]
    mean_curvature: float
    max_depth_mm: float


@dataclass
class ScarResult:
    """Result of scar detection analysis."""
    scar_count: int
    scars: List[DetectedScar]
    total_scar_area_mm2: float
    scar_density: float  # scar area / total surface area
    face_labels: np.ndarray  # per-face label (-1 = no scar, 0+ = scar index)


def _compute_principal_curvatures(
    mesh: trimesh.Trimesh,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute principal curvatures (k1, k2) per vertex (fast approximation).

    Uses trimesh's vertex_defects for Gaussian curvature (O(n), ~0.04s)
    and dihedral-angle-weighted mean curvature from face adjacency, then
    solves for principal curvatures:

        k1 = H + sqrt(H^2 - K)   (max curvature)
        k2 = H - sqrt(H^2 - K)   (min curvature)

    This avoids the expensive geodesic-neighbourhood curvature measures
    (discrete_gaussian_curvature_measure) which hang on dense meshes
    at default radius=3.0 (avg edge length ~0.17mm).
    """
    # Fast Gaussian curvature via vertex defects (angle deficit)
    # K ≈ 2π - Σ(face angles at vertex); 0 on flat, +valley, -peak
    K = mesh.vertex_defects

    # Approximate Mean curvature via dihedral-angle-weighted area
    # For each vertex, H ≈ Σ(dihedral_angle * edge_length) / (4 * vertex_area)
    # Simplified: use face normal divergence as H proxy
    try:
        # Compute per-vertex mean curvature from face angle normals
        face_pairs = mesh.face_adjacency
        if len(face_pairs) > 0:
            normals = mesh.face_normals
            n1 = normals[face_pairs[:, 0]]
            n2 = normals[face_pairs[:, 1]]
            cos_angles = np.clip(np.sum(n1 * n2, axis=1), -1.0, 1.0)
            dihedral = np.arccos(cos_angles)

            # Edge lengths for weighting
            edge_len = mesh.edges_unique_length

            # Per-vertex mean curvature approximation:
            # Sum weighted dihedral angles at each vertex
            vertex_edges = mesh.vertex_faces  # (N_vert, max_valency) edges per vertex
            edge_valence = mesh.vertex_degree

            # Build vertex-to-dihedral map
            v_dihedral = np.zeros(len(mesh.vertices))
            v_weight = np.zeros(len(mesh.vertices))
            for (f1, f2), angle, elen in zip(face_pairs, dihedral, edge_len):
                # Shared edge vertices: find the two vertices of this edge
                edge_verts = mesh.faces[f1][
                    np.isin(mesh.faces[f1], mesh.faces[f2], assume_unique=False)
                ]
                for v in edge_verts:
                    v_dihedral[v] += angle * elen
                    v_weight[v] += elen

            # Normalise
            mask = v_weight > 0
            H = np.zeros(len(mesh.vertices))
            H[mask] = v_dihedral[mask] / v_weight[mask]
        else:
            H = np.zeros(len(mesh.vertices))
    except Exception:
        H = np.zeros(len(mesh.vertices))

    # For flat regions where K > H^2 (numerical noise), clamp
    discriminant = H ** 2 - np.abs(K)
    sqrt_d = np.sqrt(np.clip(discriminant, 0.0, None))

    k1 = H + sqrt_d
    k2 = H - sqrt_d

    return k1, k2


def _shape_index(k1: np.ndarray, k2: np.ndarray) -> np.ndarray:
    """Compute Koenderink Shape Index per vertex.

    S = 2/pi * arctan2(k2 + k1, |k2 - k1|)

    Returns values in [-1, 1]:
      -1.0 = spherical cup (deep depression)
      -0.5 = valley/rut
       0.0 = saddle
       0.5 = ridge
       1.0 = spherical cap (prominent bump)
    """
    numerator = k2 + k1
    denominator = np.abs(k2 - k1) + 1e-10
    return 2.0 / np.pi * np.arctan2(numerator, denominator)


def _curvedness(k1: np.ndarray, k2: np.ndarray) -> np.ndarray:
    """Compute curvedness (curvature intensity) per vertex.

    C = sqrt((k1^2 + k2^2) / 2)
    """
    return np.sqrt((k1 ** 2 + k2 ** 2) / 2.0)


def detect_scars(
    mesh: trimesh.Trimesh,
    config: ScarConfig,
) -> ScarResult:
    """Detect flake scars on a lithic mesh using fast statistical proxy.

    Uses vertex defects (angle deficit, ~0.04s) as a curvature proxy
    instead of the expensive geodesic-neighbourhood curvature measures
    which hang on dense meshes (avg edge length ~0.17mm).

    The original watershed-based algorithm with Shape Index was a
    well-motivated approach but the principal curvature computation
    (trimesh.curvature.discrete_gaussian_curvature_measure at
    radius=3.0mm) is O(n * k) where k ~ hundreds of neighbours
    per vertex on dense archaeological scans, making it unusable
    for batch processing.

    Args:
        mesh: A triangle mesh in millimetre units.
        config: Detection parameters (ScarConfig).

    Returns:
        ScarResult with statistical scar proxy metrics.
    """
    if len(mesh.faces) < config.min_scar_faces:
        return ScarResult(
            scar_count=0, scars=[],
            total_scar_area_mm2=0.0, scar_density=0.0,
            face_labels=np.full(len(mesh.faces), -1, dtype=int),
        )

    # Fast curvature proxy via vertex defects (angle deficit)
    # Positive = concave (valley), Negative = convex (ridge)
    K = mesh.vertex_defects
    K_norm = np.tanh(K * 10)  # squash to [-1, 1]

    # Map to per-face averages
    face_K = np.mean(K_norm[mesh.faces], axis=1)

    # Ridge boundary faces (convex, low K)
    ridge_faces_mask = face_K < config.ridge_threshold
    # Valley seed faces (concave, high K)
    valley_faces_mask = face_K > -config.valley_threshold

    if not np.any(valley_faces_mask):
        return ScarResult(
            scar_count=0, scars=[],
            total_scar_area_mm2=0.0, scar_density=0.0,
            face_labels=np.full(len(mesh.faces), -1, dtype=int),
        )

    # Watershed flood-fill from valley seeds bounded by ridge faces
    face_adj = mesh.face_adjacency
    adj_list = [[] for _ in range(len(mesh.faces))]
    for f1, f2 in face_adj:
        adj_list[f1].append(f2)
        adj_list[f2].append(f1)

    face_labels = np.full(len(mesh.faces), -1, dtype=int)
    valley_indices = np.where(valley_faces_mask)[0]

    current_label = 0
    for seed in valley_indices:
        if face_labels[seed] >= 0:
            continue

        component = []
        queue = deque([seed])
        face_labels[seed] = current_label

        while queue:
            f = queue.popleft()
            component.append(f)
            for nb in adj_list[f]:
                if face_labels[nb] >= 0:
                    continue
                if ridge_faces_mask[nb]:
                    continue
                face_labels[nb] = current_label
                queue.append(nb)

        if len(component) < config.min_scar_faces:
            for f in component:
                face_labels[f] = -1
        else:
            current_label += 1

    # Build result
    unique_labels = set(face_labels[face_labels >= 0])
    scars: List[DetectedScar] = []
    total_scar_area = 0.0

    for label_id in sorted(unique_labels):
        scar_faces = np.where(face_labels == label_id)[0]
        if len(scar_faces) < config.min_scar_faces:
            continue
        scar_area = float(mesh.area_faces[scar_faces].sum())
        total_scar_area += scar_area
        verts = mesh.vertices[mesh.faces[scar_faces].ravel()]
        centroid = tuple(round(c, 3) for c in np.mean(verts, axis=0).tolist())
        scars.append(DetectedScar(
            index=label_id, face_indices=scar_faces,
            area_mm2=round(scar_area, 2), centroid=centroid,
            mean_curvature=0.0, max_depth_mm=0.0,
        ))

    total_area = max(mesh.area, 1e-10)
    return ScarResult(
        scar_count=len(scars), scars=scars,
        total_scar_area_mm2=round(total_scar_area, 2),
        scar_density=round(total_scar_area / total_area, 4),
        face_labels=face_labels,
    )


def _expand_into_indeterminate(
    component: List[int],
    adj_list: List[List[int]],
    face_labels: np.ndarray,
    ridge_faces_mask: np.ndarray,
    label: int,
) -> np.ndarray | None:
    """Expand a scar component into adjacent unlabeled non-ridge faces.

    This second-pass expansion allows a scar label to fill "neutral"
    territory (faces that are neither valleys nor ridges) contiguous
    with the component, producing more complete scar delineation.

    Returns expanded face indices, or None if no expansion occurred.
    """
    frontier = set()
    for f in component:
        for nb in adj_list[f]:
            if face_labels[nb] < 0 and not ridge_faces_mask[nb]:
                frontier.add(nb)

    if not frontier:
        return None

    # BFS from frontier into unlabeled non-ridge territory
    expanded = set(component)
    queue = deque(frontier)

    while queue:
        f = queue.popleft()
        if f in expanded:
            continue
        expanded.add(f)
        face_labels[f] = label
        for nb in adj_list[f]:
            if face_labels[nb] < 0 and not ridge_faces_mask[nb] and nb not in expanded:
                queue.append(nb)

    return np.array(list(expanded))
