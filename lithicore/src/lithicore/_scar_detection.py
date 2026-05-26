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
    curvature_radius: float = 3.0
    ridge_threshold: float = 0.4
    valley_threshold: float = -0.3
    min_scar_faces: int = 20
    curvedness_percentile: float = 70.0


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
    """Compute principal curvatures (k1, k2) per vertex.

    Uses trimesh's discrete curvature measures to derive mean (H) and
    Gaussian (K) curvature per vertex, then solves for principal curvatures:

        k1 = H + sqrt(H^2 - K)   (max curvature)
        k2 = H - sqrt(H^2 - K)   (min curvature)

    For flat regions where K > H^2 (numerical noise), both principal
    curvatures are set to H.
    """
    verts = mesh.vertices

    # Discrete curvature measures via trimesh
    K = trimesh.curvature.discrete_gaussian_curvature_measure(
        mesh, verts, radius,
    )
    H = trimesh.curvature.discrete_mean_curvature_measure(
        mesh, verts, radius,
    )

    # Solve for principal curvatures from mean and Gaussian
    # H = (k1 + k2) / 2,  K = k1 * k2
    # discriminant = H^2 - K
    discriminant = H ** 2 - K
    # Clamp to avoid sqrt of negative values from numerical noise
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
    """Detect flake scars on a lithic mesh using Shape Index + watershed.

    Algorithm:
    1. Compute per-vertex principal curvatures via discrete mean/Gaussian
       curvature measures.
    2. Compute Shape Index (SI) and Curvedness (C) per vertex.
    3. Map per-vertex values to per-face (average of incident vertices).
    4. Identify ridge vertices (SI > ridge_threshold AND C > curvedness
       percentile threshold). Ridge faces are those containing ridge vertices.
    5. Identify valley seed faces (face-averaged SI < valley_threshold).
    6. Watershed flood-fill from valley seeds across face adjacency,
       treating ridge faces as stop boundaries.
    7. Filter scars by minimum face count.

    Args:
        mesh: A triangle mesh in millimetre units.
        config: Detection parameters (ScarConfig).

    Returns:
        ScarResult with per-face labels and per-scar metrics.
    """
    if len(mesh.faces) < config.min_scar_faces:
        return ScarResult(
            scar_count=0,
            scars=[],
            total_scar_area_mm2=0.0,
            scar_density=0.0,
            face_labels=np.full(len(mesh.faces), -1, dtype=int),
        )

    # ----------------------------------------------------------------
    # Step 1-2: Curvature analysis
    # ----------------------------------------------------------------
    k1, k2 = _compute_principal_curvatures(mesh, config.curvature_radius)
    si = _shape_index(k1, k2)
    cv = _curvedness(k1, k2)

    # ----------------------------------------------------------------
    # Step 3: Map per-vertex values to per-face averages
    # ----------------------------------------------------------------
    face_si = np.mean(si[mesh.faces], axis=1)
    face_cv = np.mean(cv[mesh.faces], axis=1)

    # ----------------------------------------------------------------
    # Step 4: Identify ridge boundaries
    # ----------------------------------------------------------------
    cv_threshold = np.percentile(cv, config.curvedness_percentile)
    ridge_vertices = np.where(
        (cv > cv_threshold) & (si > config.ridge_threshold)
    )[0]
    # A face is a ridge boundary if any of its vertices is a ridge vertex
    ridge_faces_mask = np.any(np.isin(mesh.faces, ridge_vertices), axis=1)

    # ----------------------------------------------------------------
    # Step 5: Valley seeds (interior of scars)
    # ----------------------------------------------------------------
    valley_faces_mask = face_si < config.valley_threshold

    if not np.any(valley_faces_mask):
        return ScarResult(
            scar_count=0,
            scars=[],
            total_scar_area_mm2=0.0,
            scar_density=0.0,
            face_labels=np.full(len(mesh.faces), -1, dtype=int),
        )

    # ----------------------------------------------------------------
    # Step 6: Watershed flood-fill
    # ----------------------------------------------------------------
    # Build adjacency list: for each face, which faces share an edge
    face_adj = mesh.face_adjacency  # (N, 2) array of adjacent face index pairs
    adj_list = [[] for _ in range(len(mesh.faces))]
    for f1, f2 in face_adj:
        adj_list[f1].append(f2)
        adj_list[f2].append(f1)

    face_labels = np.full(len(mesh.faces), -1, dtype=int)
    valley_indices = np.where(valley_faces_mask)[0]

    current_label = 0
    for seed in valley_indices:
        if face_labels[seed] >= 0:
            continue  # already claimed by an earlier seed

        # BFS flood fill from this seed, stopping at ridge faces
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
                    continue  # ridge faces act as watershed boundaries
                face_labels[nb] = current_label
                queue.append(nb)

        # Optional: also expand into adjacent non-ridge, non-valley faces
        # (second-pass expansion to fill "indeterminate" areas between scars)
        expanded = _expand_into_indeterminate(
            component, adj_list, face_labels, ridge_faces_mask, current_label,
        )

        all_faces = expanded if expanded is not None else component

        # Step 7: filter by minimum size
        if len(all_faces) < config.min_scar_faces:
            for f in all_faces:
                face_labels[f] = -1
        else:
            current_label += 1

    # ----------------------------------------------------------------
    # Build scar result list
    # ----------------------------------------------------------------
    unique_labels = set(face_labels[face_labels >= 0])
    scars: List[DetectedScar] = []
    total_scar_area = 0.0

    for label_id in sorted(unique_labels):
        scar_faces = np.where(face_labels == label_id)[0]
        if len(scar_faces) < config.min_scar_faces:
            continue

        scar_area = float(mesh.area_faces[scar_faces].sum())
        total_scar_area += scar_area

        # Centroid: average of incident vertices
        verts_of_scar = mesh.vertices[mesh.faces[scar_faces].ravel()]
        scar_centroid = tuple(
            round(c, 3) for c in np.mean(verts_of_scar, axis=0).tolist()
        )

        # Depth proxy: mean curvedness across scar faces
        scar_depth = float(np.mean(face_cv[scar_faces]))

        scars.append(DetectedScar(
            index=label_id,
            face_indices=scar_faces,
            area_mm2=round(scar_area, 2),
            centroid=scar_centroid,
            mean_curvature=round(float(np.mean(face_si[scar_faces])), 4),
            max_depth_mm=round(scar_depth, 3),
        ))

    total_area = max(mesh.area, 1e-10)

    return ScarResult(
        scar_count=len(scars),
        scars=scars,
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
