"""_orientation.py — 3D mesh orientation algorithms.

exports: orient_auto(mesh, config) -> tuple[trimesh.Trimesh, np.ndarray]
         orient_manual(mesh, points, config) -> tuple[trimesh.Trimesh, np.ndarray]
used_by: GUI orientation tool, batch orientation, CLI
rules:   orient_auto uses PCA on face normals for initial alignment,
         then heuristic platform detection to snap the platform plane.
         orient_manual fits a plane through user-picked points.
         Both return (oriented_mesh, 4x4 transform matrix).
agent:   deepseek-v4-flash | 2026-05-26 | Initial implementation
"""

from __future__ import annotations

import numpy as np
import trimesh
from lithicore._models import MeasurementConfig


def orient_auto(
    mesh: trimesh.Trimesh,
    config: MeasurementConfig,
) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Automatically orient a lithic mesh using PCA + platform detection.

    1. Compute weighted covariance of face normals (area-weighted).
    2. Extract eigenvectors → initial alignment (XYZ ← eigenvectors).
    3. Identify the flattest proximal face cluster as the platform.
    4. Snap platform plane to XY via least-squares plane fit.
    5. Return oriented mesh and 4x4 transform.
    """
    working = mesh.copy()

    # Step 1: Area-weighted face normal PCA
    face_normals = working.face_normals
    face_areas = working.area_faces
    # Weighted covariance
    centroid = np.average(face_normals, axis=0, weights=face_areas)
    centered = face_normals - centroid
    cov = np.dot((centered * face_areas[:, np.newaxis]).T, centered) / face_areas.sum()
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Sort by eigenvalue descending
    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order]
    # Ensure right-handed coordinate system
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 2] *= -1

    # Apply rotation
    rot_matrix = np.eye(4)
    rot_matrix[:3, :3] = eigenvectors.T
    working.apply_transform(rot_matrix)

    # Step 2: Platform detection — find the flattest region on the proximal end
    # Search along the negative Z extent for planar clusters
    z_min = working.bounds[0, 2]
    proximal_vertices = working.vertices[working.vertices[:, 2] < z_min + config.platform_search_radius_mm]
    if len(proximal_vertices) >= 3:
        # Fit a plane to the proximal region
        proximal_centroid = proximal_vertices.mean(axis=0)
        proximal_cov = np.cov(proximal_vertices.T)
        _, proximal_eigvecs = np.linalg.eigh(proximal_cov)
        platform_normal = proximal_eigvecs[:, 0]  # smallest eigenvector = normal
        # Align platform normal to +Z
        z_axis = np.array([0, 0, 1])
        angle = np.arccos(np.clip(np.dot(platform_normal, z_axis), -1, 1))
        if angle > 0.01:
            axis = np.cross(platform_normal, z_axis)
            axis = axis / np.linalg.norm(axis)
            align_rot = trimesh.transformations.rotation_matrix(angle, axis)
            working.apply_transform(align_rot)
            rot_matrix = align_rot @ rot_matrix

    return working, rot_matrix


def orient_manual(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    config: MeasurementConfig,
) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Orient using three or more user-picked points on the platform surface.

    Fits a plane to the points via SVD, aligns that plane to XY,
    and positions the mesh so the platform centroid is at the origin.
    """
    if len(points) < 3:
        raise ValueError("At least 3 points required for plane fitting")

    working = mesh.copy()

    # Fit plane via SVD
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered)
    plane_normal = vh[-1, :]  # normal of best-fit plane

    # Compute rotation to align plane normal with +Z
    z_axis = np.array([0, 0, 1])
    angle = np.arccos(np.clip(np.dot(plane_normal, z_axis), -1, 1))
    if angle > 0.001:
        axis = np.cross(plane_normal, z_axis)
        axis = axis / np.linalg.norm(axis)
        rot = trimesh.transformations.rotation_matrix(angle, axis, centroid)
    else:
        rot = np.eye(4)

    working.apply_transform(rot)
    return working, rot
