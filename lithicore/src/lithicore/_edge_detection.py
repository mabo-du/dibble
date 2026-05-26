"""_edge_detection.py — 3D mesh edge detection via dihedral angle thresholding.

exports: detect_edges(mesh, config) -> tuple[np.ndarray, np.ndarray]
used_by: Viewer edge overlay, CLI export
rules:   Returns (edge_vertex_indices, face_is_edge_mask).
         Edges are detected where the angle between adjacent face
         normals exceeds the configurable threshold.
agent:   deepseek-v4-flash | 2026-05-26 | Initial implementation
"""

from __future__ import annotations

import numpy as np
import trimesh
from lithicore._models import MeasurementConfig


def detect_edges(
    mesh: trimesh.Trimesh,
    config: MeasurementConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect edges on a mesh using dihedral angle thresholding.

    Computes the angle between adjacent face normals. Faces whose
    shared edge dihedral angle exceeds the threshold are marked as edges.
    Returns (edge_vertex_indices, face_is_edge_mask).
    """
    if len(mesh.faces) < 3:
        return np.array([], dtype=int), np.zeros(len(mesh.faces), dtype=bool)

    # Get face adjacency
    face_adjacency = mesh.face_adjacency
    if len(face_adjacency) == 0:
        return np.array([], dtype=int), np.zeros(len(mesh.faces), dtype=bool)

    # Compute dihedral angles between adjacent faces
    face_normals = mesh.face_normals
    adj_faces = face_adjacency
    n1 = face_normals[adj_faces[:, 0]]
    n2 = face_normals[adj_faces[:, 1]]
    dots = np.clip(np.einsum("ij,ij->i", n1, n2), -1.0, 1.0)
    dihedral_angles = np.degrees(np.arccos(np.abs(dots)))

    # Find edges above threshold
    sharp_edges = dihedral_angles > config.edge_threshold_degrees

    # Map to face mask
    face_is_edge = np.zeros(len(mesh.faces), dtype=bool)
    sharp_adj_faces = adj_faces[sharp_edges]
    if len(sharp_adj_faces) > 0:
        face_is_edge[np.unique(sharp_adj_faces.ravel())] = True

    # Map to vertex indices
    edge_vertices = np.unique(mesh.faces[face_is_edge].ravel())

    return edge_vertices, face_is_edge
