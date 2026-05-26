"""_scale_detection.py — Automatic scale detection + mesh rescaling.

exports: ScaleResult
         detect_scale_aruco(photos, sparse_cloud_dir, marker_size_mm) -> Optional[ScaleResult]
         detect_scale_ruler(photos, sparse_cloud_dir) -> Optional[ScaleResult]
         apply_scale_to_mesh(mesh, scale_factor) -> trimesh.Trimesh
used_by: lithicore photogrammetry pipeline
rules:   Pure functions, no GUI imports. Scale detection operates on sparse cloud
         + source photos, not dense mesh.
agent:   deepseek-v4-flash | 2026-05-27 | Initial — dataclass + mesh transform
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh


@dataclass
class ScaleResult:
    """Result of automatic scale detection.

    Attributes:
        scale_factor: Multiply COLMAP unit coordinates by this to get mm.
        method: Detection method — 'aruco', 'ruler', or 'manual'.
        confidence: Estimated reliability (0 = none, 1 = certain).
        detected_length_mm: Physical length detected (mm).
        warnings: Non-fatal issues during detection.
    """
    scale_factor: float
    method: str
    confidence: float
    detected_length_mm: float = 0.0
    warnings: list[str] = field(default_factory=list)


def apply_scale_to_mesh(
    mesh: trimesh.Trimesh,
    scale_factor: float,
) -> trimesh.Trimesh:
    """Apply a uniform scale factor to all mesh vertices.

    Args:
        mesh: Input mesh (unchanged).
        scale_factor: Multiplier for vertex coordinates. Must be positive.

    Returns:
        A new trimesh.Trimesh with scaled vertices. Face topology is preserved.
        Normal arrays are invalidated (recomputed on next access).

    Raises:
        ValueError: If scale_factor is zero or negative.
    """
    if scale_factor <= 0:
        raise ValueError(
            f"Scale factor must be positive, got {scale_factor}"
        )
    scaled = mesh.copy()
    scaled.vertices = mesh.vertices * scale_factor
    # Invalidate cached normals — they will be recomputed by trimesh on demand
    scaled.face_normals = None
    scaled.vertex_normals = None
    return scaled
