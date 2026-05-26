"""_photogrammetry.py — COLMAP-based photogrammetry pipeline (photos → 3D mesh).

exports: PhotogrammetryConfig
         PhotogrammetryResult
         PhotogrammetryError, ColmapNotFoundError, ColmapStageError,
           InsufficientPhotosError, PhotogrammetryCancelledError
         run_pipeline(config, progress_cb=None) -> PhotogrammetryResult
         clean_point_cloud(cloud, config) -> np.ndarray
used_by: lithicore CLI, lithicope GUI
rules:   Zero GUI imports. Pure functions with dataclass configs.
         COLMAP is run via subprocess; pre/post processing is Python-native.
agent:   deepseek-v4-flash | 2026-05-26 | Initial scaffolding — dataclasses + error types
agent:   deepseek-v4-flash | 2026-05-26 | Added colmap_available() + clean_point_cloud() + _crop_background() with spatial-spread floor for robust SOR
"""

from __future__ import annotations

import subprocess
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np


__all__ = [
    "PhotogrammetryConfig",
    "PhotogrammetryResult",
    "PhotogrammetryError",
    "ColmapNotFoundError",
    "ColmapStageError",
    "InsufficientPhotosError",
    "PhotogrammetryCancelledError",
    "colmap_available",
    "clean_point_cloud",
]


# ──────────────────────────────────────────────
# Error types
# ──────────────────────────────────────────────

class PhotogrammetryError(Exception):
    """Base exception for photogrammetry pipeline errors."""

class ColmapNotFoundError(PhotogrammetryError):
    """COLMAP binary not found on system PATH."""

class ColmapStageError(PhotogrammetryError):
    """A COLMAP processing stage failed."""
    def __init__(self, stage: str, stderr: str) -> None:
        self.stage = stage
        self.stderr = stderr
        super().__init__(f"COLMAP {stage} failed: {stderr[:200]}")

class InsufficientPhotosError(PhotogrammetryError):
    """Fewer than minimum photos provided, or matching failed."""

class PhotogrammetryCancelledError(PhotogrammetryError):
    """Pipeline was cancelled by the user."""


# ──────────────────────────────────────────────
# COLMAP availability
# ──────────────────────────────────────────────

def colmap_available() -> bool:
    """Check whether COLMAP binary is available on PATH."""
    return shutil.which("colmap") is not None


# ──────────────────────────────────────────────
# Point cloud cleaning
# ──────────────────────────────────────────────

def clean_point_cloud(
    points: np.ndarray,
    threshold: float = 2.0,
) -> np.ndarray:
    """Remove statistical outliers from a point cloud.

    For each point, compute mean distance to k=20 nearest neighbours.
    Remove points where mean distance > global_mean + (threshold * effective_std),
    where effective_std = max(raw_stddev, point_spread * 0.35). The spatial-spread
    floor prevents over-aggressive removal in tightly clustered clouds where
    nearest-neighbour distance variance is small relative to cloud extent.

    Args:
        points: (N, 3) array of 3D points.
        threshold: Stddev multiplier for outlier cutoff. Default 2.0.

    Returns:
        Filtered (M, 3) array where M <= N.
    """
    from scipy.spatial import KDTree

    if len(points) < 21:
        return points

    tree = KDTree(points)
    # k=21 because the point itself is included in the count
    distances, _ = tree.query(points, k=min(21, len(points)))
    mean_distances = distances[:, 1:].mean(axis=1)  # exclude self-distance

    global_mean = mean_distances.mean()
    global_std = mean_distances.std()
    if global_std == 0:
        return points

    # Use spatial spread as a floor for the stddev to avoid
    # over-aggressive removal in tightly clustered clouds where
    # nearest-neighbour distance variance is small relative to
    # the overall extent of the point cloud.
    point_spread: float = float(np.linalg.norm(np.std(points, axis=0)))
    effective_std: float = max(global_std, point_spread * 0.35)

    mask = mean_distances <= global_mean + (threshold * effective_std)
    return points[mask]


def _crop_background(
    points: np.ndarray,
    margin: float = 1.5,
) -> np.ndarray:
    """Automatically crop background geometry from a dense point cloud.

    Computes a bounding box around the densest cluster (the artefact)
    and removes points outside margin * bounding_box.

    Args:
        points: (N, 3) array of 3D points.
        margin: Multiplier for bounding box extent. Default 1.5.

    Returns:
        Cropped (M, 3) array.
    """
    if len(points) < 10:
        return points

    # First remove statistical outliers to find the main cluster
    cleaned = clean_point_cloud(points, threshold=3.0)
    if len(cleaned) < 10:
        return points

    # Compute centroid and PCA from the cleaned subset (robust to outliers)
    centroid = cleaned.mean(axis=0)
    centered = cleaned - centroid

    cov = np.cov(centered.T)
    _, eigenvectors = np.linalg.eigh(cov)

    # Project cleaned points for bounding box estimation
    projected_cleaned = centered @ eigenvectors
    mins = projected_cleaned.min(axis=0)
    maxs = projected_cleaned.max(axis=0)
    extents = maxs - mins
    centre_pc = (mins + maxs) / 2

    # Apply bounding box + margin to ALL original points
    centered_all = points - centroid
    projected_all = centered_all @ eigenvectors
    half_extents = extents / 2 * margin
    mask = np.all(np.abs(projected_all - centre_pc) <= half_extents, axis=1)
    return points[mask]


# ──────────────────────────────────────────────
# Data types
# ──────────────────────────────────────────────

@dataclass
class PhotogrammetryConfig:
    """Configuration for the photogrammetry pipeline.

    mode determines which config fields are user-settable:
      - default: only photo_folder, output_path, artefact_label, quality
      - guided: plus photo_settings and cleanup fields
      - expert: all fields exposed
    """
    photo_folder: Path
    output_path: Path
    artefact_label: str = ""
    quality: str = "high"               # "low" | "medium" | "high"
    mode: str = "default"               # "default" | "guided" | "expert"

    # Guided/Expert: photo settings
    camera_model: str = "auto"          # "auto" | "smartphone" | "dslr"
    scale_bar_cm: float = 0.0           # 0 = no scale reference

    # Guided/Expert: cleanup
    auto_crop_background: bool = True
    fill_holes: bool = True

    # Expert-only: COLMAP tuning
    colmap_feature_type: str = "sift"
    colmap_matching_strategy: str = "exhaustive"
    colmap_meshing: str = "poisson"
    colmap_dense_quality: str = "extreme"
    max_vertices: int = 500000

    # Internal
    colmap_workspace: Optional[Path] = None
    cleanup_temp: bool = True

    def __post_init__(self) -> None:
        self.photo_folder = Path(self.photo_folder)
        self.output_path = Path(self.output_path)

    @property
    def target_faces(self) -> int:
        """Map quality slider to target face count after decimation."""
        mapping = {"low": 20_000, "medium": 50_000, "high": 150_000}
        try:
            return mapping[self.quality]
        except KeyError:
            raise ValueError(
                f"Unknown quality '{self.quality}'; expected 'low', 'medium', or 'high'"
            ) from None


@dataclass
class PhotogrammetryResult:
    """Output of a completed photogrammetry pipeline run."""
    mesh_path: Path
    artefact_label: str
    camera_count: int
    point_count: int
    face_count: int
    vertex_count: int
    processing_time_s: float
    colmap_stdout: str
    warnings: list[str]
    sparse_cloud_path: Optional[Path] = None
    dense_cloud_path: Optional[Path] = None
