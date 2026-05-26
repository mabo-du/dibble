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
"""

from __future__ import annotations

import subprocess
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np


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
        return {"low": 20_000, "medium": 50_000, "high": 150_000}[self.quality]


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
