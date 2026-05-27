"""_models.py — Core data types for lithicore measurement pipeline.

exports: MeasurementConfig, MeasurementResult, ArtefactResult, Landmark, MeshQualityReport, MeshGrade, FeatureImportance, ClassificationResult, LithicFeatureVector
used_by: Every lithicore module imports these dataclasses
rules:   All dataclasses frozen except ClassificationResult (mutable container).
         MeasurementResult.confidence is 0.0-1.0.
         MeshGrade is a StrEnum (Pass, Warn, Fail).
agent:   deepseek-v4-flash | 2026-05-26 | Initial data model
agent:   deepseek-v4-flash | 2026-05-27 | Added FeatureImportance and ClassificationResult dataclasses
agent:   deepseek-v4-flash | 2026-05-27 | Added LithicFeatureVector with to_array() and from_array()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import ClassVar, List

import numpy as np


class MeshGrade(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True)
class MeasurementConfig:
    """Configuration shared across all measurement algorithms.

    Attributes:
        repair_mesh: Apply auto-repair before measurement (default True).
        edge_threshold_degrees: Dihedral angle threshold for edge detection (default 50.0).
        min_face_count: Minimum faces for reliable edge angle measurement (default 2000).
        platform_search_radius_mm: Search radius for platform detection (default 5.0).
    """
    repair_mesh: bool = True
    edge_threshold_degrees: float = 50.0
    min_face_count: int = 2000
    platform_search_radius_mm: float = 5.0


@dataclass(frozen=True)
class MeasurementResult:
    """A single measurement value with metadata."""
    name: str
    value: float
    unit: str
    confidence: float


@dataclass(frozen=True)
class Landmark:
    """A 3D landmark in oriented coordinate space."""
    name: str
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class MeshQualityReport:
    """Result of mesh validation and repair."""
    original_vertex_count: int
    original_face_count: int
    repaired_vertex_count: int = 0
    repaired_face_count: int = 0
    holes_filled: int = 0
    non_manifold_edges_fixed: int = 0
    isolated_components_removed: int = 0
    grade: MeshGrade = MeshGrade.PASS
    warnings: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ArtefactResult:
    """Complete measurement results for one artefact."""
    file_path: Path
    label: str
    measurements: List[MeasurementResult]
    landmarks: List[Landmark]
    warnings: List[str]


@dataclass(frozen=True)
class FeatureImportance:
    """A single feature's contribution to a classification decision."""
    name: str
    value: float
    contribution_pct: float  # % of trees that split on this feature
    expected_range: tuple[float, float]  # typical range for the predicted class
    passed: bool  # whether value falls within expected range


@dataclass
class ClassificationResult:
    """Result of a typology classification pipeline run."""
    label: str
    confidence: float
    probabilities: dict[str, float]  # class -> probability
    top_features: list[FeatureImportance]
    alternatives: list[tuple[str, float]]  # (label, confidence) for other classes
    typology_name: str  # e.g. "basic", "bordes", "technological", "custom"
    processing_time_s: float
    warnings: list[str]


@dataclass
class LithicFeatureVector:
    """Numerical fingerprint of a lithic artefact (20+ dimensional feature vector).

    All measurements in mm or mm²/mm³. Computed from an oriented mesh.
    """
    # Raw metrics
    length_mm: float = 0.0
    width_mm: float = 0.0
    thickness_mm: float = 0.0
    surface_area_mm2: float = 0.0
    volume_mm3: float = 0.0

    # Derived ratios
    elongation: float = 0.0       # L/W
    flatness: float = 0.0         # W/T
    compactness: float = 0.0      # V/L³
    relative_thickness: float = 0.0  # T/L

    # Morphological features
    scar_count: int = 0
    mean_scar_area_mm2: float = 0.0
    platform_angle_deg: float = 0.0
    edge_angle_mean_deg: float = 0.0
    edge_angle_std_deg: float = 0.0
    curvature_index: float = 0.0
    cross_section_profile: float = 0.0  # 0=flat, 1=triangular, 2=round
    symmetry_score: float = 0.0
    com_z_ratio: float = 0.0
    dorsal_ridge_count: int = 0
    surface_roughness: float = 0.0

    FEATURE_NAMES: ClassVar[list[str]] = [
        "length_mm", "width_mm", "thickness_mm", "surface_area_mm2", "volume_mm3",
        "elongation", "flatness", "compactness", "relative_thickness",
        "scar_count", "mean_scar_area_mm2", "platform_angle_deg",
        "edge_angle_mean_deg", "edge_angle_std_deg", "curvature_index",
        "cross_section_profile", "symmetry_score", "com_z_ratio",
        "dorsal_ridge_count", "surface_roughness",
    ]

    def to_array(self) -> np.ndarray:
        """Return feature vector as a 1D numpy array in FEATURE_NAMES order."""
        return np.array([getattr(self, name) for name in self.FEATURE_NAMES], dtype=float)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> LithicFeatureVector:
        """Construct from a 20-element array in FEATURE_NAMES order."""
        return cls(**dict(zip(cls.FEATURE_NAMES, arr)))
