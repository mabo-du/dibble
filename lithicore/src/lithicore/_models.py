"""_models.py — Core data types for lithicore measurement pipeline.

exports: MeasurementConfig, MeasurementResult, ArtefactResult, Landmark, MeshQualityReport, MeshGrade
used_by: Every lithicore module imports these dataclasses
rules:   All dataclasses frozen, with typed fields.
         MeasurementResult.confidence is 0.0-1.0.
         MeshGrade is a StrEnum (Pass, Warn, Fail).
agent:   deepseek-v4-flash | 2026-05-26 | Initial data model
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import List


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
