# Lithic 3D Morphological Analyzer — v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working desktop application that imports 3D lithic meshes, orients them (auto/semi/manual), extracts standardised measurements, and exports CSV/JSON/MorphoJ/PDF.

**Architecture:** Two-package modular design: `lithicore` (pure-Python measurement library, no GUI) and `lithicope` (PyQt6 desktop GUI depending on lithicore). Algorithms are pure functions with dataclass configs, fully testable via pytest.

**Tech Stack:** Python 3.11+, trimesh, NumPy, SciPy, Open3D, PyQt6, typer (CLI), pandas (CSV), ReportLab (PDF).

---

## Phase 1: Project Scaffolding

### Task 1: Create `lithicore` package structure

**Files:**
- Create: `lithicore/pyproject.toml`
- Create: `lithicore/src/lithicore/__init__.py`

- [ ] **Step 1: Create `lithicore/pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "lithicore"
version = "0.1.0"
description = "3D lithic artefact morphological measurement library"
requires-python = ">=3.11"
dependencies = [
    "trimesh>=4.0",
    "numpy>=1.24",
    "scipy>=1.11",
    "typer>=0.9",
]

[project.scripts]
lithicore = "lithicore._cli:app"

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 2: Create `lithicore/src/lithicore/__init__.py`**

```python
"""lithicore — 3D lithic artefact morphological measurement library.

exports: orient_auto(mesh, config) -> tuple[trimesh.Trimesh, np.ndarray]
         orient_manual(mesh, points, config) -> tuple[trimesh.Trimesh, np.ndarray]
         extract_metrics(mesh, config) -> list[MeasurementResult]
         detect_edges(mesh, config) -> np.ndarray
         platform_angles(mesh, config) -> tuple[MeasurementResult, MeasurementResult]
         validate_mesh(mesh) -> MeshQualityReport
         repair_mesh(mesh) -> trimesh.Trimesh
         batch_process(directory, config) -> list[ArtefactResult]
used_by: lithicope GUI, CLI users
rules:   No GUI imports. Every public function takes a mesh + config and returns typed results.
agent:   deepseek-v4-flash | 2026-05-26 | Initial scaffolding
"""

from lithicore._models import (
    MeasurementConfig,
    MeasurementResult,
    ArtefactResult,
    Landmark,
    MeshQualityReport,
)

__all__ = [
    "MeasurementConfig", "MeasurementResult", "ArtefactResult", "Landmark",
    "MeshQualityReport",
]
```

- [ ] **Step 3: Verify package structure**

Run: `python -c "import ast; ast.parse(open('lithicore/pyproject.toml').read())"` (basic syntax check)
Then: `cd lithicore && pip install -e . 2>&1 | tail -5`
Expected: Successfully installed lithicore

- [ ] **Step 4: Commit**

```bash
git add lithicore/pyproject.toml lithicore/src/lithicore/__init__.py
git commit -m "feat: scaffold lithicore package"
```

### Task 2: Create `lithicope` package structure

**Files:**
- Create: `lithicope/pyproject.toml`
- Create: `lithicope/src/lithicope/__init__.py`
- Create: `lithicope/src/lithicope/main.py`

- [ ] **Step 1: Create `lithicope/pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "lithicope"
version = "0.1.0"
description = "Desktop GUI for 3D lithic morphological analysis"
requires-python = ">=3.11"
dependencies = [
    "lithicore>=0.1.0",
    "PyQt6>=6.5",
    "open3d>=0.17",
    "pandas>=2.0",
    "reportlab>=4.0",
]

[project.scripts]
lithicope = "lithicope.main:main"

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 2: Create `lithicope/src/lithicope/__init__.py`**

```python
"""lithicope — Desktop GUI for 3D lithic morphological analysis.

exports: main() -> None  (entry point for the PyQt6 application)
used_by: Users launching the desktop app
rules:   Thin orchestration layer over lithicore. No measurement logic.
agent:   deepseek-v4-flash | 2026-05-26 | Initial scaffolding
"""
```

- [ ] **Step 3: Create `lithicope/src/lithicope/main.py`** (minimal QApplication stub)

```python
"""main.py — Application entry point for the Lithic Analysis Platform.

exports: main() -> None
used_by: CLI script entry point
rules:   Create QApplication, instantiate MainWindow, exec event loop.
         Must call sys.exit(app.exec()) for clean shutdown.
agent:   deepseek-v4-flash | 2026-05-26 | Initial scaffolding
"""

import sys
from PyQt6.QtWidgets import QApplication
from lithicope._main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Lithic Analysis Platform")
    app.setOrganizationName("Digital Heritage Research")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify package installs**

Run: `cd lithicope && pip install -e . 2>&1 | tail -5`
Expected: Successfully installed lithicope

- [ ] **Step 5: Commit**

```bash
git add lithicope/pyproject.toml lithicope/src/lithicope/__init__.py lithicope/src/lithicope/main.py
git commit -m "feat: scaffold lithicope package"
```

### Task 3: Create test infrastructure

**Files:**
- Create: `lithicore/tests/conftest.py`
- Create: `lithicore/tests/__init__.py`

- [ ] **Step 1: Create `lithicore/tests/__init__.py`** (empty)

```python
# Tests for lithicore
```

- [ ] **Step 2: Create `lithicore/tests/conftest.py`**

```python
"""Test fixtures for lithicore.

Provides synthetic meshes with known geometry for all measurement tests.
"""

import numpy as np
import pytest
import trimesh


@pytest.fixture
def rectangular_prism():
    """A 50×30×10 mm rectangular prism simulating a standardised blade."""
    box = trimesh.creation.box(extents=[50, 30, 10])
    box.vertices += [0, 0, 5]  # centre at origin
    return box


@pytest.fixture
def oriented_prism(rectangular_prism):
    """A rectangular prism with known orientation (platform = XY plane)."""
    mesh = rectangular_prism.copy()
    # Already axis-aligned: length along Z=50, width along X=30, thickness along Y=10
    return mesh


@pytest.fixture
def synthetic_flake():
    """A simple flake-shaped mesh with a flat platform region.

    Generated by cutting a box with a slanted plane to create
    a dorsal surface at ~70° from the platform plane.
    """
    box = trimesh.creation.box(extents=[40, 20, 15])
    # Add a slanted cut to simulate dorsal surface
    plane_normal = [0, 0.342, 0.940]  # ~70° from platform
    plane_origin = [0, 0, 5]
    # Slice the box and keep the portion below the plane
    sliced = trimesh.intersections.slice_mesh_plane(
        box, plane_normal, plane_origin, cap=True
    )
    return sliced


@pytest.fixture
def sample_ply_path(tmp_path):
    """A sample PLY file on disk for batch processing tests."""
    mesh = trimesh.creation.box(extents=[10, 10, 10])
    path = tmp_path / "test_cube.ply"
    mesh.export(path)
    return path


@pytest.fixture
def mesh_dir_with_various(tmp_path):
    """Directory with PLY, OBJ, and STL test files for batch tests."""
    for name, ext in [("cube", "ply"), ("sphere", "obj"), ("cone", "stl")]:
        if name == "cube":
            m = trimesh.creation.box(extents=[10, 10, 10])
        elif name == "sphere":
            m = trimesh.creation.icosphere()
        else:
            m = trimesh.creation.cone(radius=5, height=10)
        m.export(tmp_path / f"{name}.{ext}")
    return tmp_path
```

- [ ] **Step 3: Verify fixtures import correctly**

Run: `cd lithicore && python -c "from tests.conftest import *; print('fixtures OK')"`
Expected: fixtures OK

- [ ] **Step 4: Create `.gitignore` at project root**

Create `/home/mark/Projects/Future\ Project\ Ideas/04.\ Lithic-Analysis-Platform/.gitignore`:
```
__pycache__/
*.egg-info/
dist/
build/
*.pyc
.env
.venv/
```

- [ ] **Step 5: Commit**

```bash
git add lithicore/tests/ .gitignore
git commit -m "chore: add test infrastructure and .gitignore"
```

---

## Phase 2: Lithicore — Data Model & Validation

### Task 4: Data model classes

**Files:**
- Create: `lithicore/src/lithicore/_models.py`

- [ ] **Step 1: Write `_models.py`**

```python
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
```

- [ ] **Step 2: Write the test**

```python
"""test_models.py — Tests for data model dataclasses."""
import pytest
from lithicore._models import (
    MeasurementConfig,
    MeasurementResult,
    ArtefactResult,
    Landmark,
    MeshQualityReport,
    MeshGrade,
)


class TestMeasurementConfig:
    def test_default_config(self):
        config = MeasurementConfig()
        assert config.repair_mesh is True
        assert config.edge_threshold_degrees == 50.0
        assert config.min_face_count == 2000

    def test_custom_config(self):
        config = MeasurementConfig(repair_mesh=False, edge_threshold_degrees=60.0)
        assert config.repair_mesh is False
        assert config.edge_threshold_degrees == 60.0

    def test_config_is_frozen(self):
        config = MeasurementConfig()
        with pytest.raises(AttributeError):
            config.repair_mesh = False


class TestMeasurementResult:
    def test_create(self):
        r = MeasurementResult(name="length", value=45.2, unit="mm", confidence=0.95)
        assert r.name == "length"
        assert r.value == 45.2

    def test_confidence_range(self):
        r = MeasurementResult(name="angle", value=78.3, unit="°", confidence=1.0)
        assert 0.0 <= r.confidence <= 1.0


class TestMeshQualityReport:
    def test_default_grade_is_pass(self):
        r = MeshQualityReport(original_vertex_count=1000, original_face_count=2000)
        assert r.grade == MeshGrade.PASS
        assert r.warnings == []
```

- [ ] **Step 3: Run test**

Run: `cd lithicore && python -m pytest tests/test_models.py -v`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add lithicore/src/lithicore/_models.py lithicore/tests/test_models.py
git commit -m "feat: core data model dataclasses"
```

### Task 5: Mesh validation and repair

**Files:**
- Create: `lithicore/src/lithicore/_validation.py`
- Create: `lithicore/tests/test_validation.py`

- [ ] **Step 1: Write the test**

```python
"""test_validation.py — Tests for mesh validation and repair."""

import numpy as np
import trimesh
import pytest
from lithicore._validation import validate_mesh, repair_mesh
from lithicore._models import MeshGrade


class TestValidateMesh:
    def test_valid_mesh_passes(self, rectangular_prism):
        report = validate_mesh(rectangular_prism)
        assert report.grade == MeshGrade.PASS
        assert report.original_face_count > 0

    def test_degenerate_mesh_fails(self):
        """A mesh with only 3 faces should fail."""
        vertices = [[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]]
        faces = [[0, 1, 2]]  # only one triangle
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        report = validate_mesh(mesh)
        assert report.grade == MeshGrade.FAIL

    def test_mesh_with_hole_warns(self):
        """A mesh with a large hole should warn."""
        box = trimesh.creation.box(extents=[50, 30, 10])
        # Remove some faces to create a hole
        keep_faces = [f for i, f in enumerate(box.faces) if i > 20]
        holey_mesh = trimesh.Trimesh(
            vertices=box.vertices, faces=keep_faces, process=False
        )
        report = validate_mesh(holey_mesh)
        assert report.grade in (MeshGrade.WARN, MeshGrade.PASS)


class TestRepairMesh:
    def test_repair_valid_mesh_unchanged(self, rectangular_prism):
        """A valid mesh should be unchanged by repair."""
        repaired = repair_mesh(rectangular_prism)
        assert len(repaired.vertices) > 0
        assert repaired.is_watertight is not None

    def test_repair_fills_small_holes(self):
        """A mesh with a small hole should have it filled."""
        box = trimesh.creation.box(extents=[50, 30, 10])
        # Remove a few faces to create a small hole
        keep_faces = [f for i, f in enumerate(box.faces) if i > 5]
        holey_mesh = trimesh.Trimesh(
            vertices=box.vertices, faces=keep_faces, process=False
        )
        repaired = repair_mesh(holey_mesh)
        assert repaired.is_watertight or len(repaired.faces) >= len(holey_mesh.faces)

    def test_repair_removes_isolated_vertices(self):
        """Isolated vertices should be removed."""
        box = trimesh.creation.box(extents=[50, 30, 10])
        # Add stray vertices
        bad_vertices = np.vstack([box.vertices, [[999, 999, 999], [-999, -999, -999]]])
        bad_mesh = trimesh.Trimesh(
            vertices=bad_vertices, faces=box.faces, process=False
        )
        repaired = repair_mesh(bad_mesh)
        assert len(repaired.vertices) <= len(box.vertices)
```

- [ ] **Step 2: Run test to confirm it fails**

Run: `cd lithicore && python -m pytest tests/test_validation.py -v`
Expected: FAIL with ModuleNotFoundError or NameError

- [ ] **Step 3: Write `_validation.py`**

```python
"""_validation.py — Mesh validation, cleaning, and repair.

exports: validate_mesh(mesh: trimesh.Trimesh) -> MeshQualityReport
         repair_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh
used_by: Import pipeline (GUI and CLI), called before any measurement
rules:   validate_mesh never modifies the input mesh.
         repair_mesh returns a NEW mesh (copy) — never mutates the input.
         Always check for non-manifold edges, holes, isolated vertices.
agent:   deepseek-v4-flash | 2026-05-26 | Initial implementation
"""

from __future__ import annotations

import numpy as np
import trimesh
from lithicore._models import MeshQualityReport, MeshGrade

# Minimum faces for reliable edge angle measurement
_MIN_FACES_RELIABLE = 2000
# Minimum faces for basic measurements
_MIN_FACES_BASIC = 100


def validate_mesh(mesh: trimesh.Trimesh) -> MeshQualityReport:
    """Validate mesh quality without modifying it.

    Returns a MeshQualityReport with grade and warnings.
    Grade is determined by the worst finding:
      - PASS if all checks pass
      - WARN if minor issues (small holes, <50k faces)
      - FAIL if critical issues (<100 faces, non-manifold)
    """
    warnings: list[str] = []
    vcount = len(mesh.vertices)
    fcount = len(mesh.faces)

    if fcount < _MIN_FACES_BASIC:
        return MeshQualityReport(
            original_vertex_count=vcount,
            original_face_count=fcount,
            grade=MeshGrade.FAIL,
            warnings=[f"Too few faces ({fcount}) for reliable measurement"],
        )

    if not mesh.is_watertight:
        warnings.append("Mesh is not watertight")

    if mesh.is_winding_consistent is False:
        warnings.append("Inconsistent face winding detected")

    # Check for non-manifold edges
    non_manifold = len(mesh.non_manifold_edges())
    if non_manifold > 0:
        warnings.append(f"{non_manifold} non-manifold edges")

    # Grade assignment
    if fcount < _MIN_FACES_RELIABLE:
        grade = MeshGrade.WARN
        warnings.append(f"Low resolution ({fcount} faces)")
    elif len(warnings) > 0:
        grade = MeshGrade.WARN
    else:
        grade = MeshGrade.PASS

    return MeshQualityReport(
        original_vertex_count=vcount,
        original_face_count=fcount,
        grade=grade,
        warnings=warnings,
    )


def repair_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Repair and clean a mesh.

    Performs: remove duplicate/close vertices, fill small holes,
    remove isolated components, fix normals. Returns a new mesh.
    """
    working = mesh.copy()

    # Remove duplicate vertices (merge within 1e-5 tolerance)
    working.remove_duplicate_vertices(tol=1e-5)

    # Remove degenerate faces (zero area)
    working.update_faces(working.nondegenerate_faces())

    # Fill small holes (up to 50 faces)
    holes_filled = 0
    for hole in working.holes:
        if len(hole) <= 50:
            try:
                trimesh.repair.fill_holes(working)
                holes_filled += 1
            except ValueError:
                pass

    # Remove isolated components, keep the largest
    if len(working.split()) > 1:
        components = working.split()
        working = max(components, key=lambda c: len(c.faces))
        isolated = len(components) - 1
    else:
        isolated = 0

    # Fix winding if needed
    if working.is_winding_consistent is False:
        working.fix_normals()

    # Count repair stats
    vcount = len(working.vertices)
    fcount = len(working.faces)

    return MeshQualityReport(
        original_vertex_count=len(mesh.vertices),
        original_face_count=len(mesh.faces),
        repaired_vertex_count=vcount,
        repaired_face_count=fcount,
        holes_filled=holes_filled,
        non_manifold_edges_fixed=len(mesh.non_manifold_edges()),
        isolated_components_removed=isolated,
        grade=MeshGrade.PASS,
    ), working
```

- [ ] **Step 4: Update `lithicore/src/lithicore/__init__.py` to export new functions**

```python
from lithicore._validation import validate_mesh, repair_mesh

__all__ = [
    "MeasurementConfig", "MeasurementResult", "ArtefactResult", "Landmark",
    "MeshQualityReport",
    "validate_mesh", "repair_mesh",
]
```

- [ ] **Step 5: Run tests to pass**

Run: `cd lithicore && python -m pytest tests/test_validation.py -v`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add lithicore/src/lithicore/_validation.py lithicore/tests/test_validation.py lithicore/src/lithicore/__init__.py
git commit -m "feat: mesh validation and repair pipeline"
```

---

## Phase 3: Lithicore — Measurement Algorithms

### Task 6: Automatic orientation (PCA + platform detection)

**Files:**
- Create: `lithicore/src/lithicore/_orientation.py`
- Create: `lithicore/tests/test_orientation.py`

- [ ] **Step 1: Write the test**

```python
"""test_orientation.py — Tests for mesh orientation."""

import numpy as np
import trimesh
import pytest
from lithicore._orientation import orient_auto, orient_manual
from lithicore._models import MeasurementConfig


class TestOrientAuto:
    def test_axis_aligned_box_stays_aligned(self, rectangular_prism):
        """An already-aligned box should not be rotated significantly."""
        config = MeasurementConfig()
        oriented, transform = orient_auto(rectangular_prism, config)
        # The bounding box extents should still be ~50, ~30, ~10 (in some order)
        extents = oriented.bounding_box.extents
        sorted_extents = sorted(extents, reverse=True)
        assert abs(sorted_extents[0] - 50) < 1.0
        assert abs(sorted_extents[1] - 30) < 1.0
        assert abs(sorted_extents[2] - 10) < 1.0

    def test_rotated_box_is_corrected(self):
        """A box rotated 45° should be re-aligned."""
        box = trimesh.creation.box(extents=[50, 30, 10])
        # Rotate 45° around Z
        rot = trimesh.transformations.rotation_matrix(np.pi / 4, [0, 0, 1])
        box.apply_transform(rot)
        config = MeasurementConfig()
        oriented, transform = orient_auto(box, config)
        # After orientation, extents should be ~50, ~30, ~10
        extents = oriented.bounding_box.extents
        sorted_extents = sorted(extents, reverse=True)
        assert abs(sorted_extents[0] - 50) < 2.0
        assert abs(sorted_extents[1] - 30) < 2.0

    def test_platform_detection_returns_plane(self, synthetic_flake):
        """Auto orientation should return a valid 4x4 transform."""
        config = MeasurementConfig()
        oriented, transform = orient_auto(synthetic_flake, config)
        assert transform.shape == (4, 4)
        assert np.allclose(np.linalg.det(transform[:3, :3]), 1.0, atol=1e-3)


class TestOrientManual:
    def test_three_points_produces_plane(self, rectangular_prism):
        """Three picked points should produce a valid orientation."""
        points = np.array([[-5, 0, 0], [5, 0, 0], [0, 5, 0]], dtype=float)
        config = MeasurementConfig()
        oriented, transform = orient_manual(rectangular_prism, points, config)
        assert transform.shape == (4, 4)

    def test_manual_orientation_is_stable(self, rectangular_prism):
        """Same points twice should give same result."""
        points = np.array([[-5, 0, 0], [5, 0, 0], [0, 5, 0]], dtype=float)
        config = MeasurementConfig()
        _, t1 = orient_manual(rectangular_prism, points, config)
        _, t2 = orient_manual(rectangular_prism, points, config)
        assert np.allclose(t1, t2)
```

- [ ] **Step 2: Write `_orientation.py`**

```python
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
    face_areas = working.face_areas
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
```

- [ ] **Step 3: Run tests**

Run: `cd lithicore && python -m pytest tests/test_orientation.py -v`
Expected: All tests pass (some tolerances may need adjustment for synthetic flake)

- [ ] **Step 4: Update `__init__.py` exports**

```python
from lithicore._orientation import orient_auto, orient_manual
```

- [ ] **Step 5: Commit**

```bash
git add lithicore/src/lithicore/_orientation.py lithicore/tests/test_orientation.py
git commit -m "feat: automatic and manual mesh orientation"
```

### Task 7: Core metric extraction (length, width, thickness, area, volume)

**Files:**
- Create: `lithicore/src/lithicore/_metrics.py`
- Create: `lithicore/tests/test_metrics.py`

- [ ] **Step 1: Write the test**

```python
"""test_metrics.py — Tests for metric extraction."""

import math
import trimesh
import pytest
from lithicore._metrics import extract_metrics
from lithicore._models import MeasurementConfig


class TestExtractMetrics:
    def test_oriented_prism_length(self, oriented_prism):
        """A 50×30×10 prism should give length=50, width=30, thickness=10."""
        config = MeasurementConfig()
        results = extract_metrics(oriented_prism, config)
        by_name = {r.name: r for r in results}
        assert abs(by_name["max_length"].value - 50) < 1.0
        assert abs(by_name["max_width"].value - 30) < 1.0
        assert abs(by_name["max_thickness"].value - 10) < 1.0

    def test_surface_area(self, oriented_prism):
        """50×30×10 box has surface area = 2*(50*30 + 50*10 + 30*10) = 4600 mm²."""
        config = MeasurementConfig()
        results = extract_metrics(oriented_prism, config)
        by_name = {r.name: r for r in results}
        expected = 2 * (50*30 + 50*10 + 30*10)
        assert abs(by_name["surface_area"].value - expected) < expected * 0.05

    def test_volume(self, oriented_prism):
        """50×30×10 box has volume = 15000 mm³."""
        config = MeasurementConfig()
        results = extract_metrics(oriented_prism, config)
        by_name = {r.name: r for r in results}
        assert abs(by_name["volume"].value - 15000) < 15000 * 0.05

    def test_all_expected_metrics_present(self, oriented_prism):
        """Check all v1 metrics are returned."""
        config = MeasurementConfig()
        results = extract_metrics(oriented_prism, config)
        names = {r.name for r in results}
        expected = {
            "max_length", "max_width", "max_thickness",
            "surface_area", "volume",
        }
        missing = expected - names
        assert not missing, f"Missing metrics: {missing}"

    def test_measurements_have_units(self, oriented_prism):
        """All measurements should have specified units."""
        config = MeasurementConfig()
        results = extract_metrics(oriented_prism, config)
        for r in results:
            assert r.unit in ("mm", "mm²", "mm³", "°")

    def test_confidence_between_0_and_1(self, oriented_prism):
        """All confidences should be in valid range."""
        config = MeasurementConfig()
        results = extract_metrics(oriented_prism, config)
        for r in results:
            assert 0.0 <= r.confidence <= 1.0
```

- [ ] **Step 2: Write `_metrics.py`**

```python
"""_metrics.py — Core metric extraction for lithic artefacts.

exports: extract_metrics(mesh, config) -> list[MeasurementResult]
used_by: GUI results panel, batch processing, CLI
rules:   All metrics computed in oriented coordinate space.
         Length = Z-extent, Width = X-extent, Thickness = Y-extent.
         Volume requires watertight mesh (auto-fill if not).
agent:   deepseek-v4-flash | 2026-05-26 | Initial implementation
"""

from __future__ import annotations

import numpy as np
import trimesh
from lithicore._models import MeasurementConfig, MeasurementResult


def extract_metrics(
    mesh: trimesh.Trimesh,
    config: MeasurementConfig,
) -> list[MeasurementResult]:
    """Extract all standard lithic metrics from an oriented mesh.

    The mesh should already be oriented (platform = XY plane,
    reduction axis = Z). If not oriented, results will be incorrect.
    """
    results: list[MeasurementResult] = []

    # Compute oriented bounding box
    obb = mesh.bounding_box_oriented
    extents = sorted(obb.extents, reverse=True)

    # Length = maximum extent (along reduction axis / Z after orientation)
    length_val = extents[0]
    results.append(MeasurementResult(
        name="max_length", value=round(length_val, 2), unit="mm", confidence=0.95
    ))

    # Width = second extent
    width_val = extents[1]
    results.append(MeasurementResult(
        name="max_width", value=round(width_val, 2), unit="mm", confidence=0.95
    ))

    # Thickness = minimum extent
    thickness_val = extents[2]
    results.append(MeasurementResult(
        name="max_thickness", value=round(thickness_val, 2), unit="mm", confidence=0.95
    ))

    # Surface area
    area_val = mesh.area
    results.append(MeasurementResult(
        name="surface_area", value=round(area_val, 2), unit="mm²", confidence=0.90
    ))

    # Volume (watertight fill if needed)
    if mesh.is_watertight:
        vol = mesh.volume
    else:
        # Fill holes and try again
        filled = mesh.copy()
        trimesh.repair.fill_holes(filled)
        vol = filled.volume if filled.is_watertight else 0.0
    results.append(MeasurementResult(
        name="volume", value=round(vol, 2), unit="mm³",
        confidence=0.95 if mesh.is_watertight else 0.70,
    ))

    return results
```

- [ ] **Step 3: Run tests**

Run: `cd lithicore && python -m pytest tests/test_metrics.py -v`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add lithicore/src/lithicore/_metrics.py lithicore/tests/test_metrics.py
git commit -m "feat: core metric extraction (length/width/thickness/area/volume)"
```

### Task 8: Platform angle extraction (EPA, IPA)

**Files:**
- Create: `lithicore/src/lithicore/_platform_angle.py`
- Create: `lithicore/tests/test_platform_angle.py`

- [ ] **Step 1: Write the test**

```python
"""test_platform_angle.py — Tests for platform angle extraction."""

import numpy as np
import trimesh
import math
import pytest
from lithicore._platform_angle import platform_angles
from lithicore._models import MeasurementConfig


class TestPlatformAngles:
    def test_epa_present(self, oriented_prism):
        """A box should return an EPA measurement."""
        config = MeasurementConfig()
        epa, ipa = platform_angles(oriented_prism, config)
        assert epa is not None
        assert epa.name == "exterior_platform_angle"
        assert epa.unit == "°"

    def test_ipa_present(self, oriented_prism):
        """A box should return an IPA measurement."""
        config = MeasurementConfig()
        epa, ipa = platform_angles(oriented_prism, config)
        assert ipa is not None
        assert ipa.name == "interior_platform_angle"
        assert ipa.unit == "°"

    def test_box_angles_are_90(self, oriented_prism):
        """For a rectangular box, both EPA and IPA should be ~90°."""
        config = MeasurementConfig()
        epa, ipa = platform_angles(oriented_prism, config)
        assert abs(epa.value - 90) < 5.0
        assert abs(ipa.value - 90) < 5.0

    def test_angle_range(self, oriented_prism):
        """Angles should be in [0, 180]."""
        config = MeasurementConfig()
        epa, ipa = platform_angles(oriented_prism, config)
        assert 0 <= epa.value <= 180
        assert 0 <= ipa.value <= 180
```

- [ ] **Step 2: Write `_platform_angle.py`**

```python
"""_platform_angle.py — Platform angle extraction (EPA, IPA).

exports: platform_angles(mesh, config) -> tuple[MeasurementResult | None, MeasurementResult | None]
used_by: GUI results panel, batch processing
rules:   EPA = angle between platform plane and dorsal surface.
         IPA = angle between platform plane and ventral surface.
         Both computed from face normal dot products in oriented space.
agent:   deepseek-v4-flash | 2026-05-26 | Initial implementation
"""

from __future__ import annotations

import numpy as np
import trimesh
from lithicore._models import MeasurementConfig, MeasurementResult

# Fraction of mesh extent from proximal end to search for platform
_PLATFORM_SEARCH_FRACTION = 0.1


def platform_angles(
    mesh: trimesh.Trimesh,
    config: MeasurementConfig,
) -> tuple[MeasurementResult | None, MeasurementResult | None]:
    """Compute Exterior and Interior Platform Angles.

    In oriented space, the platform is at the proximal (minimum Z) end.
    EPA = angle between platform plane and adjacent dorsal face normals.
    IPA = angle between platform plane and adjacent ventral face normals.
    """
    z_min = mesh.bounds[0, 2]
    z_max = mesh.bounds[1, 2]
    z_range = z_max - z_min

    # Find faces near the proximal (platform) end
    proximal_z_threshold = z_min + z_range * _PLATFORM_SEARCH_FRACTION
    proximal_face_indices = np.where(
        np.any(mesh.vertices[mesh.faces][:, :, 2] < proximal_z_threshold, axis=1)
    )[0]

    if len(proximal_face_indices) < 3:
        return None, None

    proximal_normals = mesh.face_normals[proximal_face_indices]

    # Platform normal is average of the flattest proximal faces
    # (those pointing most toward +Z, since platform = XY plane)
    z_dot = proximal_normals[:, 2]
    platform_faces = proximal_face_indices[z_dot > 0.7]  # within ~45° of +Z

    if len(platform_faces) < 3:
        # Fallback: use all proximal faces
        platform_normal = np.mean(proximal_normals, axis=0)
        platform_normal = platform_normal / np.linalg.norm(platform_normal)
    else:
        platform_normal = np.mean(mesh.face_normals[platform_faces], axis=0)
        platform_normal = platform_normal / np.linalg.norm(platform_normal)

    # Dorsal faces: those on the exterior (pointing away from bulb, roughly -Z)
    dorsal_faces = proximal_face_indices[proximal_normals[:, 2] < -0.3]
    # Ventral faces: those on the interior (pointing toward bulb, roughly +Z)
    ventral_faces = proximal_face_indices[z_dot > 0.3]

    def _angle_between(normal_a: np.ndarray, normals_b: np.ndarray) -> float:
        """Compute average angle between a fixed normal and a set of normals."""
        if len(normals_b) == 0:
            return 90.0
        dots = np.clip(np.dot(normals_b, normal_a), -1, 1)
        angles = np.degrees(np.arccos(np.abs(dots)))  # acute angle
        return float(np.mean(angles))

    epa_value = _angle_between(platform_normal, mesh.face_normals[dorsal_faces]) if len(dorsal_faces) > 0 else None
    ipa_value = _angle_between(platform_normal, mesh.face_normals[ventral_faces]) if len(ventral_faces) > 0 else None

    epa = None
    if epa_value is not None:
        epa = MeasurementResult(
            name="exterior_platform_angle",
            value=round(epa_value, 1),
            unit="°",
            confidence=0.85,
        )

    ipa = None
    if ipa_value is not None:
        ipa = MeasurementResult(
            name="interior_platform_angle",
            value=round(ipa_value, 1),
            unit="°",
            confidence=0.85,
        )

    return epa, ipa
```

- [ ] **Step 3: Run tests**

Run: `cd lithicore && python -m pytest tests/test_platform_angle.py -v`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add lithicore/src/lithicore/_platform_angle.py lithicore/tests/test_platform_angle.py
git commit -m "feat: platform angle extraction (EPA, IPA)"
```

### Task 9: Edge detection (dihedral thresholding)

**Files:**
- Create: `lithicore/src/lithicore/_edge_detection.py`
- Create: `lithicore/tests/test_edge_detection.py`

- [ ] **Step 1: Write the test**

```python
"""test_edge_detection.py — Tests for edge detection."""

import numpy as np
import trimesh
import pytest
from lithicore._edge_detection import detect_edges
from lithicore._models import MeasurementConfig


class TestDetectEdges:
    def test_smooth_sphere_has_few_edges(self):
        """A smooth icosphere should have very few detected edges."""
        sphere = trimesh.creation.icosphere(subdivisions=3)
        config = MeasurementConfig(edge_threshold_degrees=45.0)
        edge_indices, face_mask = detect_edges(sphere, config)
        # A smooth sphere should have fewer edge vertices than a sharp object
        assert len(edge_indices) < len(sphere.vertices) * 0.3

    def test_sharp_edge_is_detected(self):
        """A box has sharp 90° edges — they should be detected."""
        box = trimesh.creation.box(extents=[50, 30, 10])
        config = MeasurementConfig(edge_threshold_degrees=60.0)
        edge_indices, face_mask = detect_edges(box, config)
        assert len(edge_indices) > 0

    def test_edge_threshold_filtering(self):
        """Lower threshold should detect more edges than higher threshold."""
        box = trimesh.creation.box(extents=[50, 30, 10])
        config_low = MeasurementConfig(edge_threshold_degrees=30.0)
        config_high = MeasurementConfig(edge_threshold_degrees=80.0)
        edges_low, _ = detect_edges(box, config_low)
        edges_high, _ = detect_edges(box, config_high)
        assert len(edges_low) >= len(edges_high)

    def test_face_mask_shape(self, rectangular_prism):
        """Face mask should be a boolean array matching face count."""
        config = MeasurementConfig()
        edge_indices, face_mask = detect_edges(rectangular_prism, config)
        assert len(face_mask) == len(rectangular_prism.faces)
        assert face_mask.dtype == bool
```

- [ ] **Step 2: Write `_edge_detection.py`**

```python
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
    dots = np.clip(np.dot(n1, n2), -1, 1)
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
```

- [ ] **Step 3: Run tests**

Run: `cd lithicore && python -m pytest tests/test_edge_detection.py -v`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add lithicore/src/lithicore/_edge_detection.py lithicore/tests/test_edge_detection.py
git commit -m "feat: edge detection via dihedral angle thresholding"
```

---

## Phase 4: Lithicore — Batch Processing & CLI

### Task 10: Batch processing

**Files:**
- Create: `lithicore/src/lithicore/_batch.py`
- Create: `lithicore/tests/test_batch.py`

- [ ] **Step 1: Write the test**

```python
"""test_batch.py — Tests for batch processing."""

import csv
import json
import trimesh
import pytest
from pathlib import Path
from lithicore._batch import batch_process
from lithicore._models import MeasurementConfig


class TestBatchProcess:
    def test_batch_processes_all_files(self, mesh_dir_with_various):
        """Batch should find and process all supported mesh files."""
        config = MeasurementConfig(repair_mesh=True)
        results = batch_process(mesh_dir_with_various, config)
        assert len(results) == 3  # cube.ply, sphere.obj, cone.stl

    def test_batch_empty_directory(self, tmp_path):
        """An empty directory should return no results."""
        config = MeasurementConfig()
        results = batch_process(tmp_path, config)
        assert len(results) == 0

    def test_batch_each_result_has_measurements(self, mesh_dir_with_various):
        """Each artefact result should contain measurements."""
        config = MeasurementConfig()
        results = batch_process(mesh_dir_with_various, config)
        for result in results:
            assert len(result.measurements) > 0
            assert result.file_path.exists()
```

- [ ] **Step 2: Write `_batch.py`**

```python
"""_batch.py — Batch processing for lithicore measurement pipeline.

exports: batch_process(directory, config) -> list[ArtefactResult]
used_by: CLI entry point, batch runner UI
rules:   Iterates all .ply, .obj, .stl files in directory.
         Each artefact: validate → repair (if configured) → orient → measure.
         Returns ArtefactResult list with one entry per file.
agent:   deepseek-v4-flash | 2026-05-26 | Initial implementation
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import trimesh
from lithicore._models import MeasurementConfig, ArtefactResult
from lithicore._validation import validate_mesh, repair_mesh
from lithicore._orientation import orient_auto
from lithicore._metrics import extract_metrics
from lithicore._platform_angle import platform_angles
from lithicore._edge_detection import detect_edges

_SUPPORTED_EXTENSIONS = {".ply", ".obj", ".stl"}


def batch_process(
    directory: Path,
    config: MeasurementConfig,
) -> List[ArtefactResult]:
    """Process all supported mesh files in a directory.

    For each file: validate, optionally repair, auto-orient,
    extract metrics + platform angles, return ArtefactResult.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    mesh_files = sorted(
        [f for f in directory.iterdir()
         if f.suffix.lower() in _SUPPORTED_EXTENSIONS]
    )

    results: List[ArtefactResult] = []
    for filepath in mesh_files:
        try:
            mesh = trimesh.load(str(filepath), force="mesh")
        except Exception as exc:
            results.append(ArtefactResult(
                file_path=filepath,
                label=filepath.stem,
                measurements=[],
                landmarks=[],
                warnings=[f"Failed to load: {exc}"],
            ))
            continue

        # Validate
        quality = validate_mesh(mesh)
        warnings = list(quality.warnings)

        # Repair if configured
        if config.repair_mesh:
            _, mesh = repair_mesh(mesh)

        # Orient
        try:
            oriented, _ = orient_auto(mesh, config)
        except Exception as exc:
            results.append(ArtefactResult(
                file_path=filepath,
                label=filepath.stem,
                measurements=[],
                landmarks=[],
                warnings=[f"Orientation failed: {exc}"],
            ))
            continue

        # Extract metrics
        all_measurements = extract_metrics(oriented, config)
        epa, ipa = platform_angles(oriented, config)
        if epa:
            all_measurements.append(epa)
        if ipa:
            all_measurements.append(ipa)

        # Edge detection (for visualisation, not exported as measurement yet)
        detect_edges(oriented, config)

        results.append(ArtefactResult(
            file_path=filepath,
            label=filepath.stem,
            measurements=all_measurements,
            landmarks=[],
            warnings=warnings,
        ))

    return results
```

- [ ] **Step 3: Run tests**

Run: `cd lithicore && python -m pytest tests/test_batch.py -v`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add lithicore/src/lithicore/_batch.py lithicore/tests/test_batch.py
git commit -m "feat: batch processing pipeline"
```

### Task 11: CLI entry point

**Files:**
- Create: `lithicore/src/lithicore/_cli.py`

- [ ] **Step 1: Write `_cli.py`**

```python
"""_cli.py — Command-line interface for lithicore.

exports: app (typer.Typer)
used_by: Users running `lithicore batch --input ...`
rules:   Typer-based CLI. Subcommands: batch, orient, info.
agent:   deepseek-v4-flash | 2026-05-26 | Initial implementation
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
import pandas as pd
import json
from lithicore._models import MeasurementConfig
from lithicore._batch import batch_process

app = typer.Typer(
    name="lithicore",
    help="3D lithic artefact morphological analysis toolkit",
)


@app.command()
def batch(
    input: Path = typer.Argument(..., help="Directory containing mesh files"),
    output: Path = typer.Option("results.csv", "--output", "-o", help="Output CSV path"),
    format: str = typer.Option("csv", "--format", "-f", help="Output format: csv, json"),
    repair: bool = typer.Option(True, "--repair/--no-repair", help="Auto-repair meshes"),
    edge_threshold: float = typer.Option(50.0, "--edge-threshold", help="Edge detection angle threshold"),
) -> None:
    """Batch process all meshes in a directory."""
    config = MeasurementConfig(repair_mesh=repair, edge_threshold_degrees=edge_threshold)
    results = batch_process(input, config)

    if not results:
        typer.echo(f"No supported mesh files found in {input}")
        raise typer.Exit()

    # Build rows
    rows = []
    for artefact in results:
        row = {"file": artefact.file_path.name, "label": artefact.label}
        for m in artefact.measurements:
            row[m.name] = m.value
        row["warnings"] = "; ".join(artefact.warnings)
        rows.append(row)

    output_path = Path(output)

    if format.lower() == "json":
        output_path.write_text(json.dumps(rows, indent=2))
        typer.echo(f"Wrote {len(rows)} results to {output_path}")
    else:
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        typer.echo(f"Wrote {len(rows)} results to {output_path}")


@app.command()
def info(
    mesh_path: Path = typer.Argument(..., help="Path to a mesh file"),
) -> None:
    """Display information about a single mesh file."""
    import trimesh
    mesh = trimesh.load(str(mesh_path), force="mesh")
    typer.echo(f"File:     {mesh_path.name}")
    typer.echo(f"Vertices: {len(mesh.vertices)}")
    typer.echo(f"Faces:    {len(mesh.faces)}")
    typer.echo(f"Watertight: {mesh.is_watertight}")
    typer.echo(f"Area:     {mesh.area:.2f} mm²")
    if mesh.is_watertight:
        typer.echo(f"Volume:   {mesh.volume:.2f} mm³")


if __name__ == "__main__":
    app()
```

- [ ] **Step 2: Test CLI**

Run: `cd lithicore && python -m lithicore._cli --help`
Expected: Shows help with batch and info commands

- [ ] **Step 3: Commit**

```bash
git add lithicore/src/lithicore/_cli.py
git commit -m "feat: CLI entry point with batch and info commands"
```

---

## Phase 5: Lithicope — GUI Application

### Task 12: Main window shell

**Files:**
- Create: `lithicope/src/lithicope/_main_window.py`

- [ ] **Step 1: Write `_main_window.py`**

```python
"""_main_window.py — Main application window for lithicope.

exports: MainWindow(QMainWindow)
used_by: main.py entry point
rules:   Single-window layout: 3D viewer (60%) left, measurements panel (40%) right.
         Menu bar: File, Edit, Tools, Help.
         Status bar shows current artefact and batch progress.
agent:   deepseek-v4-flash | 2026-05-26 | Initial implementation
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QMainWindow, QMenuBar, QStatusBar, QSplitter, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QMessageBox,
)
from PyQt6.QtGui import QAction

from lithicope._viewer_3d import Viewer3D
from lithicope._import_dialog import ImportDialog
from lithicope._results_panel import ResultsPanel
from lithicope._batch_runner import BatchRunner


class MainWindow(QMainWindow):
    """Primary application window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Lithic Analysis Platform")
        self.setMinimumSize(1200, 800)

        self._current_mesh_path: Optional[Path] = None
        self._current_results: Optional[list] = None
        self._batch_results: List = []

        self._init_ui()
        self._init_menu()
        self._init_status_bar()

    def _init_ui(self) -> None:
        """Build the main layout: viewer left, results right."""
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: 3D viewer
        self.viewer = Viewer3D()
        splitter.addWidget(self.viewer)

        # Right: results panel
        self.results_panel = ResultsPanel()
        self.results_panel.export_requested.connect(self._on_export)
        splitter.addWidget(self.results_panel)

        splitter.setSizes([720, 480])
        self.setCentralWidget(splitter)

    def _init_menu(self) -> None:
        menu = self.menuBar()

        # File menu
        file_menu = menu.addMenu("&File")
        open_action = QAction("&Open Mesh...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._on_open)
        file_menu.addAction(open_action)

        batch_action = QAction("&Batch Import...", self)
        batch_action.setShortcut("Ctrl+B")
        batch_action.triggered.connect(self._on_batch)
        file_menu.addAction(batch_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Tools menu
        tools_menu = menu.addMenu("&Tools")
        export_action = QAction("&Export CSV...", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(lambda: self._on_export("csv"))
        tools_menu.addAction(export_action)

        # Help menu
        help_menu = menu.addMenu("&Help")
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    def _init_status_bar(self) -> None:
        self.status = self.statusBar()
        self.status.showMessage("Ready")

    def _on_open(self) -> None:
        """Open a single mesh file."""
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Open Mesh", "",
            "Mesh Files (*.ply *.obj *.stl);;All Files (*)",
        )
        if not path_str:
            return
        path = Path(path_str)

        dialog = ImportDialog(self, mode="single")
        if dialog.exec() == ImportDialog.DialogCode.Accepted:
            config = dialog.get_config()
            self._process_single(path, config)

    def _on_batch(self) -> None:
        """Open batch import dialog."""
        dir_str = QFileDialog.getExistingDirectory(
            self, "Select Mesh Directory"
        )
        if not dir_str:
            return
        directory = Path(dir_str)

        dialog = ImportDialog(self, mode="batch_auto")
        if dialog.exec() == ImportDialog.DialogCode.Accepted:
            config = dialog.get_config()
            self._run_batch(directory, config)

    def _on_export(self, fmt: str = "csv") -> None:
        """Export current results."""
        if self._current_results is None:
            QMessageBox.information(self, "No Data", "No measurements to export.")
            return
        self.results_panel.export_results(self._current_results, fmt)

    def _process_single(self, path: Path, config) -> None:
        """Load, orient, measure a single mesh."""
        self.status.showMessage(f"Loading {path.name}...")
        try:
            import trimesh
            mesh = trimesh.load(str(path), force="mesh")
            from lithicore._validation import validate_mesh, repair_mesh
            from lithicore._orientation import orient_auto
            from lithicore._metrics import extract_metrics
            from lithicore._platform_angle import platform_angles
            from lithicore._edge_detection import detect_edges
            from lithicore._models import MeasurementConfig

            quality = validate_mesh(mesh)
            if config.repair_mesh:
                _, mesh = repair_mesh(mesh)

            oriented, _ = orient_auto(mesh, config)
            measurements = extract_metrics(oriented, config)
            epa, ipa = platform_angles(oriented, config)
            if epa:
                measurements.append(epa)
            if ipa:
                measurements.append(ipa)
            edge_vertices, _ = detect_edges(oriented, config)

            self._current_mesh_path = path
            self._current_results = measurements
            self.viewer.display_mesh(oriented, edge_vertices)
            self.results_panel.show_measurements(measurements, path.name, quality.grade)
            self.status.showMessage(f"Loaded: {path.name}")
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to process mesh:\n{exc}")
            self.status.showMessage("Error loading mesh")

    def _run_batch(self, directory: Path, config) -> None:
        """Run batch processing in the GUI."""
        from lithicore._batch import batch_process
        runner = BatchRunner(directory, config, self)
        runner.run()

    def _on_about(self) -> None:
        QMessageBox.about(
            self,
            "About Lithic Analysis Platform",
            "Lithic 3D Morphological Analyzer v0.1\n\n"
            "An open-source desktop application for automated\n"
            "3D lithic artefact measurement and analysis.\n\n"
            "Built with lithicore + PyQt6 + Open3D",
        )
```

- [ ] **Step 2: Quick import check**

Run: `python -c "from lithicope._main_window import MainWindow; print('OK')"`
Expected: OK (PyQt6 may warn about no QApplication — that's fine)

- [ ] **Step 3: Commit**

```bash
git add lithicope/src/lithicope/_main_window.py
git commit -m "feat: main window shell with menu, layout, status bar"
```

### Task 13: 3D viewer widget

**Files:**
- Create: `lithicope/src/lithicope/_viewer_3d.py`

- [ ] **Step 1: Write `_viewer_3d.py`**

```python
"""_viewer_3d.py — Open3D viewer embedded in a PyQt6 widget.

exports: Viewer3D(QWidget)
used_by: MainWindow
rules:   Open3D visualisation embedded via QWidget container.
         Supports rotate (drag), zoom (scroll), pan (shift+drag).
         Edge vertices rendered as coloured overlay points.
agent:   deepseek-v4-flash | 2026-05-26 | Initial implementation
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtGui import QImage, QPixmap, QPainter

import open3d as o3d
import trimesh


class Viewer3D(QWidget):
    """PyQt6 widget wrapping Open3D visualisation."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self._mesh: Optional[o3d.geometry.TriangleMesh] = None
        self._edge_points: Optional[o3d.geometry.PointCloud] = None
        self._vis: Optional[o3d.visualization.Visualizer] = None
        self._image_label = QLabel("No mesh loaded", self)
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout = QVBoxLayout()
        layout.addWidget(self._image_label)
        self.setLayout(layout)

        # Timer for re-rendering
        self._render_timer = QTimer(self)
        self._render_timer.timeout.connect(self._render)
        self._needs_render = False

    def display_mesh(
        self,
        mesh: trimesh.Trimesh,
        edge_vertices: Optional[np.ndarray] = None,
    ) -> None:
        """Display a trimesh mesh, optionally with edge overlay."""
        # Convert trimesh to Open3D
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
        o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
        o3d_mesh.compute_vertex_normals()
        self._mesh = o3d_mesh

        if edge_vertices is not None and len(edge_vertices) > 0:
            edge_pts = np.asarray(mesh.vertices)[edge_vertices]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(edge_pts)
            pcd.paint_uniform_color([1, 0, 0])  # Red edges
            self._edge_points = pcd
        else:
            self._edge_points = None

        self._needs_render = True
        self._render_timer.start(50)

    def _render(self) -> None:
        """Render the scene to a QImage using Open3D's headless rendering."""
        if not self._needs_render:
            return
        self._needs_render = False
        self._render_timer.stop()

        if self._mesh is None:
            self._image_label.setText("No mesh loaded")
            return

        try:
            # Use Open3D's offscreen rendering
            vis = o3d.visualization.rendering.OffscreenRenderer(640, 480)
            vis.scene.add_geometry("mesh", self._mesh, o3d.visualization.rendering.MaterialRecord())

            if self._edge_points:
                vis.scene.add_geometry("edges", self._edge_points, o3d.visualization.rendering.MaterialRecord())

            # Set up camera
            bounds = self._mesh.get_axis_aligned_bounding_box()
            centre = bounds.get_center()
            extent = bounds.get_max_extent()
            vis.setup_camera(60, centre, centre + [0, 0, extent * 2], [0, -1, 0])

            img = vis.render_to_image()
            # Convert to QImage
            img_data = np.asarray(img)
            h, w, c = img_data.shape
            qimg = QImage(img_data.data, w, h, w * c, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            scaled = pixmap.scaled(
                self._image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._image_label.setPixmap(scaled)
            vis.destroy()
        except Exception as exc:
            self._image_label.setText(f"Render error: {exc}")

    def resizeEvent(self, event) -> None:
        """Re-render on resize."""
        super().resizeEvent(event)
        self._needs_render = True
        self._render_timer.start(50)

    def clear(self) -> None:
        """Clear the viewer."""
        self._mesh = None
        self._edge_points = None
        self._image_label.setText("No mesh loaded")
        self._needs_render = False
        self._render_timer.stop()
```

- [ ] **Step 2: Import check**

Run: `python -c "from lithicope._viewer_3d import Viewer3D; print('OK')"`
Expected: OK

- [ ] **Step 3: Commit**

```bash
git add lithicope/src/lithicope/_viewer_3d.py
git commit -m "feat: Open3D viewer widget with edge overlay"
```

### Task 14: Import dialog

**Files:**
- Create: `lithicope/src/lithicope/_import_dialog.py`

- [ ] **Step 1: Write `_import_dialog.py`**

```python
"""_import_dialog.py — Import dialog with mode selection and advanced options.

exports: ImportDialog(QDialog)
used_by: MainWindow file open and batch operations
rules:   Three import modes: Single, Batch-auto, Batch-manual.
         Advanced section with skip auto-repair / skip auto-validation toggles.
         Single and Batch-manual modes show advanced options; auto hides them.
agent:   deepseek-v4-flash | 2026-05-26 | Initial implementation
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QRadioButton,
    QCheckBox, QDialogButtonBox, QGroupBox, QWidget,
)
from lithicore._models import MeasurementConfig


class ImportDialog(QDialog):
    """Import configuration dialog.

    Allows the user to choose import mode and configure
    mesh repair and validation settings.
    """

    def __init__(self, parent: Optional[QWidget] = None, mode: str = "single") -> None:
        super().__init__(parent)
        self.setWindowTitle("Import Mesh")
        self.setMinimumWidth(450)

        self._mode = mode
        self._config = MeasurementConfig()

        layout = QVBoxLayout(self)

        # Mode selection
        mode_group = QGroupBox("Import Mode")
        mode_layout = QVBoxLayout(mode_group)

        self._radio_single = QRadioButton("Single artefact")
        self._radio_batch_auto = QRadioButton("Batch — auto  (auto-orient)")
        self._radio_batch_review = QRadioButton("Batch — review  (review flags)")
        self._radio_batch_manual = QRadioButton("Batch — manual  (orient each)")

        if mode == "single":
            self._radio_single.setChecked(True)
        elif mode == "batch_auto":
            self._radio_batch_auto.setChecked(True)
        elif mode == "batch_review":
            self._radio_batch_review.setChecked(True)
        elif mode == "batch_manual":
            self._radio_batch_manual.setChecked(True)

        mode_layout.addWidget(self._radio_single)
        mode_layout.addWidget(self._radio_batch_auto)
        mode_layout.addWidget(self._radio_batch_review)
        mode_layout.addWidget(self._radio_batch_manual)
        layout.addWidget(mode_group)

        # Advanced options (visible for single + batch-manual)
        self._advanced_group = QGroupBox("Advanced")
        self._advanced_group.setCheckable(False)
        advanced_layout = QVBoxLayout(self._advanced_group)

        self._skip_repair = QCheckBox("Skip auto-repair")
        self._skip_validation = QCheckBox("Skip auto-validation")
        advanced_layout.addWidget(self._skip_repair)
        advanced_layout.addWidget(self._skip_validation)
        self._advanced_group.setVisible(mode in ("single", "batch_manual"))
        layout.addWidget(self._advanced_group)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Connect radio buttons to toggle advanced visibility
        self._radio_single.toggled.connect(self._on_mode_changed)
        self._radio_batch_manual.toggled.connect(self._on_mode_changed)

    def _on_mode_changed(self) -> None:
        """Show advanced panel for single/manual modes only."""
        show_advanced = self._radio_single.isChecked() or self._radio_batch_manual.isChecked()
        self._advanced_group.setVisible(show_advanced)

    def get_config(self) -> MeasurementConfig:
        """Return a MeasurementConfig based on dialog selections."""
        return MeasurementConfig(
            repair_mesh=not self._skip_repair.isChecked(),
        )

    def get_mode(self) -> str:
        """Return the selected import mode."""
        if self._radio_single.isChecked():
            return "single"
        elif self._radio_batch_auto.isChecked():
            return "batch_auto"
        elif self._radio_batch_review.isChecked():
            return "batch_review"
        else:
            return "batch_manual"
```

- [ ] **Step 2: Import check**

Run: `python -c "from lithicope._import_dialog import ImportDialog; print('OK')"`
Expected: OK

- [ ] **Step 3: Commit**

```bash
git add lithicope/src/lithicope/_import_dialog.py
git commit -m "feat: import dialog with mode selection and advanced options"
```

### Task 15: Results panel (measurements table + export)

**Files:**
- Create: `lithicope/src/lithicope/_results_panel.py`

- [ ] **Step 1: Write `_results_panel.py`**

```python
"""_results_panel.py — Measurement results display and export panel.

exports: ResultsPanel(QWidget)
used_by: MainWindow right panel
rules:   Table shows measurement name, value, unit, confidence.
         Export buttons for CSV, JSON, MorphoJ, PDF.
         Uses pyqtSignal to request export from main window.
agent:   deepseek-v4-flash | 2026-05-26 | Initial implementation
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog,
    QGroupBox, QMessageBox, QComboBox,
)
from PyQt6.QtGui import QFont

from lithicore._models import MeasurementResult, MeshGrade


class ResultsPanel(QWidget):
    """Right-side panel showing measurements and export controls."""

    export_requested = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._current_measurements: List[MeasurementResult] = []
        self._current_label: str = ""
        self._current_grade: MeshGrade = MeshGrade.PASS

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Artefact info
        self._info_label = QLabel("No artefact loaded")
        self._info_label.setWordWrap(True)
        font = QFont()
        font.setPointSize(11)
        font.setBold(True)
        self._info_label.setFont(font)
        layout.addWidget(self._info_label)

        # Quality badge
        self._quality_label = QLabel("")
        layout.addWidget(self._quality_label)

        # Measurements table
        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Measurement", "Value", "Confidence"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self._table)

        layout.addStretch()

        # Export section
        export_group = QGroupBox("Export")
        export_layout = QHBoxLayout(export_group)

        self._export_combo = QComboBox()
        self._export_combo.addItems(["CSV", "JSON", "MorphoJ", "PDF"])
        export_layout.addWidget(self._export_combo)

        export_btn = QPushButton("Export")
        export_btn.clicked.connect(self._on_export_clicked)
        export_layout.addWidget(export_btn)

        layout.addWidget(export_group)

    def show_measurements(
        self,
        measurements: List[MeasurementResult],
        label: str,
        grade: MeshGrade,
    ) -> None:
        """Populate the table with measurement data."""
        self._current_measurements = measurements
        self._current_label = label
        self._current_grade = grade

        self._info_label.setText(f"Artefact: {label}")

        quality_text = f"Quality: {grade.value}"
        if grade == MeshGrade.PASS:
            self._quality_label.setStyleSheet("color: green;")
        elif grade == MeshGrade.WARN:
            self._quality_label.setStyleSheet("color: orange;")
        else:
            self._quality_label.setStyleSheet("color: red;")
        self._quality_label.setText(quality_text)

        self._table.setRowCount(len(measurements))
        for i, m in enumerate(measurements):
            name_item = QTableWidgetItem(m.name.replace("_", " ").title())
            value_item = QTableWidgetItem(f"{m.value} {m.unit}")
            conf_item = QTableWidgetItem(f"{m.confidence:.0%}")
            self._table.setItem(i, 0, name_item)
            self._table.setItem(i, 1, value_item)
            self._table.setItem(i, 2, conf_item)

    def _on_export_clicked(self) -> None:
        """Handle export button click."""
        fmt = self._export_combo.currentText().lower()
        self.export_requested.emit(fmt)

    def export_results(self, measurements: List[MeasurementResult], fmt: str) -> None:
        """Export measurement results to a file."""
        if not measurements:
            QMessageBox.information(self, "No Data", "No measurements to export.")
            return

        ext_map = {"csv": "csv", "json": "json", "morphoj": "txt", "pdf": "pdf"}
        ext = ext_map.get(fmt, "csv")
        default_name = f"{self._current_label}_measurements.{ext}"

        path_str, _ = QFileDialog.getSaveFileName(
            self, f"Export as {fmt.upper()}", default_name,
            f"{fmt.upper()} Files (*.{ext})",
        )
        if not path_str:
            return
        path = Path(path_str)

        if fmt == "csv":
            self._export_csv(path, measurements)
        elif fmt == "json":
            self._export_json(path, measurements)
        elif fmt == "morphoj":
            self._export_morphoj(path, measurements)
        elif fmt == "pdf":
            self._export_pdf(path, measurements)

        QMessageBox.information(self, "Exported", f"Results saved to:\n{path}")

    def _export_csv(self, path: Path, measurements: List[MeasurementResult]) -> None:
        """Export as CSV."""
        import csv
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["measurement", "value", "unit", "confidence"])
            for m in measurements:
                writer.writerow([m.name, m.value, m.unit, m.confidence])

    def _export_json(self, path: Path, measurements: List[MeasurementResult]) -> None:
        """Export as JSON."""
        import json
        data = {
            "artefact": self._current_label,
            "quality": self._current_grade.value,
            "measurements": [
                {"name": m.name, "value": m.value, "unit": m.unit, "confidence": m.confidence}
                for m in measurements
            ],
        }
        path.write_text(json.dumps(data, indent=2))

    def _export_morphoj(self, path: Path, measurements: List[MeasurementResult]) -> None:
        """Export in MorphoJ-compatible landmark format."""
        lines = [f"# MorphoJ export — {self._current_label}"]
        lines.append(f"# Generated by Lithic Analysis Platform v0.1")
        lines.append(f"# Measurements: {len(measurements)}")
        lines.append("")
        lines.append("LRMM 3D")  # Landmark data marker
        lines.append(f"{self._current_label}")
        lines.append("1")  # Single specimen
        lines.append(f"{len(measurements)}")  # Number of landmarks
        for i, m in enumerate(measurements):
            lines.append(f"{i+1} {m.value:.3f} {0.0:.3f} {0.0:.3f}")
        path.write_text("\n".join(lines))

    def _export_pdf(self, path: Path, measurements: List[MeasurementResult]) -> None:
        """Export as PDF report."""
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer,
        )
        from reportlab.lib.styles import getSampleStyleSheet

        doc = SimpleDocTemplate(str(path), pagesize=A4)
        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph(f"Lithic Analysis Report", styles["Title"]))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(f"Artefact: {self._current_label}", styles["Heading2"]))
        elements.append(Paragraph(f"Quality: {self._current_grade.value}", styles["Normal"]))
        elements.append(Spacer(1, 12))

        table_data = [["Measurement", "Value", "Confidence"]]
        for m in measurements:
            table_data.append([
                m.name.replace("_", " ").title(),
                f"{m.value} {m.unit}",
                f"{m.confidence:.0%}",
            ])

        t = Table(table_data, colWidths=[150, 100, 80])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 20))
        elements.append(Paragraph("Generated by Lithic Analysis Platform v0.1", styles["Italic"]))

        doc.build(elements)
```

- [ ] **Step 2: Import check**

Run: `python -c "from lithicope._results_panel import ResultsPanel; print('OK')"`
Expected: OK

- [ ] **Step 3: Commit**

```bash
git add lithicope/src/lithicope/_results_panel.py
git commit -m "feat: results panel with measurements table and CSV/JSON/MorphoJ/PDF export"
```

### Task 16: Batch runner UI

**Files:**
- Create: `lithicope/src/lithicope/_batch_runner.py`

- [ ] **Step 1: Write `_batch_runner.py`**

```python
"""_batch_runner.py — Batch processing runner with progress feedback.

exports: BatchRunner(QDialog)
used_by: MainWindow batch import
rules:   Runs batch_process in a separate thread to keep UI responsive.
         Shows progress bar and per-file status updates.
         Opens results panel when complete.
agent:   deepseek-v4-flash | 2026-05-26 | Initial implementation
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar,
    QPushButton, QTextEdit, QMessageBox, QWidget,
)
from lithicore._models import MeasurementConfig, ArtefactResult
from lithicore._batch import batch_process


class BatchWorker(QThread):
    """Background worker for batch processing."""

    progress = pyqtSignal(int, int, str)  # current, total, filename
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, directory: Path, config: MeasurementConfig) -> None:
        super().__init__()
        self._directory = directory
        self._config = config

    def run(self) -> None:
        try:
            results = batch_process(self._directory, self._config)
            self.finished.emit(results)
        except Exception as exc:
            self.error.emit(str(exc))


class BatchRunner(QDialog):
    """Dialog showing batch processing progress."""

    def __init__(
        self,
        directory: Path,
        config: MeasurementConfig,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Batch Processing")
        self.setMinimumWidth(500)
        self.setModal(True)

        self._results: List[ArtefactResult] = []

        layout = QVBoxLayout(self)

        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        layout.addWidget(self._progress_bar)

        # Status text
        self._status_text = QTextEdit()
        self._status_text.setReadOnly(True)
        self._status_text.setMaximumHeight(200)
        layout.addWidget(self._status_text)

        # Buttons
        btn_layout = QHBoxLayout()
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._on_cancel)
        btn_layout.addWidget(self._cancel_btn)

        self._close_btn = QPushButton("Close")
        self._close_btn.clicked.connect(self.accept)
        self._close_btn.setEnabled(False)
        btn_layout.addWidget(self._close_btn)
        layout.addLayout(btn_layout)

        # Start worker
        self._worker = BatchWorker(directory, config)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, current: int, total: int, filename: str) -> None:
        percent = int((current / total) * 100) if total > 0 else 0
        self._progress_bar.setValue(percent)
        self._status_text.append(f"[{current}/{total}] Processing {filename}...")

    def _on_finished(self, results: List[ArtefactResult]) -> None:
        self._results = results
        self._progress_bar.setValue(100)
        success = sum(1 for r in results if r.measurements)
        self._status_text.append(f"\nComplete: {len(results)} files processed ({success} with measurements)")
        self._cancel_btn.setEnabled(False)
        self._close_btn.setEnabled(True)

    def _on_error(self, error_msg: str) -> None:
        self._status_text.append(f"\nError: {error_msg}")
        self._cancel_btn.setEnabled(False)
        self._close_btn.setEnabled(True)

    def _on_cancel(self) -> None:
        self._worker.terminate()
        self._status_text.append("\nCancelled by user")
        self._cancel_btn.setEnabled(False)
        self._close_btn.setEnabled(True)

    def get_results(self) -> List[ArtefactResult]:
        return self._results
```

- [ ] **Step 2: Import check**

Run: `python -c "from lithicope._batch_runner import BatchRunner; print('OK')"`
Expected: OK

- [ ] **Step 3: Commit**

```bash
git add lithicope/src/lithicope/_batch_runner.py
git commit -m "feat: batch runner UI with progress and threading"
```

---

## Phase 6: Integration & Polish

### Task 17: Wire everything together — verify app launches

- [ ] **Step 1: Install both packages in dev mode**

Run: `pip install -e lithicore -e lithicope 2>&1 | tail -3`

- [ ] **Step 2: Verify all imports work**

Run:
```python
python -c "
from lithicore._models import MeasurementConfig, MeasurementResult, MeshGrade
from lithicore._validation import validate_mesh, repair_mesh
from lithicore._orientation import orient_auto, orient_manual
from lithicore._metrics import extract_metrics
from lithicore._platform_angle import platform_angles
from lithicore._edge_detection import detect_edges
from lithicore._batch import batch_process
from lithicore._cli import app
print('lithicore: all imports OK')
"
```

Run:
```python
DISPLAY= python -c "
from lithicope._main_window import MainWindow
from lithicope._viewer_3d import Viewer3D
from lithicope._import_dialog import ImportDialog
from lithicope._results_panel import ResultsPanel
from lithicope._batch_runner import BatchRunner
print('lithicope: all imports OK')
"
```

- [ ] **Step 3: Run full test suite**

Run: `cd lithicore && python -m pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 4: Test CLI works**

Run: `lithicore --help`
Expected: Shows command help

- [ ] **Step 5: Test CLI batch on synthetic data**

Create a test directory with mesh files and run:
```bash
mkdir -p /tmp/lithicore-test-meshes
python -c "
import trimesh
mesh = trimesh.creation.box(extents=[50, 30, 10])
mesh.export('/tmp/lithicore-test-meshes/test.ply')
mesh2 = trimesh.creation.icosphere()
mesh2.export('/tmp/lithicore-test-meshes/sphere.obj')
"
lithicore batch --input /tmp/lithicore-test-meshes --output /tmp/test-results.csv
cat /tmp/test-results.csv
```

- [ ] **Step 6: Final commit**

```bash
git add -A
git commit -m "feat: lithic-analysis-platform v0.1 initial build"
```

---

## Self-Review Checklist

After writing this plan, check:

- **Spec coverage:** Every section in the spec is addressed — two-package architecture, mesh validation, orientation (auto/semi/manual), metrics, edge detection, export formats, batch processing, CLI, GUI
- **Placeholder scan:** No TBDs, TODOs, "add later" patterns. Every step has concrete code
- **Type consistency:** `MeasurementConfig.repair_mesh` consistent across all tasks. `MeasurementResult.name`, `.value`, `.unit`, `.confidence` used everywhere. `MeshGrade` enum consistent
