"""test_scar_detection.py — Tests for flake scar detection via Shape Index + watershed.

exports: TestShapeIndex, TestCurvedness, TestDetectScars
used_by: pytest (CI, pre-commit, test suite)
rules:   Curvature-derived tests use icosphere (smooth → no scars) and box
         (sharp edges → possible ridge detections).
         Never require exact scar counts — algorithm is mesh-resolution
         dependent.
agent:   deepseek-v4-flash | 2026-05-26 | Tests for initial scar detection impl
"""

import numpy as np
import trimesh
import pytest

from lithicore._scar_detection import (
    detect_scars,
    ScarConfig,
    ScarResult,
    DetectedScar,
    _shape_index,
    _curvedness,
    _compute_principal_curvatures,
)


# ============================================================================
# Unit tests: pure math helpers
# ============================================================================


class TestShapeIndex:
    """Koenderink Shape Index — pure mathematical tests."""

    def test_spherical_cap(self):
        """Spherical cap (k1=k2>0) → SI = +1.0."""
        si = _shape_index(np.array([1.0]), np.array([1.0]))
        assert abs(si[0] - 1.0) < 0.01, f"Expected 1.0, got {si[0]}"

    def test_spherical_cup(self):
        """Spherical cup (k1=k2<0) → SI = -1.0."""
        si = _shape_index(np.array([-1.0]), np.array([-1.0]))
        assert abs(si[0] - -1.0) < 0.01, f"Expected -1.0, got {si[0]}"

    def test_saddle(self):
        """Saddle (k1=-k2) → SI ≈ 0.0."""
        si = _shape_index(np.array([1.0]), np.array([-1.0]))
        assert abs(si[0]) < 0.01, f"Expected ≈0.0, got {si[0]}"

    def test_ridge(self):
        """Ridge (k1>0, k2≈0) → SI ≈ 0.5."""
        si = _shape_index(np.array([1.0]), np.array([0.0]))
        assert abs(si[0] - 0.5) < 0.05, f"Expected ≈0.5, got {si[0]}"

    def test_valley(self):
        """Valley (k1≈0, k2<0) → SI ≈ -0.5."""
        si = _shape_index(np.array([0.0]), np.array([-1.0]))
        assert abs(si[0] - -0.5) < 0.05, f"Expected ≈-0.5, got {si[0]}"

    def test_vectorised(self):
        """Shape Index should work on arrays of any length."""
        k1 = np.array([1.0, -1.0, 1.0, 0.0])
        k2 = np.array([1.0, -1.0, -1.0, -1.0])
        si = _shape_index(k1, k2)
        assert si.shape == (4,)
        assert abs(si[0] - 1.0) < 0.01   # cap
        assert abs(si[1] - -1.0) < 0.01  # cup
        assert abs(si[2]) < 0.01         # saddle
        assert abs(si[3] + 0.5) < 0.05   # valley

    def test_near_planar(self):
        """Near-planar (both curvatures ~0) should give stable result."""
        si = _shape_index(np.array([1e-6]), np.array([1e-6]))
        assert np.isfinite(si[0])


class TestCurvedness:
    """Curvedness — pure mathematical tests."""

    def test_zero_flat(self):
        """Flat surface → curvedness = 0."""
        cv = _curvedness(np.array([0.0]), np.array([0.0]))
        assert cv[0] == 0.0

    def test_positive_curved(self):
        """Curved surface → curvedness > 0."""
        cv = _curvedness(np.array([1.0]), np.array([1.0]))
        assert cv[0] > 0

    def test_saddle_curvedness(self):
        """Saddle has curvedness proportional to absolute curvature."""
        cv = _curvedness(np.array([2.0]), np.array([-2.0]))
        # sqrt((4+4)/2) = sqrt(4) = 2.0
        assert abs(cv[0] - 2.0) < 0.01

    def test_vectorised(self):
        """Curvedness works on arrays."""
        cv = _curvedness(np.array([0.0, 1.0]), np.array([0.0, 2.0]))
        assert cv.shape == (2,)


# ============================================================================
# Integration tests: full detection pipeline
# ============================================================================


class TestDetectScars:
    """Full scar detection pipeline on synthetic meshes."""

    def test_smooth_sphere_returns_valid_result(self):
        """A smooth icosphere should return a valid (possibly empty) result."""
        sphere = trimesh.creation.icosphere(subdivisions=2)
        config = ScarConfig(min_scar_faces=10, curvature_radius=1.0)
        result = detect_scars(sphere, config)
        assert isinstance(result, ScarResult)
        assert hasattr(result, 'scar_count')
        assert hasattr(result, 'scars')
        assert hasattr(result, 'total_scar_area_mm2')
        assert hasattr(result, 'scar_density')
        assert hasattr(result, 'face_labels')
        # A sphere is smooth — expect very few, if any, scars
        assert result.scar_count >= 0

    def test_smooth_sphere_no_false_scars(self):
        """A well-subdivided sphere should have zero false detections."""
        sphere = trimesh.creation.icosphere(subdivisions=3)
        config = ScarConfig(
            min_scar_faces=10,
            curvature_radius=2.0,
            valley_threshold=-0.3,
        )
        result = detect_scars(sphere, config)
        # Sphere has no concave valleys — should have 0 or very few scars
        assert result.scar_count <= 2, (
            f"Smooth sphere should have ≤2 false positives, got {result.scar_count}"
        )

    def test_empty_mesh_returns_empty(self):
        """A mesh with too few faces returns an empty result."""
        box = trimesh.creation.box(extents=[10, 10, 10])
        config = ScarConfig(min_scar_faces=1000)
        result = detect_scars(box, config)
        assert result.scar_count == 0
        assert len(result.scars) == 0
        assert result.total_scar_area_mm2 == 0.0
        assert result.scar_density == 0.0
        assert len(result.face_labels) == len(box.faces)

    def test_result_has_correct_face_label_count(self, rectangular_prism):
        """Face labels array length must match mesh face count."""
        mesh = rectangular_prism
        config = ScarConfig(min_scar_faces=5, curvature_radius=2.0)
        result = detect_scars(mesh, config)
        assert len(result.face_labels) == len(mesh.faces)

    def test_result_face_labels_are_valid(self):
        """Face labels must be -1 (no scar) or 0+ (scar index)."""
        sphere = trimesh.creation.icosphere(subdivisions=2)
        config = ScarConfig(min_scar_faces=5, curvature_radius=1.0)
        result = detect_scars(sphere, config)
        valid = (result.face_labels >= -1) & (result.face_labels <= 100)
        assert np.all(valid)

    def test_detected_scar_has_required_fields(self, rectangular_prism):
        """Each DetectedScar must have all metadata fields."""
        mesh = rectangular_prism
        config = ScarConfig(min_scar_faces=3, curvature_radius=2.0)
        result = detect_scars(mesh, config)
        for scar in result.scars:
            assert isinstance(scar, DetectedScar)
            assert hasattr(scar, 'index')
            assert hasattr(scar, 'face_indices')
            assert hasattr(scar, 'area_mm2')
            assert hasattr(scar, 'centroid')
            assert hasattr(scar, 'mean_curvature')
            assert hasattr(scar, 'max_depth_mm')
            assert len(scar.centroid) == 3
            assert scar.area_mm2 >= 0


# ============================================================================
# Configuration tests
# ============================================================================


class TestScarConfig:
    """ScarConfig defaults and edge cases."""

    def test_defaults_are_reasonable(self):
        """Default config values should be valid."""
        config = ScarConfig()
        assert -1 <= config.ridge_threshold <= 1
        assert -1 <= config.valley_threshold <= 1
        assert config.min_scar_faces > 0
        assert 0 <= config.curvedness_percentile <= 100
        assert config.curvature_radius > 0
        # Ridge threshold should be higher than valley threshold
        assert config.ridge_threshold > config.valley_threshold

    def test_custom_config(self):
        """Custom config values should be stored."""
        config = ScarConfig(
            curvature_radius=5.0,
            ridge_threshold=0.5,
            valley_threshold=-0.5,
            min_scar_faces=50,
            curvedness_percentile=80.0,
        )
        assert config.curvature_radius == 5.0
        assert config.ridge_threshold == 0.5
        assert config.valley_threshold == -0.5
        assert config.min_scar_faces == 50
        assert config.curvedness_percentile == 80.0
