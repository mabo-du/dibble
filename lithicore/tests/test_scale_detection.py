"""test_scale_detection.py — Unit tests for scale detection.

exports: TestScaleResult
         TestApplyScale
used_by: pytest
rules:   No COLMAP dependency required for unit tests.
         Synthetic data only.
agent:   deepseek-v4-flash | 2026-05-27 | Initial test skeleton
"""

import numpy as np
import trimesh
import pytest

from lithicore._scale_detection import (
    ScaleResult,
    apply_scale_to_mesh,
)


class TestScaleResult:
    """ScaleResult dataclass construction."""

    def test_scale_result_defaults(self):
        result = ScaleResult(
            scale_factor=5.0,
            method="aruco",
            confidence=0.99,
            detected_length_mm=20.0,
            warnings=[],
        )
        assert result.scale_factor == 5.0
        assert result.method == "aruco"
        assert result.confidence == 0.99

    def test_scale_result_warning_defaults(self):
        result = ScaleResult(
            scale_factor=1.0,
            method="manual",
            confidence=0.5,
            detected_length_mm=0.0,
        )
        assert result.warnings == []


class TestApplyScale:
    """Mesh vertex scaling."""

    @pytest.fixture
    def unit_cube(self):
        return trimesh.creation.box(extents=[1, 1, 1])

    def test_apply_scale_doubles_vertices(self, unit_cube):
        scaled = apply_scale_to_mesh(unit_cube, scale_factor=2.0)
        assert np.allclose(scaled.vertices.max(axis=0), [1.0, 1.0, 1.0])
        assert np.allclose(scaled.vertices.min(axis=0), [-1.0, -1.0, -1.0])

    def test_apply_scale_halves_vertices(self, unit_cube):
        scaled = apply_scale_to_mesh(unit_cube, scale_factor=0.5)
        assert np.allclose(scaled.vertices.max(axis=0), [0.25, 0.25, 0.25])

    def test_apply_scale_identity(self, unit_cube):
        scaled = apply_scale_to_mesh(unit_cube, scale_factor=1.0)
        original = unit_cube.vertices.copy()
        assert np.allclose(scaled.vertices, original)

    def test_apply_scale_zero_raises(self, unit_cube):
        with pytest.raises(ValueError, match="Scale factor must be positive"):
            apply_scale_to_mesh(unit_cube, scale_factor=0.0)

    def test_apply_scale_negative_raises(self, unit_cube):
        with pytest.raises(ValueError, match="Scale factor must be positive"):
            apply_scale_to_mesh(unit_cube, scale_factor=-1.0)

    def test_apply_scale_preserves_faces(self, unit_cube):
        scaled = apply_scale_to_mesh(unit_cube, scale_factor=3.0)
        assert len(scaled.faces) == len(unit_cube.faces)
