"""test_scale_detection.py — Unit tests for scale detection.

exports: TestScaleResult
         TestApplyScale
used_by: pytest
rules:   No COLMAP dependency required for unit tests.
         Synthetic data only.
agent:   deepseek-v4-flash | 2026-05-27 | Initial test skeleton
agent:   deepseek-v4-flash | 2026-05-27 | Added ArUco marker detection tests
"""

from pathlib import Path

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


class TestDetectScaleArUco:
    """ArUco marker detection with synthetic images."""

    def test_aruco_no_markers_returns_none(self, tmp_path):
        """When no markers in photos, should return None."""
        from lithicore._scale_detection import detect_scale_aruco
        photo_dir = tmp_path / "photos"
        photo_dir.mkdir()
        import cv2
        blank = np.zeros((480, 640), dtype=np.uint8)
        cv2.imwrite(str(photo_dir / "img_001.jpg"), blank)
        sparse_dir = tmp_path / "sparse"
        sparse_dir.mkdir()
        result = detect_scale_aruco(photo_dir, sparse_dir, marker_size_mm=20.0)
        assert result is None

    def test_aruco_empty_photo_dir_returns_none(self, tmp_path):
        from lithicore._scale_detection import detect_scale_aruco
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        sparse_dir = tmp_path / "sparse"
        sparse_dir.mkdir()
        result = detect_scale_aruco(empty_dir, sparse_dir, marker_size_mm=20.0)
        assert result is None

    def test_aruco_returns_scale_warning_when_opencv_missing(self, tmp_path, monkeypatch):
        """If OpenCV is not installed, should return ScaleResult with warning."""
        monkeypatch.setattr("lithicore._scale_detection._read_photos", lambda d: [Path("fake.jpg")])
        # Mock ImportError for cv2
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "cv2":
                raise ImportError("No module named cv2")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        from lithicore._scale_detection import detect_scale_aruco
        result = detect_scale_aruco(tmp_path, tmp_path)
        assert result is not None
        assert result.method == "aruco"
        assert result.confidence == 0.0
        assert any("OpenCV" in w for w in result.warnings)
