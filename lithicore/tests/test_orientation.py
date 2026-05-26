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

    def test_less_than_3_points_raises(self, rectangular_prism):
        """Orient with < 3 points should raise ValueError."""
        points = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        config = MeasurementConfig()
        with pytest.raises(ValueError, match="At least 3 points"):
            orient_manual(rectangular_prism, points, config)
