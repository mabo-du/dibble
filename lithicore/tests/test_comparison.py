"""test_comparison.py — Tests for mesh comparison metrics."""

import trimesh
import pytest
from lithicore._comparison import compare_meshes, _hausdorff_distance


class TestCompareMeshes:
    def test_identical_meshes(self, oriented_prism):
        """Two copies of the same mesh should have zero differences."""
        result = compare_meshes(oriented_prism, oriented_prism)
        assert result.hausdorff_distance_mm == 0.0
        assert result.volume_difference_mm3 == 0.0
        assert result.centroid_distance_mm == 0.0

    def test_different_meshes(self, oriented_prism):
        """Two different meshes should have non-zero differences."""
        sphere = trimesh.creation.icosphere(subdivisions=2)
        result = compare_meshes(oriented_prism, sphere)
        # They're clearly different shapes
        assert result.volume_difference_mm3 > 0
        assert result.surface_area_difference_mm2 > 0

    def test_all_metrics_present(self, oriented_prism):
        """Result should contain all expected metrics."""
        result = compare_meshes(oriented_prism, oriented_prism)
        expected = {
            "hausdorff_distance_mm", "volume_difference_mm3",
            "volume_a_mm3", "volume_b_mm3",
            "surface_area_difference_mm2", "centroid_distance_mm",
            "length_diff_mm", "width_diff_mm", "thickness_diff_mm",
        }
        assert set(result.metrics) == expected


class TestHausdorffDistance:
    def test_identical_is_zero(self, oriented_prism):
        """Hausdorff distance of a mesh to itself should be ~0."""
        dist = _hausdorff_distance(oriented_prism, oriented_prism)
        assert dist < 0.01

    def test_different_is_positive(self, oriented_prism):
        """Different meshes should have positive Hausdorff distance."""
        sphere = trimesh.creation.icosphere(subdivisions=2)
        dist = _hausdorff_distance(oriented_prism, sphere)
        assert dist > 0
