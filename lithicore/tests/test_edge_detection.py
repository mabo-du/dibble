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
