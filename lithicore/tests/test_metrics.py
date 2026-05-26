"""test_metrics.py — Tests for metric extraction."""

import math
import trimesh
import pytest
from lithicore._metrics import extract_metrics
from lithicore._models import MeasurementConfig


class TestExtractMetrics:
    def test_oriented_prism_length(self, oriented_prism):
        """A 50x30x10 prism should give length=50, width=30, thickness=10."""
        config = MeasurementConfig()
        results = extract_metrics(oriented_prism, config)
        by_name = {r.name: r for r in results}
        assert abs(by_name["max_length"].value - 50) < 1.0
        assert abs(by_name["max_width"].value - 30) < 1.0
        assert abs(by_name["max_thickness"].value - 10) < 1.0

    def test_surface_area(self, oriented_prism):
        """50x30x10 box has surface area = 2*(50*30 + 50*10 + 30*10) = 4600 mm2."""
        config = MeasurementConfig()
        results = extract_metrics(oriented_prism, config)
        by_name = {r.name: r for r in results}
        expected = 2 * (50*30 + 50*10 + 30*10)
        assert abs(by_name["surface_area"].value - expected) < expected * 0.05

    def test_volume(self, oriented_prism):
        """50x30x10 box has volume = 15000 mm3."""
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
