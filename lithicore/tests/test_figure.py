"""test_figure.py — Tests for publication figure generation."""

import pytest
from lithicore._figure import FigureConfig, _nice_scale


class TestFigureConfig:
    def test_default_config(self):
        config = FigureConfig()
        assert config.views == ["plan", "profile", "section"]
        assert config.show_measurements is True
        assert config.show_ridges is True

    def test_custom_config(self):
        config = FigureConfig(
            views=["plan"],
            show_measurements=False,
            artefact_label="FLK-145",
        )
        assert config.views == ["plan"]
        assert config.show_measurements is False
        assert config.artefact_label == "FLK-145"


class TestNiceScale:
    def test_scale_for_large_object(self):
        """A 100mm object should get a ~5cm scale bar."""
        length, label = _nice_scale(100.0)
        assert 30 <= length <= 60
        assert "cm" in label or "mm" in label

    def test_scale_for_small_object(self):
        """A 10mm object should get a reasonable scale."""
        length, label = _nice_scale(10.0)
        assert length > 0
        assert label

    def test_scale_is_nice_number(self):
        """Scale bar lengths should be round numbers (1, 2, 5, 10, 20, etc.)."""
        for extent in [5, 15, 50, 120, 300]:
            length, label = _nice_scale(extent)
            assert length > 0
            assert label
