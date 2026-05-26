"""test_platform_angle.py — Tests for platform angle extraction (EPA, IPA)."""

import numpy as np
import pytest

from lithicore._platform_angle import platform_angles
from lithicore._models import MeasurementConfig


class TestPlatformAngles:
    """Verify EPA / IPA extraction with known geometries."""

    def test_epa_present(self, oriented_prism):
        """A rectangular prism should return an EPA measurement."""
        config = MeasurementConfig()
        epa, ipa = platform_angles(oriented_prism, config)
        assert epa is not None
        assert epa.name == "exterior_platform_angle"
        assert epa.unit == "°"

    def test_ipa_present(self, oriented_prism):
        """A rectangular prism should return an IPA measurement."""
        config = MeasurementConfig()
        epa, ipa = platform_angles(oriented_prism, config)
        assert ipa is not None
        assert ipa.name == "interior_platform_angle"
        assert ipa.unit == "°"

    def test_box_angles_are_90(self, oriented_prism):
        """For a rectangular box, both EPA and IPA should be ~90°.

        The side faces of a box are perpendicular to the platform (XY) plane,
        so both the exterior and interior angles should be 90°.
        """
        config = MeasurementConfig()
        epa, ipa = platform_angles(oriented_prism, config)
        assert abs(epa.value - 90) < 5.0
        assert abs(ipa.value - 90) < 5.0

    def test_angle_range(self, oriented_prism):
        """Reported angles should be in [0, 180] degrees."""
        config = MeasurementConfig()
        epa, ipa = platform_angles(oriented_prism, config)
        assert 0 <= epa.value <= 180
        assert 0 <= ipa.value <= 180

    def test_ep_vs_ip_distinct(self, oriented_prism):
        """EPA and IPA should be distinct MeasurementResult objects."""
        config = MeasurementConfig()
        epa, ipa = platform_angles(oriented_prism, config)
        assert epa is not ipa
        assert epa.name != ipa.name
