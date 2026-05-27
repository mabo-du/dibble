"""test_annotations.py — Unit tests for annotation data model.

exports: TestAnnotation, TestAnnotationSet
used_by: pytest
rules:   Pure dataclass tests, no GUI dependencies.
agent:   deepseek-v4-flash | 2026-05-27 | Test skeleton
"""

import json
from pathlib import Path

import pytest

from lithicore._annotations import Annotation, AnnotationSet


class TestAnnotation:
    """Annotation dataclass construction."""

    def test_annotation_defaults(self):
        ann = Annotation(
            point=(1.0, 2.0, 3.0),
            title="Test annotation",
        )
        assert ann.point == (1.0, 2.0, 3.0)
        assert ann.title == "Test annotation"
        assert ann.description == ""
        assert ann.category == ""
        assert ann.measurement_mm == 0.0
        assert ann.confidence == 1.0
        assert ann.author == ""
        assert ann.timestamp == ""
        assert ann.attached_photos == []
        assert ann.sub_annotations == []

    def test_annotation_custom_values(self):
        sub = Annotation(point=(4.0, 5.0, 6.0), title="sub")
        ann = Annotation(
            point=(1.0, 2.0, 3.0),
            title="Main",
            description="A test annotation",
            category="scar",
            measurement_mm=14.2,
            confidence=0.95,
            author="mark",
            timestamp="2026-05-27T12:00:00",
            attached_photos=["photo1.png"],
            sub_annotations=[sub],
        )
        assert ann.category == "scar"
        assert ann.measurement_mm == 14.2
        assert ann.confidence == 0.95
        assert len(ann.sub_annotations) == 1

    def test_annotation_point_tuple(self):
        ann = Annotation(point=(1.234, 5.678, 9.012), title="p")
        x, y, z = ann.point
        assert isinstance(x, float)
        assert isinstance(y, float)
        assert isinstance(z, float)
