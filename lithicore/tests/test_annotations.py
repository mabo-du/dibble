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


class TestAnnotationSet:
    """AnnotationSet JSON round-trip and merge."""

    def test_round_trip_json(self):
        ann = Annotation(point=(1.0, 2.0, 3.0), title="Test")
        ann_set = AnnotationSet(
            artefact_label="Flake_42",
            author="mark",
            created="2026-05-27T12:00:00",
            annotations=[ann],
        )
        json_str = ann_set.to_json()
        restored = AnnotationSet.from_json(json_str)
        assert restored.artefact_label == "Flake_42"
        assert len(restored.annotations) == 1
        assert list(restored.annotations[0].point) == [1.0, 2.0, 3.0]
        assert restored.annotations[0].title == "Test"

    def test_merge_disjoint_positions(self):
        a1 = Annotation(point=(1.0, 2.0, 3.0), title="A")
        a2 = Annotation(point=(4.0, 5.0, 6.0), title="B")
        base = AnnotationSet(artefact_label="test", annotations=[a1])
        incoming = AnnotationSet(artefact_label="test", annotations=[a2])
        merged, warnings = base.merge(incoming)
        assert len(merged.annotations) == 2
        assert warnings == []

    def test_merge_same_position(self):
        a1 = Annotation(point=(1.0, 2.0, 3.0), title="Scar A",
                        author="mark", timestamp="2026-05-27T12:00:00")
        a2 = Annotation(point=(1.0, 2.0, 3.0), title="Scar A",
                        author="anna", timestamp="2026-05-28T12:00:00")
        base = AnnotationSet(artefact_label="test", annotations=[a1])
        incoming = AnnotationSet(artefact_label="test", annotations=[a2])
        merged, warnings = base.merge(incoming)
        assert len(merged.annotations) == 1
        # Should keep the newer timestamp
        assert merged.annotations[0].timestamp == "2026-05-28T12:00:00"

    def test_merge_conflict(self):
        a1 = Annotation(point=(1.0, 2.0, 3.0), title="Scar A")
        a2 = Annotation(point=(1.0, 2.0, 3.0), title="Scar B (different opinion)",
                        author="anna")
        base = AnnotationSet(artefact_label="test", annotations=[a1])
        incoming = AnnotationSet(artefact_label="test", annotations=[a2])
        merged, warnings = base.merge(incoming)
        assert len(merged.annotations) == 1
        assert len(warnings) >= 1
        assert "(anna)" in merged.annotations[0].title or "(imported)" in merged.annotations[0].title

    def test_merge_into_empty(self):
        ann = Annotation(point=(1.0, 2.0, 3.0), title="Only")
        empty = AnnotationSet()
        incoming = AnnotationSet(annotations=[ann])
        merged, warnings = empty.merge(incoming)
        assert len(merged.annotations) == 1
        assert warnings == []

    def test_round_trip_empty_set(self):
        s = AnnotationSet()
        restored = AnnotationSet.from_json(s.to_json())
        assert restored.annotations == []

    def test_round_trip_with_sub_annotations(self):
        sub = Annotation(point=(4.0, 5.0, 6.0), title="Sub")
        main = Annotation(point=(1.0, 2.0, 3.0), title="Main",
                          sub_annotations=[sub])
        s = AnnotationSet(annotations=[main])
        restored = AnnotationSet.from_json(s.to_json())
        assert len(restored.annotations) == 1
        assert len(restored.annotations[0].sub_annotations) == 1
        assert restored.annotations[0].sub_annotations[0].title == "Sub"
