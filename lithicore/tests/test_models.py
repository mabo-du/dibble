"""test_models.py — Tests for data model dataclasses."""
import pytest
from lithicore._models import (
    MeasurementConfig,
    MeasurementResult,
    ArtefactResult,
    Landmark,
    MeshQualityReport,
    MeshGrade,
)


class TestMeasurementConfig:
    def test_default_config(self):
        config = MeasurementConfig()
        assert config.repair_mesh is True
        assert config.edge_threshold_degrees == 50.0
        assert config.min_face_count == 2000

    def test_custom_config(self):
        config = MeasurementConfig(repair_mesh=False, edge_threshold_degrees=60.0)
        assert config.repair_mesh is False
        assert config.edge_threshold_degrees == 60.0

    def test_config_is_frozen(self):
        config = MeasurementConfig()
        with pytest.raises(AttributeError):
            config.repair_mesh = False


class TestMeasurementResult:
    def test_create(self):
        r = MeasurementResult(name="length", value=45.2, unit="mm", confidence=0.95)
        assert r.name == "length"
        assert r.value == 45.2

    def test_confidence_range(self):
        r = MeasurementResult(name="angle", value=78.3, unit="°", confidence=1.0)
        assert 0.0 <= r.confidence <= 1.0


class TestMeshQualityReport:
    def test_default_grade_is_pass(self):
        r = MeshQualityReport(original_vertex_count=1000, original_face_count=2000)
        assert r.grade == MeshGrade.PASS
        assert r.warnings == []
