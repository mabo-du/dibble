"""test_classification.py — Unit tests for lithic classification.

exports: TestExtractFeatures, TestClassifierModel, TestTraining
used_by: pytest
rules:   Synthetic meshes only. Pre-trained models must be generated first.
agent:   deepseek-v4-flash | 2026-05-27 | Initial implementation
"""

import numpy as np
import trimesh
import pytest

from lithicore import (
    ClassificationResult, FeatureImportance, LithicFeatureVector,
    ClassifierModel, extract_features, extract_diagnostic_coordinates,
)


@pytest.fixture
def blade_mesh():
    """A synthetic blade-like mesh (elongated, triangular cross-section)."""
    box = trimesh.creation.box(extents=[10, 30, 5])
    return box


class TestExtractFeatures:
    """Feature extraction from synthetic meshes."""

    def test_extract_returns_feature_vector(self, blade_mesh):
        fv = extract_features(blade_mesh)
        assert isinstance(fv, LithicFeatureVector)
        assert fv.length_mm > 0

    def test_feature_vector_has_20_features(self, blade_mesh):
        fv = extract_features(blade_mesh)
        arr = fv.to_array()
        assert len(arr) == 20

    def test_feature_vector_no_nan(self, blade_mesh):
        fv = extract_features(blade_mesh)
        arr = fv.to_array()
        assert not np.any(np.isnan(arr))

    def test_feature_vector_no_inf(self, blade_mesh):
        fv = extract_features(blade_mesh)
        arr = fv.to_array()
        assert not np.any(np.isinf(arr))

    def test_elongation_of_blade_mesh(self, blade_mesh):
        fv = extract_features(blade_mesh)
        assert fv.elongation > 1.0  # length > width


class TestClassifierModel:
    """Classifier predictions."""

    def test_load_pre_trained_basic(self):
        model = ClassifierModel.load_pre_trained("basic")
        assert model.is_loaded()

    def test_load_pre_trained_bordes(self):
        model = ClassifierModel.load_pre_trained("bordes")
        assert model.is_loaded()

    def test_load_pre_trained_all(self):
        for name in ["basic", "bordes", "technological"]:
            model = ClassifierModel.load_pre_trained(name)
            assert model.is_loaded()

    def test_predict_returns_classification_result(self, blade_mesh):
        model = ClassifierModel.load_pre_trained("basic")
        fv = extract_features(blade_mesh)
        result = model.predict(fv)
        assert isinstance(result, ClassificationResult)
        assert isinstance(result.label, str)
        assert 0 <= result.confidence <= 1

    def test_predict_probabilities_sum_to_one(self, blade_mesh):
        model = ClassifierModel.load_pre_trained("basic")
        fv = extract_features(blade_mesh)
        result = model.predict(fv)
        total = sum(result.probabilities.values())
        assert abs(total - 1.0) < 0.01

    def test_predict_has_top_features(self, blade_mesh):
        model = ClassifierModel.load_pre_trained("basic")
        fv = extract_features(blade_mesh)
        result = model.predict(fv)
        assert len(result.top_features) == 5
        for f in result.top_features:
            assert isinstance(f, FeatureImportance)

    def test_predict_without_model_raises(self):
        model = ClassifierModel(typology_name="test")
        with pytest.raises(RuntimeError, match="No model loaded"):
            model.predict(LithicFeatureVector())

    def test_active_learning_queue(self, blade_mesh):
        model = ClassifierModel.load_pre_trained("basic")
        fv = extract_features(blade_mesh)
        count = model.queue_correction(fv, "Blade")
        assert count == 1

    def test_retrain_after_threshold(self, blade_mesh):
        model = ClassifierModel.load_pre_trained("basic")
        fv = extract_features(blade_mesh)
        labels = ["Blade", "Bladelet", "Core"]
        for i in range(15):
            model.queue_correction(fv, labels[i % len(labels)])
        assert model.retrain_if_ready(threshold=10) is True

    def test_retrain_below_threshold(self, blade_mesh):
        model = ClassifierModel.load_pre_trained("basic")
        fv = extract_features(blade_mesh)
        model.queue_correction(fv, "Blade")
        assert model.retrain_if_ready(threshold=10) is False


class TestDiagnosticCoordinates:
    """Viewer overlay coordinate extraction."""

    def test_extract_returns_dict(self, blade_mesh):
        coords = extract_diagnostic_coordinates(blade_mesh)
        assert isinstance(coords, dict)
        assert "ridges" in coords
        assert "platform" in coords
        assert "retouched_edges" in coords

    def test_coordinates_are_valid(self, blade_mesh):
        coords = extract_diagnostic_coordinates(blade_mesh)
        for key, points in coords.items():
            if len(points) > 0:
                assert points.shape[1] == 3  # (N, 3)


class TestTraining:
    """Custom typology training."""

    def test_train_minimal_model(self):
        from lithicore import train_model

        fvs = []
        labels = []
        for label in ["TypeA", "TypeB", "TypeC"]:
            for _ in range(4):
                fv = LithicFeatureVector(
                    length_mm=50, width_mm=30, thickness_mm=10,
                    elongation=2.0, flatness=3.0,
                )
                fvs.append(fv)
                labels.append(label)

        model = train_model(fvs, labels, typology_name="custom_test")
        assert model.is_loaded()

        result = model.predict(LithicFeatureVector(
            length_mm=50, width_mm=30, thickness_mm=10,
            elongation=2.0, flatness=3.0,
        ))
        assert result.label in ["TypeA", "TypeB", "TypeC"]
