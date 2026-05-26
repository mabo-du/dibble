"""tests/test_photogrammetry.py — Unit tests for photogrammetry pipeline.

exports: test_photogrammetry_config_defaults
         test_photogrammetry_config_target_faces
         test_photogrammetry_result_fields
used_by: pytest
rules:   COLMAP integration tests are marked @pytest.mark.skipif(colmap_missing).
         Unit tests should never require COLMAP.
agent:   deepseek-v4-flash | 2026-05-26 | Initial test skeleton
agent:   deepseek-v4-flash | 2026-05-26 | Added TestColmapCheck + TestCleanPointCloud with synthetic data
"""

from pathlib import Path
import numpy as np
import pytest

from lithicore._photogrammetry import (
    PhotogrammetryConfig,
    PhotogrammetryResult,
)


class TestPhotogrammetryConfig:
    """Config dataclass defaults and property behaviour."""

    def test_default_mode_is_default(self):
        config = PhotogrammetryConfig(
            photo_folder=Path("/photos"),
            output_path=Path("/out.ply"),
        )
        assert config.mode == "default"
        assert config.quality == "high"

    def test_target_faces_high(self):
        config = PhotogrammetryConfig(
            photo_folder=Path("/photos"),
            output_path=Path("/out.ply"),
            quality="high",
        )
        assert config.target_faces == 150_000

    def test_target_faces_medium(self):
        config = PhotogrammetryConfig(
            photo_folder=Path("/photos"),
            output_path=Path("/out.ply"),
            quality="medium",
        )
        assert config.target_faces == 50_000

    def test_target_faces_low(self):
        config = PhotogrammetryConfig(
            photo_folder=Path("/photos"),
            output_path=Path("/out.ply"),
            quality="low",
        )
        assert config.target_faces == 20_000

    def test_invalid_quality_raises(self):
        config = PhotogrammetryConfig(
            photo_folder=Path("/photos"),
            output_path=Path("/out.ply"),
            quality="invalid",
        )
        with pytest.raises(ValueError, match="Unknown quality"):
            _ = config.target_faces

    def test_post_init_coerces_paths(self):
        """Str inputs should be coerced to Path."""
        config = PhotogrammetryConfig(
            photo_folder="/photos",
            output_path="/out.ply",
        )
        assert isinstance(config.photo_folder, Path)
        assert isinstance(config.output_path, Path)

    def test_expert_fields_accept_custom_values(self):
        """Expert mode should accept colmap-specific fields."""
        config = PhotogrammetryConfig(
            photo_folder=Path("/photos"),
            output_path=Path("/out.ply"),
            mode="expert",
            colmap_feature_type="sift",
            colmap_matching_strategy="vocab_tree",
        )
        assert config.colmap_feature_type == "sift"
        assert config.colmap_matching_strategy == "vocab_tree"

    def test_artefact_label_default_empty(self):
        config = PhotogrammetryConfig(
            photo_folder=Path("/photos"),
            output_path=Path("/out.ply"),
        )
        assert config.artefact_label == ""


class TestPhotogrammetryResult:
    """Result dataclass construction."""

    def test_result_holds_all_fields(self):
        result = PhotogrammetryResult(
            mesh_path=Path("/out.ply"),
            artefact_label="FLK-145",
            camera_count=12,
            point_count=250000,
            face_count=98432,
            vertex_count=49123,
            processing_time_s=222.0,
            colmap_stdout="[info] all done",
            warnings=["2 photos failed extraction"],
        )
        assert result.mesh_path == Path("/out.ply")
        assert result.artefact_label == "FLK-145"
        assert result.camera_count == 12
        assert result.processing_time_s == 222.0
        assert len(result.warnings) == 1

    def test_optional_paths_default_to_none(self):
        result = PhotogrammetryResult(
            mesh_path=Path("/out.ply"),
            artefact_label="test",
            camera_count=3,
            point_count=100,
            face_count=50,
            vertex_count=30,
            processing_time_s=10.0,
            colmap_stdout="",
            warnings=[],
        )
        assert result.sparse_cloud_path is None
        assert result.dense_cloud_path is None


class TestColmapCheck:
    """COLMAP availability detection."""

    def test_colmap_available_returns_bool(self):
        from lithicore._photogrammetry import colmap_available
        result = colmap_available()
        # Should return True or False, never raise
        assert isinstance(result, bool)


class TestCleanPointCloud:
    """Point cloud cleaning functions with synthetic data."""

    @pytest.fixture
    def clean_cloud(self):
        """A dense cloud of points around origin (the artefact)."""
        rng = np.random.default_rng(42)
        points = rng.normal(0, 5, size=(1000, 3))
        return points

    @pytest.fixture
    def noisy_cloud(self, clean_cloud):
        """Same as clean_cloud but with distant outlier points."""
        rng = np.random.default_rng(99)
        outliers = rng.uniform(-100, 100, size=(50, 3))
        return np.vstack([clean_cloud, outliers])

    def test_removes_statistical_outliers(self, noisy_cloud):
        from lithicore._photogrammetry import clean_point_cloud
        cleaned = clean_point_cloud(noisy_cloud, threshold=2.0)
        assert len(cleaned) < len(noisy_cloud)
        # The main cluster (~1000 points) should be preserved
        assert len(cleaned) >= 950

    def test_preserves_clean_cloud(self, clean_cloud):
        from lithicore._photogrammetry import clean_point_cloud
        cleaned = clean_point_cloud(clean_cloud, threshold=2.0)
        # No outliers = no removal
        assert len(cleaned) == len(clean_cloud)

    def test_crop_background_removes_distant_points(self, noisy_cloud):
        from lithicore._photogrammetry import clean_point_cloud, _crop_background
        cropped = _crop_background(noisy_cloud, margin=1.5)
        # Distant outliers removed
        assert len(cropped) < len(noisy_cloud)
        assert len(cropped) >= 950
