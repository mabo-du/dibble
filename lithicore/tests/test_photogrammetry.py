"""tests/test_photogrammetry.py — Unit tests for photogrammetry pipeline.

exports: test_photogrammetry_config_defaults
         test_photogrammetry_config_target_faces
         test_photogrammetry_result_fields
used_by: pytest
rules:   COLMAP integration tests are marked @pytest.mark.skipif(colmap_missing).
         Unit tests should never require COLMAP.
agent:   deepseek-v4-flash | 2026-05-26 | Initial test skeleton
"""

from pathlib import Path
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
        with pytest.raises(KeyError):
            _ = config.target_faces

    def test_mode_validates_expert_fields_present(self):
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
