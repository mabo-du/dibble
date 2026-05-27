"""test_photo_preprocessing.py — Unit tests for photo pre-processing.

exports: TestPreprocessingConfig
         TestComputeLaplacianVariance
         TestComputeBlurScores
         TestPreprocessPhotos
used_by: pytest
rules:   Synthetic images only. No real photos required.
         OpenCV must be available (already a project dependency).
agent:   deepseek-v4-flash | 2026-05-27 | Initial implementation
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from lithicore._photo_preprocessing import (
    PreprocessingConfig,
    PreprocessingResult,
    compute_laplacian_variance,
    compute_blur_scores,
    preprocess_photos,
)


# ── Helpers ──


def _create_sharp_image(size: tuple[int, int] = (200, 200)) -> np.ndarray:
    """Create a synthetic sharp image with high-frequency content (noise + edges)."""
    rng = np.random.default_rng(42)
    img = rng.integers(0, 255, size=(*size, 3), dtype=np.uint8)
    # Add sharp edges for high Laplacian response
    img[50:60, :] = 255
    img[:, 80:90] = 0
    img[120:130, :] = 0
    return img


def _create_blurry_image(size: tuple[int, int] = (200, 200)) -> np.ndarray:
    """Create a synthetic blurry image using heavy Gaussian blur."""
    sharp = _create_sharp_image(size)
    return cv2.GaussianBlur(sharp, (31, 31), 15.0)


def _create_uniform_image(size: tuple[int, int] = (200, 200)) -> np.ndarray:
    """Create a completely uniform image (zero Laplacian response)."""
    return np.full((*size, 3), 128, dtype=np.uint8)


# ── Tests ──


class TestPreprocessingConfig:
    """PreprocessingConfig dataclass construction."""

    def test_defaults(self):
        cfg = PreprocessingConfig()
        assert cfg.blur_threshold == 100.0
        assert cfg.auto_remove_blurry is True
        assert cfg.normalize_exposure is True
        assert cfg.color_calibration is False
        assert cfg.max_image_dimension == 0

    def test_custom_values(self):
        cfg = PreprocessingConfig(
            blur_threshold=50.0,
            auto_remove_blurry=False,
            normalize_exposure=False,
            color_calibration=True,
            max_image_dimension=2000,
        )
        assert cfg.blur_threshold == 50.0
        assert cfg.auto_remove_blurry is False
        assert cfg.normalize_exposure is False
        assert cfg.color_calibration is True
        assert cfg.max_image_dimension == 2000


class TestComputeLaplacianVariance:
    """Laplacian variance blur detection."""

    def test_sharp_image_high_score(self, tmp_path):
        """Sharp images should have high Laplacian variance."""
        img = _create_sharp_image((100, 100))
        path = tmp_path / "sharp.png"
        cv2.imwrite(str(path), img)
        score = compute_laplacian_variance(path)
        assert score > 100, f"Expected sharp score > 100, got {score}"

    def test_blurry_image_low_score(self, tmp_path):
        """Blurry images should have low Laplacian variance."""
        img = _create_blurry_image((100, 100))
        path = tmp_path / "blurry.png"
        cv2.imwrite(str(path), img)
        score = compute_laplacian_variance(path)
        assert score < 50, f"Expected blurry score < 50, got {score}"

    def test_uniform_image_zero_score(self, tmp_path):
        """Uniform images should have near-zero Laplacian variance."""
        img = _create_uniform_image((100, 100))
        path = tmp_path / "uniform.png"
        cv2.imwrite(str(path), img)
        score = compute_laplacian_variance(path)
        assert score < 1.0, f"Expected uniform score ~0, got {score}"

    def test_sharp_vs_blurry_separation(self, tmp_path):
        """Sharp image score should be significantly higher than blurry."""
        sharp = _create_sharp_image((100, 100))
        blurry = _create_blurry_image((100, 100))
        sharp_path = tmp_path / "sharp.png"
        blurry_path = tmp_path / "blurry.png"
        cv2.imwrite(str(sharp_path), sharp)
        cv2.imwrite(str(blurry_path), blurry)
        sharp_score = compute_laplacian_variance(sharp_path)
        blurry_score = compute_laplacian_variance(blurry_path)
        assert sharp_score > blurry_score * 3, (
            f"Sharp ({sharp_score:.1f}) should be >> blurry ({blurry_score:.1f})"
        )

    def test_unreadable_file_raises(self, tmp_path):
        """Non-image file should raise ValueError."""
        path = tmp_path / "not_an_image.txt"
        path.write_text("not an image")
        with pytest.raises(ValueError, match="Cannot read image"):
            compute_laplacian_variance(path)

    def test_missing_file_raises(self):
        """Non-existent file should raise."""
        with pytest.raises(ValueError, match="Cannot read image"):
            compute_laplacian_variance(Path("/nonexistent/image.png"))


class TestComputeBlurScores:
    """Batch blur scoring."""

    def test_scores_all_images(self, tmp_path):
        """Should compute scores for all supported images."""
        for name in ("a.jpg", "b.png", "c.tiff"):
            img = _create_sharp_image((50, 50))
            cv2.imwrite(str(tmp_path / name), img)
        scores = compute_blur_scores(tmp_path)
        assert len(scores) == 3
        for name in ("a.jpg", "b.png", "c.tiff"):
            assert name in scores

    def test_skips_unsupported_extensions(self, tmp_path):
        """Should ignore non-image files."""
        (tmp_path / "readme.txt").write_text("hello")
        img = _create_sharp_image((50, 50))
        cv2.imwrite(str(tmp_path / "photo.jpg"), img)
        scores = compute_blur_scores(tmp_path)
        assert len(scores) == 1
        assert "readme.txt" not in scores

    def test_missing_folder_raises(self):
        """Non-existent folder should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Photo folder does not exist"):
            compute_blur_scores(Path("/nonexistent/folder"))


class TestPreprocessPhotos:
    """Full pre-processing pipeline."""

    def test_accepts_sharp_photos(self, tmp_path):
        """Sharp photos should pass through pre-processing."""
        photo_dir = tmp_path / "photos"
        photo_dir.mkdir()
        for i in range(5):
            img = _create_sharp_image((100, 100))
            cv2.imwrite(str(photo_dir / f"photo_{i}.jpg"), img)

        result = preprocess_photos(
            photo_dir,
            PreprocessingConfig(blur_threshold=100.0),
            output_dir=tmp_path / "out",
        )
        assert result.total_photos == 5
        assert result.accepted_photos == 5
        assert len(result.rejected_photos) == 0
        assert result.processed_dir.exists()

    def test_rejects_blurry_photos(self, tmp_path):
        """Blurry photos should be rejected."""
        photo_dir = tmp_path / "photos"
        photo_dir.mkdir()
        for i in range(3):
            img = _create_sharp_image((100, 100))
            cv2.imwrite(str(photo_dir / f"sharp_{i}.jpg"), img)
        for i in range(2):
            img = _create_blurry_image((100, 100))
            cv2.imwrite(str(photo_dir / f"blurry_{i}.jpg"), img)

        result = preprocess_photos(
            photo_dir,
            PreprocessingConfig(blur_threshold=100.0),
            output_dir=tmp_path / "out",
        )
        assert result.total_photos == 5
        assert result.accepted_photos == 3
        assert len(result.rejected_photos) == 2

    def test_rejected_folder_created(self, tmp_path):
        """Rejected photos should be copied to rejected/ subfolder."""
        photo_dir = tmp_path / "photos"
        photo_dir.mkdir()
        for i in range(3):
            img = _create_sharp_image((100, 100))
            cv2.imwrite(str(photo_dir / f"sharp_{i}.jpg"), img)
        img = _create_blurry_image((100, 100))
        cv2.imwrite(str(photo_dir / "blurry.jpg"), img)

        result = preprocess_photos(
            photo_dir,
            PreprocessingConfig(blur_threshold=100.0),
            output_dir=tmp_path / "out",
        )
        rejected_dir = tmp_path / "out" / "rejected"
        assert rejected_dir.exists()
        assert len(list(rejected_dir.iterdir())) == 1

    def test_auto_recover_when_few_photos(self, tmp_path):
        """When too few photos pass, should auto-recover the least-blurry ones."""
        photo_dir = tmp_path / "photos"
        photo_dir.mkdir()
        for i in range(2):
            img = _create_sharp_image((100, 100))
            cv2.imwrite(str(photo_dir / f"sharp_{i}.jpg"), img)
        img = _create_blurry_image((100, 100))
        cv2.imwrite(str(photo_dir / "blurry.jpg"), img)

        result = preprocess_photos(
            photo_dir,
            PreprocessingConfig(blur_threshold=100.0),
            output_dir=tmp_path / "out",
        )
        # Should have recovered at least 3 photos total
        assert result.accepted_photos >= 3

    def test_blur_skipped_when_disabled(self, tmp_path):
        """When auto_remove_blurry is False, all photos should pass."""
        photo_dir = tmp_path / "photos"
        photo_dir.mkdir()
        for i in range(3):
            img = _create_blurry_image((100, 100))
            cv2.imwrite(str(photo_dir / f"blurry_{i}.jpg"), img)

        result = preprocess_photos(
            photo_dir,
            PreprocessingConfig(blur_threshold=100.0, auto_remove_blurry=False),
            output_dir=tmp_path / "out",
        )
        assert result.accepted_photos == 3
        assert len(result.rejected_photos) == 0

    def test_exposure_normalization_produces_output(self, tmp_path):
        """Processed images should exist after exposure normalisation."""
        photo_dir = tmp_path / "photos"
        photo_dir.mkdir()
        for i in range(3):
            img = _create_sharp_image((100, 100))
            cv2.imwrite(str(photo_dir / f"photo_{i}.jpg"), img)

        result = preprocess_photos(
            photo_dir,
            PreprocessingConfig(normalize_exposure=True),
            output_dir=tmp_path / "out",
        )
        processed_files = list(result.processed_dir.iterdir())
        assert len(processed_files) == 3

    def test_resize_works(self, tmp_path):
        """Images should be resized when max_image_dimension is set."""
        img = _create_sharp_image((400, 200))
        photo_dir = tmp_path / "photos"
        photo_dir.mkdir()
        cv2.imwrite(str(photo_dir / "large.jpg"), img)

        result = preprocess_photos(
            photo_dir,
            PreprocessingConfig(max_image_dimension=100),
            output_dir=tmp_path / "out",
        )
        processed = list(result.processed_dir.iterdir())[0]
        loaded = cv2.imread(str(processed))
        h, w = loaded.shape[:2]
        assert max(h, w) <= 100

    def test_empty_folder_raises(self, tmp_path):
        """Empty folder should raise FileNotFoundError."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="No supported images found"):
            preprocess_photos(
                empty_dir,
                PreprocessingConfig(),
                output_dir=tmp_path / "out",
            )

    def test_non_existent_folder_raises(self, tmp_path):
        """Non-existent folder should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Photo folder does not exist"):
            preprocess_photos(
                tmp_path / "nonexistent",
                PreprocessingConfig(),
            )

    def test_blur_scores_in_result(self, tmp_path):
        """Result should include blur scores for all photos."""
        photo_dir = tmp_path / "photos"
        photo_dir.mkdir()
        for i in range(3):
            img = _create_sharp_image((100, 100))
            cv2.imwrite(str(photo_dir / f"photo_{i}.jpg"), img)

        result = preprocess_photos(
            photo_dir,
            PreprocessingConfig(),
            output_dir=tmp_path / "out",
        )
        assert len(result.blur_scores) == 3
        for name, score in result.blur_scores.items():
            assert score > 0

    def test_warnings_included(self, tmp_path):
        """Warnings about blurry photo removal should be in result."""
        photo_dir = tmp_path / "photos"
        photo_dir.mkdir()
        for i in range(3):
            img = _create_sharp_image((100, 100))
            cv2.imwrite(str(photo_dir / f"sharp_{i}.jpg"), img)
        img = _create_blurry_image((100, 100))
        cv2.imwrite(str(photo_dir / "blurry.jpg"), img)

        result = preprocess_photos(
            photo_dir,
            PreprocessingConfig(blur_threshold=100.0),
            output_dir=tmp_path / "out",
        )
        assert any("blurry" in w.lower() for w in result.warnings)

    def test_accepted_dir_only_contains_accepted(self, tmp_path):
        """Only accepted photos should be in the processed dir."""
        photo_dir = tmp_path / "photos"
        photo_dir.mkdir()
        for i in range(3):
            img = _create_sharp_image((100, 100))
            cv2.imwrite(str(photo_dir / f"sharp_{i}.jpg"), img)
        img = _create_blurry_image((100, 100))
        cv2.imwrite(str(photo_dir / "blurry.jpg"), img)

        result = preprocess_photos(
            photo_dir,
            PreprocessingConfig(blur_threshold=100.0),
            output_dir=tmp_path / "out",
        )
        accepted_names = {p.name for p in result.processed_dir.iterdir()}
        assert "blurry.jpg" not in accepted_names
        for i in range(3):
            assert f"sharp_{i}.jpg" in accepted_names
