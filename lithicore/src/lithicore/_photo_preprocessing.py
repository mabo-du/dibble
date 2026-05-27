"""_photo_preprocessing.py — Photo pre-processing before COLMAP pipeline.

Analyses and cleans raw photographs before photogrammetric reconstruction:
  1. Blur detection via Laplacian variance (rejects blurry images)
  2. Exposure normalisation via CLAHE on LAB colour space (consistent lighting)
  3. Optional colour calibration via ColorChecker detection (guided/expert mode)

The module operates on copies — originals are never modified. Rejected
blurry images are moved to a "rejected" subfolder in the processed output
so the user can inspect what was filtered.

exports: PreprocessingConfig
         PreprocessingResult
         preprocess_photos(photo_folder, config, progress_cb) -> PreprocessingResult
used_by: lithicore photogrammetry pipeline
rules:   Pure image-processing functions, no GUI dependencies.
         OpenCV must be available (already a project dependency).
         Creates processed copies; never modifies originals.
agent:   deepseek-v4-flash | 2026-05-27 | Initial implementation
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}


@dataclass
class PreprocessingConfig:
    """Configuration for the photo pre-processing stage.

    Attributes:
        blur_threshold: Minimum Laplacian variance for an image to be
            considered sharp. Lower values are more tolerant. Default 100.0.
            Typical ranges: <50 = very blurry, 50-100 = blurry,
            100-200 = acceptable, 200+ = sharp.
        auto_remove_blurry: If True, images below blur_threshold are
            moved to a "rejected" subfolder and excluded from reconstruction.
            If False, all images are kept regardless.
        normalize_exposure: If True, applies CLAHE equalisation to the
            L channel in LAB colour space for consistent brightness across
            the image set. Default True.
        color_calibration: If True, attempts to detect an X-Rite
            ColorChecker Classic in the first image and applies the derived
            colour correction matrix to all images. Experimental. Default False.
        max_image_dimension: Maximum pixel dimension for the longer edge.
            Images larger than this are resized (preserving aspect ratio)
            to reduce COLMAP processing time. 0 = no resize. Default 0.
    """
    blur_threshold: float = 100.0
    auto_remove_blurry: bool = True
    normalize_exposure: bool = True
    color_calibration: bool = False
    max_image_dimension: int = 0


@dataclass
class PreprocessingResult:
    """Result of the photo pre-processing stage.

    Attributes:
        processed_dir: Directory containing pre-processed photos.
            This should be passed to COLMAP as the image input.
        total_photos: Number of input photos found.
        accepted_photos: Number of photos passing quality checks.
        rejected_photos: List of filenames that were rejected as blurry.
        blur_scores: Dict mapping filename -> Laplacian variance score.
        processing_time_s: Wall-clock time for pre-processing.
        warnings: Non-fatal issues encountered during processing.
    """
    processed_dir: Path
    total_photos: int
    accepted_photos: int
    rejected_photos: list[str] = field(default_factory=list)
    blur_scores: dict[str, float] = field(default_factory=dict)
    processing_time_s: float = 0.0
    warnings: list[str] = field(default_factory=list)


ProgressCallback = Callable[[str, float, str], None]


# ──────────────────────────────────────────────
# Sharpness / blur detection
# ──────────────────────────────────────────────


def compute_laplacian_variance(image_path: Path) -> float:
    """Compute the variance of the Laplacian as a blur/sharpness metric.

    A higher variance indicates a sharper image. The metric is based on the
    second derivative (Laplacian) of the grayscale image; blurry images have
    low variation in their gradient response.

    Args:
        image_path: Path to an image file readable by OpenCV.

    Returns:
        Laplacian variance (float). Typical ranges:
            < 50   — very blurry (motion blur, severe defocus)
            50-100 — blurry (acceptable only as a last resort)
            100-200 — acceptable (most field photography)
            200+   — sharp (tripod, controlled lighting)

    Raises:
        ValueError: If the image cannot be read by OpenCV.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def compute_blur_scores(
    photo_folder: Path,
    extensions: set[str] = SUPPORTED_EXTENSIONS,
) -> dict[str, float]:
    """Compute Laplacian blur scores for all supported images in a folder.

    Args:
        photo_folder: Directory containing photo files.
        extensions: Set of valid file extensions (case-insensitive).

    Returns:
        Dict mapping filename -> blur score.

    Raises:
        FileNotFoundError: If photo_folder does not exist.
    """
    if not photo_folder.is_dir():
        raise FileNotFoundError(f"Photo folder does not exist: {photo_folder}")

    scores: dict[str, float] = {}
    for path in sorted(photo_folder.iterdir()):
        if path.suffix.lower() not in extensions:
            continue
        try:
            scores[path.name] = compute_laplacian_variance(path)
        except (ValueError, cv2.error) as exc:
            # Skip unreadable files but record them
            scores[path.name] = 0.0

    return scores


# ──────────────────────────────────────────────
# Exposure normalisation
# ──────────────────────────────────────────────


def _normalize_exposure(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE equalisation to the L channel of LAB colour space.

    Converts BGR → LAB, applies CLAHE to the L (lightness) channel,
    then converts back to BGR. This normalises brightness while preserving
    colour relationships, resulting in consistent exposure across a photo set.

    Args:
        image: BGR image array (H, W, 3) as read by cv2.imread().

    Returns:
        Exposure-normalised BGR image array (same dtype and shape).
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # CLAHE with clip limit 2.0 and tile grid 8x8 — conservative default
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_channel)

    merged = cv2.merge([l_eq, a_channel, b_channel])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# ──────────────────────────────────────────────
# Colour calibration (ColorChecker)
# ──────────────────────────────────────────────


def _detect_color_checker(image: np.ndarray) -> Optional[np.ndarray]:
    """Attempt to detect an X-Rite ColorChecker Classic in the image.

    Uses OpenCV's mcc module to locate the 24-patch colour target and
    compute a 3x3 colour correction matrix (CCM).

    Args:
        image: BGR image array.

    Returns:
        3x3 colour correction matrix as ndarray, or None if not detected.
    """
    try:
        # OpenCV 4.x: mcc module for ColorChecker detection
        detector = cv2.mcc.CCheckerDetector_create()
        detected = detector.process(image, cv2.mcc.MCC24, 1)
        if not detected:
            return None

        checkers = detector.getListColorChecker()
        if not checkers:
            return None

        # Build colour correction model from the first detected checker
        ccm_model = cv2.ccm.ColorCorrectionModel(
            checkers[1].getChartsRGB(),  # observed RGB values
            cv2.ccm.COLOR_SPACE_sRGB,
        )
        ccm_model.run()
        return ccm_model.getCCM()  # 3x3 correction matrix

    except (AttributeError, cv2.error, Exception):
        # mcc module may not be available in all OpenCV builds;
        # fall through gracefully.
        return None


def _apply_ccm(image: np.ndarray, ccm: np.ndarray) -> np.ndarray:
    """Apply a 3x3 colour correction matrix to an image.

    Args:
        image: BGR image array (H, W, 3), uint8.
        ccm: 3x3 colour correction matrix.

    Returns:
        Colour-corrected BGR image array.
    """
    img_float = image.astype(np.float32)
    corrected = img_float @ ccm.T
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return corrected


# ──────────────────────────────────────────────
# Image resizing
# ──────────────────────────────────────────────


def _maybe_resize(image: np.ndarray, max_dim: int) -> np.ndarray:
    """Downscale an image if the longer edge exceeds max_dim.

    Preserves aspect ratio. No upscaling.

    Args:
        image: BGR image array.
        max_dim: Maximum pixel length for the longer edge. 0 = no resize.

    Returns:
        Resized (or original) image array.
    """
    if max_dim <= 0:
        return image

    height, width = image.shape[:2]
    longer = max(height, width)
    if longer <= max_dim:
        return image

    scale = max_dim / longer
    new_width = int(width * scale)
    new_height = int(height * scale)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


# ──────────────────────────────────────────────
# Main pre-processing orchestrator
# ──────────────────────────────────────────────


def preprocess_photos(
    photo_folder: Path,
    config: PreprocessingConfig,
    output_dir: Optional[Path] = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> PreprocessingResult:
    """Run the full photo pre-processing pipeline.

    Stages:
        1. Scan input directory for supported images
        2. Compute blur scores for all images (Laplacian variance)
        3. Reject blurry images if auto_remove_blurry is enabled
        4. Apply exposure normalisation (CLAHE) if enabled
        5. Apply colour calibration if enabled and ColorChecker detected
        6. Resize oversized images if max_image_dimension set
        7. Write processed copies to output directory

    Original files are never modified. Rejected blurry images are copied
    to a ``rejected/`` subfolder for user inspection.

    Args:
        photo_folder: Directory containing source photos.
        config: Pre-processing configuration.
        output_dir: Directory for processed copies. If None, a temp
            directory is created alongside photo_folder.
        progress_cb: Optional progress callback (stage, progress, message).

    Returns:
        PreprocessingResult with processed_dir pointing to clean photos.

    Raises:
        FileNotFoundError: If photo_folder does not exist.
    """
    start_time = time.time()
    warnings: list[str] = []

    # Resolve input
    if not photo_folder.is_dir():
        raise FileNotFoundError(f"Photo folder does not exist: {photo_folder}")

    photo_folder = photo_folder.resolve()

    # Create output directory
    if output_dir is None:
        output_dir = photo_folder.parent / f"{photo_folder.name}_preprocessed"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rejected_dir = output_dir / "rejected"
    accepted_dir = output_dir / "accepted"
    accepted_dir.mkdir(parents=True, exist_ok=True)

    if progress_cb:
        progress_cb("preprocessing", 0.0, "Scanning photos...")

    # Stage 1: List input photos
    photo_paths: list[Path] = sorted(
        p for p in photo_folder.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not photo_paths:
        raise FileNotFoundError(
            f"No supported images found in {photo_folder} "
            f"(supported: {', '.join(SUPPORTED_EXTENSIONS)})"
        )

    total = len(photo_paths)
    if progress_cb:
        progress_cb("preprocessing", 0.05, f"Found {total} photos")

    # Stage 2: Blur scoring
    if progress_cb:
        progress_cb("preprocessing", 0.1, "Computing sharpness scores...")

    blur_scores = compute_blur_scores(photo_folder)
    accepted: list[Path] = []
    rejected_names: list[str] = []

    for p in photo_paths:
        score = blur_scores.get(p.name, 0.0)
        if config.auto_remove_blurry and score < config.blur_threshold:
            rejected_names.append(p.name)
        else:
            accepted.append(p)

    n_rejected = len(rejected_names)
    n_accepted = len(accepted)

    if n_rejected > 0 and config.auto_remove_blurry:
        warnings.append(
            f"Removed {n_rejected} blurry photo(s) "
            f"(Laplacian variance < {config.blur_threshold})"
        )
        if n_accepted < 3:
            # Too few photos for reconstruction — lower threshold automatically
            warnings.append(
                f"Only {n_accepted} photos remain after blur rejection "
                f"(need ≥3). Relaxing threshold to keep enough photos."
            )
            accepted = [p for p in photo_paths if p not in rejected_names]
            # Re-add the least blurry rejected images
            rejected_sorted = sorted(
                rejected_names,
                key=lambda n: blur_scores.get(n, 0.0),
                reverse=True,
            )
            needed = 3 - n_accepted
            for name in rejected_sorted[:needed]:
                p = photo_folder / name
                if p in photo_paths:
                    accepted.append(p)
                    rejected_names.remove(name)
                    n_rejected -= 1
                    n_accepted += 1
            warnings.append(
                f"Recovered {needed} least-blurry photo(s) to meet minimum count."
            )

    # Stage 3-6: Process each image
    color_correction_matrix: Optional[np.ndarray] = None
    ccm_derived = False

    if config.color_calibration:
        if progress_cb:
            progress_cb("preprocessing", 0.2, "Detecting colour reference...")

        # Try to find ColorChecker in the first accepted image
        first_accepted = accepted[0] if accepted else photo_paths[0]
        img_ref = cv2.imread(str(first_accepted))
        if img_ref is not None:
            ccm_matrix = _detect_color_checker(img_ref)
            if ccm_matrix is not None:
                color_correction_matrix = ccm_matrix
                ccm_derived = True
                warnings.append(
                    f"Colour calibration applied using ColorChecker "
                    f"from {first_accepted.name}"
                )
            else:
                warnings.append(
                    "ColorChecker detection requested but not found in any image."
                )

    if progress_cb:
        progress_cb("preprocessing", 0.3, "Processing photos...")

    for i, src_path in enumerate(accepted):
        if progress_cb and i % max(1, len(accepted) // 5) == 0:
            progress_cb(
                "preprocessing",
                0.3 + 0.6 * (i / max(1, len(accepted))),
                f"Processing {src_path.name} ({i+1}/{len(accepted)})...",
            )

        img = cv2.imread(str(src_path))
        if img is None:
            warnings.append(f"Cannot read {src_path.name} — skipping")
            continue

        img_processed = img.copy()

        # Exposure normalisation
        if config.normalize_exposure:
            img_processed = _normalize_exposure(img_processed)

        # Colour correction
        if config.color_calibration and color_correction_matrix is not None:
            img_processed = _apply_ccm(img_processed, color_correction_matrix)

        # Resize
        if config.max_image_dimension > 0:
            img_processed = _maybe_resize(img_processed, config.max_image_dimension)

        # Write processed copy
        dst_path = accepted_dir / src_path.name
        cv2.imwrite(str(dst_path), img_processed)

    # Write rejected photos to rejected/ subfolder (for inspection)
    if rejected_names and config.auto_remove_blurry:
        rejected_dir.mkdir(parents=True, exist_ok=True)
        for name in rejected_names:
            src = photo_folder / name
            if src.exists():
                shutil.copy2(str(src), str(rejected_dir / name))

    elapsed = time.time() - start_time

    if progress_cb:
        progress_cb(
            "preprocessing", 1.0,
            f"Done — {n_accepted} accepted, {n_rejected} rejected "
            f"({elapsed:.1f}s)",
        )

    return PreprocessingResult(
        processed_dir=accepted_dir,
        total_photos=total,
        accepted_photos=n_accepted,
        rejected_photos=rejected_names,
        blur_scores=blur_scores,
        processing_time_s=round(elapsed, 1),
        warnings=warnings,
    )
