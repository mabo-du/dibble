"""_photogrammetry.py — COLMAP-based photogrammetry pipeline (photos → 3D mesh).

exports: PhotogrammetryConfig
         PhotogrammetryResult
         PhotogrammetryError, ColmapNotFoundError, ColmapStageError,
           InsufficientPhotosError, PhotogrammetryCancelledError
         run_pipeline(config, progress_cb=None) -> PhotogrammetryResult
         colmap_available() -> bool
         clean_point_cloud(points, threshold=2.0) -> np.ndarray
         ProgressCallback
used_by: lithicore CLI, lithicope GUI
rules:   Zero GUI imports. Pure functions with dataclass configs.
         COLMAP is run via subprocess; pre/post processing is Python-native.
agent:   deepseek-v4-flash | 2026-05-26 | Initial scaffolding — dataclasses + error types
agent:   deepseek-v4-flash | 2026-05-26 | Added colmap_available() + clean_point_cloud() + _crop_background() with spatial-spread floor for robust SOR
agent:   deepseek-v4-flash | 2026-05-26 | Added ProgressCallback, _validate_inputs, _run_colmap_stage, run_pipeline with COLMAP subprocess orchestration
"""

from __future__ import annotations

import subprocess
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np


__all__ = [
    "PhotogrammetryConfig",
    "PhotogrammetryResult",
    "PhotogrammetryError",
    "ColmapNotFoundError",
    "ColmapStageError",
    "InsufficientPhotosError",
    "PhotogrammetryCancelledError",
    "colmap_available",
    "clean_point_cloud",
    "run_pipeline",
    "ProgressCallback",
]


# ──────────────────────────────────────────────
# Progress callback type
# ──────────────────────────────────────────────

ProgressCallback = Callable[[str, float, str], None]

# ──────────────────────────────────────────────
# Error types
# ──────────────────────────────────────────────

class PhotogrammetryError(Exception):
    """Base exception for photogrammetry pipeline errors."""

class ColmapNotFoundError(PhotogrammetryError):
    """COLMAP binary not found on system PATH."""

class ColmapStageError(PhotogrammetryError):
    """A COLMAP processing stage failed."""
    def __init__(self, stage: str, stderr: str) -> None:
        self.stage = stage
        self.stderr = stderr
        super().__init__(f"COLMAP {stage} failed: {stderr[:200]}")

class InsufficientPhotosError(PhotogrammetryError):
    """Fewer than minimum photos provided, or matching failed."""

class PhotogrammetryCancelledError(PhotogrammetryError):
    """Pipeline was cancelled by the user."""


# ──────────────────────────────────────────────
# COLMAP availability
# ──────────────────────────────────────────────

def colmap_available() -> bool:
    """Check whether COLMAP binary is available on PATH."""
    return shutil.which("colmap") is not None


# ──────────────────────────────────────────────
# Point cloud cleaning
# ──────────────────────────────────────────────

def clean_point_cloud(
    points: np.ndarray,
    threshold: float = 2.0,
) -> np.ndarray:
    """Remove statistical outliers from a point cloud.

    For each point, compute mean distance to k=20 nearest neighbours.
    Remove points where mean distance > global_mean + (threshold * effective_std),
    where effective_std = max(raw_stddev, point_spread * 0.35). The spatial-spread
    floor prevents over-aggressive removal in tightly clustered clouds where
    nearest-neighbour distance variance is small relative to cloud extent.

    Args:
        points: (N, 3) array of 3D points.
        threshold: Stddev multiplier for outlier cutoff. Default 2.0.

    Returns:
        Filtered (M, 3) array where M <= N.
    """
    from scipy.spatial import KDTree

    if len(points) < 21:
        return points

    tree = KDTree(points)
    # k=21 because the point itself is included in the count
    distances, _ = tree.query(points, k=min(21, len(points)))
    mean_distances = distances[:, 1:].mean(axis=1)  # exclude self-distance

    global_mean = mean_distances.mean()
    global_std = mean_distances.std()
    if global_std == 0:
        return points

    # Use spatial spread as a floor for the stddev to avoid
    # over-aggressive removal in tightly clustered clouds where
    # nearest-neighbour distance variance is small relative to
    # the overall extent of the point cloud.
    point_spread: float = float(np.linalg.norm(np.std(points, axis=0)))
    effective_std: float = max(global_std, point_spread * 0.35)

    mask = mean_distances <= global_mean + (threshold * effective_std)
    return points[mask]


def _crop_background(
    points: np.ndarray,
    margin: float = 1.5,
) -> np.ndarray:
    """Automatically crop background geometry from a dense point cloud.

    Computes a bounding box around the densest cluster (the artefact)
    and removes points outside margin * bounding_box.

    Args:
        points: (N, 3) array of 3D points.
        margin: Multiplier for bounding box extent. Default 1.5.

    Returns:
        Cropped (M, 3) array.
    """
    if len(points) < 10:
        return points

    # First remove statistical outliers to find the main cluster
    cleaned = clean_point_cloud(points, threshold=3.0)
    if len(cleaned) < 10:
        return points

    # Compute centroid and PCA from the cleaned subset (robust to outliers)
    centroid = cleaned.mean(axis=0)
    centered = cleaned - centroid

    cov = np.cov(centered.T)
    _, eigenvectors = np.linalg.eigh(cov)

    # Project cleaned points for bounding box estimation
    projected_cleaned = centered @ eigenvectors
    mins = projected_cleaned.min(axis=0)
    maxs = projected_cleaned.max(axis=0)
    extents = maxs - mins
    centre_pc = (mins + maxs) / 2

    # Apply bounding box + margin to ALL original points
    centered_all = points - centroid
    projected_all = centered_all @ eigenvectors
    half_extents = extents / 2 * margin
    mask = np.all(np.abs(projected_all - centre_pc) <= half_extents, axis=1)
    return points[mask]


# ──────────────────────────────────────────────
# Input validation
# ──────────────────────────────────────────────

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}


def _validate_inputs(config: PhotogrammetryConfig) -> int:
    """Validate photo folder and count.

    Returns:
        Number of valid photos found.

    Raises:
        InsufficientPhotosError if fewer than 3 valid photos found,
        or if photo folder does not exist.
    """
    if not config.photo_folder.is_dir():
        raise InsufficientPhotosError(
            f"Photo folder must be an existing directory: {config.photo_folder}"
        )

    photos = [
        p for p in sorted(config.photo_folder.iterdir())
        if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]

    if len(photos) < 3:
        raise InsufficientPhotosError(
            f"Found {len(photos)} valid photos in {config.photo_folder}. "
            "At least 3 photos are required for photogrammetry."
        )

    return len(photos)


# ──────────────────────────────────────────────
# Data types
# ──────────────────────────────────────────────

@dataclass
class PhotogrammetryConfig:
    """Configuration for the photogrammetry pipeline.

    mode determines which config fields are user-settable:
      - default: only photo_folder, output_path, artefact_label, quality
      - guided: plus photo_settings and cleanup fields
      - expert: all fields exposed
    """
    photo_folder: Path
    output_path: Path
    artefact_label: str = ""
    quality: str = "high"               # "low" | "medium" | "high"
    mode: str = "default"               # "default" | "guided" | "expert"

    # Guided/Expert: photo settings
    camera_model: str = "auto"          # "auto" | "smartphone" | "dslr"
    scale_bar_cm: float = 0.0           # 0 = no scale reference

    # Guided/Expert: cleanup
    auto_crop_background: bool = True
    fill_holes: bool = True

    # Expert-only: COLMAP tuning
    colmap_feature_type: str = "sift"
    colmap_matching_strategy: str = "exhaustive"
    colmap_meshing: str = "poisson"
    colmap_dense_quality: str = "extreme"
    max_vertices: int = 500000

    # Internal
    colmap_workspace: Optional[Path] = None
    cleanup_temp: bool = True

    def __post_init__(self) -> None:
        self.photo_folder = Path(self.photo_folder)
        self.output_path = Path(self.output_path)

    @property
    def target_faces(self) -> int:
        """Map quality slider to target face count after decimation."""
        mapping = {"low": 20_000, "medium": 50_000, "high": 150_000}
        try:
            return mapping[self.quality]
        except KeyError:
            raise ValueError(
                f"Unknown quality '{self.quality}'; expected 'low', 'medium', or 'high'"
            ) from None


@dataclass
class PhotogrammetryResult:
    """Output of a completed photogrammetry pipeline run.

    Note: camera_count is the number of input photos, not necessarily
    the number of successfully registered cameras. COLMAP may fail to
    register some images (blurry, no feature matches).
    """
    mesh_path: Path
    artefact_label: str
    camera_count: int
    point_count: int
    face_count: int
    vertex_count: int
    processing_time_s: float
    colmap_stdout: str
    warnings: list[str]
    sparse_cloud_path: Optional[Path] = None
    dense_cloud_path: Optional[Path] = None


# ──────────────────────────────────────────────
# COLMAP subprocess
# ──────────────────────────────────────────────

def _run_colmap_stage(
    stage_name: str,
    command_args: list[str],
    progress_cb: Optional[ProgressCallback],
    workspace: Path,
) -> str:
    """Run a single COLMAP stage as a subprocess.

    Args:
        stage_name: Human-readable stage name for progress/debugging.
        command_args: COLMAP subcommand and flags (without "colmap" prefix).
        progress_cb: Optional callback(stage, progress, message).
        workspace: Working directory for COLMAP.

    Returns:
        Captured stdout.

    Raises:
        ColmapNotFoundError if COLMAP binary is missing.
        ColmapStageError if COLMAP returns non-zero.
    """
    if not colmap_available():
        raise ColmapNotFoundError(
            "COLMAP is not installed or not on PATH. "
            "Install via: brew install colmap  (macOS)  |  "
            "sudo apt install colmap  (Linux)  |  "
            "conda install -c conda-forge colmap  (Conda)"
        )

    full_cmd = ["colmap"] + command_args

    if progress_cb:
        progress_cb(stage_name, 0.0, f"Starting {stage_name}...")

    try:
        proc = subprocess.run(
            full_cmd,
            capture_output=True,
            text=True,
            cwd=str(workspace),
        )
    except FileNotFoundError as exc:
        raise ColmapNotFoundError(
            "COLMAP binary not found. Tried running: " + " ".join(full_cmd)
        ) from exc

    if proc.returncode != 0:
        raise ColmapStageError(stage_name, proc.stderr)

    if progress_cb:
        progress_cb(stage_name, 1.0, f"{stage_name} complete")

    return proc.stdout


# ──────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────

def run_pipeline(
    config: PhotogrammetryConfig,
    progress_cb: Optional[ProgressCallback] = None,
) -> PhotogrammetryResult:
    """Run the full photogrammetry pipeline: photos → cleaned 3D mesh.

    Stages:
        1. Validate inputs
        2. COLMAP feature extraction
        3. COLMAP feature matching
        4. COLMAP sparse reconstruction
        5. COLMAP dense MVS (undistort → dense stereo → fusion)
        6. Point cloud cleaning (Python)
        7. Meshing (COLMAP Poisson or trimesh fallback)
        8. Decimation (if quality < high)
        9. Output

    Args:
        config: Pipeline configuration.
        progress_cb: Optional callback(stage, progress, message).

    Returns:
        PhotogrammetryResult.

    Raises:
        PhotogrammetryError subclasses on failure.
    """
    import time
    import trimesh

    start_time = time.time()
    warnings: list[str] = []

    # Stage 1: Validate
    photo_count = _validate_inputs(config)

    # Create workspace
    workspace = config.colmap_workspace or Path(
        tempfile.mkdtemp(prefix="dibble_photogrammetry_")
    )
    workspace.mkdir(parents=True, exist_ok=True)

    photo_dir = config.photo_folder.resolve()
    database_path = workspace / "database.db"
    sparse_path = workspace / "sparse"
    dense_path = workspace / "dense"
    fused_ply = workspace / "fused.ply"
    mesh_ply = workspace / "mesh.ply"

    colmap_stdout_parts: list[str] = []
    point_count = 0

    try:
        # Stage 2: Feature extraction
        stdout = _run_colmap_stage(
            "feature_extraction",
            [
                "feature_extractor",
                "--database_path", str(database_path),
                "--image_path", str(photo_dir),
                "--ImageReader.camera_model", "SIMPLE_RADIAL",
            ],
            progress_cb,
            workspace,
        )
        colmap_stdout_parts.append(f"=== feature_extractor ===\n{stdout}")

        # Stage 3: Feature matching
        match_cmd = "exhaustive_matcher"
        if config.colmap_matching_strategy == "sequential":
            match_cmd = "sequential_matcher"
        elif config.colmap_matching_strategy == "vocab_tree":
            match_cmd = "vocab_tree_matcher"

        stdout = _run_colmap_stage(
            "feature_matching",
            [match_cmd, "--database_path", str(database_path)],
            progress_cb,
            workspace,
        )
        colmap_stdout_parts.append(f"=== {match_cmd} ===\n{stdout}")

        # Stage 4: Sparse reconstruction
        stdout = _run_colmap_stage(
            "sparse_reconstruction",
            [
                "mapper",
                "--database_path", str(database_path),
                "--image_path", str(photo_dir),
                "--output_path", str(sparse_path),
            ],
            progress_cb,
            workspace,
        )
        colmap_stdout_parts.append(f"=== mapper ===\n{stdout}")

        # Determine sparse output directory (COLMAP creates a subdir "0")
        sparse_input = sparse_path / "0"
        if not sparse_input.exists():
            sparse_input = sparse_path

        # Stage 5: Dense MVS
        dense_images = dense_path / "images"
        dense_images.mkdir(parents=True, exist_ok=True)

        stdout = _run_colmap_stage(
            "dense_undistortion",
            [
                "image_undistorter",
                "--input_path", str(sparse_input),
                "--output_path", str(dense_path),
                "--output_type", "COLMAP",
            ],
            progress_cb,
            workspace,
        )
        colmap_stdout_parts.append(f"=== image_undistorter ===\n{stdout}")

        stdout = _run_colmap_stage(
            "dense_stereo",
            [
                "dense_stereo",
                "--workspace_path", str(dense_path),
                "--workspace_format", "COLMAP",
                "--PatchMatchStereo.geom_consistency", "true",
            ],
            progress_cb,
            workspace,
        )
        colmap_stdout_parts.append(f"=== dense_stereo ===\n{stdout}")

        stdout = _run_colmap_stage(
            "dense_fusion",
            [
                "stereo_fusion",
                "--workspace_path", str(dense_path),
                "--workspace_format", "COLMAP",
                "--output_path", str(fused_ply),
            ],
            progress_cb,
            workspace,
        )
        colmap_stdout_parts.append(f"=== stereo_fusion ===\n{stdout}")

        # Stage 6: Point cloud cleaning
        if progress_cb:
            progress_cb("cleaning", 0.0, "Cleaning point cloud...")

        cleaned_points = None
        if fused_ply.exists():
            cloud_mesh = trimesh.load(str(fused_ply))
            points = np.asarray(cloud_mesh.vertices)
            cleaned = clean_point_cloud(points, threshold=2.0)
            if config.auto_crop_background:
                cleaned = _crop_background(cleaned, margin=1.5)
            point_count = len(cleaned)
            cleaned_points = cleaned

            if progress_cb:
                progress_cb("cleaning", 1.0, f"Cleaned to {point_count} points")
        else:
            warnings.append("Dense point cloud not found; skipping cleaning")

        # Stage 7: Meshing
        if progress_cb:
            progress_cb("meshing", 0.0, "Generating mesh...")

        if fused_ply.exists() and config.colmap_meshing == "poisson":
            try:
                _run_colmap_stage(
                    "meshing",
                    [
                        "poisson_mesher",
                        "--input_path", str(fused_ply),
                        "--output_path", str(mesh_ply),
                    ],
                    progress_cb,
                    workspace,
                )
            except ColmapStageError as exc:
                warnings.append(f"Poisson meshing failed: {exc}. Using convex hull fallback.")
                if cleaned_points is not None and len(cleaned_points) > 3:
                    from trimesh.points import PointCloud
                    cloud = PointCloud(cleaned_points)
                    mesh = trimesh.creation.convex_hull(cloud)
                    mesh.export(str(mesh_ply))
        elif cleaned_points is not None and len(cleaned_points) > 3:
            # Delaunay or fallback — use convex hull
            from trimesh.points import PointCloud
            cloud = PointCloud(cleaned_points)
            mesh = trimesh.creation.convex_hull(cloud)
            mesh.export(str(mesh_ply))

        # Stage 8: Decimation + Stage 9: Output
        mesh_output = None
        if mesh_ply.exists():
            mesh_output = trimesh.load(str(mesh_ply))
            face_count = len(mesh_output.faces)
            vertex_count = len(mesh_output.vertices)

            if config.quality != "high" and face_count > config.target_faces:
                if progress_cb:
                    progress_cb("decimation", 0.0, f"Decimating {face_count} -> {config.target_faces} faces...")
                mesh_output = mesh_output.simplify_quadric_decimation(config.target_faces)
                face_count = len(mesh_output.faces)
                vertex_count = len(mesh_output.vertices)
        else:
            face_count = 0
            vertex_count = 0
            warnings.append("Meshing failed — creating placeholder mesh")
            from trimesh.creation import box
            mesh_output = box(extents=[10, 10, 10])

        # Write output
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        mesh_output.export(str(config.output_path))

        elapsed = time.time() - start_time

        # Find sparse cloud
        sparse_cloud = None
        for candidate in [sparse_path / "0" / "points3D.bin", sparse_path / "0" / "points3D.txt"]:
            if candidate.exists():
                sparse_cloud = candidate
                break

        result = PhotogrammetryResult(
            mesh_path=config.output_path.resolve(),
            artefact_label=config.artefact_label,
            camera_count=photo_count,
            point_count=point_count,
            face_count=face_count,
            vertex_count=vertex_count,
            processing_time_s=round(elapsed, 1),
            colmap_stdout="\n".join(colmap_stdout_parts),
            warnings=warnings,
            sparse_cloud_path=sparse_cloud,
            dense_cloud_path=fused_ply if fused_ply.exists() else None,
        )

        return result

    finally:
        # Clean up temp workspace
        if config.cleanup_temp and config.colmap_workspace is None:
            import shutil as _shutil
            _shutil.rmtree(workspace, ignore_errors=True)
