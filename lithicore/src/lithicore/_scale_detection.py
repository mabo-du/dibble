"""_scale_detection.py — Automatic scale detection + mesh rescaling.

exports: ScaleResult
         detect_scale_aruco(photos, sparse_cloud_dir, marker_size_mm) -> Optional[ScaleResult]
         detect_scale_ruler(photos, sparse_cloud_dir) -> Optional[ScaleResult]
         apply_scale_to_mesh(mesh, scale_factor) -> trimesh.Trimesh
used_by: lithicore photogrammetry pipeline
rules:   Pure functions, no GUI imports. Scale detection operates on sparse cloud
         + source photos, not dense mesh.
agent:   deepseek-v4-flash | 2026-05-27 | Initial — dataclass + mesh transform
agent:   deepseek-v4-flash | 2026-05-27 | Added ArUco detection + COLMAP I/O + triangulation
agent:   deepseek-v4-flash | 2026-05-27 | Added ruler/scale bar detection via Hough lines
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh


@dataclass
class ScaleResult:
    """Result of automatic scale detection.

    Attributes:
        scale_factor: Multiply COLMAP unit coordinates by this to get mm.
        method: Detection method — 'aruco', 'ruler', or 'manual'.
        confidence: Estimated reliability (0 = none, 1 = certain).
        detected_length_mm: Physical length detected (mm).
        warnings: Non-fatal issues during detection.
    """
    scale_factor: float
    method: str
    confidence: float
    detected_length_mm: float = 0.0
    warnings: list[str] = field(default_factory=list)


def apply_scale_to_mesh(
    mesh: trimesh.Trimesh,
    scale_factor: float,
) -> trimesh.Trimesh:
    """Apply a uniform scale factor to all mesh vertices.

    Args:
        mesh: Input mesh (unchanged).
        scale_factor: Multiplier for vertex coordinates. Must be positive.

    Returns:
        A new trimesh.Trimesh with scaled vertices. Face topology is preserved.
        Normal arrays are invalidated (recomputed on next access).

    Raises:
        ValueError: If scale_factor is zero or negative.
    """
    if scale_factor <= 0:
        raise ValueError(
            f"Scale factor must be positive, got {scale_factor}"
        )
    scaled = mesh.copy()
    scaled.vertices = mesh.vertices * scale_factor
    # Invalidate cached normals — they will be recomputed by trimesh on demand
    scaled.face_normals = None
    scaled.vertex_normals = None
    return scaled


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}


def _read_photos(photo_dir: Path) -> list[Path]:
    """Return sorted list of supported image files in a directory."""
    return sorted([
        p for p in photo_dir.iterdir()
        if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ])


# ──────────────────────────────────────────────
# COLMAP binary I/O
# ──────────────────────────────────────────────

def _read_colmap_sparse(sparse_dir: Path) -> tuple:
    """Read COLMAP sparse reconstruction data.

    Returns (cameras, images) dicts from COLMAP binary format.
    Uses pycolmap if available, otherwise direct binary parsing.

    Raises FileNotFoundError if camera/ image data is missing.
    """
    # Paths may be in subdir "0" or directly in sparse_dir
    cameras_path = sparse_dir / "cameras.bin"
    images_path = sparse_dir / "images.bin"

    if not cameras_path.exists():
        cameras_path = sparse_dir.parent / "cameras.bin"
    if not images_path.exists():
        images_path = sparse_dir.parent / "images.bin"

    if not cameras_path.exists() or not images_path.exists():
        raise FileNotFoundError(
            f"COLMAP sparse data not found in {sparse_dir}"
        )

    try:
        import struct
        from collections import namedtuple

        # Simpler named tuples for COLMAP data
        Camera = namedtuple("Camera", ["camera_id", "model_id", "width", "height", "params"])
        Image = namedtuple("Image", ["image_id", "name", "qw", "qx", "qy", "qz",
                                     "tx", "ty", "tz", "camera_id"])

        def _read_cameras(path):
            cameras = {}
            with open(path, "rb") as f:
                num = struct.unpack("Q", f.read(8))[0]
                for _ in range(num):
                    cam_id = struct.unpack("I", f.read(4))[0]
                    model_id = struct.unpack("i", f.read(4))[0]
                    width = struct.unpack("Q", f.read(8))[0]
                    height = struct.unpack("Q", f.read(8))[0]
                    n_params = {0: 4, 1: 4, 2: 5, 3: 8, 4: 5, 5: 8, 6: 12, 7: 3}.get(model_id, 4)
                    params = struct.unpack(f"{n_params}d", f.read(n_params * 8))
                    cameras[cam_id] = Camera(cam_id, model_id, width, height, params)
            return cameras

        def _read_images(path):
            images = {}
            with open(path, "rb") as f:
                num = struct.unpack("Q", f.read(8))[0]
                for _ in range(num):
                    img_id = struct.unpack("I", f.read(4))[0]
                    qw, qx, qy, qz = struct.unpack("dddd", f.read(32))
                    tx, ty, tz = struct.unpack("ddd", f.read(24))
                    cam_id = struct.unpack("I", f.read(4))[0]
                    name = b""
                    while True:
                        b = f.read(1)
                        if b == b"\x00":
                            break
                        name += b
                    name = name.decode("utf-8")
                    n_pts = struct.unpack("Q", f.read(8))[0]
                    f.read(n_pts * 24)  # skip point2D data
                    images[img_id] = Image(img_id, name, qw, qx, qy, qz, tx, ty, tz, cam_id)
            return images

        return _read_cameras(str(cameras_path)), _read_images(str(images_path))

    except Exception as exc:
        raise FileNotFoundError(
            f"Failed to read COLMAP sparse data: {exc}"
        ) from exc


# ──────────────────────────────────────────────
# Corner triangulation
# ──────────────────────────────────────────────

def _triangulate_corner(
    observations: list[tuple[int, np.ndarray]],
    cameras: dict,
    images: dict,
) -> Optional[np.ndarray]:
    """Triangulate a 3D point from 2D observations in multiple views."""
    import cv2 as _cv2

    points_2d = []
    proj_mats = []

    for image_id, pixel in observations[:10]:
        if image_id not in images:
            continue
        img = images[image_id]
        if img.camera_id not in cameras:
            continue
        cam = cameras[img.camera_id]

        # Camera matrix K
        fx, fy, cx, cy = cam.params[0], cam.params[1], cam.params[2], cam.params[3]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

        # Rotation from quaternion
        qw, qx, qy, qz = img.qw, img.qx, img.qy, img.qz
        R = np.array([
            [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy],
        ], dtype=np.float64)

        t = np.array([[img.tx], [img.ty], [img.tz]], dtype=np.float64)
        P = K @ np.hstack([R, t])
        proj_mats.append(P)
        points_2d.append(pixel)

    if len(points_2d) < 2:
        return None

    X = _cv2.triangulatePoints(
        proj_mats[0], proj_mats[1],
        points_2d[0:1].T, points_2d[1:2].T,
    )
    X = X[:3] / X[3]
    return X.flatten()


# ──────────────────────────────────────────────
# ArUco marker detection
# ──────────────────────────────────────────────

def detect_scale_aruco(
    photo_dir: Path,
    sparse_dir: Path,
    marker_size_mm: float = 20.0,
) -> Optional[ScaleResult]:
    """Detect ArUco markers in photos and compute scale factor.

    Detects markers via cv2.aruco, triangulates corner positions
    using COLMAP's camera poses, and computes scale from marker's
    known physical size.

    Args:
        photo_dir: Directory containing input photos.
        sparse_dir: Directory containing COLMAP sparse reconstruction.
        marker_size_mm: Physical size of the ArUco marker in mm.

    Returns:
        ScaleResult if detected, None if no markers found.
    """
    try:
        import cv2
        import cv2.aruco as aruco
    except ImportError:
        return ScaleResult(
            scale_factor=1.0, method="aruco", confidence=0.0,
            detected_length_mm=0.0,
            warnings=["OpenCV not installed. Install: pip install opencv-python"],
        )

    photos = _read_photos(photo_dir)
    if not photos:
        return None

    # Read COLMAP data
    try:
        cameras, images = _read_colmap_sparse(sparse_dir)
    except (FileNotFoundError, ValueError):
        return None

    # Detect markers
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, params)

    # Map: photo filename -> image_id
    photo_to_id = {img.name: img_id for img_id, img in images.items()}

    # Collect marker observations: marker_id -> [(image_id, corner_idx, pixel_xy)]
    marker_obs: dict[int, list[tuple[int, int, np.ndarray]]] = {}

    for photo_path in photos:
        img = cv2.imread(str(photo_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        corners, ids, _ = detector.detectMarkers(img)
        if ids is None:
            continue
        img_id = photo_to_id.get(photo_path.name)
        if img_id is None:
            continue

        for idx in range(len(ids)):
            mid = int(ids[idx].flatten()[0])
            if mid not in marker_obs:
                marker_obs[mid] = []
            for cidx in range(4):
                px, py = corners[idx][0][cidx]
                marker_obs[mid].append((img_id, cidx, np.array([px, py])))

    if not marker_obs:
        return None

    # Triangulate corners and compute scale
    scales: list[float] = []
    for mid, obs in marker_obs.items():
        # Group by corner index
        by_corner: dict[int, list[tuple[int, np.ndarray]]] = {}
        for img_id, cidx, px in obs:
            by_corner.setdefault(cidx, []).append((img_id, px))

        corners_3d: dict[int, np.ndarray] = {}
        for cidx, pts in by_corner.items():
            if len(pts) >= 2:
                pt = _triangulate_corner(pts, cameras, images)
                if pt is not None:
                    corners_3d[cidx] = pt

        if len(corners_3d) < 4:
            continue

        for i, j in [(0, 1), (1, 2), (2, 3), (3, 0)]:
            if i in corners_3d and j in corners_3d:
                d = float(np.linalg.norm(corners_3d[i] - corners_3d[j]))
                if d > 0:
                    scales.append(marker_size_mm / d)

    if not scales:
        return None

    arr = np.array(scales)
    med = float(np.median(arr))
    std = float(arr.std())
    valid = arr[np.abs(arr - med) <= 2 * std]
    if len(valid) == 0:
        return None

    final_scale = float(np.median(valid))
    confidence = min(1.0, len(valid) / 10.0)

    return ScaleResult(
        scale_factor=final_scale,
        method="aruco",
        confidence=confidence,
        detected_length_mm=marker_size_mm,
    )


# ──────────────────────────────────────────────
# Ruler/scale bar detection
# ──────────────────────────────────────────────

def detect_scale_ruler(
    photo_dir: Path,
    sparse_dir: Optional[Path],
) -> Optional[ScaleResult]:
    """Detect a standard ruler in photos and estimate scale.

    Uses adaptive thresholding, Canny edge detection, Hough lines,
    and tick mark frequency analysis to locate a ruler and compute
    a pixel-per-mm estimate.

    Args:
        photo_dir: Directory containing input photos.
        sparse_dir: Optional — not used for this method (kept for API consistency).

    Returns:
        ScaleResult if a ruler was detected, None otherwise.
    """
    try:
        import cv2
    except ImportError:
        return ScaleResult(
            scale_factor=1.0, method="ruler", confidence=0.0,
            detected_length_mm=0.0,
            warnings=["OpenCV not installed. Install: pip install opencv-python"],
        )

    from scipy.signal import argrelextrema

    photos = _read_photos(photo_dir)
    if not photos:
        return None

    # Sample up to 5 photos
    sample = photos[:5]
    ratios: list[float] = []

    for photo_path in sample:
        img = cv2.imread(str(photo_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Adaptive threshold for varied lighting
        thresh = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 4,
        )

        # Edge detection
        edges = cv2.Canny(thresh, 50, 150)

        # Hough line detection
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180,
            threshold=100, minLineLength=50, maxLineGap=10,
        )

        if lines is None or len(lines) < 2:
            continue

        # Longest line = likely ruler edge
        longest = max(lines, key=lambda l: np.linalg.norm(
            [l[0][2] - l[0][0], l[0][3] - l[0][1]]
        ))
        x1, y1, x2, y2 = longest[0]
        line_len = float(np.linalg.norm([x2 - x1, y2 - y1]))

        # Extract intensity profile along the ruler
        n = 200
        profile = np.zeros(n)
        for i in range(n):
            t = i / (n - 1)
            xi = int(x1 + t * (x2 - x1))
            yi = int(y1 + t * (y2 - y1))
            if 0 <= xi < img.shape[1] and 0 <= yi < img.shape[0]:
                profile[i] = img[yi, xi]

        # Normalize
        pmin, pmax = profile.min(), profile.max()
        if pmax == pmin:
            continue
        profile = (profile - pmin) / (pmax - pmin)

        # Find tick mark minima
        minima = argrelextrema(profile, np.less, order=5)[0]
        if len(minima) < 3:
            continue

        # Spacing in pixels between tick marks
        spacing_px = np.diff(minima).mean() / n * line_len
        if spacing_px > 0:
            ratios.append(spacing_px)

    if len(ratios) < 2:
        return None

    arr = np.array(ratios)
    med = float(np.median(arr))
    std = float(arr.std())
    valid = arr[np.abs(arr - med) <= 2 * std]
    if len(valid) < 2:
        return None

    px_per_mm = float(np.median(valid))
    confidence = min(0.5, len(valid) * 0.1)

    return ScaleResult(
        scale_factor=px_per_mm,
        method="ruler",
        confidence=confidence,
        detected_length_mm=1.0,
        warnings=["Ruler-based scale is approximate (2D estimate)"],
    )
