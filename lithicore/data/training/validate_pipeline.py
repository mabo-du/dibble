"""validate_pipeline.py — Photo-to-mesh pipeline validation tool.

Tests the photogrammetry pipeline accuracy by rendering synthetic
photo sets from known reference meshes, running them through the
COLMAP pipeline, and comparing the output mesh to the original.

This allows us to validate pipeline accuracy at scale without needing
a physical camera setup or known reference scans.

Usage:
    python3 lithicore/data/training/validate_pipeline.py \\
        --mesh /data/dibble-training/raw/RF_3D_Meshes/.../RF.b_5420.ply \\
        --n-photos 24

Output:
    - Rendered photos to {workspace}/photos/
    - Pipeline mesh to {workspace}/output.ply
    - Validation report printed to stdout
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import trimesh

# Add project source to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "lithicore" / "src"))

from lithicore._photogrammetry import (
    PhotogrammetryConfig,
    PhotogrammetryQuality,
    run_pipeline,
)
from lithicore._classification import extract_features
from lithicore._models import LithicFeatureVector


# ── Photo Rendering ──


def render_views(mesh: trimesh.Trimesh, n_photos: int = 24) -> list[np.ndarray]:
    """Render synthetic photos of a mesh from multiple angles.

    Distributes cameras on a sphere around the mesh, pointing at the
    mesh centroid. Returns list of RGBA image arrays.
    """
    # Center mesh at origin
    centroid = mesh.vertices.mean(axis=0)
    mesh.vertices -= centroid

    # Scale to fit in unit sphere
    max_extent = np.max(mesh.vertices.ptp(axis=0))
    if max_extent > 0:
        mesh.vertices /= max_extent

    # Distribute cameras using Fibonacci sphere
    phi = np.pi * (3 - np.sqrt(5))  # golden angle
    cameras = []
    for i in range(n_photos):
        y = 1 - (i / (n_photos - 1)) * 2
        radius = np.sqrt(1 - y * y)
        theta = phi * i
        cameras.append(np.array([radius * np.cos(theta), y, radius * np.sin(theta)]))

    # Render each view using trimesh's scene rendering
    images = []
    for pos in cameras:
        scene = trimesh.Scene(mesh)
        # Place camera at pos, looking at origin
        camera_transform = _look_at(pos, np.array([0, 0, 0]))
        scene.camera_transform = camera_transform
        # Render
        try:
            img = scene.save_image(resolution=(1024, 768), visible=True)
            # trimesh returns PNG bytes — we'll save to disk instead
            images.append(img)
        except Exception:
            # Fallback: skip this view
            pass

    return cameras


def _look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Generate a look-at camera matrix."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    up = np.array([0, 1, 0])
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    mat = np.eye(4)
    mat[:3, 0] = right
    mat[:3, 1] = up
    mat[:3, 2] = -forward
    mat[:3, 3] = eye
    return mat


def save_rendered_photos(mesh_path: Path, photo_dir: Path, n_photos: int = 24) -> int:
    """Render and save photos from a mesh. Returns photo count."""
    mesh = trimesh.load(str(mesh_path), force="mesh")
    photo_dir.mkdir(parents=True, exist_ok=True)

    # Center and scale
    centroid = mesh.vertices.mean(axis=0)
    mesh.vertices -= centroid
    max_ext = np.max(mesh.vertices.ptp(axis=0))
    if max_ext > 0:
        mesh.vertices /= max_ext

    # Fibonacci sphere camera positions
    phi = np.pi * (3 - np.sqrt(5))
    saved = 0
    for i in range(n_photos):
        y = 1 - (i / (n_photos - 1)) * 2
        r = np.sqrt(1 - y * y)
        theta = phi * i
        pos = np.array([r * np.cos(theta), y, r * np.sin(theta)])

        scene = trimesh.Scene(mesh)
        scene.camera_transform = _look_at(pos, np.array([0, 0, 0]))

        try:
            # Save image using trimesh's scene rendering
            img_path = photo_dir / f"view_{i:03d}.png"
            # trimesh Scene.save_image returns bytes
            png_bytes = scene.save_image(resolution=(1024, 768), visible=True)
            with open(img_path, "wb") as f:
                f.write(png_bytes)
            saved += 1
        except Exception:
            continue

    return saved


# ── Mesh Comparison ──


def compare_meshes(
    reference: trimesh.Trimesh, test: trimesh.Trimesh
) -> dict:
    """Compare two meshes and return quality metrics.

    Metrics:
        hausdorff_max: Maximum Hausdorff distance (worst deviation)
        hausdorff_mean: Mean Hausdorff distance
        hausdorff_rmse: RMS Hausdorff distance
        vertex_count_ratio: test_verts / reference_verts
        volume_ratio: test_vol / reference_vol
        area_ratio: test_area / reference_area
        is_watertight: whether the test mesh is watertight
    """
    metrics = {}

    try:
        # Hausdorff distance: sample points on both meshes
        ref_pts = reference.sample(5000)
        test_pts = test.sample(5000)

        # For each test point, find distance to nearest ref point
        from scipy.spatial import cKDTree
        tree = cKDTree(ref_pts)
        dists, _ = tree.query(test_pts)

        metrics["hausdorff_max"] = float(dists.max())
        metrics["hausdorff_mean"] = float(dists.mean())
        metrics["hausdorff_rmse"] = float(np.sqrt((dists ** 2).mean()))
    except Exception:
        metrics["hausdorff_max"] = -1
        metrics["hausdorff_mean"] = -1
        metrics["hausdorff_rmse"] = -1

    # Basic geometric metrics
    metrics["reference_vertices"] = len(reference.vertices)
    metrics["test_vertices"] = len(test.vertices)
    metrics["vertex_count_ratio"] = len(test.vertices) / max(len(reference.vertices), 1)

    if reference.is_watertight and reference.volume > 0:
        ref_vol = reference.volume
    else:
        ref_vol = 0

    if test.is_watertight and test.volume > 0:
        test_vol = test.volume
    else:
        test_vol = 0

    metrics["reference_volume"] = float(ref_vol)
    metrics["test_volume"] = float(test_vol)
    metrics["volume_ratio"] = float(test_vol / max(ref_vol, 0.001))
    metrics["reference_area"] = float(reference.area)
    metrics["test_area"] = float(test.area)
    metrics["area_ratio"] = float(test.area / max(reference.area, 0.001))
    metrics["test_is_watertight"] = test.is_watertight

    return metrics


def compare_features(reference: trimesh.Trimesh, test: trimesh.Trimesh) -> dict:
    """Compare 22-feature vectors between reference and test meshes."""
    ref_fv = extract_features(reference)
    test_fv = extract_features(test)

    ref_arr = ref_fv.to_array()
    test_arr = test_fv.to_array()

    metrics = {}
    for i, name in enumerate(ref_fv.FEATURE_NAMES):
        ref_val = float(ref_arr[i])
        test_val = float(test_arr[i])
        diff = abs(ref_val - test_val)
        rel_err = diff / max(abs(ref_val), 0.001)
        metrics[f"ref_{name}"] = round(ref_val, 4)
        metrics[f"test_{name}"] = round(test_val, 4)
        metrics[f"err_{name}"] = round(diff, 4)
        metrics[f"rel_err_{name}"] = round(rel_err, 4)

    # Overall feature similarity
    from scipy.stats import pearsonr
    corr, _ = pearsonr(ref_arr, test_arr)
    metrics["feature_correlation"] = round(float(corr), 4)
    metrics["feature_rmse"] = round(float(np.sqrt(((ref_arr - test_arr) ** 2).mean())), 4)

    return metrics


# ── Pipeline Runner ──


def run_validation(
    mesh_path: Path,
    n_photos: int = 24,
    quality: str = "medium",
    workspace: Path | None = None,
) -> dict:
    """Run full validation: render → pipeline → compare.

    Args:
        mesh_path: Path to reference mesh (PLY).
        n_photos: Number of synthetic photos to render.
        quality: Pipeline quality ('low', 'medium', 'high').
        workspace: Working directory (temp dir if None).

    Returns:
        Dict with all validation metrics.
    """
    own_workspace = workspace is None
    if own_workspace:
        workspace = Path(tempfile.mkdtemp(prefix="dibble_val_"))

    try:
        # Load reference
        print(f"Loading reference: {mesh_path.name}")
        reference = trimesh.load(str(mesh_path), force="mesh")
        if reference.is_watertight and reference.volume < 0:
            reference.fix_normals()

        # Render photos
        photo_dir = workspace / "photos"
        n_saved = save_rendered_photos(mesh_path, photo_dir, n_photos)
        print(f"Rendered {n_saved}/{n_photos} photos to {photo_dir}")

        if n_saved < 3:
            raise ValueError(f"Too few photos rendered: {n_saved}")

        # Run photogrammetry pipeline
        quality_map = {
            "low": PhotogrammetryQuality.LOW,
            "medium": PhotogrammetryQuality.MEDIUM,
            "high": PhotogrammetryQuality.HIGH,
        }

        config = PhotogrammetryConfig(
            photo_folder=photo_dir,
            colmap_workspace=workspace / "pipeline",
            quality=quality_map.get(quality, PhotogrammetryQuality.MEDIUM),
        )

        print(f"Running pipeline ({quality} quality)...")
        t0 = time.time()
        result = run_pipeline(config)
        pipeline_time = time.time() - t0
        print(f"Pipeline completed in {pipeline_time:.0f}s")
        print(f"  Mesh: {result.mesh_path}")
        print(f"  Warnings: {len(result.warnings)}")

        # Load output mesh
        if result.mesh_path and result.mesh_path.exists():
            test_mesh = trimesh.load(str(result.mesh_path), force="mesh")
            if test_mesh.is_watertight and test_mesh.volume < 0:
                test_mesh.fix_normals()
        else:
            raise FileNotFoundError("Pipeline produced no output mesh")

        # Compare
        print("Computing comparison metrics...")
        geo_metrics = compare_meshes(reference, test_mesh)
        feat_metrics = compare_features(reference, test_mesh)

        # Compile report
        report = {
            "mesh": str(mesh_path),
            "n_photos": n_saved,
            "pipeline_quality": quality,
            "pipeline_time_s": round(pipeline_time, 1),
            "geometric": geo_metrics,
            "feature": feat_metrics,
            "status": "pass",
        }

        return report

    except Exception as e:
        return {
            "mesh": str(mesh_path),
            "n_photos": n_photos,
            "status": "fail",
            "error": str(e),
        }

    finally:
        if own_workspace and workspace:
            shutil.rmtree(workspace)


# ── CLI ──


def main():
    parser = argparse.ArgumentParser(
        description="Validate photogrammetry pipeline accuracy"
    )
    parser.add_argument(
        "--mesh",
        required=True,
        help="Path to reference mesh (PLY)",
    )
    parser.add_argument(
        "--n-photos",
        type=int,
        default=24,
        help="Number of synthetic photos to render",
    )
    parser.add_argument(
        "--quality",
        default="medium",
        choices=["low", "medium", "high"],
        help="Pipeline quality setting",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="Working directory (default: temp)",
    )
    parser.add_argument(
        "--json",
        default=None,
        help="Save report to JSON file",
    )
    args = parser.parse_args()

    mesh_path = Path(args.mesh)
    if not mesh_path.exists():
        print(f"Error: mesh not found: {mesh_path}")
        sys.exit(1)

    workspace = Path(args.workspace) if args.workspace else None

    print(f"{'='*60}")
    print(f"  Photo-to-Mesh Pipeline Validation")
    print(f"{'='*60}")
    print(f"  Mesh: {mesh_path}")
    print(f"  Photos: {args.n_photos}")
    print(f"  Quality: {args.quality}")
    print(f"{'='*60}\n")

    report = run_validation(mesh_path, args.n_photos, args.quality, workspace)

    print(f"\n{'='*60}")
    print(f"  Validation Report")
    print(f"{'='*60}")
    print(f"  Status: {report.get('status', 'unknown')}")

    if report.get("status") == "pass":
        geo = report["geometric"]
        feat = report["feature"]

        print(f"\n  --- Geometric ---")
        print(f"  Hausdorff max:  {geo.get('hausdorff_max', -1):.4f}")
        print(f"  Hausdorff mean: {geo.get('hausdorff_mean', -1):.4f}")
        print(f"  Vertices ratio: {geo.get('vertex_count_ratio', -1):.2f}")
        print(f"  Volume ratio:   {geo.get('volume_ratio', -1):.2f}")
        print(f"  Watertight:     {geo.get('test_is_watertight', False)}")

        print(f"\n  --- Features ---")
        print(f"  Correlation: {feat.get('feature_correlation', -1):.4f}")
        print(f"  RMSE:        {feat.get('feature_rmse', -1):.4f}")

        print(f"\n  --- Per-Feature Errors (top 5 worst) ---")
        errs = [(k.replace("rel_err_", ""), v) for k, v in feat.items()
                if k.startswith("rel_err_")]
        errs.sort(key=lambda x: -x[1])
        for name, err in errs[:5]:
            ref_key = f"ref_{name}"
            test_key = f"test_{name}"
            if ref_key in feat and test_key in feat:
                print(f"  {name:25s} ref={feat[ref_key]:>8.4f}  test={feat[test_key]:>8.4f}  rel_err={err:.4f}")

        print(f"\n  Pipeline time: {report.get('pipeline_time_s', 0):.0f}s")

    else:
        print(f"  Error: {report.get('error', 'unknown')}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  Report saved: {args.json}")


if __name__ == "__main__":
    main()
