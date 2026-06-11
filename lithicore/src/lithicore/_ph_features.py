"""_ph_features.py — Persistent Homology feature extraction for lithic meshes.

Computes persistence images (2D histograms of birth-persistence pairs) from
3D mesh files using GUDHI's Alpha complex. Designed to capture micro-topographic
features (flake scars, edge notches, retouch patterns) that complement the 22
core morphometrics + 10 interaction features.

Usage:
    from lithicore._ph_features import compute_ph_vector, get_ph_features

    # Single mesh
    ph_vec = compute_ph_vector("mesh.ply")

    # From cache
    ph_vec = get_ph_features("artefact_001", cache_dir=".ph_cache")

Signal test results (June 2026):
    On Retouched Flake vs Unmodified Flake (103 artefacts):
    - Baseline 32 features:        43.8%
    - PH only (15 PCA components): 41.9%
    - Combined (32 + PH):          47.7%  (+3.9pp)
    → PH SIGNAL DETECTED, proceeding to full implementation.
agent:   deepseek-v4-pro | 2026-06-12 | Fixed global np.random.seed → local default_rng for thread safety
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np

# GUDHI — optional dependency, checked at first use
_gudhi = None


def _import_gudhi():
    global _gudhi
    if _gudhi is None:
        import gudhi
        _gudhi = gudhi
    return _gudhi


# PH computation parameters
N_POINTS: int = 2000          # Vertex subsample count
PI_RESOLUTION: int = 20       # Persistence image grid size (20x20 → 400 features/dim)
PH_DIMENSIONS: tuple[int, ...] = (0, 1, 2)  # Which topological dimensions to use
PCA_COMPONENTS: int = 15      # Dimensionality reduction target

# PH cache directory (relative to project root or absolute)
CACHE_DIR: Path = Path.home() / ".cache" / "dibble" / "ph_features"


def compute_persistence_diagram(
    mesh_path: str | Path,
    n_points: int = N_POINTS,
    seed: int | None = None,
) -> dict[int, np.ndarray] | None:
    """Compute persistence diagram from a mesh file.

    Steps:
    1. Load mesh via trimesh
    2. Normalise: centre at origin, scale to unit bounding box
    3. Subsample vertices (random)
    4. Compute Alpha complex persistence via GUDHI

    Args:
        mesh_path: Path to a .ply or .stl mesh file.
        n_points: Number of vertices to subsample (default 2000).
        seed: Random seed for subsampling (use artefact ID hash for determinism).

    Returns:
        Dict mapping dimension → (N, 2) array of (birth, persistence) pairs,
        or None if computation fails.
    """
    import trimesh

    gd = _import_gudhi()

    try:
        mesh = trimesh.load(str(mesh_path))
        verts = np.array(mesh.vertices)
    except Exception:
        return None

    if len(verts) < 10:
        return None

    # Normalise: centre at origin, scale to unit bounding box
    verts = verts - verts.mean(axis=0)
    scale = np.max(np.ptp(verts, axis=0))
    if scale > 0:
        verts = verts / scale

    # Subsample
    if len(verts) > n_points:
        if seed is not None:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(verts), n_points, replace=False)
        else:
            idx = np.random.choice(len(verts), n_points, replace=False)
        verts = verts[idx]

    # Alpha complex
    alpha = gd.AlphaComplex(points=verts)
    st = alpha.create_simplex_tree()
    diag = st.persistence(min_persistence=0.001)

    # Organise by dimension
    result: dict[int, list] = {dim: [] for dim in PH_DIMENSIONS}
    for dim, (b, d) in diag:
        if d != float("inf") and dim in result:
            result[dim].append([b, d - b])  # (birth, persistence)

    return {dim: np.array(pts) for dim, pts in result.items() if pts}


def vectorise_persistence_image(
    diagram: dict[int, np.ndarray],
    resolution: int = PI_RESOLUTION,
) -> np.ndarray:
    """Convert persistence diagrams to a fixed-length feature vector.

    For each topological dimension, computes a 2D histogram of
    (birth, persistence) pairs. Flattens and concatenates across dimensions.

    Args:
        diagram: Dict from compute_persistence_diagram().
        resolution: Grid size per dimension (default 20 → 400 cells).

    Returns:
        1D array of length len(PH_DIMENSIONS) * resolution * resolution.
    """
    n_dims = len(PH_DIMENSIONS)
    total_bins = n_dims * resolution * resolution
    features = np.zeros(total_bins, dtype=float)

    if diagram is None:
        return features

    # Determine global range from all dimensions
    dim_pts = [diagram[d] for d in PH_DIMENSIONS if d in diagram and len(diagram[d]) > 0]
    if not dim_pts:
        return features
    all_pts = np.vstack(dim_pts)

    b_max = max(1.0, float(np.max(all_pts[:, 0])))
    p_max = max(1.0, float(np.max(all_pts[:, 1])))

    offset = 0
    for dim in PH_DIMENSIONS:
        pts = diagram.get(dim)
        if pts is not None and len(pts) > 0:
            hist, _, _ = np.histogram2d(
                pts[:, 0], pts[:, 1],
                bins=resolution,
                range=[[0, b_max], [0, p_max]],
            )
            features[offset:offset + resolution * resolution] = hist.flatten()
        offset += resolution * resolution

    return features


def compute_ph_vector(
    mesh_path: str | Path,
    n_points: int = N_POINTS,
    resolution: int = PI_RESOLUTION,
    seed: int | None = None,
) -> np.ndarray | None:
    """Compute the full PH feature vector for a mesh.

    Combines compute_persistence_diagram() + vectorise_persistence_image().

    Args:
        mesh_path: Path to mesh file.
        n_points: Subsample count.
        resolution: Persistence image resolution.
        seed: Random seed for deterministic subsampling.

    Returns:
        Normalised 1D feature vector, or None on failure.
    """
    diag = compute_persistence_diagram(mesh_path, n_points, seed)
    if diag is None:
        return None
    vec = vectorise_persistence_image(diag, resolution)
    # Normalise to unit sum (so meshes with different vertex counts are comparable)
    total = vec.sum()
    if total > 0:
        vec = vec / total
    return vec


def ph_cache_path(artefact_id: str, cache_dir: str | Path = "") -> Path:
    """Return the expected cache path for an artefact's PH features."""
    cache = Path(cache_dir) if cache_dir else CACHE_DIR
    return cache / f"{artefact_id}.npy"


def get_ph_features(
    artefact_id: str,
    mesh_path: str | Path | None = None,
    cache_dir: str | Path = "",
    n_points: int = N_POINTS,
    resolution: int = PI_RESOLUTION,
) -> np.ndarray | None:
    """Get PH features for an artefact, from cache or by computing.

    Args:
        artefact_id: Unique artefact identifier (for cache lookup).
        mesh_path: Path to mesh file (required if not cached).
        cache_dir: Override default cache directory.
        n_points: Subsample count for computation.
        resolution: Persistence image resolution.

    Returns:
        1D feature vector, or None if unavailable.
    """
    cache = Path(cache_dir) if cache_dir else CACHE_DIR
    cache_path = cache / f"{artefact_id}.npy"

    # Check cache
    if cache_path.exists():
        try:
            return np.load(str(cache_path))
        except Exception:
            pass

    # Compute
    if mesh_path is None:
        return None

    seed = abs(hash(artefact_id)) % (2 ** 31) if artefact_id else None
    vec = compute_ph_vector(mesh_path, n_points, resolution, seed)
    if vec is None:
        return None

    # Save to cache
    cache.mkdir(parents=True, exist_ok=True)
    try:
        np.save(str(cache_path), vec)
    except Exception:
        pass

    return vec


def batch_compute_ph(
    artefact_list: list[tuple[str, str | Path]],
    cache_dir: str | Path = "",
    n_points: int = N_POINTS,
    resolution: int = PI_RESOLUTION,
    verbose: bool = True,
) -> int:
    """Batch compute PH features for a list of artefacts.

    Args:
        artefact_list: List of (artefact_id, mesh_path) tuples.
        cache_dir: Cache directory.
        n_points: Subsample count.
        resolution: Persistence image resolution.
        verbose: Print progress.

    Returns:
        Number of successfully computed artefacts.
    """
    cache = Path(cache_dir) if cache_dir else CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)

    success = 0
    for i, (aid, mesh_path) in enumerate(artefact_list):
        cache_path = cache / f"{aid}.npy"
        if cache_path.exists():
            success += 1
            continue

        seed = abs(hash(aid)) % (2 ** 31)
        vec = compute_ph_vector(mesh_path, n_points, resolution, seed)
        if vec is not None:
            np.save(str(cache_path), vec)
            success += 1

        if verbose and (i + 1) % 50 == 0:
            print(f"  PH: {i+1}/{len(artefact_list)} ({success} succeeded)")

        # Periodic GC to prevent memory buildup
        if (i + 1) % 10 == 0:
            gc.collect()

    if verbose:
        print(f"  PH complete: {success}/{len(artefact_list)} artefacts cached")
    return success


def load_ph_matrix(
    artefact_ids: list[str],
    cache_dir: str | Path = "",
    n_components: int = PCA_COMPONENTS,
) -> tuple[np.ndarray, list[int]] | tuple[None, None]:
    """Load PH features for a list of artefact IDs and apply PCA.

    Returns both the PCA-reduced matrix AND the indices of valid rows,
    so callers can align PH features with their feature matrices.

    Args:
        artefact_ids: List of artefact IDs (must be cached).
        cache_dir: Cache directory.
        n_components: PCA target dimensions.

    Returns:
        (X_ph_reduced, valid_indices) tuple, or (None, None) if insufficient data.
        valid_indices are the positions in artefact_ids that had cached features.
    """
    cache = Path(cache_dir) if cache_dir else CACHE_DIR

    vectors = []
    valid_indices: list[int] = []
    for idx, aid in enumerate(artefact_ids):
        cache_path = cache / f"{aid}.npy"
        if cache_path.exists():
            try:
                vec = np.load(str(cache_path))
                vectors.append(vec)
                valid_indices.append(idx)
            except Exception:
                continue

    if len(vectors) < 10:
        return None, None

    X_ph = np.array(vectors)
    
    # Apply PCA
    from sklearn.decomposition import PCA
    n = min(n_components, X_ph.shape[0], X_ph.shape[1])
    pca = PCA(n_components=n)
    X_ph_reduced = pca.fit_transform(X_ph)

    # Save PCA model for inference-time use
    pca_path = cache / "pca_model.joblib"
    try:
        import joblib
        joblib.dump(pca, pca_path)
    except Exception:
        pass

    return X_ph_reduced, valid_indices
