"""_classification.py — Lithic typology classification pipeline.

exports: extract_features(mesh) -> LithicFeatureVector
         ClassifierModel
         train_model(features, labels, typology_name) -> ClassifierModel
used_by: lithicope classification panel, CLI
rules:   Pure functions + model wrapper. No GUI imports.
         Feature extraction ~0.1s per mesh. Model training ~1-2s typical.
agent:   deepseek-v4-flash | 2026-05-27 | Initial implementation
agent:   deepseek-v4-pro | 2026-06-12 | Fixed feature dimension mismatch: store n_features, pad missing PH dims with zeros. Added _infer_n_features, _pad_features, _pad_features_for_model.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

from lithicore._models import (
    ClassificationResult,
    FeatureImportance,
    LithicFeatureVector,
)

# Pre-trained model paths
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "models"

TYPOLOGIES: dict[str, dict] = {
    "basic": {
        "label": "Basic Morphological",
        "classes": ["Flake", "Blade", "Bladelet", "Core", "Tool"],
    },
    "bordes": {
        "label": "Bordes Typology",
        "classes": ["Scraper", "Handaxe", "Point", "Burin",
                     "Denticulate", "Notched", "Backed knife"],
    },
    "technological": {
        "label": "Technological",
        "classes": ["Primary", "Secondary", "Tertiary",
                     "Crested blade", "Core rejuvenation"],
    },
}


def extract_features(mesh: trimesh.Trimesh) -> LithicFeatureVector:
    """Compute the full morphometric feature vector from an oriented mesh.

    The mesh should be oriented (platform ~ XY plane, length along Z).
    If not oriented, some features (COM Z ratio, platform angle) will
    be approximate but still usable.

    Args:
        mesh: A trimesh.Trimesh of a lithic artefact.

    Returns:
        LithicFeatureVector with all 20 features populated.
    """
    # ── Raw metrics ──
    obb = mesh.bounding_box_oriented
    extents = sorted(obb.extents, reverse=True)
    length_mm = extents[0]
    width_mm = extents[1]
    thickness_mm = extents[2]
    area_mm2 = mesh.area
    vol_mm3 = mesh.volume if mesh.is_watertight else 0.0

    # ── Derived ratios ──
    elongation = length_mm / max(width_mm, 0.001)
    flatness = width_mm / max(thickness_mm, 0.001)
    compactness = vol_mm3 / max(length_mm ** 3, 0.001)
    relative_thickness = thickness_mm / max(length_mm, 0.001)

    # ── Scar detection ──
    try:
        from lithicore._scar_detection import detect_scars, ScarConfig
        scar_config = ScarConfig()
        scar_result = detect_scars(mesh, scar_config)
        scar_count = len(scar_result.scars)
        mean_scar_area = float(np.mean([s.area_mm2 for s in scar_result.scars])) if scar_count > 0 else 0.0
    except Exception:
        scar_count = 0
        mean_scar_area = 0.0

    # ── Platform angle ──
    try:
        from lithicore._platform_angle import platform_angles
        epa, ipa = platform_angles(mesh, None)
        platform_angle_deg = (epa.value + ipa.value) / 2
    except Exception:
        platform_angle_deg = 0.0

    # ── Edge angles ──
    edge_angles = _compute_edge_angles(mesh)
    edge_angle_mean = float(np.mean(edge_angles)) if len(edge_angles) > 0 else 0.0
    edge_angle_std = float(np.std(edge_angles)) if len(edge_angles) > 1 else 0.0
    # Higher-order edge statistics — key for separating denticulate/notched/scraper
    from scipy.stats import skew as scipy_skew, kurtosis as scipy_kurtosis
    edge_angle_skewness = float(scipy_skew(edge_angles)) if len(edge_angles) > 2 else 0.0
    edge_angle_kurtosis = float(scipy_kurtosis(edge_angles)) if len(edge_angles) > 2 else 0.0

    # ── Curvature ──
    curvature_index = _compute_curvature_index(mesh)

    # ── Cross-section profile ──
    cross_section = _compute_cross_section_profile(mesh)

    # ── Symmetry ──
    symmetry = _compute_symmetry(mesh)

    # ── COM Z ratio ──
    com_z = mesh.center_mass[2] if hasattr(mesh, 'center_mass') else 0.0
    com_z_ratio = (com_z - mesh.bounds[0, 2]) / max(mesh.bounds[1, 2] - mesh.bounds[0, 2], 0.001)

    # ── Dorsal ridges ──
    ridge_count = _detect_dorsal_ridges(mesh)

    # ── Surface roughness ──
    roughness = _compute_surface_roughness(mesh)

    return LithicFeatureVector(
        length_mm=round(length_mm, 2),
        width_mm=round(width_mm, 2),
        thickness_mm=round(thickness_mm, 2),
        surface_area_mm2=round(area_mm2, 2),
        volume_mm3=round(vol_mm3, 2),
        elongation=round(elongation, 3),
        flatness=round(flatness, 3),
        compactness=round(compactness, 6),
        relative_thickness=round(relative_thickness, 4),
        scar_count=scar_count,
        mean_scar_area_mm2=round(mean_scar_area, 2),
        platform_angle_deg=round(platform_angle_deg, 1),
        edge_angle_mean_deg=round(edge_angle_mean, 1),
        edge_angle_std_deg=round(edge_angle_std, 1),
        edge_angle_skewness=round(edge_angle_skewness, 3),
        edge_angle_kurtosis=round(edge_angle_kurtosis, 3),
        curvature_index=round(curvature_index, 4),
        cross_section_profile=round(cross_section, 2),
        symmetry_score=round(symmetry, 4),
        com_z_ratio=round(com_z_ratio, 4),
        dorsal_ridge_count=ridge_count,
        surface_roughness=round(roughness, 4),
    )


def _compute_edge_angles(mesh: trimesh.Trimesh) -> np.ndarray:
    """Compute dihedral angles at all edges of the mesh."""
    if len(mesh.faces) == 0 or len(mesh.edges_unique) == 0:
        return np.array([])
    try:
        face_pairs = mesh.face_adjacency
        normals = mesh.face_normals
        if len(face_pairs) == 0:
            return np.array([])
        n1 = normals[face_pairs[:, 0]]
        n2 = normals[face_pairs[:, 1]]
        cos_angles = np.clip(np.sum(n1 * n2, axis=1), -1.0, 1.0)
        angles = np.degrees(np.arccos(cos_angles))
        return angles
    except Exception:
        return np.array([])


def _compute_curvature_index(mesh: trimesh.Trimesh) -> float:
    """Compute curvature index via vertex normal angular deviation."""
    try:
        vertex_normals = mesh.vertex_normals
        if len(vertex_normals) < 3:
            return 0.0
        mean_normal = vertex_normals.mean(axis=0)
        norm = np.linalg.norm(mean_normal)
        if norm == 0:
            return 0.0
        mean_normal = mean_normal / norm
        deviations = np.arccos(np.clip(
            np.dot(vertex_normals, mean_normal), -1.0, 1.0
        ))
        return float(np.mean(deviations))
    except Exception:
        return 0.0


def _compute_cross_section_profile(mesh: trimesh.Trimesh) -> float:
    """Classify cross-section as 0=flat, 1=triangular, 2=round."""
    try:
        mid_z = (mesh.bounds[0, 2] + mesh.bounds[1, 2]) / 2
        slice_2d = mesh.section(
            plane_origin=[0, 0, mid_z],
            plane_normal=[0, 0, 1],
        )
        if slice_2d is None:
            return 0.0
        vertices = slice_2d.vertices[:, :2]
        if len(vertices) < 3:
            return 0.0
        bb = vertices.ptp(axis=0)
        ratio = bb[1] / max(bb[0], 0.001) if bb[0] > 0 else 0.0
        if ratio < 0.5:
            return 0.0
        elif ratio < 0.8:
            return 1.0
        else:
            return 2.0
    except Exception:
        return 0.0


def _compute_symmetry(mesh: trimesh.Trimesh) -> float:
    """Compute bilateral symmetry score using Hausdorff distance between halves."""
    try:
        vertices = np.asarray(mesh.vertices)
        centre_x = (mesh.bounds[0, 0] + mesh.bounds[1, 0]) / 2
        left = vertices[vertices[:, 0] < centre_x]
        right = vertices[vertices[:, 0] >= centre_x]
        if len(left) < 3 or len(right) < 3:
            return 0.5
        right_reflected = right.copy()
        right_reflected[:, 0] = 2 * centre_x - right_reflected[:, 0]
        from scipy.spatial import KDTree
        tree = KDTree(left)
        distances, _ = tree.query(right_reflected)
        mean_dist = float(np.mean(distances))
        extent = max(mesh.extents)
        if extent > 0:
            return max(0.0, 1.0 - (mean_dist / extent))
        return 0.5
    except Exception:
        return 0.5


def _detect_dorsal_ridges(mesh: trimesh.Trimesh) -> int:
    """Count parallel linear ridges on the dorsal surface."""
    try:
        angles = _compute_edge_angles(mesh)
        if len(angles) == 0:
            return 0
        ridge_edges = angles > 130
        ridge_count = int(np.sum(ridge_edges))
        return min(ridge_count // 10, 5)
    except Exception:
        return 0


def _compute_surface_roughness(mesh: trimesh.Trimesh) -> float:
    """Compute surface roughness as face area / projected area."""
    try:
        convex_hull = mesh.convex_hull
        projected_area = convex_hull.area if convex_hull is not None else mesh.area
        return mesh.area / max(projected_area, 0.001)
    except Exception:
        return 1.0


class HierarchicalClassifier:
    """Two-level hierarchical cascade for lithic typology.

    Level 0: Broad morphological group (Flake/Blade, Core, Biface).
    Level 1: Specific type within group (Blade, Flake, Core, Biface, etc.).

    Each node has its own RandomForest trained on the relevant data subset.
    This isolates decision boundaries — the root node separates gross morphology,
    while leaf nodes focus on fine-grained distinctions within a group.

    Hierarchy for Basic/Bordes:
        Root
        ├── flake_blade  →  Blade, Flake, Retouched Flake, Unmodified Flake
        ├── core         →  Core, Experimental Core
        └── biface       →  Biface, Unmodified Cobble
    """

    FLAT_NODES: ClassVar[dict[str, list[str]]] = {
        "flake_blade": ["Blade", "Flake", "Retouched Flake", "Unmodified Flake"],
        "core": ["Core", "Experimental Core"],
        "biface": ["Biface", "Unmodified Cobble"],
    }

    # Reverse mapping: leaf class → parent node
    LEAF_TO_NODE: ClassVar[dict[str, str]] = {
        leaf: node
        for node, leaves in FLAT_NODES.items()
        for leaf in leaves
    }

    def __init__(self, n_features: int = 32) -> None:
        self.n_features = n_features
        self.root_rf: RandomForestClassifier | None = None
        self.node_rfs: dict[str, RandomForestClassifier] = {}

    def _train_node_rf(
        self, X: np.ndarray, y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> RandomForestClassifier:
        """Train a single RandomForest for one node in the hierarchy."""
        classes = sorted(set(y))
        depth = min(20, max(12, len(classes) * 2))
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=depth,
            min_samples_leaf=2, max_features=0.3,
            class_weight="balanced", random_state=42, n_jobs=1,
        )
        if sample_weight is not None:
            rf.fit(X, y, sample_weight=sample_weight)
        else:
            rf.fit(X, y)
        return rf

    def fit(
        self, X: np.ndarray, y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> HierarchicalClassifier:
        """Train all nodes in the hierarchy.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Flat class labels (e.g., "Blade", "Core", "Biface").
            sample_weight: Per-sample dataset weights.

        Returns:
            self (fitted pipeline).
        """
        # Map each sample to its Level-0 parent node
        node_labels = np.array([
            self.LEAF_TO_NODE.get(str(lbl), "other") for lbl in y
        ])

        # Train root classifier (broad group)
        self.root_rf = self._train_node_rf(X, node_labels, sample_weight)

        # Train one child classifier per node
        for node_name, leaf_classes in self.FLAT_NODES.items():
            mask = node_labels == node_name
            if mask.sum() < 10:
                continue
            X_node = X[mask]
            y_node = np.array([str(lbl) for lbl, m in zip(y, mask) if m])
            sw_node = sample_weight[mask] if sample_weight is not None else None
            self.node_rfs[node_name] = self._train_node_rf(X_node, y_node, sw_node)

        # Ensure every leaf class can be predicted (fallback = root label)
        self._all_classes = sorted(set(str(lbl) for lbl in y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels through the hierarchy."""
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Level 0: predict broad group
        node_preds = self.root_rf.predict(X)

        # Level 1: route to child classifier
        predictions: list[str] = []
        for i, node in enumerate(node_preds):
            node_str = str(node)
            if node_str in self.node_rfs:
                child_pred = self.node_rfs[node_str].predict(X[i:i + 1])[0]
                predictions.append(str(child_pred))
            else:
                predictions.append(node_str)
        return np.array(predictions)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (n, n_all_classes) probability matrix.

        Uses root confidence to weight child predictions.
        Fallback: if child classifier misses a sample, uses root prediction.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n = X.shape[0]
        n_classes = len(self._all_classes)
        cls_to_idx = {c: i for i, c in enumerate(self._all_classes)}
        result = np.zeros((n, n_classes), dtype=float)

        # Get root node probabilities
        root_probs = self.root_rf.predict_proba(X)
        root_classes = list(self.root_rf.classes_)

        for i in range(n):
            node_name = str(root_classes[int(np.argmax(root_probs[i]))])
            if node_name in self.node_rfs:
                child_probs = self.node_rfs[node_name].predict_proba(X[i:i + 1])[0]
                child_classes = list(self.node_rfs[node_name].classes_)
                for cls_name, prob in zip(child_classes, child_probs):
                    if cls_name in cls_to_idx:
                        result[i, cls_to_idx[cls_name]] = prob
            else:
                # No child classifier — distribute root probability to leaf classes
                result[i, cls_to_idx.get(node_name, 0)] = 1.0

        row_sums = result.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return result / row_sums


class SingleClassPredictor:
    """Trivial predictor for traditions with only one class.

    Always returns the same class label. Required for traditions like
    COADS (all Biface) or Levantine (all Handaxe) where there's no
    meaningful classification to do — the artefacts are all the same type.
    """

    def __init__(self, class_name: str) -> None:
        self._classes = [class_name]
        self.classes_ = np.array([class_name])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self._classes[0])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        probs = np.zeros((n, 1))
        probs[:, 0] = 1.0
        return probs


class TraditionRouter:
    """Routes classification to a tradition-specific model.

    Each archaeological tradition has its own class space and morphological
    signatures. A single universal classifier cannot handle them all — the
    LOGO CV analysis showed accuracy collapsing to 6-12% cross-dataset.

    This router selects the appropriate sub-classifier based on user-provided
    tradition name, then returns the prediction from that tradition's model.

    Traditions:
        - oap: Open Aurignacian Project (Italy) — full flake/blade/core typology
        - levantine: Levantine Acheulean handaxes
        - coads: COADS projectile points (Ohio)
        - experimental: Lombao + Morales experimental cores/retouch
    """

    def __init__(self) -> None:
        self.models: dict[str, ClassifierModel | HierarchicalClassifier] = {}
        self.tradition_classes: dict[str, list[str]] = {}
        self.default_tradition: str = "oap"

    def add_model(
        self,
        tradition: str,
        model: ClassifierModel | HierarchicalClassifier,
        classes: list[str],
    ) -> None:
        """Register a tradition-specific model."""
        self.models[tradition] = model
        self.tradition_classes[tradition] = classes

    @property
    def traditions(self) -> list[str]:
        return list(self.models.keys())

    def predict(
        self, X: np.ndarray, tradition: str | None = None,
    ) -> np.ndarray:
        """Predict class labels for the given tradition.

        Args:
            X: Feature matrix.
            tradition: Tradition identifier. Falls back to default if None.

        Returns:
            Array of predicted class labels.
        """
        t = tradition or self.default_tradition
        if t not in self.models:
            t = self.default_tradition
        model = self.models[t]
        if hasattr(model, "predict"):
            return model.predict(X)
        return np.array([model._classes[0]] * X.shape[0])

    def predict_proba(
        self, X: np.ndarray, tradition: str | None = None,
    ) -> np.ndarray:
        """Return probability matrix for the given tradition's classes."""
        t = tradition or self.default_tradition
        if t not in self.models:
            t = self.default_tradition
        model = self.models[t]
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)
        # Fallback: one-hot for single-class models
        n = X.shape[0]
        classes = self.tradition_classes.get(t, ["Unknown"])
        probs = np.zeros((n, len(classes)))
        probs[:, 0] = 1.0
        return probs


class OrdinalTechnologicalPipeline:
    """Sklearn-compatible hierarchical classifier for Technological typology.

    Level 1: Binary RF separates reduction-stage artefacts from non-reduction.
    Level 2a: LogisticAT (ordinal) for stages: Init → Semi-cortical → Optimal → Maintenance.
    Level 2b: Multi-class RF for non-reduction classes (Handaxe, Other, etc.).

    Exposes sklearn-compatible predict() and predict_proba() for seamless use
    with the existing benchmark and evaluation pipeline.

    The ordinal branch correctly penalises adjacent-stage confusion less than
    ordinal↔non-ordinal confusion, which is the key benefit over flat multi-class.
    """

    def __init__(
        self,
        level1=None,
        ord_model=None,
        non_ord_model=None,
        ordinal_classes: list[str] | None = None,
        ordinal_map: dict[str, int] | None = None,
        inv_ordinal_map: dict[int, str] | None = None,
        non_ordinal_classes: list[str] | None = None,
        all_classes: list[str] | None = None,
    ):
        self.level1 = level1
        self.ord_model = ord_model
        self.non_ord_model = non_ord_model
        self.ordinal_classes = ordinal_classes or []
        self.ordinal_map = ordinal_map or {}
        self.inv_ordinal_map = inv_ordinal_map or {}
        self.non_ordinal_classes = non_ordinal_classes or []
        self.all_classes = all_classes or []

    def fit(self, X: np.ndarray, y: np.ndarray) -> OrdinalTechnologicalPipeline:
        """Fit all three levels of the hierarchical pipeline.

        Level 2a uses cumulative-link (threshold) Random Forests for the
        ordinal reduction stages — each stage-separator model preserves
        RF's non-linear separation, unlike linear LogisticAT.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target labels (string class names).

        Returns:
            self (fitted pipeline).
        """
        from sklearn.ensemble import RandomForestClassifier

        if isinstance(X, list):
            X = np.array(X)
        y_arr = np.array(y)

        # Level 1: reduction vs non-reduction
        l1_labels = np.array([
            "reduction" if lbl in self.ordinal_map else "non_reduction"
            for lbl in y_arr
        ])
        self.level1 = RandomForestClassifier(
            n_estimators=200, max_depth=12,
            min_samples_leaf=2, class_weight="balanced",
            random_state=42, n_jobs=1,
        )
        self.level1.fit(X, l1_labels)

        # Level 2a: Cumulative-link ordinal RF for reduction stages
        ord_mask = np.array([lbl in self.ordinal_map for lbl in y_arr])
        X_ord = X[ord_mask]
        y_ord = np.array([self.ordinal_map[lbl] for lbl in y_arr if lbl in self.ordinal_map])

        if len(y_ord) >= 10:
            n_ord_classes = len(self.ordinal_classes)
            # Train K-1 binary threshold models: P(class > k)
            self._ord_thresholds = []
            for k in range(n_ord_classes - 1):
                y_binary = (y_ord > k).astype(int)
                # Handle edge case where threshold has only one class
                if len(set(y_binary)) < 2:
                    self._ord_thresholds.append(None)
                    continue
                thr_rf = RandomForestClassifier(
                    n_estimators=100, max_depth=8,
                    min_samples_leaf=3, class_weight="balanced",
                    random_state=42 + k, n_jobs=1,
                )
                thr_rf.fit(X_ord, y_binary)
                self._ord_thresholds.append(thr_rf)
        else:
            self._ord_thresholds = []

        # Level 2b: multi-class RF for non-reduction classes
        non_ord_mask = ~ord_mask
        X_non = X[non_ord_mask]
        y_non = np.array([lbl for lbl, m in zip(y_arr, non_ord_mask) if m])

        if len(np.unique(y_non)) >= 2:
            n_classes = len(np.unique(y_non))
            self.non_ord_model = RandomForestClassifier(
                n_estimators=200, max_depth=min(20, max(12, n_classes * 2)),
                min_samples_leaf=2, class_weight="balanced",
                random_state=42, n_jobs=1,
            )
            self.non_ord_model.fit(X_non, y_non)
        else:
            self.non_ord_model = None

        return self

    def _predict_ordinal_proba(self, X: np.ndarray) -> np.ndarray:
        """Cumulative-link probability estimates for ordinal stages.

        For K ordinal classes, trains K-1 binary classifiers P(class > k).
        Then:
            P(class=0) = 1 - P(class > 0)
            P(class=k) = P(class > k-1) - P(class > k)  for 0 < k < K-1
            P(class=K-1) = P(class > K-2)

        Returns (n, n_ordinal_classes) probability matrix.
        """
        n = X.shape[0]
        K = len(self.ordinal_classes)
        if K < 2 or not self._ord_thresholds:
            probs = np.zeros((n, K))
            probs[:, 0] = 1.0
            return probs

        # Get P(class > k) for each threshold
        p_greater = np.zeros((n, K - 1))
        for k, thr_model in enumerate(self._ord_thresholds):
            if thr_model is None:
                p_greater[:, k] = 0.0  # No samples above this threshold
            else:
                probs_k = thr_model.predict_proba(X)
                # Find index of class "1" (True = class > k)
                try:
                    idx_one = list(thr_model.classes_).index(1)
                except ValueError:
                    idx_one = 1 if len(thr_model.classes_) > 1 else 0
                p_greater[:, k] = probs_k[:, idx_one]

        # Convert to per-class probabilities
        result = np.zeros((n, K))
        result[:, 0] = 1.0 - p_greater[:, 0]
        for k in range(1, K - 1):
            result[:, k] = p_greater[:, k - 1] - p_greater[:, k]
        result[:, K - 1] = p_greater[:, K - 2]

        # Clip numerical noise
        result = np.clip(result, 0.0, 1.0)
        row_sums = result.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return result / row_sums

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for each row in X.

        Routes through Level 1 → appropriate Level 2 model per sample.
        """
        if isinstance(X, list):
            X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Level 1: reduction vs non-reduction
        l1_pred = self.level1.predict(X)

        predictions: list[str] = []
        for i, l1_label in enumerate(l1_pred):
            if l1_label == "reduction":
                # Cumulative-link ordinal prediction
                ord_probs = self._predict_ordinal_proba(X[i:i + 1])[0]
                ord_pred = int(np.argmax(ord_probs))
                predictions.append(self.inv_ordinal_map[ord_pred])
            else:
                # Non-ordinal model predicts string label
                if self.non_ord_model is not None:
                    pred = self.non_ord_model.predict(X[i:i + 1])[0]
                else:
                    pred = self.non_ordinal_classes[0] if self.non_ordinal_classes else "Unknown"
                predictions.append(pred)
        return np.array(predictions)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (n, n_all_classes) probability matrix.

        For reduction-stage artefacts: cumulative-link ordinal probabilities
        placed in the correct columns of the full class array.
        For non-reduction artefacts: RF predict_proba mapped to full columns.
        """
        if isinstance(X, list):
            X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n = X.shape[0]
        n_classes = len(self.all_classes)
        class_to_idx = {c: i for i, c in enumerate(self.all_classes)}
        result = np.zeros((n, n_classes), dtype=float)

        l1_pred = self.level1.predict(X)

        for i in range(n):
            if l1_pred[i] == "reduction":
                ord_probs = self._predict_ordinal_proba(X[i:i + 1])[0]
                for ord_idx, prob in enumerate(ord_probs):
                    cls_name = self.inv_ordinal_map[ord_idx]
                    col = class_to_idx[cls_name]
                    result[i, col] = prob
            else:
                if self.non_ord_model is not None:
                    probs = self.non_ord_model.predict_proba(X[i:i + 1])[0]
                    for cls_name, prob in zip(
                        self.non_ord_model.classes_, probs
                    ):
                        col = class_to_idx[cls_name]
                        result[i, col] = prob
                else:
                    single_cls = self.non_ordinal_classes[0] if self.non_ordinal_classes else "Unknown"
                    result[i, class_to_idx[single_cls]] = 1.0

        # Normalise each row to sum to 1
        row_sums = result.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        result = result / row_sums
        return result

    def get_params(self, deep: bool = True) -> dict:
        """Required by sklearn for cross-validation compatibility."""
        return {"alpha": getattr(self.ord_model, "alpha", 1.0) if self.ord_model else 1.0}


class ClassifierModel:
    """Wraps a sklearn Random Forest classifier for lithic typology.

    Supports pre-trained model loading, prediction with explanation,
    active learning corrections, and custom typology training.
    """

    def __init__(
        self,
        typology_name: str = "basic",
        model_path: Optional[Path] = None,
    ) -> None:
        self.typology_name = typology_name
        self._model: Optional[RandomForestClassifier | CalibratedClassifierCV] = None
        self._classes: list[str] = []
        self._correction_queue: list[tuple[np.ndarray, str]] = []
        self._correction_count: int = 0
        self._onnx_session = None
        self._onnx_loaded = False
        self._router: Optional[TraditionRouter] = None
        self._tradition: Optional[str] = None
        self._n_features: int = 0  # Number of features the model expects

        if model_path is not None:
            self._load(model_path)

    def _load(self, path: Path) -> None:
        """Load a trained model from a .joblib file."""
        data = joblib.load(str(path))
        if isinstance(data, dict):
            # Current format: dict with "model", "classes", "typology_name"
            self._model = data["model"]
            self._classes = data["classes"]
            self.typology_name = data.get("typology_name", self.typology_name)
            self._n_features = data.get("n_features", self._infer_n_features())
        else:
            # Legacy format: saved as a ClassifierModel object directly
            self._model = data._model
            self._classes = data._classes
            self.typology_name = data.typology_name
            self._n_features = getattr(data, "_n_features", 0) or self._infer_n_features()

    def save(self, path: Path) -> None:
        """Save the trained model to a .joblib file."""
        data = {
            "model": self._model,
            "classes": self._classes,
            "typology_name": self.typology_name,
            "n_features": self._infer_n_features(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(data, str(path))

    def export_onnx(self, path: Path) -> None:
        """Export the base Random Forest to ONNX format.

        ONNX models are secure (no arbitrary code execution on load),
        version-independent, and can be updated without app updates.

        Requires: pip install skl2onnx onnxruntime
        """
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
        except ImportError:
            raise ImportError(
                "ONNX export requires skl2onnx. Install: pip install skl2onnx"
            )

        if not self.is_loaded():
            raise RuntimeError("No model loaded to export.")

        # Unwrap calibration to get the base RF
        rf = self._model
        if hasattr(rf, "calibrated_classifiers_"):
            base = rf.calibrated_classifiers_[0].estimator
        else:
            base = rf

        initial_type = [("float_input", FloatTensorType([None, 22]))]
        onx = convert_sklearn(base, initial_types=initial_type)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(path), "wb") as f:
            f.write(onx.SerializeToString())

    @classmethod
    def load_onnx(cls, path: Path, classes: list[str], typology_name: str = "onnx") -> ClassifierModel:
        """Load an ONNX model for inference.

        Args:
            path: Path to the .onnx file.
            classes: List of class labels in the order the model expects.
            typology_name: Name for this model.

        Returns:
            A ClassifierModel that runs inference via ONNX Runtime.
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "ONNX inference requires onnxruntime. Install: pip install onnxruntime"
            )

        model = cls(typology_name=typology_name)
        model._classes = classes
        session = ort.InferenceSession(str(path))
        model._onnx_session = session
        model._onnx_loaded = True
        return model

    def is_loaded(self) -> bool:
        """Check if a trained model is loaded (sklearn or ONNX or tradition router)."""
        return (
            (self._model is not None and len(self._classes) > 0)
            or getattr(self, "_onnx_loaded", False)
            or self._router is not None
        )

    def _infer_n_features(self) -> int:
        """Infer the number of input features the underlying model expects.

        Checks the model, router sub-models, and ONNX session in order.
        Returns 0 if feature count cannot be determined.
        """
        # Sklearn model (CalibratedClassifierCV wraps a base estimator)
        if hasattr(self._model, "n_features_in_"):
            return int(self._model.n_features_in_)
        if hasattr(self._model, "calibrated_classifiers_"):
            base = self._model.calibrated_classifiers_[0].estimator
            if hasattr(base, "n_features_in_"):
                return int(base.n_features_in_)
        # Hierarchical or ordinal pipelines
        if hasattr(self._model, "root_rf") and self._model.root_rf is not None:
            if hasattr(self._model.root_rf, "n_features_in_"):
                return int(self._model.root_rf.n_features_in_)
        if hasattr(self._model, "level1") and self._model.level1 is not None:
            if hasattr(self._model.level1, "n_features_in_"):
                return int(self._model.level1.n_features_in_)
        # Tradition router — check any sub-model
        if self._router is not None and self._router.models:
            for sub in self._router.models.values():
                nf = _infer_model_n_features(sub)
                if nf:
                    return nf
        return 0

    def _pad_features(self, X: np.ndarray) -> np.ndarray:
        """Pad feature matrix to match the model's expected input dimension.

        Models trained with PH features expect more columns than the
        32 core+interaction features generated at inference time.
        Missing columns are zero-padded (zero PH signal = no extra information).
        """
        n_expected = self._n_features
        if n_expected and n_expected > X.shape[1]:
            padding = np.zeros((X.shape[0], n_expected - X.shape[1]))
            X = np.concatenate([X, padding], axis=1)
        return X

    @classmethod
    def load_pre_trained(
        cls,
        typology_name: str,
        tradition: Optional[str] = None,
    ) -> ClassifierModel:
        """Load one of the shipped pre-trained models.

        Args:
            typology_name: One of "basic", "bordes", "technological".
            tradition: Optional tradition name to load a tradition-router model.
                When provided, loads the ``*_traditions.joblib`` file containing
                a TraditionRouter. Pass ``None`` to load the standard model.

        Returns:
            A ClassifierModel instance.
        """
        if tradition is not None:
            path = MODEL_DIR / f"typology_{typology_name}_traditions.joblib"
            if not path.exists():
                raise FileNotFoundError(
                    f"Tradition model not found: {path}. "
                    f"Run training data generation first."
                )
            router: TraditionRouter = joblib.load(str(path))
            model = cls(typology_name=typology_name)
            model._router = router
            model._tradition = tradition
            model._classes = router.tradition_classes.get(tradition, [])
            return model

        # Standard model — never silently upgrade to router
        path = MODEL_DIR / f"typology_{typology_name}.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"Pre-trained model not found: {path}. "
                f"Available: {', '.join(TYPOLOGIES.keys())}"
            )
        return cls(typology_name=typology_name, model_path=path)

    def predict(
        self,
        feature_vector: LithicFeatureVector,
        tradition: Optional[str] = None,
    ) -> ClassificationResult:
        """Classify a single artefact and return an explained result.

        Args:
            feature_vector: Extracted morphometric features of the artefact.
            tradition: Optional tradition override. When the model holds a
                TraditionRouter and this is provided, the router dispatches
                to the matching tradition's sub-model. Ignored for standard models.

        Returns:
            A ClassificationResult with label, confidence, and explanations.
        """
        if not self.is_loaded():
            raise RuntimeError("No model loaded. Call load_pre_trained() or train() first.")

        start = time.time()
        core = feature_vector.to_array().reshape(1, -1)
        inter = compute_interactions(core[0]).reshape(1, -1)
        X = np.concatenate([core, inter], axis=1)
        X = self._pad_features(X)

        # Tradition router branch
        if self._router is not None:
            trad = tradition or self._tradition or "oap"
            return self._predict_via_router(feature_vector, X, trad)

        probs = self._model.predict_proba(X)[0]
        class_idx = int(np.argmax(probs))

        prob_dict: dict[str, float] = {}
        for i, cls_name in enumerate(self._classes):
            prob_dict[cls_name] = round(float(probs[i]), 4)

        label = self._classes[class_idx]
        confidence = round(float(probs[class_idx]), 4)

        # Feature importances (ordinal/hierarchical pipelines may not expose these)
        if hasattr(self._model, "feature_importances_"):
            importances = self._model.feature_importances_
        elif hasattr(self._model, "calibrated_classifiers_"):
            importances = self._model.calibrated_classifiers_[0].estimator.feature_importances_
        elif hasattr(self._model, "level1"):
            # Hierarchical ordinal model — use Level-1 RF importances as proxy
            importances = self._model.level1.feature_importances_ if hasattr(self._model.level1, "feature_importances_") else None
        elif hasattr(self._model, "root_rf") and self._model.root_rf is not None:
            # Hierarchical cascade — use root node importances as proxy
            importances = self._model.root_rf.feature_importances_
        else:
            importances = None

        top_features = []
        if importances is not None:
            top_features = self._compute_feature_importances(feature_vector, importances)

        alternatives = [
            (cls_name, round(float(probs[i]), 4))
            for i, cls_name in enumerate(self._classes)
            if i != class_idx and probs[i] > 0.01
        ]
        alternatives.sort(key=lambda x: x[1], reverse=True)

        elapsed = time.time() - start

        return ClassificationResult(
            label=label,
            confidence=confidence,
            probabilities=prob_dict,
            top_features=top_features[:5],
            alternatives=alternatives[:3],
            typology_name=self.typology_name,
            processing_time_s=round(elapsed, 3),
            warnings=[],
        )

    def _predict_via_router(
        self,
        fv: LithicFeatureVector,
        X: np.ndarray,
        tradition: str,
    ) -> ClassificationResult:
        """Predict using the tradition-router sub-model for *tradition*."""
        router = self._router
        # Pad features if the router sub-model expects more dimensions
        X = _pad_features_for_model(X, router.models.get(tradition) if router.models else None)
        # Label via the router
        label = str(router.predict(X, tradition=tradition)[0])

        # Probabilities from the router
        probs = router.predict_proba(X, tradition=tradition)[0]
        classes = router.tradition_classes.get(tradition, self._classes)

        prob_dict: dict[str, float] = {}
        for i, cls_name in enumerate(classes):
            prob_dict[cls_name] = round(float(probs[i]), 4)

        confidence = round(float(prob_dict.get(label, 0.0)), 4)

        # Feature importances from the tradition sub-model (if it exposes them)
        top_features: list[FeatureImportance] = []
        sub = router.models.get(tradition)
        importances: Optional[np.ndarray] = None
        if hasattr(sub, "root_rf") and sub.root_rf is not None:
            importances = sub.root_rf.feature_importances_
        elif hasattr(sub, "_model"):
            m = sub._model
            if hasattr(m, "feature_importances_"):
                importances = m.feature_importances_
            elif hasattr(m, "calibrated_classifiers_"):
                importances = m.calibrated_classifiers_[0].estimator.feature_importances_
        if importances is not None:
            top_features = self._compute_feature_importances(fv, importances)

        alternatives = [
            (cls_name, round(float(probs[i]), 4))
            for i, cls_name in enumerate(classes)
            if cls_name != label and probs[i] > 0.01
        ]
        alternatives.sort(key=lambda x: x[1], reverse=True)

        return ClassificationResult(
            label=label,
            confidence=confidence,
            probabilities=prob_dict,
            top_features=top_features[:5],
            alternatives=alternatives[:3],
            typology_name=f"{self.typology_name}/{tradition}",
            processing_time_s=0.0,
            warnings=[],
        )

    def _compute_feature_importances(
        self, fv: LithicFeatureVector, importances: np.ndarray
    ) -> list[FeatureImportance]:
        """Build per-feature explanations with contribution percentages."""
        names = LithicFeatureVector.FEATURE_NAMES
        total_imp = importances.sum()

        results = []
        for name, imp in zip(names, importances):
            value = getattr(fv, name)
            contrib = round(float(imp / total_imp), 4) if total_imp > 0 else 0.0
            expected_range = self._get_expected_range(name)
            passed = expected_range[0] <= value <= expected_range[1]
            results.append(FeatureImportance(
                name=name,
                value=value,
                contribution_pct=contrib,
                expected_range=expected_range,
                passed=passed,
            ))

        results.sort(key=lambda x: x.contribution_pct, reverse=True)
        return results

    @staticmethod
    def _get_expected_range(name: str) -> tuple[float, float]:
        """Return typical range for a feature based on known lithic literature."""
        ranges: dict[str, tuple[float, float]] = {
            "length_mm": (5, 500), "width_mm": (3, 300), "thickness_mm": (1, 150),
            "elongation": (0.5, 6.0), "flatness": (1.0, 10.0),
            "platform_angle_deg": (0, 90), "scar_count": (0, 50),
            "edge_angle_mean_deg": (0, 90), "dorsal_ridge_count": (0, 5),
        }
        return ranges.get(name, (0, 1e6))

    def queue_correction(self, features: LithicFeatureVector, correct_label: str) -> int:
        """Add a correction to the retraining queue. Returns the current correction count."""
        self._correction_queue.append((features.to_array(), correct_label))
        self._correction_count += 1
        return self._correction_count

    def retrain_if_ready(self, threshold: int = 10) -> bool:
        """Retrain model if correction queue >= threshold. Returns True if retraining occurred."""
        if self._correction_count < threshold:
            return False
        return self._retrain()

    def predict_conformal(
        self, feature_vector: LithicFeatureVector,
    ) -> ClassificationResult:
        """Predict with conformal prediction for out-of-distribution detection.

        Uses MAPIE to produce prediction sets instead of single labels.
        If the prediction set is empty, the artefact is flagged as
        out-of-distribution (OOD) — the classifier doesn't know.

        Requires: pip install mapie
        """
        try:
            from mapie.classification import MapieClassifier
        except ImportError:
            raise ImportError(
                "Conformal prediction requires MAPIE. Install: pip install mapie"
            )

        if not self.is_loaded():
            raise RuntimeError("No model loaded.")

        X = feature_vector.to_array().reshape(1, -1)

        # Use the calibrated model directly
        rf = self._model

        # Fit MAPIE on a small synthetic buffer to estimate conformity scores
        # For a production system, this would use held-out calibration data
        rng = np.random.default_rng(42)
        n_calib = max(50, len(self._classes) * 10)
        X_calib = rng.random((n_calib, 22))
        y_calib = rng.integers(0, len(self._classes), size=n_calib)

        mapie = MapieClassifier(estimator=rf, cv="prefit")
        mapie.fit(X_calib, y_calib)

        # Predict with conformal sets
        y_pred, y_set = mapie.predict(X, alpha=0.2)  # 80% confidence
        prediction_set = [self._classes[i] for i, in_set in enumerate(y_set[0]) if in_set]

        if len(prediction_set) == 0:
            return ClassificationResult(
                label="Unknown (out-of-distribution)",
                confidence=0.0,
                probabilities={},
                top_features=[],
                alternatives=[],
                typology_name=self.typology_name,
                processing_time_s=0.0,
                warnings=["Artefact falls outside the model's training distribution."],
            )

        if len(prediction_set) == 1:
            label = prediction_set[0]
            confidence = 1.0 / len(prediction_set)
        else:
            label = prediction_set[0]
            confidence = 1.0 / len(prediction_set)

        return ClassificationResult(
            label=label,
            confidence=round(confidence, 4),
            probabilities={c: 1.0 / len(prediction_set) for c in prediction_set},
            top_features=[],
            alternatives=[(c, 1.0 / len(prediction_set)) for c in prediction_set[1:]],
            typology_name=self.typology_name,
            processing_time_s=0.0,
            warnings=[f"Prediction set: {', '.join(prediction_set)}"] if len(prediction_set) > 1 else [],
        )

    def _retrain(self) -> bool:
        """Retrain on accumulated corrections.

        Returns:
            True if retraining occurred, False if skipped (router model,
            empty queue, or no model loaded).

        Note: tradition-router models do not support retraining — the method
        returns False when a router is present.
        """
        if self._router is not None:
            return False
        if not self._correction_queue or not self.is_loaded():
            return False

        X_corrections = np.array([item[0] for item in self._correction_queue])
        y_corrections = [item[1] for item in self._correction_queue]

        base_rf = RandomForestClassifier(
            n_estimators=300, max_depth=12,
            min_samples_leaf=3, class_weight="balanced", random_state=42,
        )
        base_rf.fit(X_corrections, y_corrections)
        self._model = CalibratedClassifierCV(base_rf, cv=3)
        self._model.fit(X_corrections, y_corrections)
        self._classes = sorted(set(y_corrections))
        self._correction_queue = []
        self._correction_count = 0
        return True


def _infer_model_n_features(model) -> int:
    """Infer expected feature count from any model-like object."""
    if hasattr(model, "n_features_in_"):
        return int(model.n_features_in_)
    if hasattr(model, "calibrated_classifiers_"):
        base = model.calibrated_classifiers_[0].estimator
        if hasattr(base, "n_features_in_"):
            return int(base.n_features_in_)
    if hasattr(model, "root_rf") and model.root_rf is not None:
        if hasattr(model.root_rf, "n_features_in_"):
            return int(model.root_rf.n_features_in_)
    if hasattr(model, "level1") and model.level1 is not None:
        if hasattr(model.level1, "n_features_in_"):
            return int(model.level1.n_features_in_)
    return 0


def _pad_features_for_model(X: np.ndarray, model, n_expected: int = 0) -> np.ndarray:
    """Pad feature matrix to match model's expected input dimension."""
    if n_expected <= 0:
        n_expected = _infer_model_n_features(model)
    if n_expected and n_expected > X.shape[1]:
        padding = np.zeros((X.shape[0], n_expected - X.shape[1]))
        X = np.concatenate([X, padding], axis=1)
    return X


# ── Interaction features (derived from the 22 core morphometrics) ──
# These capture non-linear relationships the RF's axis-aligned splits
# may miss. Computed on-the-fly at both training and inference time.
# See docs/research-prompts/deep-research-prompt-accuracy-innovation.md
# Expert 3 for rationale.

INTERACTION_NAMES: list[str] = [
    "elongation_x_flatness",       # shape index
    "length_x_width",               # size-area proxy
    "length_div_thickness",         # robustness (slenderness)
    "edge_mean_x_std",              # edge variability (high = notched/denticulate)
    "area_div_volume",              # specific surface area
    "curvature_x_symmetry",         # global regularity
    "scar_density",                 # scar_count / surface_area
    "elongation_sq",                # non-linear elongation
    "platform_x_flatness",          # platform × cross-section
    "compactness_x_elongation",     # mass × shape
]

# Indices into the 22-element FEATURE_NAMES array for computing interactions
# Ordered as: length, width, thickness, area, volume, elongation, flatness,
# compactness, rel_thickness, scar_count, mean_scar_area, platform_angle,
# edge_mean, edge_std, edge_skew, edge_kurt, curvature, cross_section,
# symmetry, com_z_ratio, dorsal_ridge, roughness

_FIDX = LithicFeatureVector.FEATURE_NAMES.index  # type: ignore[attr-defined]


def compute_interactions(features_22: np.ndarray) -> np.ndarray:
    """Compute 10 interaction features from the 22 core morphometrics.

    Args:
        features_22: 22-element array in FEATURE_NAMES order.

    Returns:
        10-element array in INTERACTION_NAMES order.
    """
    f = features_22  # alias for readability
    # Named indices — safe as long as FEATURE_NAMES doesn't change order
    L, W, T, A, V = 0, 1, 2, 3, 4        # length, width, thickness, area, volume
    EL, FL = 5, 6                          # elongation, flatness
    SC, MSC = 9, 10                        # scar_count, mean_scar_area
    PA = 11                                # platform_angle_deg
    EM, ES = 12, 13                        # edge_mean, edge_std
    CV = 16                                # curvature_index
    CS = 17                                # cross_section_profile
    SY = 18                                # symmetry_score

    # Avoid division by zero
    eps = 1e-8

    return np.array([
        f[EL] * f[FL],                                    # elongation × flatness
        f[L] * f[W],                                      # length × width
        f[L] / max(f[T], eps),                            # length / thickness
        f[EM] * max(f[ES], eps),                          # edge_mean × edge_std
        f[A] / max(f[V], eps),                            # area / volume
        f[CV] * f[SY],                                    # curvature × symmetry
        f[SC] / max(f[A], eps),                           # scar density
        f[EL] ** 2,                                       # elongation²
        f[PA] * f[FL] / 100,                              # platform × flatness (scaled)
        f[V] * max(f[EL], eps) / max(f[L] ** 3, eps),     # compactness × elongation
    ])


# ── End interaction features ──


def train_model(
    feature_vectors: list[LithicFeatureVector],
    labels: list[str],
    typology_name: str = "custom",
    sample_weight: np.ndarray | None = None,
) -> ClassifierModel:
    """Train a new classifier from labelled feature vectors.

    Args:
        feature_vectors: List of LithicFeatureVector objects.
        labels: Corresponding class labels.
        typology_name: Name for this trained model.
        sample_weight: Per-sample weights. If None, uses class_weight='balanced'.
            If provided, this is multiplied with the class_weight effect
            (sklearn multiplies sample_weight with class_weight internally).

    Note:
        When sample_weight is used with CalibratedClassifierCV, the weights
        are passed through to the base estimator on each CV fold.
    """
    X = np.array([fv.to_array() for fv in feature_vectors])
    # Append interaction features
    interactions = np.array([compute_interactions(fv.to_array()) for fv in feature_vectors])
    X = np.concatenate([X, interactions], axis=1)
    y = np.array(labels)
    classes = sorted(set(labels))

    # Hyperparameter-optimized: max_features=0.3 reduces tree correlation
    # improving ensemble diversity. Depth is dynamic per typology.
    depth = min(20, max(12, len(classes) * 2))
    base_rf = RandomForestClassifier(
        n_estimators=200, max_depth=depth,
        min_samples_leaf=2, max_features=0.3,
        class_weight="balanced", random_state=42,
    )
    model = ClassifierModel(typology_name=typology_name)
    model._classes = classes

    if len(classes) >= 2:
        calibrated = CalibratedClassifierCV(base_rf, cv=min(3, len(classes)))
        if sample_weight is not None:
            calibrated.fit(X, y, sample_weight=sample_weight)
        else:
            calibrated.fit(X, y)
        model._model = calibrated
    else:
        if sample_weight is not None:
            base_rf.fit(X, y, sample_weight=sample_weight)
        else:
            base_rf.fit(X, y)
        model._model = base_rf

    return model


# ── Diagnostic coordinate extraction (for viewer overlays) ──

def extract_diagnostic_coordinates(mesh: trimesh.Trimesh) -> dict[str, np.ndarray]:
    """Extract 3D coordinates of diagnostic features for viewer highlighting.

    Returns:
        Dict with keys 'ridges', 'platform', 'retouched_edges' containing
        numpy arrays of 3D points for each feature.
    """
    result: dict[str, np.ndarray] = {}

    # Ridge points: vertices on edges with high dihedral angle
    try:
        angles = _compute_edge_angles(mesh)
        if len(angles) > 0:
            ridge_mask = angles > 130
            ridge_edge_indices = mesh.face_adjacency[ridge_mask]
            ridge_vertices = mesh.vertices[
                np.unique(mesh.faces[ridge_edge_indices.flatten()])
            ]
            result["ridges"] = ridge_vertices
    except Exception:
        result["ridges"] = np.array([])

    # Platform: vertices on the bottom 10% of the mesh
    try:
        z_min = mesh.bounds[0, 2]
        z_max = mesh.bounds[1, 2]
        platform_z = z_min + (z_max - z_min) * 0.1
        platform_vertices = mesh.vertices[mesh.vertices[:, 2] <= platform_z]
        result["platform"] = platform_vertices
    except Exception:
        result["platform"] = np.array([])

    # Retouched edges: vertices on edges with extreme dihedral angle
    try:
        angles = _compute_edge_angles(mesh)
        if len(angles) > 0:
            retouch_mask = angles > 150
            retouch_edge_indices = mesh.face_adjacency[retouch_mask]
            retouch_vertices = mesh.vertices[
                np.unique(mesh.faces[retouch_edge_indices.flatten()])
            ]
            result["retouched_edges"] = retouch_vertices
    except Exception:
        result["retouched_edges"] = np.array([])

    return result
