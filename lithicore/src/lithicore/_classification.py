"""_classification.py — Lithic typology classification pipeline.

exports: extract_features(mesh) -> LithicFeatureVector
         ClassifierModel
         train_model(features, labels, typology_name) -> ClassifierModel
used_by: lithicope classification panel, CLI
rules:   Pure functions + model wrapper. No GUI imports.
         Feature extraction ~0.1s per mesh. Model training ~1-2s typical.
agent:   deepseek-v4-flash | 2026-05-27 | Initial implementation
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
MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"

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

        if model_path is not None:
            self._load(model_path)

    def _load(self, path: Path) -> None:
        """Load a trained model from a .joblib file."""
        data = joblib.load(str(path))
        self._model = data["model"]
        self._classes = data["classes"]
        self.typology_name = data.get("typology_name", self.typology_name)

    def save(self, path: Path) -> None:
        """Save the trained model to a .joblib file."""
        data = {
            "model": self._model,
            "classes": self._classes,
            "typology_name": self.typology_name,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(data, str(path))

    def is_loaded(self) -> bool:
        """Check if a trained model is loaded."""
        return self._model is not None and len(self._classes) > 0

    @classmethod
    def load_pre_trained(cls, typology_name: str) -> ClassifierModel:
        """Load one of the shipped pre-trained models."""
        path = MODEL_DIR / f"typology_{typology_name}.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"Pre-trained model not found: {path}. "
                f"Available: {', '.join(TYPOLOGIES.keys())}"
            )
        return cls(typology_name=typology_name, model_path=path)

    def predict(self, feature_vector: LithicFeatureVector) -> ClassificationResult:
        """Classify a single artefact and return an explained result."""
        if not self.is_loaded():
            raise RuntimeError("No model loaded. Call load_pre_trained() or train() first.")

        start = time.time()
        X = feature_vector.to_array().reshape(1, -1)

        probs = self._model.predict_proba(X)[0]
        class_idx = int(np.argmax(probs))

        prob_dict: dict[str, float] = {}
        for i, cls_name in enumerate(self._classes):
            prob_dict[cls_name] = round(float(probs[i]), 4)

        label = self._classes[class_idx]
        confidence = round(float(probs[class_idx]), 4)

        # Feature importances
        if hasattr(self._model, "calibrated_classifiers_"):
            importances = self._model.calibrated_classifiers_[0].base_estimator.feature_importances_
        else:
            importances = self._model.feature_importances_

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
        self._retrain()
        return True

    def _retrain(self) -> None:
        """Retrain on accumulated corrections."""
        if not self._correction_queue or not self.is_loaded():
            return

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


def train_model(
    feature_vectors: list[LithicFeatureVector],
    labels: list[str],
    typology_name: str = "custom",
) -> ClassifierModel:
    """Train a new classifier from labelled feature vectors."""
    X = np.array([fv.to_array() for fv in feature_vectors])
    y = np.array(labels)
    classes = sorted(set(labels))

    base_rf = RandomForestClassifier(
        n_estimators=500, max_depth=12,
        min_samples_leaf=3, class_weight="balanced", random_state=42,
    )
    model = ClassifierModel(typology_name=typology_name)
    model._classes = classes

    if len(classes) >= 2:
        calibrated = CalibratedClassifierCV(base_rf, cv=min(3, len(classes)))
        calibrated.fit(X, y)
        model._model = calibrated
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
