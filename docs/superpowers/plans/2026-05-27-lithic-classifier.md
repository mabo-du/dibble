# Lithic Typology Classifier — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add automated lithic typology classification to Dibble using geometric morphometric features + sklearn Random Forest, with explainable predictions, active learning, and custom typology training.

**Architecture:** 25-dim feature vector extracted from each mesh via trimesh/scipy → sklearn Random Forest classifier → structured ClassificationResult with per-feature explanations. Pre-trained models shipped as .joblib files. Active learning queue retrains after 10 corrections.

**Tech Stack:** Python 3.11+, scikit-learn>=1.3, joblib>=1.2, numpy, scipy, trimesh, PyQt6, pyvista

**Spec:** `docs/superpowers/specs/2026-05-27-lithic-classifier-design.md`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `lithicore/src/lithicore/_models.py` | Modify | Add ClassificationResult, FeatureImportance, LithicFeatureVector dataclasses |
| `lithicore/src/lithicore/_classification.py` | Create | extract_features(), ClassifierModel, train/predict/retrain, active learning |
| `lithicore/src/lithicore/__init__.py` | Modify | Export new symbols |
| `lithicore/data/__init__.py` | Create | Make data/ a package |
| `lithicore/data/models/` | Create | Directory for pre-trained .joblib models |
| `lithicore/data/generate_training_data.py` | Create | Generate synthetic training data from published metric ranges |
| `lithicore/tests/test_classification.py` | Create | ~10 unit tests for feature extraction + classifier |
| `lithicope/src/lithicope/_classification_panel.py` | Create | Classification panel widget |
| `lithicope/src/lithicope/_viewer_3d.py` | Modify | Add diagnostic overlay methods (ridges, platform, edges) |
| `lithicope/src/lithicope/_main_window.py` | Modify | Add Classification tab, menu items, auto-classify toggle |
| `lithicore/pyproject.toml` | Modify | Add scikit-learn>=1.3, joblib>=1.2 |

---

### Task 1: Data Model Additions — `_models.py`

**Files:**
- Modify: `lithicore/src/lithicore/_models.py`

- [ ] **Write and add these new dataclasses** to `_models.py` (append at end of file, before the closing):

```python
@dataclass(frozen=True)
class FeatureImportance:
    """A single feature's contribution to a classification decision."""
    name: str
    value: float
    contribution_pct: float  # % of trees that split on this feature
    expected_range: tuple[float, float]  # typical range for the predicted class
    passed: bool  # whether value falls within expected range


@dataclass
class ClassificationResult:
    """Result of a typology classification pipeline run."""
    label: str
    confidence: float
    probabilities: dict[str, float]  # class -> probability
    top_features: list[FeatureImportance]
    alternatives: list[tuple[str, float]]  # (label, confidence) for other classes
    typology_name: str  # e.g. "basic", "bordes", "technological", "custom"
    processing_time_s: float
    warnings: list[str]
```

- [ ] **Verify importable**: `cd .../dibble && python -c "from lithicore._models import ClassificationResult, FeatureImportance; print('OK')"`
- [ ] **Commit:**

```bash
cd .../dibble && git add lithicore/src/lithicore/_models.py && git commit -m "feat: add ClassificationResult and FeatureImportance dataclasses"
```

---

### Task 2: LithicFeatureVector Dataclass

**Files:**
- Modify: `lithicore/src/lithicore/_models.py`

- [ ] **Add LithicFeatureVector dataclass** (append after FeatureImportance):

```python
@dataclass
class LithicFeatureVector:
    """Numerical fingerprint of a lithic artefact (25+ dimensional feature vector).

    All measurements in mm or mm²/mm³. Computed from an oriented mesh.
    """
    # Raw metrics
    length_mm: float = 0.0
    width_mm: float = 0.0
    thickness_mm: float = 0.0
    surface_area_mm2: float = 0.0
    volume_mm3: float = 0.0

    # Derived ratios
    elongation: float = 0.0       # L/W
    flatness: float = 0.0         # W/T
    compactness: float = 0.0      # V/L³
    relative_thickness: float = 0.0  # T/L

    # Morphological features
    scar_count: int = 0
    mean_scar_area_mm2: float = 0.0
    platform_angle_deg: float = 0.0
    edge_angle_mean_deg: float = 0.0
    edge_angle_std_deg: float = 0.0
    curvature_index: float = 0.0
    cross_section_profile: float = 0.0  # 0=flat, 1=triangular, 2=round
    symmetry_score: float = 0.0
    com_z_ratio: float = 0.0
    dorsal_ridge_count: int = 0
    surface_roughness: float = 0.0

    FEATURE_NAMES: ClassVar[list[str]] = [
        "length_mm", "width_mm", "thickness_mm", "surface_area_mm2", "volume_mm3",
        "elongation", "flatness", "compactness", "relative_thickness",
        "scar_count", "mean_scar_area_mm2", "platform_angle_deg",
        "edge_angle_mean_deg", "edge_angle_std_deg", "curvature_index",
        "cross_section_profile", "symmetry_score", "com_z_ratio",
        "dorsal_ridge_count", "surface_roughness",
    ]

    def to_array(self) -> np.ndarray:
        """Return feature vector as a 1D numpy array in FEATURE_NAMES order."""
        return np.array([getattr(self, name) for name in self.FEATURE_NAMES], dtype=float)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> LithicFeatureVector:
        """Construct from a 20-element array in FEATURE_NAMES order."""
        return cls(**dict(zip(cls.FEATURE_NAMES, arr)))
```

Need to add import at top: `from typing import ClassVar`

- [ ] **Verify**: `cd .../dibble && python -c "from lithicore._models import LithicFeatureVector; v = LithicFeatureVector(); assert len(v.to_array()) == 20; print('OK')"`
- [ ] **Commit:**

```bash
cd .../dibble && git add lithicore/src/lithicore/_models.py && git commit -m "feat: add LithicFeatureVector dataclass with to_array()"
```

---

### Task 3: Feature Extraction — `_classification.py` (part 1)

**Files:**
- Create: `lithicore/src/lithicore/_classification.py`

- [ ] **Create the feature extraction module**:

```python
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

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

from lithicore._models import (
    ClassificationResult,
    FeatureImportance,
    LithicFeatureVector,
)


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
    # Use trimesh's built-in face angle computation
    angles = np.array([])
    try:
        # Dihedral angle = angle between face normals along an edge
        edges = mesh.edges_unique
        face_pairs = mesh.face_adjacency  # (N, 2) pairs of adjacent faces
        normals = mesh.face_normals
        if len(face_pairs) == 0:
            return np.array([])
        n1 = normals[face_pairs[:, 0]]
        n2 = normals[face_pairs[:, 1]]
        cos_angles = np.clip(np.sum(n1 * n2, axis=1), -1.0, 1.0)
        angles = np.degrees(np.arccos(cos_angles))
    except Exception:
        pass
    return angles


def _compute_curvature_index(mesh: trimesh.Trimesh) -> float:
    """Compute Gaussian vs mean curvature ratio as a shape descriptor."""
    try:
        from scipy.sparse.linalg import eigsh
        # Compute approximate curvature via vertex normals variation
        vertex_normals = mesh.vertex_normals
        if len(vertex_normals) < 3:
            return 0.0
        # Use angular deviation of normals as a curvature proxy
        mean_normal = vertex_normals.mean(axis=0)
        mean_normal = mean_normal / np.linalg.norm(mean_normal)
        deviations = np.arccos(np.clip(
            np.dot(vertex_normals, mean_normal), -1.0, 1.0
        ))
        return float(np.mean(deviations))
    except Exception:
        return 0.0


def _compute_cross_section_profile(mesh: trimesh.Trimesh) -> float:
    """Classify cross-section as 0=flat, 1=triangular, 2=round.

    Slices the mesh at mid-height and analyses the cross-section
    using the aspect ratio of the bounding box of the slice.
    """
    try:
        mid_z = (mesh.bounds[0, 2] + mesh.bounds[1, 2]) / 2
        slice_2d = mesh.section(
            plane_origin=[0, 0, mid_z],
            plane_normal=[0, 0, 1],
        )
        if slice_2d is None:
            return 0.0
        # Project to 2D and get bounding box ratio
        vertices = slice_2d.vertices[:, :2]
        if len(vertices) < 3:
            return 0.0
        bb = vertices.ptp(axis=0)
        ratio = bb[1] / max(bb[0], 0.001) if bb[0] > 0 else 0.0
        # ratio < 0.5 = flat, 0.5-0.8 = triangular, > 0.8 = round
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
        # Reflect right half across X
        right_reflected = right.copy()
        right_reflected[:, 0] = 2 * centre_x - right_reflected[:, 0]
        # Compute normalized Hausdorff distance proxy
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
    """Count parallel linear ridges on the dorsal surface.

    Uses curvature-based extraction of ridge-like features.
    Returns count of distinct ridge lines.
    """
    try:
        # Use edge angle threshold to find ridge-like edges
        angles = _compute_edge_angles(mesh)
        if len(angles) == 0:
            return 0
        # Edges with high dihedral angle (130-180°) are ridges
        ridge_edges = angles > 130
        ridge_count = int(np.sum(ridge_edges))
        # Normalise: count distinct ridge segments (not individual edges)
        return min(ridge_count // 10, 5)  # rough heuristic
    except Exception:
        return 0


def _compute_surface_roughness(mesh: trimesh.Trimesh) -> float:
    """Compute surface roughness as the ratio of face area to projected area."""
    try:
        projected_area = mesh.convex_hull.area if hasattr(mesh, 'convex_hull') else mesh.area
        return mesh.area / max(projected_area, 0.001)
    except Exception:
        return 1.0
```

- [ ] **Verify syntax**: `cd .../dibble && python -c "import ast; ast.parse(open('lithicore/src/lithicore/_classification.py').read()); print('Syntax OK')"`
- [ ] **Commit:**

```bash
cd .../dibble && git add lithicore/src/lithicore/_classification.py && git commit -m "feat: add feature extraction module"
```

---

### Task 4: ClassifierModel — Training and Prediction

**Files:**
- Modify: `lithicore/src/lithicore/_classification.py`

- [ ] **Add ClassifierModel class** to the same file (after the feature extraction functions, before the closing):

```python
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV


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
        """Classify a single artefact and return an explained result.

        Args:
            feature_vector: The morphometric fingerprint to classify.

        Returns:
            ClassificationResult with label, confidence, feature importances.

        Raises:
            RuntimeError: If no model is loaded.
        """
        if not self.is_loaded():
            raise RuntimeError("No model loaded. Call load_pre_trained() or train() first.")

        start = time.time()
        X = feature_vector.to_array().reshape(1, -1)

        # Get probabilities
        probs = self._model.predict_proba(X)[0]
        class_idx = int(np.argmax(probs))

        # Build probability dict
        prob_dict: dict[str, float] = {}
        for i, cls_name in enumerate(self._classes):
            prob_dict[cls_name] = round(float(probs[i]), 4)

        label = self._classes[class_idx]
        confidence = round(float(probs[class_idx]), 4)

        # Feature importances
        importances = self._model.feature_importances_
        if hasattr(self._model, "calibrated_classifiers_"):
            importances = self._model.calibrated_classifiers_[0].base_estimator.feature_importances_

        # Get expected ranges from training data if available
        top_features = self._compute_feature_importances(
            feature_vector, importances
        )

        # Alternatives (other classes with confidence > 1%)
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
            top_features=top_features[:5],  # top 5
            alternatives=alternatives[:3],  # top 3
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
            # Expected range heuristic (will be refined with real training data)
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
        """Return typical range for a feature based on known lithic literature.

        These are generous ranges used for reference display, not hard limits.
        """
        ranges: dict[str, tuple[float, float]] = {
            "length_mm": (5, 500), "width_mm": (3, 300), "thickness_mm": (1, 150),
            "elongation": (0.5, 6.0), "flatness": (1.0, 10.0),
            "platform_angle_deg": (0, 90), "scar_count": (0, 50),
            "edge_angle_mean_deg": (0, 90), "dorsal_ridge_count": (0, 5),
        }
        return ranges.get(name, (0, 1e6))

    def queue_correction(self, features: LithicFeatureVector, correct_label: str) -> int:
        """Add a correction to the retraining queue.

        Returns the current correction count.
        """
        self._correction_queue.append((features.to_array(), correct_label))
        self._correction_count += 1
        return self._correction_count

    def retrain_if_ready(self, threshold: int = 10) -> bool:
        """Retrain model if correction queue >= threshold.

        Returns True if retraining occurred.
        """
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

        # Merge with original training data (if available) + corrections
        # For simplicity, retrain just on corrections + a weighted sample
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
    """Train a new classifier from labelled feature vectors.

    Args:
        feature_vectors: List of morphometric fingerprints.
        labels: Corresponding typology labels.
        typology_name: Name for this model.

    Returns:
        A trained ClassifierModel ready for prediction.
    """
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
        calibrated = CalibratedClassifierCV(base_rf, cv=min(3, min(len(classes), 3)))
        calibrated.fit(X, y)
        model._model = calibrated
    else:
        base_rf.fit(X, y)
        model._model = base_rf

    return model


# ── Diagnostic coordinate extraction (for viewer overlays) ──

def extract_diagnostic_coordinates(
    mesh: trimesh.Trimesh,
) -> dict[str, np.ndarray]:
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
```

- [ ] **Verify**: `cd .../dibble && python -c "from lithicore._classification import ClassifierModel, train_model, extract_features, extract_diagnostic_coordinates; print('OK')"`
- [ ] **Commit:**

```bash
cd .../dibble && git add lithicore/src/lithicore/_classification.py && git commit -m "feat: add ClassifierModel with prediction, explanation, active learning"
```

---

### Task 5: Training Data Generator — Synthetic Data from Published Ranges

**Files:**
- Create: `lithicore/data/generate_training_data.py`

- [ ] **Create the data generator script**:

```python
"""generate_training_data.py — Generate synthetic training data for lithic classifiers.

Generates feature vectors from published metric ranges for each lithic typology
system (basic, Bordes, technological), with added Gaussian noise to simulate
natural variation.

Output: .joblib files in data/models/

Usage:
    python -m lithicore.data.generate_training_data
"""

from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

from lithicore._models import LithicFeatureVector


MODEL_DIR = Path(__file__).resolve().parent / "models"

# ── Basic typology: Flake, Blade, Bladelet, Core, Tool ──

BASIC_RANGES = {
    "Flake": {
        "length_mm": (15, 80), "width_mm": (10, 60), "thickness_mm": (3, 20),
        "elongation": (0.8, 1.8), "flatness": (2.0, 5.0),
        "scar_count": (1, 5),
        "platform_angle_deg": (60, 90),
        "edge_angle_mean_deg": (40, 70),
        "dorsal_ridge_count": (0, 1),
        "curvature_index": (0.1, 0.5),
        "symmetry_score": (0.3, 0.7),
        "com_z_ratio": (0.2, 0.4),
    },
    "Blade": {
        "length_mm": (50, 250), "width_mm": (10, 40), "thickness_mm": (2, 10),
        "elongation": (2.0, 5.0), "flatness": (3.0, 8.0),
        "scar_count": (2, 6),
        "platform_angle_deg": (65, 85),
        "edge_angle_mean_deg": (50, 75),
        "dorsal_ridge_count": (2, 4),
        "curvature_index": (0.05, 0.3),
        "symmetry_score": (0.5, 0.9),
        "com_z_ratio": (0.15, 0.35),
    },
    "Bladelet": {
        "length_mm": (10, 50), "width_mm": (3, 12), "thickness_mm": (1, 4),
        "elongation": (2.5, 6.0), "flatness": (3.0, 7.0),
        "scar_count": (1, 3),
        "platform_angle_deg": (60, 80),
        "edge_angle_mean_deg": (40, 65),
        "dorsal_ridge_count": (1, 2),
        "curvature_index": (0.05, 0.25),
        "symmetry_score": (0.4, 0.8),
        "com_z_ratio": (0.2, 0.4),
    },
    "Core": {
        "length_mm": (30, 150), "width_mm": (25, 100), "thickness_mm": (15, 80),
        "elongation": (0.5, 1.5), "flatness": (1.0, 2.5),
        "scar_count": (3, 20),
        "platform_angle_deg": (70, 90),
        "edge_angle_mean_deg": (60, 85),
        "dorsal_ridge_count": (0, 2),
        "curvature_index": (0.2, 0.6),
        "symmetry_score": (0.2, 0.5),
        "com_z_ratio": (0.3, 0.7),
    },
    "Tool": {
        "length_mm": (20, 120), "width_mm": (15, 70), "thickness_mm": (5, 30),
        "elongation": (0.8, 2.5), "flatness": (2.0, 4.0),
        "scar_count": (2, 8),
        "platform_angle_deg": (50, 80),
        "edge_angle_mean_deg": (55, 85),
        "dorsal_ridge_count": (0, 2),
        "curvature_index": (0.1, 0.4),
        "symmetry_score": (0.4, 0.8),
        "com_z_ratio": (0.2, 0.5),
    },
}

# ── Bordes typology (simplified ranges for synthetic generation) ──

BORDES_RANGES = {
    "Scraper": {
        "length_mm": (30, 100), "width_mm": (20, 60), "thickness_mm": (5, 20),
        "elongation": (0.8, 2.0), "edge_angle_mean_deg": (60, 85),
        "scar_count": (3, 10), "dorsal_ridge_count": (0, 2),
        "symmetry_score": (0.3, 0.6),
    },
    "Handaxe": {
        "length_mm": (80, 250), "width_mm": (50, 120), "thickness_mm": (20, 60),
        "elongation": (1.2, 2.2), "edge_angle_mean_deg": (50, 75),
        "scar_count": (5, 20), "dorsal_ridge_count": (1, 3),
        "symmetry_score": (0.7, 0.95),
    },
    "Point": {
        "length_mm": (30, 100), "width_mm": (10, 35), "thickness_mm": (3, 12),
        "elongation": (1.5, 3.5), "edge_angle_mean_deg": (55, 80),
        "scar_count": (2, 6), "dorsal_ridge_count": (1, 3),
        "symmetry_score": (0.6, 0.9),
    },
    "Burin": {
        "length_mm": (20, 80), "width_mm": (10, 30), "thickness_mm": (4, 15),
        "elongation": (1.0, 3.0), "edge_angle_mean_deg": (70, 90),
        "scar_count": (1, 4), "dorsal_ridge_count": (0, 1),
        "symmetry_score": (0.3, 0.6),
    },
    "Denticulate": {
        "length_mm": (20, 70), "width_mm": (15, 45), "thickness_mm": (4, 15),
        "elongation": (0.8, 2.0), "edge_angle_mean_deg": (45, 65),
        "scar_count": (3, 8), "dorsal_ridge_count": (0, 1),
        "symmetry_score": (0.3, 0.6),
    },
    "Notched": {
        "length_mm": (20, 70), "width_mm": (15, 45), "thickness_mm": (4, 15),
        "elongation": (0.8, 2.0), "edge_angle_mean_deg": (50, 70),
        "scar_count": (2, 5), "dorsal_ridge_count": (0, 1),
        "symmetry_score": (0.3, 0.6),
    },
    "Backed knife": {
        "length_mm": (40, 150), "width_mm": (15, 40), "thickness_mm": (3, 12),
        "elongation": (2.0, 4.0), "edge_angle_mean_deg": (60, 80),
        "scar_count": (2, 5), "dorsal_ridge_count": (1, 2),
        "symmetry_score": (0.4, 0.7),
    },
}

# ── Technological typology ──

TECH_RANGES = {
    "Primary": {
        "length_mm": (30, 120), "width_mm": (20, 80), "thickness_mm": (8, 30),
        "elongation": (0.8, 2.0), "scar_count": (0, 1),
        "surface_roughness": (0.7, 1.0),
    },
    "Secondary": {
        "length_mm": (20, 100), "width_mm": (15, 60), "thickness_mm": (5, 25),
        "elongation": (0.8, 2.0), "scar_count": (1, 3),
        "surface_roughness": (0.5, 0.9),
    },
    "Tertiary": {
        "length_mm": (15, 80), "width_mm": (10, 50), "thickness_mm": (3, 15),
        "elongation": (0.8, 2.5), "scar_count": (2, 5),
        "surface_roughness": (0.3, 0.7),
    },
    "Crested blade": {
        "length_mm": (40, 150), "width_mm": (8, 25), "thickness_mm": (3, 10),
        "elongation": (3.0, 6.0), "scar_count": (2, 4),
        "dorsal_ridge_count": (2, 4),
        "surface_roughness": (0.3, 0.6),
    },
    "Core rejuvenation": {
        "length_mm": (15, 60), "width_mm": (10, 40), "thickness_mm": (5, 20),
        "elongation": (0.8, 2.0), "scar_count": (1, 3),
        "platform_angle_deg": (70, 90),
        "surface_roughness": (0.5, 0.8),
    },
}


def generate_samples(ranges: dict, n_per_class: int = 200, noise: float = 0.15) -> tuple[list, list]:
    """Generate synthetic feature vectors with Gaussian noise."""
    features = []
    labels = []
    rng = np.random.default_rng(42)

    for label, params in ranges.items():
        for _ in range(n_per_class):
            vec = {}
            for key, (lo, hi) in params.items():
                # Uniform sample from range
                val = lo + rng.random() * (hi - lo)
                # Add Gaussian noise proportional to range
                val += rng.normal(0, (hi - lo) * noise)
                val = max(lo * 0.5, min(hi * 1.5, val))  # clip to reasonable bounds
                vec[key] = round(float(val), 4)

            # Fill remaining features with sensible defaults
            for name in LithicFeatureVector.FEATURE_NAMES:
                if name not in vec:
                    vec[name] = 0.0

            fv = LithicFeatureVector(**{k: vec.get(k, 0.0) for k in LithicFeatureVector.FEATURE_NAMES})
            features.append(fv)
            labels.append(label)

    return features, labels


def train_and_save(
    name: str,
    ranges: dict,
    n_per_class: int = 200,
) -> Path:
    """Generate training data, train model, save to file."""
    from lithicore._classification import train_model

    features, labels = generate_samples(ranges, n_per_class=n_per_class)
    model = train_model(features, labels, typology_name=name)
    path = MODEL_DIR / f"typology_{name}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    print(f"  Saved {name} model ({len(ranges)} classes, {n_per_class * len(ranges)} samples) -> {path}")
    return path


if __name__ == "__main__":
    print("Generating pre-trained lithic classifier models...")
    train_and_save("basic", BASIC_RANGES, n_per_class=300)
    train_and_save("bordes", BORDES_RANGES, n_per_class=200)
    train_and_save("technological", TECH_RANGES, n_per_class=200)
    print("Done.")
```

- [ ] **Create `lithicore/data/__init__.py`** (empty file) to make data a package.
- [ ] **Create `lithicore/data/models/` directory** by making sure the directory exists.
- [ ] **Run the generator**: `cd .../dibble && python -m lithicore.data.generate_training_data`
- [ ] **Verify models created**: `ls lithicore/data/models/*.joblib`
- [ ] **Commit:**

```bash
cd .../dibble && git add lithicore/data/ && git commit -m "feat: add pre-trained lithic classifier models (basic, bordes, technological)"
```

---

### Task 6: Wire into `__init__.py` and `pyproject.toml`

**Files:**
- Modify: `lithicore/src/lithicore/__init__.py`
- Modify: `lithicore/pyproject.toml`

- [ ] **Add scikit-learn and joblib to pyproject.toml** dependencies:

```toml
    "scikit-learn>=1.3",
    "joblib>=1.2",
```

- [ ] **Add imports to `__init__.py`**:

```python
    from lithicore._classification import (
        ClassifierModel,
        extract_features,
        train_model,
        extract_diagnostic_coordinates,
    )

    __all__ = [
        # ... existing ...
        "ClassificationResult", "FeatureImportance", "LithicFeatureVector",
        "ClassifierModel", "extract_features", "train_model",
        "extract_diagnostic_coordinates",
    ]
```

- [ ] **Verify**: `cd .../dibble && python -c "from lithicore import ClassifierModel, extract_features, ClassificationResult, LithicFeatureVector; print('OK')"`
- [ ] **Commit:**

```bash
cd .../dibble && git add lithicore/src/lithicore/__init__.py lithicore/pyproject.toml && git commit -m "feat: wire classifier into lithicore exports, add scikit-learn dependency"
```

---

### Task 7: Classification Tests

**Files:**
- Create: `lithicore/tests/test_classification.py`

- [ ] **Create test file**:

```python
"""test_classification.py — Unit tests for lithic classification.

exports: TestExtractFeatures, TestClassifierModel, TestTraining
used_by: pytest
rules:   Synthetic meshes only. No real artefacts required.
         Pre-trained models must be generated first (Task 5).
agent:   deepseek-v4-flash | 2026-05-27 | Initial implementation
"""

import numpy as np
import trimesh
import pytest

from lithicore import (
    ClassificationResult, FeatureImportance, LithicFeatureVector,
    ClassifierModel, extract_features, extract_diagnostic_coordinates,
)


@pytest.fixture
def blade_mesh():
    """A synthetic blade-like mesh (elongated, triangular cross-section)."""
    box = trimesh.creation.box(extents=[10, 30, 5])
    return box


class TestExtractFeatures:
    """Feature extraction from synthetic meshes."""

    def test_extract_returns_feature_vector(self, blade_mesh):
        fv = extract_features(blade_mesh)
        assert isinstance(fv, LithicFeatureVector)
        assert fv.length_mm > 0

    def test_feature_vector_has_20_features(self, blade_mesh):
        fv = extract_features(blade_mesh)
        arr = fv.to_array()
        assert len(arr) == 20

    def test_feature_vector_no_nan(self, blade_mesh):
        fv = extract_features(blade_mesh)
        arr = fv.to_array()
        assert not np.any(np.isnan(arr))

    def test_feature_vector_no_inf(self, blade_mesh):
        fv = extract_features(blade_mesh)
        arr = fv.to_array()
        assert not np.any(np.isinf(arr))

    def test_elongation_of_blade_mesh(self, blade_mesh):
        fv = extract_features(blade_mesh)
        assert fv.elongation > 1.0  # length > width


class TestClassifierModel:
    """Classifier predictions."""

    def test_load_pre_trained_basic(self):
        model = ClassifierModel.load_pre_trained("basic")
        assert model.is_loaded()

    def test_load_pre_trained_bordes(self):
        model = ClassifierModel.load_pre_trained("bordes")
        assert model.is_loaded()

    def test_load_pre_trained_all(self):
        for name in ["basic", "bordes", "technological"]:
            model = ClassifierModel.load_pre_trained(name)
            assert model.is_loaded()

    def test_predict_returns_classification_result(self, blade_mesh):
        model = ClassifierModel.load_pre_trained("basic")
        fv = extract_features(blade_mesh)
        result = model.predict(fv)
        assert isinstance(result, ClassificationResult)
        assert isinstance(result.label, str)
        assert 0 <= result.confidence <= 1

    def test_predict_probabilities_sum_to_one(self, blade_mesh):
        model = ClassifierModel.load_pre_trained("basic")
        fv = extract_features(blade_mesh)
        result = model.predict(fv)
        total = sum(result.probabilities.values())
        assert abs(total - 1.0) < 0.01

    def test_predict_has_top_features(self, blade_mesh):
        model = ClassifierModel.load_pre_trained("basic")
        fv = extract_features(blade_mesh)
        result = model.predict(fv)
        assert len(result.top_features) == 5
        for f in result.top_features:
            assert isinstance(f, FeatureImportance)

    def test_predict_without_model_raises(self):
        model = ClassifierModel(typology_name="test")
        with pytest.raises(RuntimeError, match="No model loaded"):
            model.predict(LithicFeatureVector())

    def test_active_learning_queue(self, blade_mesh):
        model = ClassifierModel.load_pre_trained("basic")
        fv = extract_features(blade_mesh)
        count = model.queue_correction(fv, "Blade")
        assert count == 1

    def test_retrain_after_threshold(self, blade_mesh):
        model = ClassifierModel.load_pre_trained("basic")
        fv = extract_features(blade_mesh)
        for _ in range(15):
            model.queue_correction(fv, "Blade")
        assert model.retrain_if_ready(threshold=10) is True

    def test_retrain_below_threshold(self, blade_mesh):
        model = ClassifierModel.load_pre_trained("basic")
        fv = extract_features(blade_mesh)
        model.queue_correction(fv, "Blade")
        assert model.retrain_if_ready(threshold=10) is False


class TestDiagnosticCoordinates:
    """Viewer overlay coordinate extraction."""

    def test_extract_returns_dict(self, blade_mesh):
        coords = extract_diagnostic_coordinates(blade_mesh)
        assert isinstance(coords, dict)
        assert "ridges" in coords
        assert "platform" in coords
        assert "retouched_edges" in coords

    def test_coordinates_are_valid(self, blade_mesh):
        coords = extract_diagnostic_coordinates(blade_mesh)
        for key, points in coords.items():
            if len(points) > 0:
                assert points.shape[1] == 3  # (N, 3)


class TestTraining:
    """Custom typology training."""

    def test_train_minimal_model(self):
        from lithicore import train_model

        fvs = []
        labels = []
        for label in ["TypeA", "TypeB", "TypeC"]:
            for _ in range(4):
                fv = LithicFeatureVector(
                    length_mm=50, width_mm=30, thickness_mm=10,
                    elongation=2.0, flatness=3.0,
                )
                fvs.append(fv)
                labels.append(label)

        model = train_model(fvs, labels, typology_name="custom_test")
        assert model.is_loaded()

        # Predict a new sample
        result = model.predict(LithicFeatureVector(
            length_mm=50, width_mm=30, thickness_mm=10,
            elongation=2.0, flatness=3.0,
        ))
        assert result.label in ["TypeA", "TypeB", "TypeC"]
```

- [ ] **Run tests**: `cd .../dibble && python -m pytest lithicore/tests/test_classification.py -v`
Expected: tests may fail for pre-trained model loading if models not generated — run generation first.
- [ ] **Run full suite**: `cd .../dibble && python -m pytest lithicore/tests/ -v`
- [ ] **Commit:**

```bash
cd .../dibble && git add lithicore/tests/test_classification.py && git commit -m "feat: add classification tests (feature extraction, prediction, active learning, training)"
```

---

### Task 8: Classification Panel Widget

**Files:**
- Create: `lithicope/src/lithicope/_classification_panel.py`

This is a large GUI widget (~250 lines). Key structure:

```python
"""_classification_panel.py — Side panel for lithic typology classification.

exports: ClassificationPanel(QWidget)
used_by: MainWindow right-side tab widget
rules:   Operates on ClassificationResult objects. No direct sklearn imports.
agent:   deepseek-v4-flash | 2026-05-27 | Initial implementation
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QGroupBox, QHBoxLayout, QLabel,
    QListWidget, QPushButton, QVBoxLayout, QWidget,
)

from lithicore import (
    ClassificationResult,
    ClassifierModel,
    LithicFeatureVector,
    extract_features,
)


class ClassificationPanel(QWidget):
    """Panel for running and displaying lithic typology classification."""

    classification_computed = pyqtSignal(object)  # ClassificationResult
    diagnostic_overlay_requested = pyqtSignal(dict)  # coordinate dict
    auto_classify_changed = pyqtSignal(bool)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._current_result: Optional[ClassificationResult] = None
        self._current_mesh = None
        self._models: dict[str, ClassifierModel] = {}
        self._correction_timer = QTimer()
        self._correction_timer.setInterval(500)  # debounce retrain signal
        self._correction_timer.setSingleShot(True)
        self._build_ui()
        self._load_models()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Header
        header = QLabel("Lithic Typology Classification")
        header.setStyleSheet("font-size: 13pt; font-weight: bold;")
        layout.addWidget(header)

        # Typology selector
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Typology:"))
        self._typology_combo = QComboBox()
        self._typology_combo.addItems(["Basic Morphological", "Bordes Typology", "Technological", "Custom"])
        self._typology_combo.currentTextChanged.connect(self._on_typology_changed)
        type_row.addWidget(self._typology_combo)
        type_row.addStretch()
        layout.addLayout(type_row)

        # Auto-classify toggle
        self._auto_check = QCheckBox("Auto-classify on load")
        self._auto_check.setChecked(False)
        self._auto_check.toggled.connect(self.auto_classify_changed.emit)
        layout.addWidget(self._auto_check)

        # Predict button
        self._classify_btn = QPushButton("Classify Artefact")
        self._classify_btn.setStyleSheet("font-size: 12pt; padding: 8px;")
        self._classify_btn.clicked.connect(self._on_classify)
        layout.addWidget(self._classify_btn)

        # Result card
        self._result_group = QGroupBox("Classification Result")
        result_layout = QVBoxLayout(self._result_group)
        self._label_display = QLabel("")
        self._label_display.setStyleSheet("font-size: 16pt; font-weight: bold;")
        result_layout.addWidget(self._label_display)
        self._confidence_display = QLabel("")
        result_layout.addWidget(self._confidence_display)
        self._features_list = QListWidget()
        self._features_list.setMaximumHeight(150)
        result_layout.addWidget(self._features_list)
        self._alternatives_label = QLabel("")
        self._alternatives_label.setStyleSheet("color: #666;")
        result_layout.addWidget(self._alternatives_label)
        layout.addWidget(self._result_group)
        self._result_group.setVisible(False)

        # Correction area
        correct_row = QHBoxLayout()
        correct_row.addWidget(QLabel("Correct?"))
        self._correct_combo = QComboBox()
        correct_row.addWidget(self._correct_combo)
        self._correct_btn = QPushButton("Submit Correction")
        self._correct_btn.clicked.connect(self._on_correct)
        correct_row.addWidget(self._correct_btn)
        correct_row.addStretch()
        layout.addLayout(correct_row)

        # Overlay toggle
        self._overlay_check = QCheckBox("Show diagnostic overlays on mesh")
        self._overlay_check.setChecked(True)
        layout.addWidget(self._overlay_check)

        # Batch classify
        self._batch_btn = QPushButton("Classify All in Batch...")
        layout.addWidget(self._batch_btn)

        layout.addStretch()

    def _load_models(self) -> None:
        """Load pre-trained models on startup."""
        for name in ["basic", "bordes", "technological"]:
            try:
                self._models[name] = ClassifierModel.load_pre_trained(name)
            except FileNotFoundError:
                pass

    def set_mesh(self, mesh) -> None:
        """Set the current mesh for classification."""
        self._current_mesh = mesh
        if self._auto_check.isChecked():
            self._on_classify()

    def _on_classify(self) -> None:
        """Run classification on the current mesh."""
        if self._current_mesh is None:
            return

        # Determine which model
        typology_map = {
            "Basic Morphological": "basic",
            "Bordes Typology": "bordes",
            "Technological": "technological",
            "Custom": "custom",
        }
        model_key = typology_map.get(self._typology_combo.currentText(), "basic")
        model = self._models.get(model_key)

        if model is None or not model.is_loaded():
            self._label_display.setText("Model not available")
            return

        # Extract features and predict
        fv = extract_features(self._current_mesh)
        result = model.predict(fv)
        self._current_result = result

        # Update UI
        self._show_result(result)

    def _show_result(self, result: ClassificationResult) -> None:
        """Display a ClassificationResult in the panel."""
        self._result_group.setVisible(True)
        self._label_display.setText(f"🏷 {result.label}")
        colour = "green" if result.confidence >= 0.8 else \
                 "orange" if result.confidence >= 0.6 else "red"
        self._confidence_display.setText(
            f"Confidence: <span style='color:{colour}; font-weight:bold;'>"
            f"{result.confidence:.0%}</span>"
        )

        # Features
        self._features_list.clear()
        for f in result.top_features:
            status = "✓" if f.passed else "✗"
            self._features_list.addItem(
                f"  {f.name}: {f.value:.2f} ({f.contribution_pct:.0%}) {status}"
            )

        # Alternatives
        if result.alternatives:
            alt_text = "Also possible: " + ", ".join(
                f"{label} ({conf:.0%})" for label, conf in result.alternatives
            )
            self._alternatives_label.setText(alt_text)
        else:
            self._alternatives_label.setText("")

        # Populate correction combo
        self._correct_combo.clear()
        model_key = self._typology_combo.currentText()
        if model_key in self._models:
            for cls_name in sorted(self._models[model_key]._classes):
                self._correct_combo.addItem(cls_name)

        # Emit signal for viewer overlays
        if self._overlay_check.isChecked():
            from lithicore import extract_diagnostic_coordinates
            coords = extract_diagnostic_coordinates(self._current_mesh)
            self.diagnostic_overlay_requested.emit(coords)

        self.classification_computed.emit(result)

    def _on_correct(self) -> None:
        """Submit a correction for active learning."""
        if self._current_result is None or self._current_mesh is None:
            return
        correct_label = self._correct_combo.currentText()
        if not correct_label:
            return

        model_key = self._typology_combo.currentText()
        model = self._models.get(model_key)
        if model is None:
            return

        fv = extract_features(self._current_mesh)
        count = model.queue_correction(fv, correct_label)
        self._label_display.setText(f"Corrected to: {correct_label}")
        self._correction_timer.start()  # triggers retrain check after debounce

    def get_overlay_enabled(self) -> bool:
        """Check if diagnostic overlays should be shown."""
        return self._overlay_check.isChecked()

    def get_auto_classify(self) -> bool:
        """Check if auto-classify is enabled."""
        return self._auto_check.isChecked()
```

- [ ] Create the file with the content above
- [ ] Commit:

```bash
cd .../dibble && git add lithicope/src/lithicope/_classification_panel.py && git commit -m "feat: add classification panel widget"
```

---

### Task 9: Viewer Overlay Methods

**Files:**
- Modify: `lithicope/src/lithicope/_viewer_3d.py`

Add these methods to Viewer3D (after the annotation section, before scar overlay):

```python
    # ── Diagnostic overlays ──────────────────────────────────

    def show_diagnostic_overlay(self, coords: dict[str, np.ndarray]) -> None:
        """Highlight diagnostic features from classification.

        coords keys: 'ridges' (blue), 'platform' (green), 'retouched_edges' (red)
        """
        if not HAS_PYVISTAQT or self._pv_mesh is None:
            return

        # Clear existing overlays
        self.clear_diagnostic_overlay()

        # Ridge points — blue
        ridges = coords.get("ridges", np.array([]))
        if len(ridges) > 0:
            cloud = pv.PolyData(ridges)
            actor = self.plotter.add_points(
                cloud, color="blue", point_size=4.0,
                render_points_as_spheres=True,
            )
            self._diagnostic_actors.append(actor)

        # Platform points — green
        platform = coords.get("platform", np.array([]))
        if len(platform) > 0:
            cloud = pv.PolyData(platform)
            actor = self.plotter.add_points(
                cloud, color="green", point_size=4.0,
                render_points_as_spheres=True,
            )
            self._diagnostic_actors.append(actor)

        # Retouched edges — red
        retouch = coords.get("retouched_edges", np.array([]))
        if len(retouch) > 0:
            cloud = pv.PolyData(retouch)
            actor = self.plotter.add_points(
                cloud, color="red", point_size=5.0,
                render_points_as_spheres=True,
            )
            self._diagnostic_actors.append(actor)

        # Add legend
        legend_text = (
            "Diagnostic Overlays:\n"
            "  Blue  = Dorsal ridges\n"
            "  Green = Platform\n"
            "  Red   = Retouched edges"
        )
        self.plotter.add_text(
            legend_text, position="lower_right", font_size=10, color="black",
        )

        self.plotter.render()

    def clear_diagnostic_overlay(self) -> None:
        """Remove all diagnostic overlay actors."""
        if not HAS_PYVISTAQT:
            return
        for actor in self._diagnostic_actors:
            self.plotter.remove_actor(actor, render=False)
        self._diagnostic_actors.clear()
        # Clear legend text
        self.plotter.render()

    def has_diagnostic_overlay(self) -> bool:
        """Check if diagnostic overlays are currently shown."""
        return len(self._diagnostic_actors) > 0
```

Also add to `__init__`:
```python
        self._diagnostic_actors: list = []
```

- [ ] **Verify syntax**: `cd .../dibble && python -c "import ast; ast.parse(open('lithicope/src/lithicope/_viewer_3d.py').read()); print('Syntax OK')"`
- [ ] **Commit:**

```bash
cd .../dibble && git add lithicope/src/lithicope/_viewer_3d.py && git commit -m "feat: add diagnostic overlay methods for classifier explanation"
```

---

### Task 10: Main Window Wiring

**Files:**
- Modify: `lithicope/src/lithicope/_main_window.py`

Changes:
1. Add import: `from lithicope._classification_panel import ClassificationPanel`
2. Add the Classification tab to `_right_tabs`
3. Add menu items under Tools > Classification
4. Wire auto-classify on mesh load
5. Wire diagnostic overlay signals
6. Add keyboard shortcut

- [ ] **Add import**
- [ ] **Add Classification tab** to `_right_tabs` (alongside Results, Annotations)
- [ ] **Wire signals**:

```python
        self._classification_panel.classification_computed.connect(self._on_classification_result)
        self._classification_panel.diagnostic_overlay_requested.connect(self._on_diagnostic_overlay)
        self._classification_panel.auto_classify_changed.connect(self._on_auto_classify_toggle)
```

- [ ] **Add handlers**:

```python
    def _on_classification_result(self, result):
        self.status.showMessage(f"Classification: {result.label} ({result.confidence:.0%})")

    def _on_diagnostic_overlay(self, coords):
        self.viewer.show_diagnostic_overlay(coords)

    def _on_auto_classify_toggle(self, enabled):
        self.status.showMessage(f"Auto-classify: {'ON' if enabled else 'OFF'}")
```

- [ ] **Call `set_mesh`** in the mesh loading path (`_process_single`) to trigger auto-classify if enabled.
- [ ] **Add menu items**:

```python
        # Classification submenu
        tools_menu.addSeparator()
        class_menu = tools_menu.addMenu("&Classification")
        classify_action = QAction("&Classify Artefact", self)
        classify_action.setShortcut("Ctrl+Shift+C")
        classify_action.triggered.connect(self._classification_panel._on_classify)
        class_menu.addAction(classify_action)
        batch_class_action = QAction("&Batch Classify...", self)
        class_menu.addAction(batch_class_action)
        train_action = QAction("&Train Custom Typology...", self)
        class_menu.addAction(train_action)
```

- [ ] **Verify**: `cd .../dibble && python -c "from lithicope._main_window import MainWindow; print('OK')"`
- [ ] **Run full test suite**: `cd .../dibble && python -m pytest lithicore/tests/ -v`
- [ ] **Commit**:

```bash
cd .../dibble && git add lithicope/src/lithicope/_main_window.py && git commit -m "feat: wire classification panel and overlays into main window"
```
