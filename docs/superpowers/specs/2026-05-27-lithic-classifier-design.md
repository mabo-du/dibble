# AI Lithic Typology Classifier — Design Spec

**Date:** 2026-05-27
**Status:** Approved for implementation

## Overview

Add automated lithic artefact typology classification to Dibble using geometric morphometric features extracted from 3D meshes, with explainable predictions, active learning, and custom typology support.

## Architecture

**Hybrid approach:** Geometric feature vector (25+ dimensions) → sklearn Random Forest classifier → explainable output with confidence, diagnostic features, and viewer overlays. No GPU, no cloud, fully local.

**Key innovation:** "Morphometric Fingerprint™" — a dense numerical summary of every diagnostic aspect of a lithic artefact, computed from existing measurement pipeline + new shape descriptors.

---

## 1. Feature Engineering — `LithicFeatureVector`

### Raw metrics (from existing pipeline)

| Feature | Source | Range |
|---|---|---|
| Max length | `extract_metrics()` via oriented bounding box | 0–500 mm |
| Max width | `extract_metrics()` | 0–300 mm |
| Max thickness | `extract_metrics()` | 0–150 mm |
| Surface area | `extract_metrics()` via trimesh.area | 0–100,000 mm² |
| Volume | `extract_metrics()` via trimesh.volume | 0–1,000,000 mm³ |
| Scar count | `detect_scars()` | 0–50 |
| Mean scar area | Total scar area / scar count | — |
| Platform angle | `platform_angles()` (IPA / EPA) | 0–90° |
| Edge angle (mean) | Face dihedral angles at boundary edges | 0–90° |
| Edge angle (std) | Standard deviation of edge angles | — |

### Derived ratios

| Ratio | Formula | Diagnostic |
|---|---|---|
| Elongation | Length / Width | ≥2.0 = blade; <2.0 = flake |
| Flatness | Width / Thickness | High = flat blade; low = chunky core |
| Compactness | Volume / Length³ | Massivity / robusticity |
| Relative thickness | Thickness / Length | Thin = blade; thick = core |

### New shape descriptors

| Feature | Method | What it captures |
|---|---|---|
| Curvature index | Mean Gaussian curvature / mean absolute curvature | Dorsal curvature — flakes vs blades |
| Cross-section profile | PCA of mid-section slice; 0=flat, 1=tri, 2=round | Blade (triangular) vs flake (flat) vs core (round) |
| Symmetry score | Hausdorff distance between mesh halves | Bilateral symmetry — bifaces score high |
| COM Z ratio | Centre-of-mass Z / total height | Platform position on tool |
| Dorsal ridge count | Detect linear curvature ridges | 2+ parallel ridges → blade |
| Surface roughness | Face area / projected area (fractal dimension) | Overall texture |
| Cross-section area | Area of mid-coronal slice | — |

**Total: ~25 dimensional feature vector.**

### Code structure

```python
@dataclass
class LithicFeatureVector:
    """Numerical fingerprint of a lithic artefact."""
    # Raw metrics
    length_mm: float
    width_mm: float
    thickness_mm: float
    surface_area_mm2: float
    volume_mm3: float

    # Derived ratios
    elongation: float
    flatness: float
    compactness: float
    relative_thickness: float

    # Morphological
    scar_count: int
    mean_scar_area_mm2: float
    platform_angle_deg: float
    edge_angle_mean_deg: float
    edge_angle_std_deg: float
    curvature_index: float
    cross_section_profile: float  # 0=flat, 1=triangular, 2=round
    symmetry_score: float
    com_z_ratio: float
    dorsal_ridge_count: int
    surface_roughness: float

    def to_array(self) -> np.ndarray: ...
    @classmethod
    def from_mesh(cls, mesh: trimesh.Trimesh) -> LithicFeatureVector: ...
```

---

## 2. Classifier — `ClassifierModel`

### Core

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=12,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
)
```

Choice rationale: Random Forest wins on interpretability (feature importances), small-sample performance, multi-class support, and zero GPU requirement.

### Pre-trained models

| Name | Classes | Source data |
|---|---|---|
| `typology_basic` | Flake, Blade, Bladelet, Core, Tool | Published metric ranges (Andrefsky 2005, Inizan 1999) |
| `typology_bordes` | Scraper, Handaxe, Point, Burin, Denticulate, Notched, Backed knife | Bordes typology (Bordes 1961) |
| `typology_technological` | Primary, Secondary, Tertiary, Crested blade, Core rejuvenation | Reduction stage diagnostics |

Each pre-trained model is trained on **synthetic datasets** generated from published metric ranges with injected variance. Models are saved via `joblib.dump()` (~50 KB each) and shipped in `lithicore/data/models/`.

### Prediction output

```python
@dataclass
class ClassificationResult:
    label: str                     # e.g. "Blade"
    confidence: float              # calibrated probability (0–1)
    probabilities: dict[str, float]  # per-class probabilities
    top_features: list[FeatureImportance]  # [(name, value, contribution%)]
    alternatives: list[tuple[str, float]]  # other likely labels
```

### Confidence calibration

Uses `CalibratedClassifierCV(sigmoid)` on the raw `predict_proba()` output. Predictions with confidence <0.6 are flagged "uncertain — manual review recommended."

---

## 3. Explainability

Traces which features drove the decision through each decision tree in the forest.

```python
@dataclass
class FeatureImportance:
    name: str
    value: float
    contribution_pct: float  # how many trees split on this feature
    expected_range: tuple[float, float]  # typical range for the predicted class
    passed: bool  # value falls in expected range
```

### Viewer overlays

When a classification result is shown, the viewer highlights:

| Colour | Feature |
|---|---|
| Blue | Dorsal ridges (curvature-based line detection) |
| Green | Platform (from existing platform detection) |
| Red | Retouched edges (edge angle > threshold) |

---

## 4. Active Learning

### Correction loop

1. User clicks "Correct" → selects correct label → corrected example saved
2. Every 10 corrections → background retrain (QTimer, ~1-2s)
3. Confidence scores improve over time for that user's specific assemblage

### Custom typology training

**Tools → Classification → Train Custom Typology** opens a dialog:

```
Type name    | Examples loaded | Minimum
─────────────|─────────────────|────────
Type A       | 12 meshes       |  3
Type B       | 8 meshes        |  3
Type C       | 3 meshes        |  3
[+ Add Type]                     [Start Training]
```

- Each type needs ≥3 classified meshes
- Click "Start Training" → trains in ~1s → new model saved to `~/.dibble/models/typology_custom.joblib`
- Custom model appears in dropdown alongside built-ins

### Model persistence

- Pre-trained: `lithicore/data/models/*.joblib`
- User-trained: `~/.dibble/models/*.joblib`
- Models are shareable as standalone files

---

## 5. UI Integration

### Classification panel (new tab alongside Results/Annotations)

```
┌──────────────────────────────────────────┐
│ Results │ Annotations │ Classification   │
├──────────────────────────────────────────┤
│  🏷  BLADE                        92%   │
│  via: Basic Morphological               │
│                                          │
│  Diagnostic Features:                    │
│  Elongation    3.2  ━━━░░  82%          │
│  Ridges        2    ━━━━━  95%          │
│  Platform     74°   ━━━░░  65%          │
│  Edge angle   68°   ━━━░░  60%          │
│                                          │
│  Alternatives: Flake (25%)  Tool (18%)  │
│                                          │
│  Typology: [Basic ▼]                     │
│  Correct?  [Scraper ▼]    [Retrain]     │
│  ☑ Auto-classify on load                 │
│  [Classify All in Batch]                  │
└──────────────────────────────────────────┘
```

### Viewer overlays mode

When classification is active:
- Dorsal ridges highlighted in blue
- Platform highlighted in green
- Retouched edges highlighted in red
- Legend in corner

### Menu items

```
Tools
├── ...
├── ─────
├── Classification
│   ├── Classify Artefact         Ctrl+Shift+C
│   ├── Batch Classify...
│   ├── Train Custom Typology...
│   └── Edit Typology Sets...
└── ...
```

---

## 6. Files to Create/Modify

| File | Action | Purpose |
|---|---|---|
| `lithicore/src/lithicore/_classification.py` | Create | Feature vector, classifier model, training + prediction |
| `lithicore/src/lithicore/__init__.py` | Modify | Export new symbols |
| `lithicore/src/lithicore/_models.py` | Modify | Add ClassificationResult, FeatureImportance, LithicFeatureVector dataclasses |
| `lithicore/data/models/typology_basic.joblib` | Create | Pre-trained Basic model |
| `lithicore/data/models/typology_bordes.joblib` | Create | Pre-trained Bordes model |
| `lithicore/data/models/typology_technological.joblib` | Create | Pre-trained Technological model |
| `lithicore/data/generate_training_data.py` | Create | Script to generate synthetic training data from published metric ranges |
| `lithicore/tests/test_classification.py` | Create | Tests for feature extraction + classifier |
| `lithicope/src/lithicope/_classification_panel.py` | Create | Classification panel widget |
| `lithicope/src/lithicope/_viewer_3d.py` | Modify | Add diagnostic overlay methods |
| `lithicope/src/lithicope/_main_window.py` | Modify | Add Classification tab, menu items |
| `lithicore/pyproject.toml` | Modify | Add scikit-learn>=1.3, joblib>=1.2 |

---

## 7. Testing Strategy

### Unit tests — `test_classification.py`

| Test | Description |
|---|---|
| `test_feature_vector_from_mesh` | Creates a synthetic box mesh, verifies all features computed |
| `test_feature_vector_to_array` | 25-dim array, no NaN, no Inf |
| `test_classifier_predict_shape` | Prediction returns ClassificationResult with correct fields |
| `test_classifier_confidence_range` | Confidence is 0–1 |
| `test_classifier_alternatives_sum_to_1` | Probabilities sum to 1.0 ± epsilon |
| `test_active_learning_correction` | Correcting a label updates the training queue |
| `test_retrain_updates_model` | After retrain, the previously-wrong prediction is now correct |
| `test_custom_typology_training` | Train a 3-type model on 4 examples per type |
| `test_pre_trained_models_load` | All shipped .joblib files load without errors |
| `test_dorsal_ridge_detection` | Synthetic blade-like mesh has 2+ ridges detected |

### Integration tests — via pytest-qt

| Test | Description |
|---|---|
| `test_classification_panel_shows_result` | After classify → panel shows label + confidence |
| `test_viewer_overlays_toggle` | Overlays appear/disappear on classify/clear |
| `test_correction_updates_viewer` | Correct label → confidence updates |
