# Lithic 3D Morphological Analyzer — Design Specification

**Date:** 2026-05-26
**Status:** Approved design, pending implementation

## 1. Project Overview

A desktop application that allows archaeologists and anthropologists to load a 3D scan of a stone tool (lithic artefact) and automatically extract standardised morphological measurements — edge angles, platform dimensions, flake scar counts and orientations, maximum length/width/thickness, mass estimation — without writing any code. Bridges the gap between R/Python libraries (Lithics3D, PyLithics) and non-programming researchers.

**Target users:** Lithic analysts, Palaeolithic archaeologists, zooarchaeologists, graduate students, museum collections researchers.

## 2. Architecture

### 2.1 Two-package modular design

Separation of concerns from day one to allow independent evolution of algorithms vs GUI.

```
lithic-analysis-platform/
├── lithicore/                    # Pure-Python measurement library
│   ├── pyproject.toml
│   └── src/lithicore/
│       ├── __init__.py
│       ├── _orientation.py       # PCA + platform detection, user-guided alignment
│       ├── _metrics.py           # Length, width, thickness, volume, area
│       ├── _edge_detection.py    # Dihedral thresholding, ridge detection
│       ├── _platform_angle.py    # EPA, IPA calculations
│       ├── _scar_analysis.py     # Flake scar counting (v2)
│       ├── _batch.py             # Batch processor
│       ├── _cli.py               # Typer CLI: `lithicore batch ...`
│       ├── _models.py            # @dataclasses (MeasurementResult, ArtefactResult, etc.)
│       └── _validation.py        # Mesh validation + cleaning pipeline
│
├── lithicope/                    # PyQt6 desktop GUI (depends on lithicore)
│   ├── pyproject.toml            # lithicore as install dependency
│   └── src/lithicope/
│       ├── __init__.py
│       ├── main.py               # Application entry point
│       ├── _main_window.py       # QMainWindow: menu, layout, status bar
│       ├── _viewer_3d.py         # Open3D/vis widget embedded in Qt
│       ├── _import_dialog.py     # File + folder import with mode selection
│       ├── _orientation_tool.py  # 3-point platform picker overlay
│       ├── _results_panel.py     # Measurement table + export controls
│       └── _batch_runner.py      # Batch progress UI
│
├── docs/
│   ├── scope.md
│   ├── research-papers/
│   ├── research-prompts/
│   └── superpowers/specs/        # Design documents
│
└── README.md
```

### 2.2 Key constraint

`lithicore` has **zero GUI imports** — only `trimesh`, `numpy`, `scipy`, `typer`. This guarantees testability in headless CI and enables future CLI/server/web wrappers.

### 2.3 Measurement algorithm design

Every algorithm is a pure function with a `@dataclass` config parameter, returning typed measurement results:

```python
@dataclass
class MeasurementConfig:
    """Shared across all metric extraction functions."""
    repair_mesh: bool = True
    edge_threshold_degrees: float = 50.0
    min_face_count: int = 2000

@dataclass
class MeasurementResult:
    name: str
    value: float
    unit: str           # "mm", "mm²", "mm³", "°"
    confidence: float   # 0.0–1.0, derived from mesh quality + algorithm certainty

@dataclass
class ArtefactResult:
    file_path: Path
    label: str
    measurements: list[MeasurementResult]
    landmarks: list
    warnings: list[str]
```

## 3. User Interface

### 3.1 Main window layout

Single-window design. Left-dominant 3D viewer (60% width), measurements panel on the right.

```
┌──────────────────────────────────────────────────────┐
│  File  Edit  Tools  Help                              │
├──────────────────────────────────────────────────────┤
│ ┌────────────────────┐  ┌──────────────────────────┐  │
│ │   3D Viewer        │  │  Measurements             │  │
│ │   (Open3D+Qt)      │  │  ┌────────────────────┐  │  │
│ │                    │  │  │ Length    45.2 mm   │  │  │
│ │   rotate: drag     │  │  │ Width     28.7 mm   │  │  │
│ │   zoom: scroll     │  │  │ Thickness  8.1 mm   │  │  │
│ │   pan: shift+drag  │  │  │ Area      12.4 cm²  │  │  │
│ │                    │  │  │ Volume    22.1 cm³  │  │  │
│ │   [Edge overlay]   │  │  │ EPA        78.3°    │  │  │
│ │   [Threshold ▬]    │  │  │ IPA       112.5°    │  │  │
│ │                    │  │  └────────────────────┘  │  │
│ │                    │  │  ┌────────────────────┐  │  │
│ │                    │  │  │ Export: [CSV▼]     │  │  │
│ │                    │  │  │        [JSON▼]     │  │  │
│ │                    │  │  │        [MorphoJ▼]  │  │  │
│ │                    │  │  │        [PDF▼]      │  │  │
│ └────────────────────┘  │  └────────────────────┘  │  │
├──────────────────────────────────────────────────────┤
│  Artefact 3/12 — Complete  |  Batch: 25%              │
└──────────────────────────────────────────────────────┘
```

### 3.2 Import dialog

```
┌─────────────────────────────────────────┐
│ Import Mode                              │
│ ○ Single artefact  (File → Open)        │
│   ┌────────────────────────────────┐    │
│   │ Advanced                        ▼ │ │
│   │  ☐ Skip auto-repair             │  │
│   │  ☐ Skip auto-validation          │  │
│   └────────────────────────────────┘  │  │
│ ○ Batch — auto      (auto-orient)     │
│ ○ Batch — review    (review flags)    │
│ ○ Batch — manual    (orient each)     │
│                                         │
│ [Select Folder] → 12 meshes found       │
│                                         │
│ [      Import      ]                    │
└─────────────────────────────────────────┘
```

- Auto-repair is **on by default** for all modes
- Power users can opt out via the Advanced panel (only visible in single-artefact and batch-manual modes)
- `lithicore` API exposes `repair=True | False`, so the choice is clean at every level

### 3.3 Three-tier orientation strategy

| Mode | Trigger | Process |
|---|---|---|
| **Auto** (default) | Import → auto | PCA on surface normals + platform detection heuristic. Sub-second |
| **Semi-auto** | "Adjust →" button | Auto runs first, user corrects with 1–2 clicks |
| **Manual** | Batch-manual or single-artefact manual mode | 3-point platform picker |

After auto-orientation, the viewer shows a translucent preview with "Looks good? ✓ / Adjust →" prompt.

### 3.4 Batch processing modes

| Mode | Behaviour |
|---|---|
| **Auto** | Load folder, auto-orient every mesh, run all measurements, append to CSV. No prompts |
| **Review** | Auto-orient all, flag low-confidence results for user to tab through and confirm/adjust |
| **Manual** | User orients each artefact before measurements run |

## 4. Mesh Validation Pipeline

### 4.1 Three-stage automated pipeline

All stages are automatic on import. Power users can skip repair/validation via the import dialog.

```
Step 1: Structural integrity
  → Valid triangle mesh?
  → Manifold?
  → Watertight?
  → Has normals?

Step 2: Clean & repair
  → Remove duplicate vertices
  → Merge close vertices
  → Fill small holes (≤ threshold faces)
  → Invert misoriented normals
  → Remove isolated components

Step 3: Quality assessment
  → Vertex count / resolution grade
  → Triangle aspect ratio histogram
  → Overall score: Pass / Warn / Fail
```

### 4.2 User-facing quality feedback

Non-technical notifications:

| Grade | Criteria | Message |
|---|---|---|
| **Pass** | >50k faces, manifold, watertight | "Excellent quality" |
| **Warn** | 10–50k faces or small holes filled | Measures fine, edges ±1-2° |
| **Fail** | <2k faces, major holes, non-manifold | "Can't reliably measure — import anyway?" |

### 4.3 Error handling philosophy

- Typed exceptions in `lithicore`: `MeshValidationError`, `OrientationError`, `MeasurementError`
- GUI catches at the boundary → non-blocking toast notification
- Graceful degradation: if edge detection fails, return length/width/thickness with note "Edge angle: unavailable (insufficient resolution)"
- All warnings in CSV output as extra column

## 5. Measurement Algorithms

### 5.1 Orientation pipeline

1. PCA on face normals (weighted by area) → 3 eigenvectors as X/Y/Z axes
2. Platform detection: identify flattest proximal face cluster via dihedral thresholding
3. Snap platform plane via least-squares SVD through those faces
4. Artefact bounding box computed in oriented coordinate space

### 5.2 Extracted metrics (v1)

| Measurement | Algorithm | Basis |
|---|---|---|
| Max length | Maximum extent along Z-axis (reduction axis) | Andrefsky 2005 |
| Max width | Maximum extent perpendicular to Z, measured at intervals | Andrefsky 2005 |
| Max thickness | Maximum orthogonal distance, excluding bulb centroid | Replicability in Lithic Analysis (2023) |
| Platform width | Transverse distance across platform surface | Dibble & Whittaker |
| Platform thickness | Perpendicular to width, bisecting point of percussion | Dibble & Whittaker |
| Surface area | Sum of triangle areas | Standard geometry |
| Volume | Watertight fill + tetrahedral decomposition | Standard geometry |
| EPA | Angle between platform plane and best-fit dorsal plane | Clarkson |
| IPA | Angle between platform plane and ventral surface | Clarkson |

### 5.3 Edge detection

- **Primary:** Dihedral angle thresholding (configurable, default 40–60°)
- **V2 enhancement:** Shape Index analysis (`S = 2/π * arctan((k₂ + k₁)/(k₂ - k₁))`) classifying ridges vs valleys
- **V2 enhancement:** 3D-EdgeAngle protocol — virtual cross-sections at 2/5/10 mm using the '3-points' procedure (Schunk et al. 2023)

## 6. Export Formats

### 6.1 v1 (MVP — all shipping now)

| Format | Contents |
|---|---|
| **CSV** | One row per artefact, one column per measurement + `warnings` column. Ready for R/Python |
| **JSON** | Same data in structured JSON for programmatic pipelines |
| **MorphoJ** | Landmark/procrustes coordinates in `.txt` format for geometric morphometric analysis |
| **PDF** | One-page summary: artefact image, key metrics, measurement method notes |

### 6.2 v2 (first deliverable)

| Format | Contents |
|---|---|
| **Publication figure** | Standardised three-view technical drawing (plan, profile, section) with scale bar — vector output |

## 7. Testing Strategy

### 7.1 Tier 1 — Unit tests (lithicore)
- Every algorithm tested against synthetic meshes with known geometry
- Parameterised tests for edge detection thresholds, orientation accuracy
- No GUI, runs in CI with `pytest`

### 7.2 Tier 2 — Integration tests (lithicore)
- Curated sample meshes (5–10 PLY/OBJ/STL) with ground-truth caliper measurements
- Verify `lithicore` outputs fall within published inter-analyst variance

### 7.3 Tier 3 — GUI smoke tests (lithicope)
- `pytest-qt` for window lifecycle
- Orientation tool receives mouse clicks
- Batch mode generates correct CSV

### 7.4 Test fixtures
- 3–5 synthetic meshes (PLY) with exact known dimensions
- 1 real sample mesh (photogrammetry scan)
- Bundled in `tests/fixtures/`

## 8. Tech Stack

| Layer | Choice | Rationale |
|---|---|---|
| Language | Python 3.11+ | trimesh, Open3D, scientific ecosystem |
| Measurement library | trimesh + NumPy + SciPy | trimesh for boolean ops/manifold repair/volume; SciPy for spatial |
| GUI | PyQt6 | Mature, stable, OpenGL widget support |
| 3D rendering | Open3D | Pythonic API, built-in visualisation |
| CLI | Typer | Automatic help text, type validation |
| Export | pandas (CSV), ReportLab (PDF) | Standard data + document tools |

## 9. Technical Risks

| Risk | Mitigation |
|---|---|
| Mesh quality variance | Built-in validation + repair pipeline. Graceful degradation |
| Full-auto orientation is unsolved research | Three-tier strategy. Default auto, fallback to semi/manual |
| Edge angle accuracy depends on mesh resolution | Quality grades communicated to user. Minimum resolution documented |

## 10. Design Boundaries

- **Publication figure generator** is explicitly scoped as the first v2 deliverable, not part of v1 MVP
- All other v2 features (flake scar detection, cortex ratio, 3D landmarks, comparison mode, reference classification) remain on the roadmap
- No cloud/web component in v1 — standalone desktop only
- No MorphoJ integration of data *into* the app, only export *to* MorphoJ format

---

*Specification approved by user on 2026-05-26. Ready for implementation planning.*
