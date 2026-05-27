# Dibble

**Digital Image-Based Benchmark for Lithic Evaluation**

Dibble is a desktop application for **end-to-end lithic analysis** — from photos to classified 3D artefact. Import a 3D mesh (or photos for photogrammetry), and Dibble orients it, extracts standardised measurements, detects flake scars, identifies typology, and lets you annotate, compare, and classify — all locally, no GPU required.

---

## Features

### v1 — Core platform
- **Import** 3D meshes in PLY, OBJ, and STL formats
- **Orient** automatically (PCA + platform detection) or manually (click 3 points)
- **Validate & repair** meshes — fills holes, removes isolated components, fixes normals
- **Measure** — max length, width, thickness, surface area, volume, exterior/interior platform angles (EPA/IPA)
- **Detect edges** — configurable dihedral angle thresholding
- **Batch process** entire directories of meshes
- **Export** CSV, JSON, or PDF reports

### v2 — Advanced analysis
- **3D viewer** — interactive PyVista viewport with rotate, zoom, pan
- **Publication figures** — standardised three-view technical drawings (plan/profile/section) with scale bar and artefact ID, exported as SVG via VTK GL2PS
- **Measurement callouts** — L/W/T labels with leader lines on publication figures
- **Comparison mode** — overlay two meshes, adjust opacity, compare 9 difference metrics (Hausdorff distance, volume difference, centroid distance, L/W/T diffs)
- **3D landmarks** — click-to-place anatomical landmarks, scheme-guided naming (13-point flake, 16-point biface), MorphoJ export
- **Flake scar detection** — automated segmentation via Shape Index curvature + watershed

### v3 — Photogrammetry (May 2026)
- **Full pipeline** — photos → 3D mesh via COLMAP (9 stages: feature extraction → matching → sparse recon → dense MVS → cleaning → meshing → decimation → output)
- **Three-tier import** — Default (one-click), Guided (settings), Expert (full COLMAP control)
- **Point cloud cleaning** — statistical outlier removal + automatic background cropping (PCA)
- **Batch queue** — process multiple artefacts sequentially with live progress

### v3.5 — Scale & GPU (May 2026)
- **Automatic scale detection** — ArUco markers (±0.1%) or ruler/scale bar analysis via Hough lines
- **Manual scale** — click two points on mesh, enter known distance
- **GPU acceleration** — CUDA toggle for COLMAP stages (auto-detected)
- **Live preview** — point/face count polling during reconstruction
- **Photo pre-processing** — blur detection (Laplacian variance), CLAHE exposure normalisation, image resize

### v4 — Annotation & Classification (May 2026)
- **3D annotation** — pin notes on mesh with title, description, category, measurement, confidence
- **Three display modes** — Pin+Label, Pin Only (hover), Numbered markers
- **Photo capture** — screenshot from 3D view attached to any annotation
- **Multi-user collaboration** — JSON export/import with smart merge (conflict detection by position)
- **AI lithic classification** — 20-dim morphometric fingerprint extracts every diagnostic feature from the mesh
- **Three pre-trained typologies** — Basic (flake/blade/bladelet/core/tool), Bordes (7 tool types), Technological (5-stage reduction)
- **Explainable predictions** — "Why is this a blade?" with per-feature contribution percentages and expected ranges
- **Diagnostic viewer overlays** — ridges (blue), platform (green), retouched edges (red) highlighted on mesh
- **Active learning** — every correction retrains the model; auto-triggered at 10 corrections
- **Custom typologies** — train a classifier on your own types from your own collection, save/share models

---

## Quick start

```bash
# Install
pip install dibble dibble-gui

# CLI — batch process a folder of meshes
dibble batch ./meshes/ --output results.csv

# CLI — generate a publication figure
dibble figure artefact.ply --output figure.svg --label "FLK-145"

# GUI — launch the desktop application
dibble-gui
```

## Architecture

```
lithicore/                 # Measurement library (pure Python, no GUI)
├── pyproject.toml
├── data/
│   ├── generate_training_data.py  # Synthetic lithic training data generator
│   └── models/                    # Pre-trained classifier .joblib files
└── src/lithicore/
    ├── _models.py          # Core data types + classification dataclasses
    ├── _validation.py      # Mesh validation + repair
    ├── _orientation.py     # PCA + manual orientation
    ├── _metrics.py         # Length, width, thickness, area, volume
    ├── _edge_detection.py  # Dihedral angle thresholding
    ├── _platform_angle.py  # EPA, IPA extraction
    ├── _comparison.py      # Mesh comparison metrics
    ├── _landmarks.py       # 3D landmark placement + MorphoJ export
    ├── _scar_detection.py  # Shape Index + watershed segmentation
    ├── _figure.py          # Publication figure generator (VTK GL2PS)
    ├── _batch.py           # Batch processing pipeline
    ├── _cli.py             # CLI entry point
    ├── _photogrammetry.py  # COLMAP pipeline orchestration
    ├── _scale_detection.py # ArUco/ruler scale detection
    ├── _photo_preprocessing.py  # Blur detection, exposure normalisation
    ├── _annotations.py     # Annotation data model + merge
    └── _classification.py  # Feature extraction + classifier model

lithicope/                 # Desktop GUI (PyQt6 + PyVista)
├── pyproject.toml
└── src/lithicope/
    ├── main.py             # Application entry point
    ├── _main_window.py     # Main window shell
    ├── _viewer_3d.py       # PyVista 3D viewer (annotations, overlays, landmarks)
    ├── _import_dialog.py   # Import mode selection
    ├── _results_panel.py   # Measurement + export panel
    ├── _batch_runner.py    # Threaded batch processing
    ├── _photogrammetry_dialog.py  # 3-mode photogrammetry dialog
    ├── _annotation_panel.py      # Annotation list + edit panel
    └── _classification_panel.py  # Classification result display + active learning
```

## Requirements

- **Python 3.11+** (Python 3.13 supported)
- **PyQt6** — for the GUI
- **PyVista / VTK** — for 3D rendering and figure export
- **trimesh, NumPy, SciPy** — for mesh processing and measurements
- **opencv-python** — for photo pre-processing and scale detection
- **scikit-learn, joblib** — for lithic typology classification
- **COLMAP** — for photogrammetry (install separately: `brew install colmap`, `sudo apt install colmap`, or `conda install -c conda-forge colmap`)
- **pandas, ReportLab** — for CSV and PDF export

## Configuration

Edge detection threshold, auto-repair behaviour, and landmark schemes are configurable through the GUI import dialog or via code:

```python
from dibble import MeasurementConfig
config = MeasurementConfig(repair_mesh=True, edge_threshold_degrees=45.0)
```

## Project status

**Complete.** Dibble now covers the full end-to-end lithic analysis pipeline:
1. 📷 **Photos in** (or import existing 3D meshes)
2. 🔧 **Automatic orientation, repair, and measurement**
3. 🏷 **3D annotation and collaboration**
4. 🤖 **AI typology classification with active learning**
5. 📊 **Publication-ready figures and exports**

Active development continues with ceramic sherd classification ("Dibble: Fired") and LLM-powered natural language assemblage querying on the roadmap.

---

*Built for the archaeological community. Open source under the MIT license.*

*Named for Harold Dibble (1951–2018) — pioneering lithic technologist, open-source advocate.*
