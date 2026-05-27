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
- **AI lithic classification** — 22-dim morphometric fingerprint (length, width, scar count, platform angle, edge angle statistics, curvature, symmetry, dorsal ridges, etc.)
- **Three pre-trained typologies** — Basic (98% accuracy, 5 classes), Bordes (85% accuracy, 7 classes), Technological (90% accuracy, 5 classes)
- **Explainable predictions** — "Why is this a blade?" with per-feature contribution percentages and expected ranges
- **Diagnostic viewer overlays** — ridges (blue), platform (green), retouched edges (red) highlighted on mesh
- **Active learning** — every correction retrains the model; auto-triggered at 10 corrections
- **Custom typologies** — train a classifier on your own types from your own collection, save/share models
- **Self-validation benchmark** — run `lithicore benchmark` to generate an HTML validation report with confusion matrices and per-class metrics

### v4.1 — AI Lithic Assistant (May 2026)
- **Conversational AI** — ask questions about your collection in natural language
- **Local LLM** — Qwen3-4B runs entirely on your machine via llama.cpp (optional: `pip install llama-cpp-python`)
- **Natural language to SQL** — "show me all crested blades with platform angles over 75°" → instant results
- **Self-correcting** — the assistant fixes its own SQL errors automatically
- **Explainable** — toggle "Show SQL" to see and verify the generated query
- **Private** — fully offline, no data leaves your machine

---

## Quick start

```bash
# Install
pip install lithicore lithicope

# CLI — batch process a folder of meshes
lithicore batch ./meshes/ --output results.csv

# CLI — generate a publication figure
lithicore figure artefact.ply --output figure.svg --label "FLK-145"

# CLI — validate classifier accuracy
lithicore benchmark

# GUI — launch the desktop application
lithicope
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
    ├── _classification.py  # Feature extraction + classifier model
    └── _assistant.py       # AI Lithic Assistant (local LLM query engine)

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
    ├── _classification_panel.py  # Classification result display + active learning
    └── _assistant_panel.py       # AI Assistant chat interface
```

## Requirements

- **Python 3.11+** (Python 3.13 supported)
- **PyQt6** — for the GUI
- **PyVista / VTK** — for 3D rendering and figure export
- **trimesh, NumPy, SciPy** — for mesh processing and measurements
- **opencv-python** — for photo pre-processing and scale detection
- **scikit-learn, joblib** — for lithic typology classification
- **llama-cpp-python** — optional, for AI Lithic Assistant (`pip install llama-cpp-python`)
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
5. 💬 **Conversational AI assistant** — query your collection in natural language
6. 📊 **Self-validation benchmark** — `lithicore benchmark` generates accuracy reports

### Classifier accuracy (synthetic test data)

| Typology | Classes | Accuracy |
|---|---|---|
| Basic Morphological | 5 | ~98% |
| Bordes Typology | 7 | ~85% |
| Technological | 5 | ~90% |

*Note: These results are on held-out synthetic test data. Real-world accuracy depends on mesh quality, orientation accuracy, and artefact condition. Run `lithicore benchmark` to reproduce on your machine.*

### Roadmap
- **Ceramic sherd classification** ("Dibble: Fired") — reusing the classifier engine for pottery
- **Community benchmark** — user-contributed repository of real 3D lithic meshes with expert-verified labels

---

*Built for the archaeological community. Open source under the MIT license.*

*Named for Harold Dibble (1951–2018) — pioneering lithic technologist, open-source advocate.*
