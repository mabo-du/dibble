# Dibble

<p align="center">
  <b>Digital Image-Based Benchmark for Lithic Evaluation</b><br>
  <i>End-to-end 3D lithic analysis — from photos to classified artefact, fully offline.</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
  <img src="https://img.shields.io/badge/platform-linux%20%7C%20macOS%20%7C%20windows-lightgrey" alt="Platform">
  <img src="https://img.shields.io/badge/GPU-none%20required-brightgreen" alt="No GPU required">
  <img src="https://img.shields.io/badge/CLI-%E2%9C%94-blueviolet" alt="CLI">
  <img src="https://img.shields.io/badge/GUI-%E2%9C%94-orange" alt="GUI">
</p>

---

## Overview

Dibble is a desktop application for **automated 3D lithic (stone tool) analysis**.
Import a 3D mesh — or a folder of photos for photogrammetric reconstruction — and
Dibble orients it, extracts standardised measurements, detects flake scars,
identifies typology, and lets you annotate, compare, and query your collection
in natural language. Everything runs locally. No GPU required. No cloud dependency.

Dibble is named for **[Harold Dibble](https://en.wikipedia.org/wiki/Harold_Dibble)**
(1951–2018), pioneering lithic technologist and open-source advocate.

---

## Features

### 📐 3D Mesh Processing

| Feature | Details |
|---|---|
| **Import** | PLY, OBJ, STL — single file or batch directory |
| **Auto-orientation** | PCA-based alignment with platform detection. Or manually orient by clicking three points. |
| **Mesh validation** | Detects non-manifold geometry, inverted normals, degenerate faces, isolated components. Auto-repair option. |
| **Measurement** | Max length, width, thickness, surface area, and volume. All measurements in mm/mm²/mm³ in oriented coordinate space. |
| **Platform angles** | Exterior platform angle (EPA) and interior platform angle (IPA) extracted automatically. |
| **Edge detection** | Configurable dihedral angle thresholding for identifying retouched edges and ridges. |
| **Batch processing** | Process entire directories of meshes with a single command. Exports CSV, JSON, or PDF. |

### 🔬 Advanced Analysis

| Feature | Details |
|---|---|
| **3D viewer** | Interactive PyVista/VTK viewport with rotate, zoom, pan. Embedded directly in the GUI — not screenshot-based. |
| **Publication figures** | Generate standardised three-view technical drawings (plan, profile, section) with scale bar, artefact ID, and measurement callouts. Exported as SVG via VTK GL2PS. |
| **Mesh comparison** | Overlay two meshes, adjust opacity, compare 9 difference metrics including Hausdorff distance, volume difference, centroid distance, and L/W/T deltas. |
| **3D landmarks** | Click-to-place anatomical landmarks on the mesh with scheme-guided naming (13-point flake, 16-point biface). Export to MorphoJ format for geometric morphometrics. |
| **Flake scar detection** | Automated scar segmentation via Shape Index curvature analysis and watershed algorithm. Each scar is labelled with its area and position. |

### 📷 Photogrammetry Pipeline

| Feature | Details |
|---|---|
| **Photos to mesh** | Full 9-stage COLMAP pipeline: feature extraction → matching → sparse reconstruction → dense MVS → fusion → cleaning → meshing → decimation → output. |
| **Three-tier interface** | Default (one-click), Guided (settings), Expert (full COLMAP control with GPU toggle and dense quality settings). |
| **Point cloud cleaning** | Statistical outlier removal with spatial-spread floor plus automatic background cropping via PCA bounding box. |
| **Scale detection** | Automatic detection via ArUco markers (±0.1% accuracy) or ruler/scale bar analysis via Hough lines + tick mark frequency. Manual mode: click two points, enter known distance. |
| **Photo pre-processing** | Blur detection (Laplacian variance filters out motion-blurred images), CLAHE exposure normalisation for consistent lighting across photo sets, and automatic image resizing. |
| **Batch queue** | Process multiple artefacts sequentially with live progress tracking and per-artefact preview. |

### 🏷 3D Annotation

| Feature | Details |
|---|---|
| **Pin annotations** | Attach rich notes to any point on the mesh surface — title, description, category, measurement, confidence. |
| **Three display modes** | Pin+Label (always visible), Pin Only (appear on hover), or Numbered markers with legend. |
| **Photo capture** | Take a screenshot of the current 3D view and attach it directly to any annotation. |
| **Collaboration** | Export annotations as JSON. Colleagues import on their machine — Dibble merges intelligently, detecting conflicts by 3D position. |

### 🤖 AI Lithic Classification

| Feature | Details |
|---|---|
| **Morphometric fingerprint** | 22-dimensional feature vector extracted from every mesh: length, width, thickness, elongation, flatness, scar count, platform angle, edge angle statistics (mean, std, skewness, kurtosis), curvature, cross-section profile, symmetry, dorsal ridge count, and surface roughness. |
| **Three pre-trained typologies** | **Basic** (8 classes: Biface, Blade, Bladelet, Core, Flake, Retouched Flake, Tool, Unmodified Flake — 80.7% 5-fold CV), **Bordes** (same morphology-based mapping — 80.7% CV), **Technological** (7 reduction stages — 65.3% CV). |
| **Explainable predictions** | Every prediction includes a per-feature breakdown showing which measurements drove the decision and whether each falls within the expected range for the predicted type. |
| **Diagnostic viewer overlays** | The 3D viewer highlights dorsal ridges (blue), platform (green), and retouched edges (red) — the physical features that the classifier used. |
| **Active learning** | Every time you correct a prediction, it queues for retraining. After 10 corrections, the model retrains automatically on 3,028 baseline artefacts plus your corrections. |
| **Custom typologies** | Define your own typology system and train a classifier on your own types. Save the model to share with colleagues. |
| **Self-validation** | Run `lithicore benchmark` to generate an interactive HTML report with confusion matrices, per-class precision/recall/F1, and accuracy scores. Reproducible on any machine. |

### 💬 AI Lithic Assistant

| Feature | Details |
|---|---|
| **Natural language queries** | Ask questions about your collection in plain English: *"Show me all crested blades with platform angles over 75°"* or *"What's the average length of scrapers?"* |
| **Local LLM** | Powered by Qwen3-4B running locally via llama.cpp. Fully offline — no data ever leaves your machine. |
| **SQL generation** | The assistant translates your question into SQL, executes it against the in-memory collection via DuckDB, and explains the results. Toggle "Show SQL" to see and verify the generated query. |
| **Self-correcting** | If the generated SQL fails, the assistant automatically fixes it and retries (up to 3 attempts). |

*The AI Assistant is optional. Install with `pip install llama-cpp-python`. Model downloads automatically on first use (~2.5 GB).*

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

---

## Architecture

```
lithicore/                          # Core library (pure Python, no GUI)
├── src/lithicore/
│   ├── _models.py                  # All shared data types
│   ├── _validation.py              # Mesh validation + repair
│   ├── _orientation.py             # PCA + manual orientation
│   ├── _metrics.py                 # Length, width, thickness, area, volume
│   ├── _edge_detection.py          # Dihedral angle edge detection
│   ├── _platform_angle.py          # EPA, IPA extraction
│   ├── _comparison.py              # Mesh comparison metrics
│   ├── _landmarks.py               # 3D landmark placement
│   ├── _scar_detection.py          # Shape Index + watershed
│   ├── _figure.py                  # Publication figure generator
│   ├── _batch.py                   # Batch processing
│   ├── _cli.py                     # CLI entry point
│   ├── _photogrammetry.py          # COLMAP pipeline
│   ├── _scale_detection.py         # ArUco + ruler detection
│   ├── _photo_preprocessing.py     # Blur + exposure correction
│   ├── _annotations.py             # Annotation data model
│   ├── _classification.py          # Feature extraction + classifier
│   └── _assistant.py               # AI Lithic Assistant
├── data/
│   ├── models/                     # Pre-trained classifiers (.joblib)
│   ├── grammars/                   # GBNF grammar for SQL generation
│   ├── training/
│   │   ├── download_and_process.py # Dataset download + feature extraction
│   │   ├── _worker.py              # Per-mesh subprocess worker (OOM-safe)
│   │   ├── process_safe.py         # Memory-safe batch processing pipeline
│   │   ├── retrain.py              # Retrain all classifiers from matrix
│   │   ├── download_coads.py       # COADS batch downloader (Zenodo API)
│   │   ├── validate_pipeline.py    # Photo-to-mesh pipeline validator
│   │   └── processed/              # Generated training_matrix.csv
│   ├── generate_training_data.py   # Legacy synthetic data generator
│   └── run_benchmark.py            # CLI benchmark entry point

lithicope/                          # Desktop GUI (PyQt6 + PyVista)
└── src/lithicope/
    ├── main.py                     # Entry point
    ├── _main_window.py             # Main window with tabbed interface
    ├── _viewer_3d.py               # 3D viewport (PyVista)
    ├── _import_dialog.py           # Import configuration
    ├── _results_panel.py           # Measurement display
    ├── _batch_runner.py            # Threaded batch processing
    ├── _batch_photogrammetry.py    # Batch photogrammetry queue
    ├── _photogrammetry_dialog.py   # 3-mode photogrammetry UI
    ├── _annotation_panel.py        # Annotation management
    ├── _classification_panel.py    # Classification + active learning
    └── _assistant_panel.py         # AI Assistant chat interface
```

---

## Requirements

| Dependency | Version | Purpose |
|---|---|---|
| Python | 3.11+ (3.13 supported) | Runtime |
| PyQt6 | ≥6.5 | GUI framework |
| PyVista / VTK | ≥0.42 | 3D rendering and figure export |
| trimesh | ≥4.0 | Mesh processing and measurements |
| NumPy / SciPy | ≥1.24 / ≥1.11 | Numerical computing |
| opencv-python | ≥4.7 | Photo pre-processing and ArUco detection |
| scikit-learn / joblib | ≥1.3 / ≥1.2 | Classifier training and prediction |
| pandas / ReportLab | ≥2.0 | CSV, JSON, PDF export |
| DuckDB | ≥1.0 | In-memory SQL queries (AI Assistant) |
| **llama-cpp-python** | *optional* | Local LLM inference (AI Assistant) |
| **COLMAP** | *separate install* | Photogrammetry reconstruction |

---

## Classifier Validation

The classifiers are trained on **3,415 real-world 3D scan meshes** from three continents:

| Data Source | Origin | Artefacts | Scanner |
|---|---|---|---|
| Open Aurignacian Project (Vols 1-4) + Broglio, Castelcivita, Cala | Italy | 2,418 | Artec Space Spider / Micro / micro-CT |
| Levantine Acheulean Handaxes | Israel / Palestine | 526 | Structured light |
| COADS (Central Ohio Arch. Digitization Survey) | Ohio, USA | 514 | Structured light |
| Lombao Experimental Cores | Spain | 254 | Structured light |
| Morales Experimental Retouch | Spain | 100 | Structured light |

| Typology | Classes | 5-Fold CV Accuracy | Training Accuracy |
|---|---|---|---|---|
| Basic Morphological | Biface, Blade, Bladelet, Core, Experimental Core, Flake, Retouched Flake, Unmodified Cobble, Unmodified Flake | **84.8%** | 96.8% |
| Bordes Typology | Same morphology-based mapping | **84.8%** | 96.8% |
| Technological | Handaxe, Initialization, Maintenance, Optimal, Other, Semi-cortical, Undetermined, Unknown | **73.6%** | 92.3% |

*Results are 5-fold cross-validation on real archaeological and experimental meshes.
Run `lithicore benchmark` to generate a full interactive HTML report with confusion
matrices, per-class precision/recall/F1 scores, and cross-validation accuracy.*

---

## Known Limitations

Dibble is an **open-source research tool**, not a production-grade commercial product.
Before using the classifier in published research, please be aware of these limitations:

### Class imbalance
The classifier performs well on well-represented classes (Biface: 1,018 samples,
Core: 751, Bladelet: 592) but is **unreliable on rare classes**:

| Class | Samples | Reliability |
|-------|---------|-------------|
| Biface, Core, Bladelet | 592–1,018 | Good |
| Blade, Flake, Experimental Core | 254–401 | Moderate |
| Retouched Flake | 57 | **Low — use predictions with caution** |
| Unmodified Flake, Unmodified Cobble | 30–50 | **Low — may misclassify** |

### Geographic bias
~70% of training data comes from Italian Aurignacian sites (Fumane, Castelcivita,
Cala, Bombrini). The classifier will be **less accurate on non-European assemblages**
or time periods not represented in training.

### Feature limitation
One of the 22 morphometric features (`edge_angle_std_deg`) was computed as zero
for the entire training corpus due to an earlier pipeline bug (CSV column name
mismatch). This has been fixed and the existing matrix patched — **88.8% of rows
now have correct non-zero values**. The 11.2% still at zero correspond to meshes
not available on disk (some Levantine, Lombao, and COADS specimens).

### Cross-validation vs real-world accuracy
The reported 81.6% CV accuracy is a **best-case estimate** from 5-fold cross-validation
on the training set. Real-world accuracy on independently collected assemblages will
likely be lower.

### Recommendations for users
- Retrain the classifier on your own assemblage before relying on predictions
  (see custom typology training in the GUI)
- Treat predictions on Retouched Flake, Unmodified Flake, and Unmodified Cobble
  as **suggestions that require expert verification**
- Use the explainable predictions panel to review which features drove each decision
- Run `lithicore benchmark` on your retrained model to assess its actual performance

---

## Citation

MIT License. See `LICENSE` for details.

*Built for the archaeological community. Named for Harold Dibble (1951–2018).*
