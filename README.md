<div align="center">

# ⛏ Dibble

**Digital Image-Based Benchmark for Lithic Evaluation**

*End-to-end 3D lithic analysis — from photos to classified artefact, fully offline.*

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square&logo=python&logoColor=white)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green?style=flat-square)]()
[![Platform](https://img.shields.io/badge/platform-linux%20|%20macOS%20|%20win-lightgrey?style=flat-square)]()
[![No GPU required](https://img.shields.io/badge/GPU-none%20required-brightgreen?style=flat-square)]()
[![CLI](https://img.shields.io/badge/CLI-%E2%9C%94-blueviolet?style=flat-square)]()
[![GUI](https://img.shields.io/badge/GUI-%E2%9C%94-orange?style=flat-square)]()
[![Status: Beta](https://img.shields.io/badge/status-beta-yellow?style=flat-square)]()
[![Benchmark: 86.1% CV](https://img.shields.io/badge/benchmark-86.1%25%20CV-4472C4?style=flat-square)]()

> **Status: Beta** — An open-source research tool in active development.
> Named for **[Harold Dibble](https://en.wikipedia.org/wiki/Harold_Dibble)** (1951–2018),
> pioneering lithic technologist and open-source advocate.

---

https://github.com/user-attachments/assets/placeholder

</div>

## 🔍 What is Dibble?

Dibble transforms 3D scans of stone tools into **standardised morphometric measurements**, **typological classifications**, and **publication-ready figures** — all on your own machine, no GPU required, no data sent to the cloud.

**A single command takes you from mesh to analysis:**

```bash
# Batch process a folder of 3D scans
lithicore batch ./excavation_meshes/ --output classified.csv

# Launch the desktop GUI
lithicope
```

### At a glance

| Capability | What it does |
|-----------|--------------|
| **📐 3D processing** | Import PLY/OBJ/STL → auto-orient → extract 22 morphometric measurements → detect flake scars → compute platform angles |
| **🧠 AI classification** | Predict typology (Basic/Bordes/Technological) with **86.1% CV accuracy** on European Upper Palaeolithic assemblages. Explainable predictions with per-feature breakdowns. |
| **📷 Photogrammetry** | 9-stage COLMAP pipeline: raw photos → cleaned 3D mesh → PLY output. Three-tier interface from one-click to full expert control. |
| **📊 Publication figures** | Standardised three-view technical drawings (plan, profile, section) with scale bars and measurement callouts. Exported as SVG via VTK GL2PS. |
| **🏷 3D annotation** | Pin notes, measurements, and screenshots to any point on the mesh surface. Export/import for collaboration. |
| **💬 AI Assistant** | Query your collection in natural language via a local LLM. Fully offline — no data ever leaves your machine. |
| **🧪 Training pipeline** | Retrain classifiers on your own data. Add new typologies. Export to ONNX for deployment. |

---

## ⚠️ Known Limitations

Dibble is a **research tool (v0.4.1-beta)**, not a commercial product. Please read these before using the classifier in published research.

### Geographic bias
~70% of training data comes from **Italian Aurignacian sites** (Fumane, Castelcivita, Cala, Bombrini). The combined model achieves **86.1% 5-fold CV** on this data but performs **significantly worse on non-European assemblages**. We now provide **tradition-specific models** (OAP, COADS, Levantine, Experimental) — select the appropriate tradition from the GUI dropdown to get honest, per-tradition accuracy.

**Dataset contributions from the archaeological community are warmly welcomed.** If you have 3D scans of lithic assemblages from under-represented regions or time periods, please open an issue or pull request — every new tradition improves the classifier's ability to serve the whole field, not just one corner of it.

### Class imbalance
| Class | Samples | Reliability |
|-------|---------|-------------|
| Biface | 1,018 | Good |
| Blade | 993 | Good |
| Core | 751 | Good |
| Flake | 262 | Moderate |
| Experimental Core | 254 | Moderate |
| Retouched Flake | 57 | Low — verify |
| Unmodified Flake | 50 | Low — verify |
| Unmodified Cobble | 30 | Low — verify |

### Recommendations for publication
1. **Retrain on your own assemblage** before relying on predictions
2. **Use tradition-specific models** via the GUI dropdown for non-OAP assemblages
3. **Treat low-confidence predictions** (<0.6) as suggestions requiring expert verification
4. **Run `lithicore benchmark`** after retraining to assess real performance
5. **Review the feature breakdown** panel to understand each decision

---

## ✨ Features

### 📐 3D Mesh Processing
- **Import formats:** PLY, OBJ, STL (single file or batch directory)
- **Auto-orientation:** PCA-based alignment with platform detection + manual fallback
- **Mesh validation:** Detects non-manifold geometry, inverted normals, degenerate faces, isolated components — with auto-repair
- **Measurement suite:** Length, width, thickness, surface area, volume, elongation, flatness, compactness; platform angles (EPA/IPA); edge angle statistics (mean, std, skewness, kurtosis); curvature index; cross-section profile; symmetry score; dorsal ridge count; surface roughness
- **Edge detection:** Configurable dihedral angle thresholding for retouched edges and ridges
- **Batch processing:** Entire directories → CSV/JSON export

### 🔬 Advanced Analysis
- **Interactive 3D viewer** — PyVista/VTK with rotate, zoom, pan, and four display modes (solid, wireframe, translucent, points)
- **Diagnostic overlays** — Viewer highlights dorsal ridges (blue), platform (green), and retouched edges (red)
- **Publication figures** — Three-view drawings (plan, profile, section) with scale bar and measurements. SVG via VTK GL2PS.
- **Mesh comparison** — Overlay two meshes with adjustable opacity, compare 9 difference metrics
- **3D landmarks** — Click-to-place anatomical landmarks (13-point flake, 16-point biface). Export to MorphoJ format.
- **Flake scar detection** — Automated segmentation via Shape Index curvature + watershed algorithm

### 🤖 AI Lithic Classification
Dibble's classifier extracts a **47-dimensional feature vector** from every mesh (22 core morphometrics + 10 interaction features + 15 topological PH features):

| Feature group | Examples |
|--------------|----------|
| **Size & shape** | Length, width, thickness, area, volume, elongation, flatness, compactness |
| **Angles** | Platform angle (EPA/IPA), edge angle statistics (mean, std, skewness, kurtosis) |
| **Surface** | Scar count, curvature index, cross-section profile, symmetry, dorsal ridge count, roughness |
| **Topological (PH)** | Multi-scale surface texture via Alpha complex persistent homology (15 PCA components) |

**Three pre-trained typologies:**

| Typology | Classes | CV Accuracy |
|----------|---------|-------------|
| Basic | 8 | **86.1%** |
| Bordes | 8 | **86.1%** |
| Technological | 8 | **75.3%** |

**Tradition-aware classification:** Select from OAP (Europe), COADS (Ohio), Levantine, or Experimental in the GUI — the model dispatches to a tradition-specific classifier that only predicts classes present in that tradition.

**Explainable predictions:** Every prediction comes with a per-feature breakdown — which measurements drove the decision, their values, and whether each falls in the expected range for the predicted type.

**Active learning:** Correct a prediction → the sample queues for retraining. After 10 corrections, the model retrains automatically on baseline data plus your corrections.

**Custom typologies:** Define your own classification system, train on your labelled meshes, save the model to share with colleagues.

### 📷 Photogrammetry Pipeline
- **9-stage COLMAP pipeline:** Feature extraction → matching → sparse reconstruction → dense MVS → fusion → cleaning → meshing → decimation → output
- **Three-tier interface:** Default (one-click), Guided (quality/speed settings), Expert (full COLMAP control)
- **Photo pre-processing:** Blur detection (Laplacian variance), CLAHE exposure normalisation, auto-resize
- **Scale detection:** ArUco markers (±0.1%), ruler/scale bar (±1%), or manual
- **Batch queue:** Process multiple artefacts sequentially with progress tracking

### 🏷 3D Annotation
- Pin notes, measurements, and screenshots to any mesh surface point
- Three display modes: Pin+Label, Pin Only, Numbered with legend
- Export/import for collaboration (intelligent merge by 3D position)

### 💬 AI Lithic Assistant
- Ask questions in plain English: *"Show me crested blades with platform angles over 75°"*
- Local Qwen3-4B LLM via llama.cpp — fully offline, no data ever leaves your machine
- Translates questions to SQL → queries in-memory DuckDB → explains results
- Self-correcting (up to 3 retries on SQL failure)
- *Optional:* `pip install llama-cpp-python` (~2.5 GB model downloads on first use)

---

## 🚀 Quick Start

```bash
# 1. Install from source
git clone https://github.com/mabo-du/dibble.git
cd dibble
pip install -e lithicore      # Core library
pip install -e lithicope       # GUI (optional)

# 2. Verify
lithicore --help
lithicore benchmark            # Validation report → browser

# 3. Process meshes
lithicore batch ./meshes/ --output results.csv
lithicore figure artefact.ply --output figure.svg --label "FLK-145"

# 4. Launch GUI
lithicope
```

---

## 🏛 Architecture

```
lithicore/                          # Core library (pure Python, no GUI)
├── src/lithicore/
│   ├── _models.py                  # All shared data types
│   ├── _classification.py          # 47-dim feature extraction + classifier
│   ├── _ph_features.py             # Persistent Homology (Alpha complex)
│   ├── _validation.py              # Mesh validation + repair
│   ├── _orientation.py             # PCA + manual orientation
│   ├── _metrics.py                 # Length, width, thickness, etc.
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
│   └── _assistant.py               # AI Lithic Assistant
├── data/
│   ├── models/                     # Pre-trained classifiers (.joblib)
│   ├── training/
│   │   ├── retrain.py              # Retrain all classifiers
│   │   ├── batch_ph.py             # Batch PH feature computation
│   │   └── processed/              # training_matrix.csv
│   └── run_benchmark.py            # CLI benchmark entry point

lithicope/                          # Desktop GUI (PyQt6 + PyVista)
└── src/lithicope/
    ├── main.py                     # Entry point
    ├── _main_window.py             # Tabbed main window
    ├── _viewer_3d.py               # 3D viewport (PyVista)
    ├── _classification_panel.py    # Classification + tradition selector
    ├── _annotation_panel.py        # Annotation management
    ├── _assistant_panel.py         # AI Assistant chat interface
    ├── _photogrammetry_dialog.py   # 3-mode photogrammetry
    ├── _results_panel.py           # Measurement display
    ├── _batch_runner.py            # Threaded batch processing
    └── _import_dialog.py           # Import configuration
```

---

## 📊 Training Data

Trained on **3,415 real-world 3D scans** from five assemblages across three continents:

| Source | Origin | Artefacts | Scanner | Years represented |
|--------|--------|-----------|---------|-------------------|
| Open Aurignacian Project | Italy | 2,010 | Artec Space Spider / micro-CT | ~42,000–33,000 BP |
| Levantine Acheulean Handaxes | Israel/Palestine | 526 | Structured light | ~1.4M–200,000 BP |
| COADS Projectile Points | Ohio, USA | 492 | Structured light | ~3,000–1,000 BP |
| Lombao Experimental Cores | Spain | 284 | Structured light | Modern (controlled) |
| Morales Experimental Retouch | Spain | 100 | Structured light | Modern (controlled) |

*The Open Aurignacian Project datasets (Fumane, Castelcivita, Cala, Bombrini) were generously made available by **[Armando Falcucci](https://www.armandofalcucci.com/)** and colleagues at the University of Southampton and the University of Tübingen. These open-access repositories of high-resolution 3D lithic models have been invaluable. If you use Dibble with Aurignacian material, please consider citing their work — full references can be found in the [research papers directory](docs/research-papers/).*

---

## 📋 Requirements

| Dependency | Version | Purpose |
|-----------|---------|---------|
| Python | 3.11+ (3.13 supported) | Runtime |
| PyQt6 | ≥6.5 | GUI framework |
| PyVista / VTK | ≥0.42 | 3D rendering + figure export |
| trimesh | ≥4.0 | Mesh processing |
| NumPy / SciPy | ≥1.24 / ≥1.11 | Numerical computing |
| scikit-learn / joblib | ≥1.3 / ≥1.2 | Classifier training |
| opencv-python | ≥4.7 | Photo pre-processing + ArUco |
| GUDHI | ≥3.12 | Persistent Homology (PH features) |
| umap-learn / hdbscan | optional | Morphospace diagnostics |
| **llama-cpp-python** | *optional* | Local LLM (AI Assistant) |
| **COLMAP** | *separate install* | Photogrammetry |

---

## 📖 Documentation

- **[User Guide](docs/User_Guide.md)** — Full documentation: installation, GUI, CLI, training pipeline
- **[Research Papers](docs/research-papers/)** — Deep Research papers on accuracy enhancement, PH, and typology
- **[Research Prompts](docs/research-prompts/)** — Prompts used for Deep Research investigations

---

## 📝 Citation

```bibtex
@software{bouck2026dibble,
  author    = {Bouck, M.},
  title     = {Dibble: Digital Image-Based Benchmark for Lithic Evaluation},
  year      = {2026},
  version   = {0.4.1-beta},
  url       = {https://github.com/mabo-du/dibble},
}
```

---

## 📝 Changelog

| Version | Date | Changes |
|---------|------|---------|
| **0.4.1-beta** | 2026-06 | Bug-fix release: feature dimension alignment for PH-augmented models, `holes_filled` counter, ruler scale guard, thread-safe numpy RNG, `AssistantResult` public API, temp file race fix, SVG XSS hardening, SQL safety validator |
| 0.4.0-beta | 2026-06 | Tradition-aware models, PH features, hierarchical cascade, GUI tradition selector, edge-angle fix |
| 0.1.0 | 2026-05 | Initial release — 3D viewer, photogrammetry, classification, AI Assistant |

Full details: [docs/User_Guide.md](docs/User_Guide.md#version-history)

---

## 📜 License

MIT License. See `LICENSE` for details.

*Built for the archaeological community. Named for Harold Dibble (1951–2018).*
