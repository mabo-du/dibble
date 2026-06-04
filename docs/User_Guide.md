# Dibble User Guide

> **Version:** 0.4.0-beta | **Last updated:** 2026-06-04

Dibble is a desktop application for automated 3D lithic (stone tool) analysis — from photos to classified artefact, fully offline. This guide covers installation, the GUI workflow, the CLI tools, training custom classifiers, and understanding the accuracy of each component.

---

## Table of Contents

1. [Installation](#1-installation)
2. [Quick Start](#2-quick-start)
3. [The 3D Viewer](#3-the-3d-viewer)
4. [Importing and Processing Meshes](#4-importing-and-processing-meshes)
5. [Photogrammetry Pipeline](#5-photogrammetry-pipeline)
6. [Publication Figures](#6-publication-figures)
7. [Lithic Classification](#7-lithic-classification)
8. [Tradition-Aware Classification](#8-tradition-aware-classification)
9. [3D Annotation](#9-3d-annotation)
10. [AI Lithic Assistant](#10-ai-lithic-assistant)
11. [CLI Reference](#11-cli-reference)
12. [Training Pipeline](#12-training-pipeline)
13. [Configuration and Troubleshooting](#13-configuration-and-troubleshooting)

---

## 1. Installation

### System requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16 GB |
| Storage | 5 GB | 50 GB (for training data) |
| OS | Linux, macOS 12+, Windows 10+ | Linux (primary target) |
| Python | 3.11 | 3.13 |
| GPU | Not required | Optional (for COLMAP dense matching) |

### Install from source

```bash
git clone https://github.com/mabo-du/dibble.git
cd dibble

# Core library (required)
pip install -e lithicore

# GUI (optional — needed for desktop app)
pip install -e lithicope

# Verify
lithicore --help
lithicope --help
```

### Optional dependencies

| Dependency | Purpose | Install command |
|------------|---------|-----------------|
| **COLMAP** | Photogrammetry reconstruction | `apt install colmap` or [colmap.github.io](https://colmap.github.io/) |
| **llama-cpp-python** | Local AI Assistant | `pip install llama-cpp-python` (~2.5 GB model downloads on first use) |
| **skl2onnx** | ONNX model export | `pip install skl2onnx onnxruntime` |
| **GUDHI** | Persistent Homology features | `pip install gudhi` (pre-installed with lithicore) |

### Verify installation

```bash
# Process a single mesh
lithicore info ./test_mesh.ply

# Validate classifier accuracy
lithicore benchmark

# Launch desktop app
lithicope
```

---

## 2. Quick Start

### Launch the GUI

```bash
lithicope
```

The main window opens with a 3D viewport (centre), tool panels (right), and a menu bar (top).

### Load a mesh

1. Click **File > Open** or press `Ctrl+O`
2. Select a `.ply`, `.obj`, or `.stl` file
3. The mesh appears in the 3D viewport. Rotate with left-click, zoom with scroll wheel.

### Run classification

1. Click the **Classification** tab in the right panel (or press `Ctrl+Shift+C`)
2. Select a **tradition** from the dropdown (OAP, COADS, Levantine, Experimental, or Auto)
3. Select a **typology system** (Basic, Bordes, or Technological)
4. Click **Classify Artefact**
5. The predicted type, confidence score, and top diagnostic features are shown

### Run photogrammetry

1. Click **Tools > Photogrammetry** or press `Ctrl+P`
2. Select a folder containing photos of a single artefact
3. Choose **Default** mode for one-click reconstruction
4. The pipeline runs: photos → sparse model → dense model → cleaned mesh → PLY output

---

## 3. The 3D Viewer

Built on PyVista/VTK. Provides interactive 3D visualisation of lithic meshes.

### Controls

| Action | Mouse / Keyboard |
|--------|-----------------|
| Rotate | Left-click + drag |
| Pan | Middle-click + drag |
| Zoom | Scroll wheel |
| Reset view | `R` |
| Toggle axes | `A` |
| Toggle wireframe | `W` |
| Screenshot | `Ctrl+S` |

### View modes

| Mode | Description |
|------|-------------|
| Solid | Opaque surface rendering (default) |
| Wireframe | Edge mesh shown as lines |
| Translucent | Semi-transparent surface |
| Points | Vertex cloud display |

Toggle modes from the **View** menu or the toolbar dropdown.

### Diagnostic overlays

When classification is run with "Show diagnostic overlays" enabled:

| Feature | Colour | Meaning |
|---------|--------|---------|
| Dorsal ridges | Blue | Linear ridges from previous flake removals |
| Platform | Green | Striking platform surface |
| Retouched edges | Red | Areas of secondary modification |

### Measurement display

The **Results** tab shows all computed measurements:
- **Size:** Length, width, thickness (mm), surface area (mm²), volume (mm³)
- **Ratios:** Elongation, flatness, compactness
- **Angles:** Exterior platform angle (EPA), interior platform angle (IPA)
- **Edge statistics:** Mean, std, skewness, kurtosis of dihedral angles
- **Surface features:** Scar count, mean scar area, curvature index, cross-section profile, symmetry score, dorsal ridge count, surface roughness

---

## 4. Importing and Processing Meshes

### Supported formats

| Format | Extension | Notes |
|--------|-----------|-------|
| Polygon File Format | `.ply` | Preferred — best compatibility |
| Wavefront OBJ | `.obj` | Supported, may need companion `.mtl` |
| Stereolithography | `.stl` | Supported |
| Virtual Reality ML | `.wrl`, `.vrml` | Supported — may be slower |
| GL Transmission | `.glb` | Supported via conversion |

### Orientation

All measurements are computed in **oriented coordinate space**:

- **Z-axis:** Maximum length (reduction axis for flakes)
- **Y-axis:** Maximum width
- **X-axis:** Thickness

**Automatic (PCA-based):**
```bash
lithicore batch ./meshes/ --output results.csv
```
Aligns the first principal component with the Z-axis with platform detection.

**Manual:** In the GUI, click **Tools > Orient Manually** and click three points on the mesh to define the platform plane. Useful for irregular artefacts where PCA fails.

### Mesh validation

Dibble automatically checks for:
- Non-manifold geometry
- Inverted normals
- Degenerate faces (zero-area triangles)
- Isolated components (floating vertices)

**Auto-repair** fixes inverted normals and removes degenerate faces.

### Available measurements

| Measurement | Unit | Method |
|-------------|------|--------|
| Maximum length | mm | Oriented bounding box longest extent |
| Maximum width | mm | OBB second extent |
| Maximum thickness | mm | OBB third extent |
| Surface area | mm² | Sum of all face areas |
| Volume | mm³ | Mesh interior volume (watertight only) |
| Elongation | ratio | Length / Width |
| Flatness | ratio | Width / Thickness |
| Compactness | ratio | Volume / Length³ |
| Platform angle (EPA) | ° | Exterior platform angle |
| Platform angle (IPA) | ° | Interior platform angle |
| Edge angle mean | ° | Mean dihedral angle across all edges |
| Edge angle std dev | ° | Standard deviation of dihedral angles |
| Edge angle skewness | — | Asymmetry of edge angle distribution |
| Edge angle kurtosis | — | Tailedness of edge angle distribution |
| Scar count | integer | Number of detected flake scars |
| Curvature index | — | Mean vertex normal angular deviation |
| Cross-section profile | 0–2 | Flat (0), triangular (1), round (2) |
| Symmetry score | 0–1 | Bilateral symmetry (1 = perfect) |
| Dorsal ridge count | integer | Count of parallel linear ridges |
| Surface roughness | ratio | Face area / projected area |

### Batch processing

Process an entire directory of meshes:

```bash
# CSV output (default)
lithicore batch ./excavation_meshes/ --output analysis.csv

# JSON output
lithicore batch ./excavation_meshes/ --output analysis.json --format json

# Custom edge threshold
lithicore batch ./excavation_meshes/ --output results.csv --edge-threshold 45
```

Output includes one row per artefact with all measurements, typology predictions, and any processing warnings.

---

## 5. Photogrammetry Pipeline

Dibble wraps COLMAP into a 9-stage pipeline: photos → 3D mesh.

### Pipeline stages

| Stage | Description |
|-------|-------------|
| 1. Validate | Check photos exist, minimum count, blur detection |
| 2. Pre-process | CLAHE exposure normalisation, auto-resize |
| 3. Feature extract | COLMAP SIFT feature extraction |
| 4. Feature match | COLMAP exhaustive or sequential matching |
| 5. Sparse reconstruct | COLMAP sparse point cloud |
| 6. Dense MVS | COLMAP dense multi-view stereo |
| 7. Fuse + clean | Point cloud fusion, statistical outlier removal |
| 8. Mesh + decimate | Poisson surface reconstruction + decimation |
| 9. Output | Export cleaned mesh as PLY |

### Three-tier interface

**Default mode — one click:**
```bash
lithicore photogrammetry ./photos/ --output artefact.ply
```

**Guided mode — quality/speed trade-off:**
In the GUI photogrammetry dialog, select **Guided** to adjust: Quality (Low/Medium/High) and expected artefact size.

**Expert mode — full COLMAP control:**
```bash
lithicore photogrammetry ./photos/ --output artefact.ply \
    --quality high \
    --colmap-feature-type sift \
    --colmap-matching exhaustive \
    --dense-quality extreme
```

### Scale detection

| Method | Accuracy | How it works |
|--------|----------|-------------|
| ArUco marker | ±0.1% | Detects printed marker of known size in photos |
| Ruler/scale bar | ±1% | Hough line detection + tick mark frequency |
| Manual | User-defined | Click two points on the mesh, enter known distance |

### Batch queue

Process multiple artefacts sequentially:
```bash
lithicore photogrammetry ./excavation_photos/ --batch --batch-output ./results/
```

Outputs one PLY per artefact with live progress tracking.

---

## 6. Publication Figures

Generate standardised three-view technical drawings suitable for publication:

```bash
lithicore figure artefact.ply --output figure.svg --label "FLK-145"
```

The figure includes:
- **Plan view** (dorsal)
- **Profile view** (lateral)
- **Section view** (cross-section)
- Scale bar
- Artefact ID label
- Measurement callouts

Options:
```bash
# Hide annotations
lithicore figure artefact.ply --output figure.svg --no-measurements

# Hide scar ridge lines
lithicore figure artefact.ply --output figure.svg --no-ridges

# Custom label
lithicore figure artefact.ply --output figure.svg --label "Unit 4, Level 3"
```

Figures are exported as SVG via VTK GL2PS for lossless scaling in publications.

---

## 7. Lithic Classification

### Feature vector

Dibble extracts a **47-dimensional feature vector** from every mesh:

| Group | Dimensions | Examples |
|-------|-----------|---------|
| Core morphometrics | 22 | Length, width, angles, scar count, curvature, symmetry |
| Interaction features | 10 | Shape indices, ratios, products of core metrics |
| Persistent Homology | 15 | Multi-scale surface texture via Alpha complex (PCA-reduced) |

### Typology systems

| System | Classes | CV Accuracy | Description |
|--------|---------|-------------|-------------|
| Basic | 8 | **86.1%** | Broad morphological categories |
| Bordes | 8 | **86.1%** | Same morphology-based mapping |
| Technological | 8 | **75.3%** | Core reduction stages |

### Running classification

**In the GUI:**
1. Open the **Classification** tab (right panel)
2. Optionally select a **tradition** (see §8)
3. Select a **typology system** (Basic, Bordes, Technological)
4. Click **Classify Artefact**
5. View: predicted type, confidence score, per-feature breakdown, alternatives

**Via CLI:**
```bash
lithicore batch ./meshes/ --output classified.csv
```

### Understanding predictions

The classification result shows:
- **Predicted type:** The most likely class
- **Confidence:** Probability (0–1) assigned to the predicted class
- **Feature breakdown:** Which measurements drove the decision, their values, and whether each falls within the expected range
- **Alternatives:** Other classes with non-trivial probability (>1%)

A confidence of **<0.6** indicates the model is uncertain — expert verification is recommended.

### Active learning

When the classifier makes a mistake:
1. Click **Submit Correction** in the Classification panel
2. Select the correct label from the dropdown
3. The correction is queued for retraining
4. After **10 corrections**, the model retrains automatically

This adapts the classifier to your specific assemblage over time.

### Custom typologies

Define and train your own typology system:

1. Prepare a CSV with artefact IDs and your labels
2. Run the training pipeline:
   ```bash
   python3 lithicore/data/training/retrain.py
   ```
3. The custom model appears in the GUI's "Custom" typology option
4. Share the `.joblib` model file with colleagues

### Model export (ONNX)

```python
from lithicore import ClassifierModel
model = ClassifierModel.load_pre_trained("basic")
model.export_onnx("./my_model.onnx")
```

ONNX models can be loaded without the full scikit-learn dependency chain.

---

## 8. Tradition-Aware Classification

Dibble now provides **tradition-specific models** that are honest about what they can and cannot classify.

### Why this matters

The training data is dominated by European Aurignacian material (70% OAP). A single universal classifier trained on all five assemblages achieves **86.1% CV**, but this drops to **6–12% cross-dataset** when tested on unseen traditions. The morphological signatures of a "Biface" in the Levantine Acheulean are fundamentally different from a COADS projectile point — yet they share the same label.

### How it works

The GUI includes a **Tradition** dropdown with these options:

| Tradition | What it classifies | Accuracy | Notes |
|-----------|-------------------|----------|-------|
| **Auto** (default) | All 8 classes | 86.1% CV | Combined model — works best on OAP-like material |
| **OAP (Europe)** | Blade, Core, Flake, Biface | **95.3%** | Italian Aurignacian — most comprehensive |
| **COADS (Ohio)** | Biface (projectile points) | 100% | Single-class tradition |
| **Levantine** | Biface (handaxes) | 100% | Single-class tradition |
| **Experimental** | Core, Experimental Core | **94.5%** | Lombao + Morales cores |

When you select a specific tradition, the model **only predicts classes present in that tradition**. This eliminates impossible predictions (like "Blade" for a COADS projectile point) and gives honest accuracy numbers.

### When to use each tradition

| Your material | Select |
|-------------|--------|
| European Upper Palaeolithic (Aurignacian, Gravettian, etc.) | **OAP** |
| North American projectile points | **COADS** |
| Levantine Acheulean handaxes | **Levantine** |
| Experimental cores or retouched pieces | **Experimental** |
| Mixed or uncertain | **Auto** |

---

## 9. 3D Annotation

### Pin annotations

1. Enable **Annotation mode** (Annotation tab or `Ctrl+Shift+A`)
2. Click on the mesh to place a pin
3. Enter a title, description, category, and confidence level

Each annotation stores:
- 3D position (oriented coordinate space)
- Title and description
- Category (for filtering)
- Optional measurement value
- Confidence assessment

### Display modes

| Mode | Behaviour |
|------|-----------|
| Pin + Label | Pins and titles always visible |
| Pin Only | Pins appear on hover |
| Numbered | Numbered markers with a legend panel |

Switch modes from the **Annotation** tab dropdown.

### Photo capture

Right-click an existing annotation → **Capture View** to attach a screenshot of the current 3D viewport state.

### Collaboration

Export annotations as JSON (Annotation tab > Export). Import on another machine — Dibble merges intelligently, detecting conflicts by 3D position (within 1 mm).

---

## 10. AI Lithic Assistant

Query your collection in natural language via a local LLM — no internet required.

### Setup

```bash
pip install llama-cpp-python
```

The Qwen3-4B model (~2.5 GB) downloads automatically on first use.

### Usage

Open the **Assistant** tab and type questions like:

- *"Show me all crested blades with platform angles over 75°"*
- *"What's the average length of scrapers?"*
- *"How many bifaces have a symmetry score above 0.8?"*
- *"Compare the mean thickness of blades vs cores"*

### How it works

1. Your question is sent to the local LLM (Qwen3-4B)
2. The LLM generates a DuckDB SQL query
3. The query runs against the in-memory collection database
4. Results are returned in natural language with optional SQL display

Toggle **Show SQL** to see and verify the generated query before execution.

### Notes

- Fully offline — no data ever leaves your machine
- If SQL generation fails, the assistant automatically retries (up to 3 attempts)
- The assistant works with the currently loaded collection — import artefacts before querying

---

## 11. CLI Reference

### `lithicore info`

Display information about a mesh file:

```bash
lithicore info mesh.ply
# File:     mesh.ply
# Vertices: 150002
# Faces:    300000
# Watertight: True
# Area:     43824.46 mm²
# Volume:   5906.45 mm³
```

### `lithicore batch`

Batch process a directory of meshes:

```bash
lithicore batch <input_directory> \
    --output results.csv \
    --format csv|json \
    --repair/--no-repair \
    --edge-threshold <degrees>
```

### `lithicore figure`

Generate a publication figure:

```bash
lithicore figure <mesh_path> \
    --output figure.svg \
    --no-measurements \
    --no-ridges \
    --label "ARTEFACT-ID"
```

### `lithicore photogrammetry`

Run the photogrammetry pipeline:

```bash
lithicore photogrammetry <photo_folder> \
    --output mesh.ply \
    --label "ARTEFACT-ID" \
    --quality low|medium|high \
    --colmap-matching exhaustive|sequential \
    --dense-quality low|medium|high|extreme \
    --batch \
    --batch-output <output_dir>
```

### `lithicore benchmark`

Run the classifier validation benchmark:

```bash
lithicore benchmark
# Output: docs/benchmark/results/report.html (opens in browser)
```

Generates an interactive HTML report with:
- Confusion matrices for all three typologies
- Per-class precision, recall, and F1 scores
- Cross-validation accuracy estimates
- **Per-tradition accuracy breakdown** (OAP, COADS, Levantine, Experimental)
- Config and metrics JSON files

---

## 12. Training Pipeline

### Data sources

The pre-trained classifiers are trained on **3,415 real-world 3D scan meshes** from five assemblages:

| Source | Origin | Artefacts | Period |
|--------|--------|-----------|--------|
| Open Aurignacian Project (Vols 1–4) | Italy | 2,010 | Upper Palaeolithic |
| Levantine Acheulean Handaxes | Israel/Palestine | 526 | Lower Palaeolithic |
| COADS | Ohio, USA | 492 | Late Prehistoric |
| Lombao Experimental Cores | Spain | 284 | Modern |
| Morales Experimental Retouch | Spain | 100 | Modern |

### Retraining with your own data

```bash
# 1. Compute features for your meshes
python3 lithicore/data/training/_worker.py \
    /path/to/mesh.ply \
    artefact_id \
    typology_label \
    "Dataset Name" \
    metadata.csv

# 2. Retrain all classifiers
python3 lithicore/data/training/retrain.py

# 3. Evaluate
lithicore benchmark
```

### Custom typology training

```python
from lithicore import LithicFeatureVector, train_model

# Prepare your data
feature_vectors = [...]
labels = [...]

# Train
model = train_model(feature_vectors, labels, typology_name="my_typology")

# Save
model.save("./my_typology.joblib")
```

### Data format

The training matrix (`training_matrix.csv`) has these columns:

| Column | Description |
|--------|-------------|
| `artefact_id` | Unique identifier matching the mesh filename |
| `typology` | Typology label |
| `dataset` | Source collection name |
| `source_csv` | Metadata filename |
| `length_mm` ... `surface_roughness` | 22 morphometric features |

### Persistent Homology (PH) features

When retraining, the pipeline automatically augments the feature matrix with **15 PCA-compressed topological features** derived from GUDHI Alpha complex persistence on the mesh surface. These capture micro-topographic information (flake scars, edge notches, surface texture) that the handcrafted morphometrics may miss — contributing approximately **+0.7pp** to classification accuracy.

PH features are cached per-artefact at `~/.cache/dibble/ph_features/`. To precompute them for batch processing:

```bash
python3 lithicore/data/training/batch_ph.py
```

### OOM-safe processing

The training pipeline uses subprocess workers per mesh, with BATCH_SIZE=10 concurrency and a 120-second timeout. Memory is freed between workers.

---

## 13. Configuration and Troubleshooting

### Configuration

| Platform | Path |
|----------|------|
| Linux | `~/.config/dibble/` |
| macOS | `~/Library/Application Support/dibble/` |
| Windows | `%APPDATA%/dibble/` |

The main config file is `settings.toml`:
- Default edge detection threshold
- Photogrammetry quality presets
- AI Assistant model path
- GUI theme and layout
- PH cache directory

### Common issues

| Issue | Solution |
|-------|----------|
| **COLMAP not found** | `apt install colmap` or download from [colmap.github.io](https://colmap.github.io/) |
| **ModuleNotFoundError: lithicore** | `pip install -e lithicore` from the project root |
| **3D viewer blank** | Ensure PyVista/VTK are installed correctly: `pip install pyvista` |
| **Photogrammetry fails** | Check photo format (jpg/png/tiff), minimum 8 photos, consistent lighting |
| **Low classification confidence** | The artefact may be from an under-represented class or tradition. Try the **Tradition** dropdown. |
| **AI Assistant not responding** | Install llama-cpp-python. Check ~2.5 GB free disk for model download on first use. |
| **Benchmark shows 0% accuracy** | Run `python3 lithicore/data/training/retrain.py` first to generate model files |
| **GUDHI import error** | `pip install gudhi` (C++ backend, pre-compiled wheels available) |

### Getting help

- **GitHub Issues:** https://github.com/mabo-du/dibble/issues
- **Documentation:** `docs/` directory for research papers, plans, and specs
- **Research papers:** `docs/research-papers/` — Deep Research papers on accuracy, Persistent Homology, and typology diagnostics

---

## Version history

| Version | Date | Changes |
|---------|------|---------|
| 0.4.0-beta | 2026-06 | Tradition-aware models, PH features, hierarchical cascade, GUI tradition selector, edge-angle fix |
| 0.1.0 | 2026-05 | Initial release — 3D viewer, photogrammetry, classification, AI Assistant |

---

*Built for the archaeological community. Named for Harold Dibble (1951–2018).*
