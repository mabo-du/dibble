# Dibble User Guide

> **Version:** 0.4.0-beta
> **Last updated:** 2026-06-03

Dibble is a desktop application for automated 3D lithic (stone tool) analysis —
from photos to classified artefact, fully offline. This guide covers installation,
the GUI workflow, the CLI tools, and how to train custom classifiers.

---

## Table of Contents

1. [Installation](#1-installation)
2. [Quick Start](#2-quick-start)
3. [The 3D Viewer](#3-the-3d-viewer)
4. [Importing and Processing Meshes](#4-importing-and-processing-meshes)
5. [Photogrammetry Pipeline](#5-photogrammetry-pipeline)
6. [Publications Figures](#6-publication-figures)
7. [Lithic Classification](#7-lithic-classification)
8. [3D Annotation](#8-3d-annotation)
9. [AI Lithic Assistant](#9-ai-lithic-assistant)
10. [CLI Reference](#10-cli-reference)
11. [Training Pipeline](#11-training-pipeline)
12. [Configuration and Troubleshooting](#12-configuration-and-troubleshooting)

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
# Clone the repository
git clone https://github.com/mabo-du/dibble.git
cd dibble

# Install the core library
pip install -e lithicore

# Install the GUI (optional — needed for desktop app)
pip install -e lithicope

# Verify installation
lithicore --help
lithicope --help
```

### Optional dependencies

| Dependency | Purpose | Install |
|------------|---------|---------|
| **COLMAP** | Photogrammetry reconstruction | `apt install colmap` or [colmap.github.io](https://colmap.github.io/) |
| **llama-cpp-python** | Local AI Assistant | `pip install llama-cpp-python` |
| **skl2onnx** | ONNX model export | `pip install skl2onnx onnxruntime` |

> **Note:** The AI Assistant model (~2.5 GB) downloads automatically on first use.

### Verify installation

```bash
# CLI — process a mesh
lithicore info ./test_mesh.ply

# GUI — launch the desktop application
lithicope

# Classifier — run the validation benchmark
lithicore benchmark
```

---

## 2. Quick Start

### Launch the GUI

```bash
lithicope
```

The main window opens with a 3D viewport, tool panels on the right, and a menu bar.

### Load a mesh

1. Click **File > Open** or press `Ctrl+O`
2. Select a `.ply`, `.obj`, or `.stl` file
3. The mesh appears in the 3D viewport

### Run classification

1. Click the **Classification** tab in the right panel (or press `Ctrl+Shift+C`)
2. Select a typology system (Basic, Bordes, or Technological)
3. Click **Classify Artefact**
4. The predicted type, confidence score, and top diagnostic features are shown

### Run photogrammetry

1. Click **Tools > Photogrammetry** or press `Ctrl+P`
2. Select a folder containing photos
3. Choose **Default** mode for one-click reconstruction
4. The pipeline runs: photos → sparse model → dense model → cleaned mesh → PLY output

---

## 3. The 3D Viewer

The viewer is built on PyVista/VTK and provides interactive 3D visualisation.

### Controls

| Action | Mouse / Keyboard |
|--------|-----------------|
| Rotate | Left-click + drag |
| Pan | Middle-click + drag |
| Zoom | Scroll wheel |
| Reset view | `R` key |
| Toggle axes | `A` key |
| Toggle wireframe | `W` key |
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

When classification is run with "Show diagnostic overlays" enabled, the viewer
highlights three feature types:

| Feature | Colour | Meaning |
|---------|--------|---------|
| Dorsal ridges | Blue | Linear ridges from previous flake removals |
| Platform | Green | Striking platform surface |
| Retouched edges | Red | Areas of secondary modification |

### Measurement display

The **Results** tab shows computed measurements:
- Length, width, thickness (mm)
- Surface area (mm²) and volume (mm³)
- Elongation, flatness, compactness ratios
- Platform angles (EPA, IPA)
- Edge angle statistics
- Scar count and mean scar area

---

## 4. Importing and Processing Meshes

### Supported formats

| Format | Extension | Notes |
|--------|-----------|-------|
| Polygon File Format | `.ply` | Preferred — best compatibility |
| Wavefront OBJ | `.obj` | Supported, may need companion `.mtl` |
| Stereolithography | `.stl` | Supported |
| Virtual Reality ML | `.wrl`, `.vrml` | Supported — may be slower |

### Orientation

All measurements are computed in **oriented coordinate space**:

- **Z-axis**: Maximum length (reduction axis for flakes)
- **Y-axis**: Maximum width
- **X-axis**: Thickness

Two orientation methods are available:

**Automatic (PCA-based):**
```bash
lithicore batch ./meshes/ --output results.csv
```
The auto-orientation aligns the first principal component with the Z-axis with
platform detection.

**Manual:**
In the GUI, click **Tools > Orient Manually** and click three points on the mesh
to define the platform plane. This is useful for irregular artefacts where PCA
fails.

### Mesh validation

Dibble automatically checks for:
- Non-manifold geometry
- Inverted normals
- Degenerate faces (zero-area triangles)
- Isolated components (floating vertices)

**Auto-repair** can fix inverted normals (`mesh.fix_normals()`) and remove
degenerate faces.

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
# CSV output
lithicore batch ./excavation_meshes/ --output analysis.csv

# JSON output
lithicore batch ./excavation_meshes/ --output analysis.json --format json

# With custom edge detection threshold
lithicore batch ./excavation_meshes/ --output results.csv --edge-threshold 45
```

Output includes one row per artefact with all measurements, typology labels,
and any processing warnings.

---

## 5. Photogrammetry Pipeline

Dibble wraps COLMAP into a 9-stage photogrammetry pipeline: photos → 3D mesh.

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
In the GUI photogrammetry dialog, select **Guided** to adjust:
- Quality: Low / Medium / High
- Expected artefact size

**Expert mode — full COLMAP control:**
```bash
lithicore photogrammetry ./photos/ --output artefact.ply \
    --quality high \
    --colmap-feature-type sift \
    --colmap-matching exhaustive \
    --dense-quality extreme
```

### Photo pre-processing

Before reconstruction, photos are automatically processed:
- **Blur detection**: Images with Laplacian variance below threshold are flagged
- **CLAHE normalisation**: Contrast Limited Adaptive Histogram Equalisation for
  consistent lighting across photo sets
- **Auto-resize**: Large images are resized for optimal COLMAP performance

### Scale detection

Three methods for determining real-world scale:

| Method | Accuracy | How it works |
|--------|----------|-------------|
| ArUco marker | ±0.1% | Detects printed marker of known size in photos |
| Ruler/scale bar | ±1% | Hough line detection + tick mark frequency |
| Manual | User-defined | Click two points on the mesh, enter known distance |

### Batch queue

Process multiple artefacts sequentially:
```bash
# Each sub-folder is one artefact
lithicore photogrammetry ./excavation_photos/ --batch --batch-output ./results/
```

Live progress tracking with per-artefact preview. Outputs one PLY per artefact.

---

## 6. Publication Figures

Generate standardised three-view technical drawings suitable for publication.

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
# Hide measurement annotations
lithicore figure artefact.ply --output figure.svg --no-measurements

# Hide scar ridge lines
lithicore figure artefact.ply --output figure.svg --no-ridges

# Custom artefact label
lithicore figure artefact.ply --output figure.svg --label "Unit 4, Level 3"
```

Figures are exported as SVG via VTK GL2PS for lossless scaling in publications.

---

## 7. Lithic Classification

### Typology systems

Dibble ships with three pre-trained typology systems:

| System | Classes | CV Accuracy | Description |
|--------|---------|-------------|-------------|
| Basic | 9 | 84.8% | Broad morphological categories |
| Bordes | 9 | 84.8% | Same morphology-based mapping |
| Technological | 8 | 73.6% | Reduction stages |

### Running classification

**In the GUI:**
1. Open the **Classification** tab (right panel)
2. Select a typology from the dropdown
3. Click **Classify Artefact**
4. View the result: predicted type, confidence, top features, alternatives

**Via CLI:**
```bash
# Batch classify all meshes in a directory
lithicore batch ./meshes/ --output classified.csv
# The output includes typology predictions alongside measurements
```

### Understanding predictions

The classification result shows:
- **Predicted type**: The most likely class
- **Confidence**: Probability (0–1) assigned to the predicted class
- **Feature breakdown**: Which measurements drove the decision, their values,
  and whether each falls within the expected range for the predicted type
- **Alternatives**: Other classes with non-trivial probability (>1%)

A confidence of <0.6 indicates the model is uncertain — expert verification
is recommended.

### Active learning

When the classifier makes a mistake:
1. Click **Submit Correction** in the Classification panel
2. Select the correct label from the dropdown
3. The correction is queued for retraining
4. After 10 corrections, the model retrains automatically

This allows the classifier to adapt to your specific assemblage over time.

### Custom typologies

Define and train your own typology system:

1. Prepare a CSV with artefact IDs and your labels
2. Run the training pipeline on your labelled meshes:
   ```bash
   python3 lithicore/data/training/retrain.py
   ```
3. The custom model appears in the GUI's "Custom" typology option
4. Share the `.joblib` model file with colleagues

### Model export (ONNX)

Trained models can be exported to ONNX format for secure, version-independent
deployment:

```python
from lithicore import ClassifierModel
model = ClassifierModel.load_pre_trained("basic")
model.export_onnx("./my_model.onnx")
```

ONNX models can be loaded without the full sklearn dependency chain.

---

## 8. 3D Annotation

### Pin annotations

Attach notes to any point on the mesh surface:
1. Enable **Annotation mode** (Annotation tab or `Ctrl+Shift+A`)
2. Click on the mesh to place a pin
3. Enter a title, description, category, and confidence level

Each annotation stores:
- 3D position (in oriented coordinate space)
- Title and description
- Category (for filtering)
- Measurement value (optional)
- Confidence assessment

### Display modes

| Mode | Behaviour |
|------|-----------|
| Pin + Label | Annotation pins and titles always visible |
| Pin Only | Pins appear only on hover |
| Numbered | Numbered markers with a legend panel |

Switch modes from the **Annotation** tab dropdown.

### Photo capture

Attach a screenshot of the current 3D view to any annotation:
1. Right-click an existing annotation
2. Select **Capture View**
3. The current viewport state is saved and attached

### Collaboration

Export annotations as JSON:
```bash
# Export all annotations
# (In GUI: Annotation tab > Export)
```

Import on another machine:
- Dibble merges intelligently, detecting conflicts by 3D position
- If two annotations are within 1 mm of each other, they're flagged as potential
  duplicates

---

## 9. AI Lithic Assistant

The AI Assistant lets you query your collection in natural language.

### Setup

```bash
pip install llama-cpp-python
```

The Qwen3-4B model (~2.5 GB) downloads automatically on first use.

### Usage

Open the **Assistant** tab and type questions like:

- "Show me all crested blades with platform angles over 75°"
- "What's the average length of scrapers?"
- "How many bifaces have a symmetry score above 0.8?"
- "Compare the mean thickness of blades vs bladelets"

### How it works

1. Your question is sent to the local LLM (Qwen3-4B)
2. The LLM generates a DuckDB SQL query
3. The query runs against the in-memory collection database
4. Results are returned in natural language with optional SQL display

You can toggle **Show SQL** to see and verify the generated query before execution.

### Notes

- Fully offline — no data ever leaves your machine
- If the generated SQL fails, the assistant automatically fixes it and retries
  (up to 3 attempts)
- The assistant works with the currently loaded collection — artefacts must be
  imported before querying

---

## 10. CLI Reference

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
    --colmap-feature-type sift \
    --colmap-matching exhaustive|sequential \
    --dense-quality low|medium|high|extreme \
    --batch \
    --batch-output <output_dir>
```

### `lithicore benchmark`

Run the classifier validation benchmark:
```bash
lithicore benchmark
# Output: docs/benchmark/results/report.html
```

Generates an interactive HTML report with:
- Confusion matrices for all three typologies
- Per-class precision, recall, and F1 scores
- Cross-validation accuracy estimates
- Config and metrics JSON files

---

## 11. Training Pipeline

### Data sources

The pre-trained classifiers are trained on **3,415 real-world 3D scan meshes**
from five continents:

| Source | Origin | Artefacts |
|--------|--------|-----------|
| Open Aurignacian Project | Italy | 2,418 |
| Levantine Acheulean Handaxes | Israel/Palestine | 526 |
| COADS | Ohio, USA | 514 |
| Lombao Experimental Cores | Spain | 254 |
| Morales Experimental Retouch | Spain | 100 |

### Retraining with your own data

1. **Prepare training data**: Run `_worker.py` on each mesh to extract the
   22-dimensional feature vector and append to `training_matrix.csv`:
   ```bash
   python3 lithicore/data/training/_worker.py \
       /path/to/mesh.ply \
       artefact_id \
       typology_label \
       "Dataset Name" \
       metadata.csv
   ```

2. **Retrain all classifiers**:
   ```bash
   python3 lithicore/data/training/retrain.py
   ```

3. **Run the benchmark** to verify accuracy:
   ```bash
   lithicore benchmark
   ```

### Adding new data sources

To integrate a new collection:

1. Download 3D meshes (PLY, STL, or OBJ format)
2. Create a metadata CSV with artefact IDs and typological classifications
3. Place files in `/data/dibble-training/raw/`
4. Run the retrain script

The `get_labels()` function in `retrain.py` maps metadata to typology labels.
You may need to extend it for your classification scheme.

### Custom typology training

For a completely custom classification system:

```python
from lithicore import LithicFeatureVector, train_model

# Prepare your feature vectors and labels
feature_vectors = [...]
labels = [...]

# Train a new model
model = train_model(feature_vectors, labels, typology_name="my_typology")

# Save and use
model.save("./my_typology.joblib")
```

### Data format

The training matrix is a CSV with these columns:

| Column | Description |
|--------|-------------|
| `artefact_id` | Unique identifier matching the mesh filename |
| `typology` | Typology label (e.g., "Biface", "Blade", "Core") |
| `dataset` | Source collection name |
| `source_csv` | Metadata filename |
| `length_mm` ... `surface_roughness` | 22 morphometric features |

### OOM-safe processing

The training pipeline uses subprocess workers to prevent memory accumulation:
- Each mesh is processed in a separate Python process
- Memory is fully freed when the process exits
- BATCH_SIZE (default 10) controls concurrency
- Workers have a 120-second timeout per mesh

---

## 12. Configuration and Troubleshooting

### Configuration

Dibble stores its configuration at:
- **Linux**: `~/.config/dibble/`
- **macOS**: `~/Library/Application Support/dibble/`
- **Windows**: `%APPDATA%/dibble/`

The main config file is `settings.toml` and includes:
- Default edge detection threshold
- Photogrammetry quality presets
- AI Assistant model path
- GUI theme and layout

### Common issues

| Issue | Solution |
|-------|----------|
| **COLMAP not found** | Install COLMAP separately (`apt install colmap`) |
| **ImportError: no module named 'lithicore'** | Run `pip install -e lithicore` from the project root |
| **3D viewer blank** | Ensure PyVista/VTK are installed correctly |
| **Photogrammetry fails** | Check photo format (jpg/png/tiff), minimum 8 photos |
| **Low classification confidence** | Artefact may be from an under-represented class — expert verification recommended |
| **AI Assistant not responding** | Install llama-cpp-python, check ~2.5 GB free disk for model download |
| **Benchmark shows 0% accuracy** | Run `python3 lithicore/data/training/retrain.py` first to generate models |

### Getting help

- **GitHub Issues**: https://github.com/mabo-du/dibble/issues
- **Documentation**: See `docs/` directory for research papers, plans, and specs
- **Training data sources**: `docs/research-papers/training-data-sources.md`

---

## Version history

| Version | Date | Changes |
|---------|------|---------|
| 0.4.0-beta | 2026-06 | Edge-angle fix, class imbalance docs, benchmark rewrite |
| 0.1.0 | 2026-05 | Initial release with full feature set |

---

*Built for the archaeological community. Named for Harold Dibble (1951–2018).*
