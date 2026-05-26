# Dibble

**Digital Image-Based Benchmark for Lithic Evaluation**

Dibble is a desktop application for automated 3D lithic (stone tool) morphological analysis. Import a 3D mesh, and Dibble orients it, extracts standardised measurements, generates publication-ready figures, detects flake scars, and places 3D landmarks — no coding required.

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
- **3D landmarks** — click-to-place anatomical landmarks on the mesh, scheme-guided naming (13-point flake, 16-point biface), export to MorphoJ
- **Flake scar detection** — automated scar segmentation via Shape Index curvature analysis and watershed algorithm

### v3 — Photogrammetry (released May 2026)
- **Photogrammetry pipeline** — photos → 3D mesh via COLMAP
- **Three-tier import** — Default (one-click), Guided (settings), Expert (full COLMAP control)
- **Point cloud cleaning** — statistical outlier removal + automatic background cropping
- **Batch queue** — process multiple artefacts sequentially with progress tracking
- **CLI integration** — `lithicore photogrammetry` for headless or batch processing

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
dibble/                    # Measurement library (pure Python, no GUI)
├── pyproject.toml
└── src/dibble/
    ├── _models.py          # Core data types
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
    └── _cli.py             # CLI entry point

dibble-gui/                # Desktop GUI (PyQt6, depends on dibble)
├── pyproject.toml
└── src/dibble_gui/
    ├── main.py             # Application entry point
    ├── _main_window.py     # Main window shell
    ├── _viewer_3d.py       # PyVista 3D viewer (comparison, landmarks, scar overlay)
    ├── _import_dialog.py   # Import mode selection
    ├── _results_panel.py   # Measurement + export panel
    └── _batch_runner.py    # Threaded batch processing
```

## Requirements

- **Python 3.11+** (Python 3.13 supported)
- **PyQt6** — for the GUI
- **PyVista / VTK** — for 3D rendering and figure export
- **trimesh, NumPy, SciPy** — for mesh processing and measurements
- **COLMAP** — for photogrammetry reconstruction (install separately: `brew install colmap`, `sudo apt install colmap`, or `conda install -c conda-forge colmap`)
- **pandas, ReportLab** — for CSV and PDF export

## Configuration

Edge detection threshold, auto-repair behaviour, and landmark schemes are configurable through the GUI import dialog or via code:

```python
from dibble import MeasurementConfig
config = MeasurementConfig(repair_mesh=True, edge_threshold_degrees=45.0)
```

## Project status

v1 and v2 features are complete. v3 (photogrammetry) is released. Active development continues with planned enhancements including scale bar auto-detection, GPU acceleration, and live point cloud preview.

---

*Built for the archaeological community. Open source under the MIT license.*

*Named for Harold Dibble (1951–2018) — pioneering lithic technologist, open-source advocate.*
