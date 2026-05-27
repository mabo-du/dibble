# Dibble v3 — Photogrammetry Pipeline Design

**Date:** 2026-05-26
**Status:** Approved design, pending implementation

## 1. Overview

Add a COLMAP-based photogrammetry pipeline to Dibble: import a folder of photos of a lithic artefact and produce a 3D mesh ready for measurement. The pipeline integrates into the existing `lithicore` / `lithicope` two-package architecture, following the same patterns (dataclass configs, pure functions, zero-GUI-import constraint for lithicore).

**Target users:** Same as Dibble v1/v2 — lithic analysts, archaeologists, graduate students, museum researchers. The v3 GUI is designed so first-time users can get a mesh with one click, while power users have full control over COLMAP parameters.

## 2. Architecture & Module Layout

### 2.1 New files

```
lithicore/src/lithicore/
├── _photogrammetry.py       # Core pipeline — COLMAP orchestration, point cloud cleaning,
│                            #   meshing, decimation. No GUI.
│                            #   Exports: run_pipeline(config, progress_cb) -> PhotogrammetryResult
│                            #           estimate_quality_from_photos(folder) -> int
│                            #           clean_point_cloud(cloud, config) -> np.ndarray
│                            #           PhotogrammetryConfig, PhotogrammetryResult

lithicope/src/lithicope/
├── _photogrammetry_dialog.py    # Multi-page dialog: mode selection → config → progress → result
├── _batch_photogrammetry.py     # Queue manager: submit multiple jobs, sequential processing
```

### 2.2 Existing files to update

| File | Change |
|---|---|
| `lithicore/__init__.py` | Add `PhotogrammetryConfig`, `PhotogrammetryResult`, `run_pipeline` to exports |
| `lithicore/_cli.py` | Add `lithicore photogrammetry` subcommand |
| `lithicore/pyproject.toml` | No new Python dependencies (trimesh + numpy already present) |
| `lithicope/_main_window.py` | Wire menu items — File → New from Photos, Tools → Photogrammetry (Guided/Expert), File → New Batch Photogrammetry |

### 2.3 Design constraint

`_photogrammetry.py` follows the same rule as every other `lithicore` module: **zero GUI imports**. Pure functions taking `PhotogrammetryConfig` and returning `PhotogrammetryResult`. Progress is reported via an optional callback — the CLI writes to console, the GUI updates a progress bar.

## 3. Core Data Types

```python
@dataclass
class PhotogrammetryConfig:
    """Configuration for the photogrammetry pipeline.

    mode determines which config fields are user-settable:
      - default: only photo_folder, output_path, artefact_label, quality
      - guided: plus photo_settings and cleanup fields
      - expert: all fields exposed
    """
    photo_folder: Path
    output_path: Path
    artefact_label: str = ""
    quality: str = "high"               # "low" | "medium" | "high"
    mode: str = "default"               # "default" | "guided" | "expert"

    # Guided/Expert: photo settings
    camera_model: str = "auto"          # "auto" | "smartphone" | "dslr"
    scale_bar_cm: float = 0.0           # 0 = no scale reference

    # Guided/Expert: cleanup
    auto_crop_background: bool = True
    fill_holes: bool = True

    # Expert-only: COLMAP tuning
    colmap_feature_type: str = "sift"
    colmap_matching_strategy: str = "exhaustive"   # "exhaustive" | "sequential" | "vocab_tree"
    colmap_meshing: str = "poisson"                # "poisson" | "delaunay"
    colmap_dense_quality: str = "extreme"          # "low" | "medium" | "high" | "extreme"
    max_vertices: int = 500000

    # Internal
    colmap_workspace: Optional[Path] = None        # temp dir, managed by pipeline
    cleanup_temp: bool = True

    @property
    def target_faces(self) -> int:
        """Map quality slider to target face count after decimation."""
        return {"low": 20_000, "medium": 50_000, "high": 150_000}[self.quality]


@dataclass
class PhotogrammetryResult:
    """Output of a completed photogrammetry pipeline run."""
    mesh_path: Path
    artefact_label: str
    camera_count: int
    point_count: int
    face_count: int
    vertex_count: int
    processing_time_s: float
    colmap_stdout: str                     # Captured for diagnostics
    warnings: list[str]
    sparse_cloud_path: Optional[Path] = None
    dense_cloud_path: Optional[Path] = None
```

## 4. Pipeline Design — `run_pipeline()`

### 4.1 Stage flow

```
photo_folder/*.jpg
    │
    ▼
┌─────────────────────────────────────┐
│ 1. Input Validation                  │  Check formats (jpg/png/tiff), minimum 3 photos
│    (Python)                          │  Estimate resolution, count
└──────────────────┬──────────────────┘
                   ▼
┌─────────────────────────────────────┐
│ 2. COLMAP Feature Extraction        │  colmap feature_extractor
│    (subprocess)                      │  --ImageReader.camera_model SIMPLE_RADIAL
└──────────────────┬──────────────────┘
                   ▼
┌─────────────────────────────────────┐
│ 3. COLMAP Feature Matching          │  colmap exhaustive_matcher (or sequential/vocab_tree)
│    (subprocess)                      │
└──────────────────┬──────────────────┘
                   ▼
┌─────────────────────────────────────┐
│ 4. COLMAP Sparse Reconstruction     │  colmap mapper
│    (subprocess)                      │  Output: sparse point cloud + camera poses
└──────────────────┬──────────────────┘
                   ▼
┌─────────────────────────────────────┐
│ 5. COLMAP Dense MVS                 │  colmap image_undistorter → dense_stereo → fusion
│    (subprocess)                      │  Output: dense point cloud (.ply)
└──────────────────┬──────────────────┘
                   ▼
┌─────────────────────────────────────┐
│ 6. Point Cloud Cleaning             │  NumPy/SciPy statistical outlier removal
│    (Python — lithicore)              │  Automatic background crop
└──────────────────┬──────────────────┘
                   ▼
┌─────────────────────────────────────┐
│ 7. Meshing                          │  colmap Poisson meshing OR trimesh Poisson
│    (subprocess or trimesh)          │
└──────────────────┬──────────────────┘
                   ▼
┌─────────────────────────────────────┐
│ 8. Decimation (if quality < High)   │  trimesh.simplify.simplify_quadric_decimation
│    (trimesh)                         │  Target: PhotogrammetryConfig.target_faces
└──────────────────┬──────────────────┘
                   ▼
┌─────────────────────────────────────┐
│ 9. Output                           │  Write final mesh (.ply)
│    (trimesh)                         │  Clean up temp workspace
└─────────────────────────────────────┘
```

### 4.2 Progress reporting

`run_pipeline(config, progress_cb=None)` accepts an optional callable:

```python
def progress_callback(stage: str, progress: float, message: str):
    # stage: "validation" | "features" | "matching" | "sparse" | "dense" | "cleaning" | "meshing" | "decimation" | "output"
    # progress: 0.0–1.0 within current stage
    # message: human-readable status, e.g. "Matched camera 4/12"
    pass
```

In the CLI, a `rich.progress` spinner displays current stage. In the GUI, the progress dialog updates the stage list and progress bars.

### 4.3 Error handling

Typed exceptions in `lithicore`:
- `PhotogrammetryError(base_exception)` — parent class
- `ColmapNotFoundError` — COLMAP binary not found on PATH
- `ColmapStageError` — a specific COLMAP stage failed (captures stderr)
- `InsufficientPhotosError` — fewer than 3 photos, or matching failed catastrophically
- `PhotogrammetryCancelledError` — user cancelled mid-pipeline

The GUI catches at the event-loop boundary and shows a non-blocking error dialog with diagnostic details.

### 4.4 Temp file management

COLMAP requires a workspace directory with a specific structure (database, sparse/, dense/). The pipeline:
1. Creates a tempdir via `tempfile.mkdtemp(prefix="dibble_photogrammetry_")`
2. Runs all COLMAP stages inside it
3. On success: copies the final mesh to `output_path`, then removes tempdir
4. On failure: keeps tempdir (user can inspect for debugging), reports full paths in error dialog
5. Option in Guided/Expert to "Keep temporary files" for debugging

## 5. Point Cloud Cleaning

Between stages 5 and 7 — the dense cloud from COLMAP is cleaned in Python before meshing.

### 5.1 Statistical outlier removal

For each point in the dense cloud, compute mean distance to k=20 nearest neighbours (using `scipy.spatial.KDTree`). Remove points where mean distance > global_mean + (threshold × stddev). Default threshold: 2.0.

Configurable in Guided mode as a "Noise reduction" slider (Low/Medium/High mapping to thresholds 3.0/2.0/1.0). Invisible in Default mode.

### 5.2 Background cropping

Lithic artefacts are photographed on a turntable or flat surface. The camera rig is typically stationary relative to the artefact. The algorithm:

1. Compute centroid and covariance of dense point cloud
2. Eigen-decomposition → principal axes of the point distribution
3. The artefact is the densest cluster — segment via DBSCAN or percentile bounding box
4. Crop everything outside 1.5× the bounding box of the main cluster

This removes the background surface and distant noise. Configurable margin via "Crop margin" slider in Expert mode.

### 5.3 Scale normalization

If `scale_bar_cm > 0`, search the sparse point cloud for pairs of points at the known scale bar distance (within tolerance). This is a future enhancement — v3 MVP will ship with unsealed meshes (COLMAP's arbitrary scale), with scale bar detection documented as a known limitation.

## 6. GUI Design

### 6.1 Menu wiring

```
File
├── Import Mesh...          (existing → file dialog, OBJ/PLY/STL)
├── New from Photos...      → _photogrammetry_dialog.py in "default" mode
├── Batch Import...         (existing → folder of meshes)
├── New Batch Photogrammetry... → _batch_photogrammetry.py
└── Exit                    (existing)

Tools
├── Publication Figure...   (existing)
├── Comparison Mode...      (existing)
├── Landmark Tool...        (existing)
├── ───────────
└── Photogrammetry
    ├── Guided...           → dialog in "guided" mode
    └── Expert...           → dialog in "expert" mode
```

### 6.2 Photogrammetry Dialog

A single `QDialog` subclass (`PhotogrammetryDialog`) with stacked widget pages:

**Page 1 — Config (varies by mode)**

Default mode — minimal:
```
┌─────────────────────────────────────────┐
│ Photogrammetry — Default                 │
├─────────────────────────────────────────┤
│ Photos folder:    [Browse...] ─────────│
│ Artefact label:   [FLK-145        ]     │
│ Mesh quality:     ● High  ○ Med  ○ Low │
│ Output file:      [Browse...] ─────────│
│                                         │
│              [  Process  ]              │
└─────────────────────────────────────────┘
```

Guided mode — expanded with settings groups:
```
┌─────────────────────────────────────────┐
│ Photogrammetry — Guided                  │
├─────────────────────────────────────────┤
│ Photos folder:    [Browse...] ─────────│
│ Artefact label:   [FLK-145        ]     │
│ Mesh quality:     ● High  ○ Med  ○ Low │
│ Output file:      [Browse...] ─────────│
│                                         │
│ ▼ Photo settings (optional)             │
│   Camera:  ○ Auto-detect  ● Phone  ○ DSLR
│   Scale:   [None, 3cm, 5cm, 10cm ▼]    │
│                                         │
│ ▼ Cleanup                               │
│   ☑ Auto-crop background                │
│   ☑ Fill holes                          │
│   Noise reduction: [Medium ▼]           │
│                                         │
│              [  Process  ]              │
└─────────────────────────────────────────┘
```

Expert mode — full COLMAP control:
```
┌─────────────────────────────────────────┐
│ Photogrammetry — Expert                  │
├─────────────────────────────────────────┤
│ Photos folder:    [Browse...] ─────────│
│ Artefact label:   [FLK-145        ]     │
│ Mesh quality:     ● High  ○ Med  ○ Low │
│ Output file:      [Browse...] ─────────│
│                                         │
│ Feature type:     [SIFT           ▼]    │
│ Matching:         [Exhaustive     ▼]    │
│ Dense quality:    [Extreme        ▼]    │
│ Meshing:          [Poisson        ▼]    │
│ Max vertices:     [500000         ]     │
│                                         │
│ ☐ Keep temporary files                  │
│ Crop margin:      [1.5x        ]        │
│                                         │
│              [  Process  ]              │
└─────────────────────────────────────────┘
```

**Page 2 — Progress (same for all modes)**

```
┌─────────────────────────────────────────┐
│ Processing — 12 photos                   │
├─────────────────────────────────────────┤
│                                         │
│ ● Validating input           [████████] │
│ ● Extracting features        [██████░░] │
│ ○ Matching features          [░░░░░░░░] │
│ ○ Sparse reconstruction      [░░░░░░░░] │
│ ○ Dense reconstruction       [░░░░░░░░] │
│ ○ Cleaning point cloud       [░░░░░░░░] │
│ ○ Meshing                    [░░░░░░░░] │
│                                         │
│  Camera 4/12 matched                    │
│                                         │
│        [  Cancel  ]                     │
└─────────────────────────────────────────┘
```

**Page 3 — Result**

```
┌─────────────────────────────────────────┐
│ Photogrammetry Complete                  │
├─────────────────────────────────────────┤
│ Artefact:    FLK-145                    │
│ Photos:      12                          │
│ Mesh size:   98,432 faces               │
│ Time:        3m 42s                     │
│                                         │
│ Warnings:                               │
│   • 2 photos failed feature extraction  │
│                                         │
│  [Open in Viewer]   [Save Mesh As...]   │
│                                         │
│          ☑ Auto-measure on open         │
└─────────────────────────────────────────┘
```

Clicking **Open in Viewer** loads the mesh into the existing `_viewer_3d.py` and (if auto-measure is checked) runs the full lithicore measurement pipeline immediately.

### 6.3 Batch Queue Dialog

**`_batch_photogrammetry.py`** — `BatchPhotogrammetryDialog`:

```
┌─────────────────────────────────────────────┐
│ Batch Photogrammetry                         │
├─────────────────────────────────────────────┤
│ Artefacts (each sub-folder = one artefact):  │
│                                             │
│ ┌─────────────────────────────────────────┐ │
│ │ □ FLK-001  (12 photos)    ✓ Complete   │ │
│ │ ■ FLK-002  (8 photos)     ████████░░░ │ │
│ │ □ FLK-003  (15 photos)   Queued       │ │
│ │ □ FLK-004  (10 photos)   Queued       │ │
│ └─────────────────────────────────────────┘ │
│                                             │
│ Output folder: [Browse...] ─────────────   │
│ Settings: [Default Preset ▼]               │
│                                             │
│   [Add Artefacts...]    [Start Batch]       │
└─────────────────────────────────────────────┘
```

Each artefact is a sub-folder within a parent folder. The batch runner processes them sequentially, saving results to `output_folder/<label>/<label>.ply`.

## 7. CLI Usage

The pipeline is also accessible via the existing `lithicore` CLI:

```bash
# Default (single artefact, folder of photos)
lithicore photogrammetry ./photos/ --output artefact.ply --label "FLK-145"

# With quality preset
lithicore photogrammetry ./photos/ --output artefact.ply --quality medium

# Expert: pass raw COLMAP options
lithicore photogrammetry ./photos/ --output artefact.ply \
    --colmap-feature-type sift \
    --colmap-matching exhaustive \
    --dense-quality extreme

# Batch mode CLI
lithicore photogrammetry ./artefacts/ --batch --output ./results/
```

The CLI uses the same `run_pipeline()` function, with console-based progress output via `print()` + ANSI spinner.

## 8. Dependencies

### 8.1 COLMAP (external binary)

Not a Python dependency — installed by the user:

| Platform | Command |
|---|---|
| **Ubuntu/Debian** | `sudo apt install colmap` |
| **macOS** | `brew install colmap` |
| **Windows** | Prebuilt binary from colmap.github.io or WSL |
| **Conda** | `conda install -c conda-forge colmap` |

A new `PHOTOGRAMMETRY.md` doc will provide installation instructions and troubleshooting.

### 8.2 Python

No new Python dependencies. `lithicore` already depends on:
- **trimesh** — mesh decimation, Poisson meshing fallback, file I/O
- **numpy** — point cloud array operations, KD-tree construction
- **scipy** — spatial KD-tree for outlier removal, statistical filtering

## 9. COLMAP Integration Details

### 9.1 Subprocess orchestration

COLMAP stages are run as subprocess calls with structured arguments:

```python
import subprocess
import shlex

def _run_colmap_stage(stage_name: str, command_args: list[str],
                      progress_cb, workspace: Path) -> str:
    """Run a single COLMAP stage, capture stdout, report progress."""
    full_cmd = ["colmap"] + command_args
    # Use shlex.join for logging / error messages
    proc = subprocess.run(
        full_cmd,
        capture_output=True,
        text=True,
        cwd=str(workspace),
    )
    if proc.returncode != 0:
        raise ColmapStageError(stage_name, proc.stderr)
    # Parse stdout for progress indicators
    return proc.stdout
```

### 9.2 Stage commands

| Stage | COLMAP command |
|---|---|
| Feature extraction | `colmap feature_extractor --database_path ... --image_path ... --ImageReader.camera_model SIMPLE_RADIAL` |
| Feature matching | `colmap exhaustive_matcher --database_path ...` (or `sequential_matcher`, `vocab_tree_matcher`) |
| Sparse reconstruction | `colmap mapper --database_path ... --image_path ... --output_path ...` |
| Dense MVS | `colmap image_undistorter --input_path ... --output_path ... --output_type COLMAP` + `colmap dense_stereo --workspace_path ... --workspace_format COLMAP --PatchMatchStereo.geom_consistency true` + `colmap stereo_fusion --workspace_path ... --workspace_format COLMAP --output_path fused.ply` |
| Poisson meshing | `colmap poisson_mesher --input_path fused.ply --output_path mesh.ply` |

### 9.3 COLMAP checks

On first launch of any photogrammetry feature, the app checks `shutil.which("colmap")`. If not found, shows a dialog with platform-specific install instructions. The check is cached per session.

## 10. Testing Strategy

### 10.1 Unit tests (lithicore)

- **`test_photogrammetry_config.py`** — dataclass defaults, target_faces property, mode validation
- **`test_clean_point_cloud.py`** — synthetic point clouds with known outliers, verify removal
- **`test_stage_detection.py`** — parse COLMAP stdout for stage detection and progress extraction

### 10.2 Integration tests

- A small (3-5 photo) test fixture bundled in `tests/fixtures/photogrammetry/` — photos of a known-geometry object (3D-printed calibration cube) with ground-truth caliper measurements
- Run full pipeline end-to-end with COLMAP (requires COLMAP installed — skipped if absent)
- Verify output mesh is manifold, watertight, and measurements match ground truth within tolerance (default: ±2mm)

### 10.3 GUI tests (lithicope)

- `pytest-qt` tests for dialog lifecycle (open, fill fields, close)
- Signal-slot tests for progress callback wiring

## 11. Future Enhancements (Post-v3)

- **Scale bar auto-detection** — detect scale reference in photos, rescale mesh to real-world units
- **Multi-GPU support** — COLMAP supports CUDA acceleration for dense MVS
- **Live preview** — intermediate point cloud viewer during processing
- **Cloud processing** — submit photogrammetry jobs to remote server for faster processing
- **Video import** — extract frames from video for quick photo sets
- **Masking** — automatic foreground/background segmentation for cleaner reconstruction

## 12. Known Limitations (v3 MVP)

- COLMAP must be installed separately via system package manager or conda
- No GPU acceleration detection or configuration (COLMAP auto-detects CUDA)
- Scale is arbitrary unless a scale bar reference is provided (future enhancement)
- Very large photo sets (100+) may take 10+ minutes on consumer hardware
- Minimum 3 photos required; fewer may produce unreliable results
- Non-lithic objects (highly reflective, transparent, or textureless surfaces) may fail to reconstruct

---

*Design approved by user on 2026-05-26. Ready for implementation planning.*
