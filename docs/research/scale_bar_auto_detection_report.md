# Scale Bar Auto-Detection — Deep Research Report

## Executive Summary

Four approaches for automatically detecting scale in a COLMAP photogrammetry pipeline, with implementation recommendations for Dibble v3.5.

---

## Approach A: EXIF Sensor Size + COLMAP Intrinsics (C1)

**How it works:** COLMAP already reads EXIF data (`Make`, `Model`, `FocalLength`, `FocalLengthIn35mmFilm`) and includes a built-in camera sensor database (`CameraDatabase`). When the user takes photos with a smartphone or DSLR, EXIF metadata contains the focal length in mm. Combined with the known sensor width (looked up from camera model), the pixel pitch can be computed. COLMAP's self-calibration produces focal length in pixels; dividing by pixel pitch gives physical focal length in mm, establishing absolute scale.

**COLMAP source evidence** (from `src/colmap/sensor/bitmap.cc`):
- `Bitmap::ExifFocalLength()` reads `Exif:FocalLengthIn35mmFilm` or `Exif:FocalLength`
- Falls back to a camera database (`CameraDatabase::QuerySensorWidth`) for known models
- Computes focal length in pixels: `focal_length_mm / sensor_width_mm * max(image_width, image_height)`

**Accuracy:** Moderate (±2-5% typical). Depends on:
- EXIF focal length accuracy (zoom lenses report nominal, not actual)
- Sensor width database coverage (well-covered for popular smartphones/DSLRs)
- COLMAP's self-calibration accuracy (prone to drift with wide baselines)

**Requirements:** Smartphone or DSLR with filled-in EXIF data. Zero user effort.

**Effort:** Very low (2-4 hours). Pipe COLMAP's EXIF-extracted focal length through sensor width lookup, compute scale factor, apply to mesh vertices.

## Approach B: ArUco Fiducial Markers (B1)

**How it works:** A printed ArUco marker of known physical size (e.g., 20×20 mm) is placed next to the artefact before photography. OpenCV's `cv2.aruco` module detects the marker corners in each photo with sub-pixel precision. From COLMAP's known camera poses and the marker's known 3D dimensions, the absolute scale is computed. This is the gold standard for accuracy.

**OpenCV evidence:** `cv2.aruco` is a mature, well-tested module with:
- Adaptive thresholding across multiple window sizes for robust detection
- Perspective correction and Otsu thresholding for bit extraction
- Sub-pixel corner refinement (`CORNER_REFINE_SUBPIX`, `CORNER_REFINE_APRILTAG`)
- Pose estimation via `solvePnP` from 4 corners with known 3D coordinates
- Error correction and false positive rejection

**Accuracy:** Excellent (±0.1-0.5% typical), limited only by:
- Sub-pixel corner detection accuracy
- Precision of printed marker dimensions
- Stability of COLMAP camera poses

**Requirements:** User prints an ArUco marker (PDF included in docs). Place next to artefact. Minimal effort.

**Effort:** Low-medium (1-2 days). Key tasks:
1. Generate ArUco marker PDF for user printing
2. Add `cv2.aruco` detection to the photo scanning step (before COLMAP runs, or post-hoc on the photos)
3. For each photo where the marker is detected, compute the 3D-2D correspondence
4. Use COLMAP's camera pose to transform marker corners into world space
5. Compute scale factor from known physical size vs arbitrary-unit distance

## Approach C: 2D Scale Bar Detection in Photos (A2)

**How it works:** A ruler or printed scale bar with known graduation spacing is photographed next to the artefact. Computer vision techniques (edge detection, Hough lines, template matching) locate the scale bar in 2D photos. The known physical spacing between graduations (e.g., millimetre marks) maps to pixel distances, which combined with COLMAP camera poses gives the 3D scale.

**Sub-approaches:**

**C1. Graduation detection** — Detect tick marks on a standard ruler using vertical edge detection + Hough line transform. Known spacing between ticks (e.g., 1 mm) provides scale in each photo. Requires the scale bar to be roughly parallel to the image plane.

**C2. Pattern matching** — A printed scale bar with a known pattern (alternating black/white bands of known width). Locate via template matching or colour segmentation. More robust than tick detection but requires a custom-printed scale.

**Accuracy:** Moderate (±1-3%). Depends heavily on:
- Blur/motion in photos
- Scale bar angle relative to camera
- Detection robustness in varying lighting

**Requirements:** Standard ruler or printed scale bar. Works retrospectively with existing photo sets if a scale bar is visible.

**Effort:** High (3-5 days). Significant computer vision development for robust detection across varied lighting, angles, and backgrounds. Fragile — likely needs per-scene tuning.

## Approach D: Sparse Point Cloud Measurement (A1)

**How it works:** After COLMAP sparse reconstruction, identify the scale bar in the 3D sparse point cloud. If two points spaced exactly 50 mm apart on the scale bar can be identified in the 3D point cloud, their distance in arbitrary units divided by 50 mm gives the scale factor.

**Detection methods:**
- Colour-based segmentation if the scale bar has a distinct colour
- Geometric filtering (points lying on a thin, flat rectangle near the scene floor)
- Manual identification (user clicks two points on the mesh)

**Accuracy:** Good (±1%). The sparse cloud's relative distances are accurate even if the absolute scale is arbitrary.

**Requirements:** Scale bar visible in enough photos to reconstruct as 3D points. Fully retrospective.

**Effort:** Medium (2-3 days) for automatic detection. Low (few hours) for manual-click approach.

---

## Comparison Table

| Approach | Accuracy | Effort | User Requirements | Retrospective? | Robustness |
|---|---|---|---|---|---|
| **A: EXIF** (C1) | ±2-5% | 2-4 hrs | Smartphone/DSLR with EXIF | Yes (if EXIF present) | Moderate |
| **B: ArUco** (B1) | ±0.1-0.5% | 1-2 days | Print marker PDF | No | Excellent |
| **C: 2D Scale bar** (A2) | ±1-3% | 3-5 days | Ruler in photos | Yes | Low-Moderate |
| **D: Sparse cloud** (A1) | ±1% | 2-3 days | Scale bar in scene | Yes | Moderate |

---

## Recommended Implementation for Dibble v3.5

**Tiered approach** (following the existing Default/Guided/Expert pattern):

### Default mode — EXIF auto-detection (Approach A)
- Zero user effort. Pipeline checks EXIF data from photos.
- If sensor width and focal length are available → compute scale automatically.
- Accuracy ~±3% which is acceptable for most lithic measurements.
- Falls back silently to unscaled if EXIF data is insufficient.

### Expert mode — ArUco markers (Approach B)
- User prints a Dibble-branded ArUco marker PDF (included in app docs).
- Place marker next to the artefact before photography.
- Pipeline detects marker corners in 2D, computes scale from known size.
- Accuracy ~±0.2% — gold standard.
- Also available as a "Guided" option with setup instructions.

### Fallback — Manual scale input (Approach D, manual variant)
- If both automatic methods fail, the result page shows "Scale unknown" with a button:
  "Select two points with known distance"
- User clicks two points on the mesh, enters the real-world distance.
- Pipeline rescales the mesh accordingly.

---

## New Python Dependencies

- `opencv-python` (for `cv2.aruco`) — needed for ArUco detection. Already a common dependency.
- Exif reading: COLMAP handles this internally via its database — no new dep needed.
- Manual fallback: mesh viewer already exists in lithicope — just need a distance measurement tool.

---

## Scale Bar Design Recommendations

For best results regardless of which method is used:

| Parameter | Recommendation |
|---|---|
| **ArUco marker size** | 20×20 mm (small artefact) to 50×50 mm (large) |
| **ArUco dictionary** | `DICT_6X6_250` (good balance of robustness and marker count) |
| **Placement** | Flat on surface, within 2 cm of the artefact, not casting shadow |
| **Visibility** | Visible in ≥ 50% of photos from multiple angles |
| **Standard ruler** | 50 mm visible length minimum, high-contrast (black on white) |

---

## Implementation Plan Summary

| Phase | What | Est. time |
|---|---|---|
| 1 | EXIF scale extraction: read COLMAP's camera model output, compute scale from sensor database | 2-4 hrs |
| 2 | Manual scale fallback: two-click distance tool in the viewer + rescale function | 4-6 hrs |
| 3 | ArUco detection: marker PDF generator, `cv2.aruco` integration in pipeline, scale computation | 1-2 days |
| 4 | Integration: wire into PhotogrammetryConfig, update `scale_bar_cm` field, progress reporting | 4 hrs |
| 5 | Testing: synthetic photo sets, ground-truth mesh comparisons, accuracy benchmarks | 4 hrs |
