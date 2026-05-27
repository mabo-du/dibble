# 3D Mesh Annotation — Design Spec

**Date:** 2026-05-27
**Status:** Approved for implementation

## Overview

Add text annotations to any point on a 3D mesh surface, with rich structured data,
attached photos, multi-user JSON collaboration, and three toggleable visual display modes.

## Approach

**Approach 1 (Integrated):** New annotation mode in `Viewer3D` + new side panel in `MainWindow`
+ data layer in `lithicore`. Follows the same pattern as existing landmark/scale/measurement systems.
Chosen over overlay-dialog or plugin approaches for minimal abstraction overhead.

---

## 1. Data Model — `lithicore._annotations`

Pure Python dataclasses in a new module following the `_scale_detection.py` pattern.

### Annotation

```python
@dataclass
class Annotation:
    point: tuple[float, float, float]   # (x, y, z) on mesh surface
    title: str                           # short label
    description: str = ""                # multi-line notes
    category: str = ""                   # e.g. "scar", "ridge", "notch", "cortex"
    measurement_mm: float = 0.0          # optional measurement
    confidence: float = 1.0              # 0–1 scale
    author: str = ""                     # who created it
    timestamp: str = ""                  # ISO datetime string
    attached_photos: list[str] = field(default_factory=list)   # file paths
    sub_annotations: list[Annotation] = field(default_factory=list)
```

Annotations are attached to a **3D world coordinate** (not face/vertex index),
so they survive mesh re-export, decimation, or orientation changes.

### AnnotationSet

```python
@dataclass
class AnnotationSet:
    format_version: int = 1
    artefact_label: str = ""
    mesh_path: str = ""                   # relative path for portability
    mesh_checksum: str = ""               # sha256 of mesh for validation
    author: str = ""
    created: str = ""                     # ISO datetime
    annotations: list[Annotation] = field(default_factory=list)
```

Key methods:
- `to_json() -> str` — serialise to JSON
- `from_json(data: str) -> AnnotationSet` — deserialise
- `merge(other: AnnotationSet) -> AnnotationSet` — multi-user merge

### Merge Logic

| Scenario | Behaviour |
|---|---|
| Same (x,y,z) within 3 decimal places | Merge metadata, prefer newer timestamp |
| Unique position not in existing set | Append as new annotation |
| Conflicting edit at same point | Keep both with `(author)` suffix, flag as conflict warning |
| Photo path not found on receiver | Shown as "(photo not found)" — no crash |

### Merge result

Returns the merged `AnnotationSet` plus a list of warning strings:
```python
merged_set, warnings = base_set.merge(other_set)
```

---

## 2. Viewer Display Modes — `Viewer3D`

Three toggleable visual modes stored as an enum on the viewer:

### A. Pin + Label (default)
- Coloured sphere at annotation point (colour by category)
- Floating text label with title next to sphere
- Same infrastructure as `refresh_landmarks()` + `add_point_labels()`

### B. Pin Only (Show on Hover)
- Coloured sphere at annotation point
- Labels hidden until hover/click
- Uses PyVista's point picking with a smaller pickable radius

### C. Numbered Markers
- Numbered labels (1, 2, 3...) next to each pin
- Legend panel linking numbers to titles
- Same pattern as existing landmark numbering

### Interaction modes

| Mode | Trigger | Behaviour |
|---|---|---|
| Place | Ctrl+Shift+A or [Add] button | Click mesh → create annotation → open edit panel |
| Select | Click existing pin | Highlight pin + scroll panel to entry |
| Focus | Double-click pin or panel entry | Camera flies to annotation point |
| Delete | Right-click pin → "Delete" | Remove annotation from set and viewer |

### Colour palette (per category)

```python
CATEGORY_COLORS = {
    "scar":     (0.9, 0.2, 0.2),   # Red
    "ridge":    (0.2, 0.6, 0.9),   # Blue
    "notch":    (0.2, 0.8, 0.3),   # Green
    "cortex":   (0.9, 0.6, 0.1),   # Orange
    "flake":    (0.6, 0.2, 0.8),   # Purple
    "breakage": (0.9, 0.4, 0.6),   # Pink
    "other":    (0.5, 0.5, 0.5),   # Gray
}
```

---

## 3. Annotation Panel — `lithicope._annotation_panel`

A new `QWidget` that sits in a tabbed widget alongside the Results Panel in the right splitter.

### Layout structure

```
┌────────────────────────┐
│ Annotations       [3]  │  ← header with count
├────────────────────────┤
│ [Add] [Export] [Import]│  ← toolbar
│ [▼ Filter by category] │
├────────────────────────┤
│ ○ 1. Scar on dorsal   │  ← scrollable list
│ ○ 2. Ridge platform   │     (click to select/focus)
│ ○ 3. Cortex remnant   │
├────────────────────────┤
│ ▲ Edit: Scar on dorsal │  ← collapsible edit section
│ Title: [____________]  │
│ Category: [scar ▼]     │
│ Point: (12.3, 4.5, ..) │
│ Meas: [14.2] mm        │
│ Conf: [0.95]           │
│ Desc: [textarea  ]     │
│ Photos: [img.jpg]  [+] │
│ Sub-annotations: [2] + │
│ Author: [mark     ]    │
│ [💾 Save] [🗑 Delete]   │
└────────────────────────┘
```

### Three states

| State | Content |
|---|---|
| **List view** | All annotations listed with coloured dots, truncated previews. Click to select and focus camera. |
| **Edit view** | Expanded form for selected annotation. Save/Delete buttons. |
| **Empty state** | "No annotations yet. Click [Add] or use Ctrl+Shift+A to place annotations on the mesh." |

### Keyboard shortcuts

| Shortcut | Action |
|---|---|
| `Ctrl+Shift+A` | Enter annotation placement mode (click on mesh) |
| `Delete` | Remove selected annotation |
| `Ctrl+S` | Export annotation set to JSON |
| `Ctrl+O` | Import annotation set from JSON |

---

## 4. Photo Capture

When editing an annotation, a **"Capture View"** button takes a screenshot of the
current 3D viewport and attaches it to the annotation.

```python
def _capture_annotation_photo(self) -> None:
    screenshot = self.viewer.plotter.screenshot(return_img=True)  # RGBA ndarray
    photo_path = self._annotation_dir / f"ann_{ann_id}_{timestamp}.png"
    cv2.imwrite(str(photo_path), cv2.cvtColor(screenshot, cv2.COLOR_RGBA2BGR))
    self._current_annotation.attached_photos.append(str(photo_path))
```

Photos appear as a thumbnail strip in the edit panel. Click to open full-size.
External photos can be attached via a [+] button that opens a file dialog.

---

## 5. JSON File Format

Single `.json` file per annotation set:

```json
{
  "format_version": 1,
  "artefact_label": "Flake_42",
  "mesh_path": "Flake_42.ply",
  "mesh_checksum": "sha256:abc123...",
  "author": "mark",
  "created": "2026-05-27T12:00:00",
  "annotations": [
    {
      "point": [12.345, 4.567, 7.890],
      "title": "Scar on dorsal",
      "description": "Prominent bulb scar on the dorsal face near the platform.",
      "category": "scar",
      "measurement_mm": 14.2,
      "confidence": 0.95,
      "author": "mark",
      "timestamp": "2026-05-27T12:05:00",
      "attached_photos": ["ann_1_view.png"],
      "sub_annotations": []
    }
  ]
}
```

Embedded photo screenshots are saved as `.png` files alongside the JSON,
referenced by filename. External photos are stored as absolute-or-relative paths.

---

## 6. Main Window Wiring

### Menu structure (Tools menu → Annotations submenu)

```
Tools
├── ...
├── ─────
├── Annotations
│   ├── Add Annotation          Ctrl+Shift+A
│   ├── Edit Annotations...
│   ├── Import from JSON...     Ctrl+O
│   ├── Export to JSON...       Ctrl+S
│   └── Merge Annotation Set...
├── ─────
├── Compare with Another Mesh...
└── ...
```

### Splitter layout

The right side of the main window gets a `QTabWidget` with two tabs:
- **Results** — existing `ResultsPanel`
- **Annotations** — new `AnnotationPanel`

### Toggle

A **"Show Annotations"** checkbox/button in the viewer toolbar toggles
annotation pin visibility without clearing them.

---

## 7. Files to Create/Modify

| File | Action | Purpose |
|---|---|---|
| `lithicore/src/lithicore/_annotations.py` | **Create** | Data model: Annotation, AnnotationSet, merge logic |
| `lithicore/src/lithicore/__init__.py` | **Modify** | Export new public symbols |
| `lithicore/tests/test_annotations.py` | **Create** | Unit tests for data model |
| `lithicope/src/lithicope/_annotation_panel.py` | **Create** | Annotation panel widget |
| `lithicope/src/lithicope/_viewer_3d.py` | **Modify** | Add annotation display modes, placement, interaction |
| `lithicope/src/lithicope/_main_window.py` | **Modify** | Add annotation menu, tabbed panel, wire signals |
| `lithicore/pyproject.toml` | **No change** | No new dependencies needed |

---

## 8. Testing

### Unit tests — `test_annotations.py`

| Test | Description |
|---|---|
| `test_annotation_defaults` | Default values are correct |
| `test_annotation_custom_values` | Custom values are stored |
| `test_round_trip_json` | to_json → from_json → identical data |
| `test_merge_disjoint` | Two sets with different positions merge cleanly |
| `test_merge_same_point` | Same position merges metadata, prefers newer timestamp |
| `test_merge_conflict` | Same position + same field conflict → both kept with suffix + warning |
| `test_merge_into_empty` | Merging into empty set appends all |
| `test_checksum_mismatch` | Import with wrong checksum → warning |
| `test_empty_annotation_set` | Empty set serialises and deserialises |
| `test_photo_path_handling` | Missing photo path → "(photo not found)" display |

### Integration tests — via pytest-qt

| Test | Description |
|---|---|
| `test_add_annotation` | Click [Add] → enters placement mode |
| `test_annotation_appears_in_list` | After placing → appears in list |
| `test_delete_annotation` | Delete removes from list and viewer |
| `test_edit_title_updates_label` | Edit title → viewer label updates |
| `test_export_import_round_trip` | Export JSON → import JSON → identical |
| `test_viewer_mode_toggle` | Switch between pin/label/numbered modes |
| `test_capture_photo` | Capture view → file created + attached to annotation |
