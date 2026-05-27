# 3D Mesh Annotation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add rich text annotations with multi-user collaboration to the 3D mesh viewer.

**Architecture:** Data model lives in `lithicore._annotations` (pure Python dataclasses + JSON I/O + merge logic). GUI lives in `lithicope._annotation_panel` (QWidget with list/edit views). Viewer gets annotation display modes built on existing point-picking infrastructure. Main window gets an Annotations tab alongside Results.

**Tech Stack:** Python 3.11+, PyQt6, PyVista/pyvistaqt, trimesh, numpy, pytest

**Spec:** `docs/superpowers/specs/2026-05-27-3d-annotation-design.md`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `lithicore/src/lithicore/_annotations.py` | Create | Annotation/AnnotationSet dataclasses, JSON serialization, merge logic |
| `lithicore/src/lithicore/__init__.py` | Modify | Export Annotation, AnnotationSet, merge |
| `lithicore/tests/test_annotations.py` | Create | Unit tests for data model |
| `lithicope/src/lithicope/_annotation_panel.py` | Create | QWidget: list view, edit form, import/export, photo capture |
| `lithicope/src/lithicope/_viewer_3d.py` | Modify | Add annotation display modes, placement, interaction |
| `lithicope/src/lithicope/_main_window.py` | Modify | Add Annotations tab, menu items, signal wiring |

---

### Task 1: Data Model — `_annotations.py`

**Files:**
- Create: `lithicore/src/lithicore/_annotations.py`
- Test: `lithicore/tests/test_annotations.py`

**Step 1: Write the failing test for Annotation defaults**

```python
"""test_annotations.py — Unit tests for annotation data model.

exports: TestAnnotation, TestAnnotationSet
used_by: pytest
rules:   Pure dataclass tests, no GUI dependencies.
agent:   initial | 2026-05-27 | Test skeleton
"""

import json
from pathlib import Path

import pytest

from lithicore._annotations import Annotation, AnnotationSet


class TestAnnotation:
    """Annotation dataclass construction."""

    def test_annotation_defaults(self):
        ann = Annotation(
            point=(1.0, 2.0, 3.0),
            title="Test annotation",
        )
        assert ann.point == (1.0, 2.0, 3.0)
        assert ann.title == "Test annotation"
        assert ann.description == ""
        assert ann.category == ""
        assert ann.measurement_mm == 0.0
        assert ann.confidence == 1.0
        assert ann.author == ""
        assert ann.timestamp == ""
        assert ann.attached_photos == []
        assert ann.sub_annotations == []

    def test_annotation_custom_values(self):
        sub = Annotation(point=(4.0, 5.0, 6.0), title="sub")
        ann = Annotation(
            point=(1.0, 2.0, 3.0),
            title="Main",
            description="A test annotation",
            category="scar",
            measurement_mm=14.2,
            confidence=0.95,
            author="mark",
            timestamp="2026-05-27T12:00:00",
            attached_photos=["photo1.png"],
            sub_annotations=[sub],
        )
        assert ann.category == "scar"
        assert ann.measurement_mm == 14.2
        assert ann.confidence == 0.95
        assert len(ann.sub_annotations) == 1

    def test_annotation_point_tuple(self):
        ann = Annotation(point=(1.234, 5.678, 9.012), title="p")
        x, y, z = ann.point
        assert isinstance(x, float)
        assert isinstance(y, float)
        assert isinstance(z, float)
```

- [ ] Write the tests above to `lithicore/tests/test_annotations.py`
- [ ] Run to verify failure: `pytest lithicore/tests/test_annotations.py::TestAnnotation -v` → FAIL
- [ ] Create `lithicore/src/lithicore/_annotations.py`:

```python
"""_annotations.py — 3D mesh annotation data model.

exports: Annotation
         AnnotationSet
used_by: lithicope annotation panel
rules:   Pure dataclasses with JSON serialization. No GUI imports.
         Coordinates are (x, y, z) floats matching mesh vertex space.
agent:   deepseek-v4-flash | 2026-05-27 | Initial implementation
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class Annotation:
    """A single annotation attached to a 3D mesh point.

    Attributes:
        point: (x, y, z) coordinates on the mesh surface.
        title: Short label for the annotation.
        description: Multi-line descriptive notes.
        category: Type classification — e.g. "scar", "ridge",
            "notch", "cortex", "flake", "breakage", "other".
        measurement_mm: Optional numeric measurement at this point.
        confidence: Estimated reliability (0 = uncertain, 1 = certain).
        author: Name or identifier of the annotator.
        timestamp: ISO 8601 datetime string of creation/last edit.
        attached_photos: List of file paths to associated images.
        sub_annotations: Child annotations nested under this one.
    """
    point: tuple[float, float, float]
    title: str
    description: str = ""
    category: str = ""
    measurement_mm: float = 0.0
    confidence: float = 1.0
    author: str = ""
    timestamp: str = ""
    attached_photos: list[str] = field(default_factory=list)
    sub_annotations: list["Annotation"] = field(default_factory=list)


@dataclass
class AnnotationSet:
    """A collection of annotations for a single artefact mesh.

    Attributes:
        format_version: Schema version for forward compatibility.
        artefact_label: Human-readable artefact identifier.
        mesh_path: Relative or absolute path to the associated mesh file.
        mesh_checksum: SHA-256 hex digest of the mesh file for validation.
        author: Name of the person who created this set.
        created: ISO 8601 datetime of initial creation.
        annotations: All top-level annotations for this artefact.
    """
    format_version: int = 1
    artefact_label: str = ""
    mesh_path: str = ""
    mesh_checksum: str = ""
    author: str = ""
    created: str = ""
    annotations: list[Annotation] = field(default_factory=list)

    def to_json(self) -> str:
        """Serialize this annotation set to a JSON string."""
        data = asdict(self)
        return json.dumps(data, indent=2, default=str)

    @classmethod
    def from_json(cls, data: str) -> AnnotationSet:
        """Deserialize a JSON string into an AnnotationSet."""
        raw = json.loads(data)
        # Reconstruct nested Annotation objects
        raw["annotations"] = [
            cls._annotation_from_dict(a) for a in raw.get("annotations", [])
        ]
        return cls(**raw)

    @staticmethod
    def _annotation_from_dict(raw: dict) -> Annotation:
        """Recursively build an Annotation from a dict, handling sub-annotations."""
        subs = raw.pop("sub_annotations", [])
        ann = Annotation(**raw)
        ann.sub_annotations = [
            AnnotationSet._annotation_from_dict(s) for s in subs
        ]
        return ann

    @staticmethod
    def _point_key(point: tuple[float, float, float]) -> tuple[float, float, float]:
        """Round a 3D point to 3 decimal places for stable merge matching."""
        return (round(point[0], 3), round(point[1], 3), round(point[2], 3))

    def merge(self, other: AnnotationSet) -> tuple[AnnotationSet, list[str]]:
        """Merge another AnnotationSet into this one.

        Annotations at the same 3D position (rounded to 3 dp) are merged.
        Unique positions are appended. Conflicts (same position, different
        data) keep both entries with an author suffix and a warning.

        Args:
            other: The incoming annotation set to merge.

        Returns:
            A tuple of (merged AnnotationSet, list of warning strings).
        """
        warnings: list[str] = []
        merged = AnnotationSet(
            format_version=self.format_version,
            artefact_label=self.artefact_label or other.artefact_label,
            mesh_path=self.mesh_path or other.mesh_path,
            mesh_checksum=self.mesh_checksum or other.mesh_checksum,
            author=f"{self.author}+{other.author}" if self.author and other.author
                   else self.author or other.author,
            created=self.created or other.created,
        )

        # Index existing annotations by position key
        existing: dict[tuple[float, float, float], Annotation] = {}
        for ann in self.annotations:
            existing[self._point_key(ann.point)] = ann

        # Add all current annotations
        merged.annotations = list(self.annotations)

        for ann in other.annotations:
            key = self._point_key(ann.point)
            if key in existing:
                existing_ann = existing[key]
                # Conflict detection: same point, different title/desc
                if (existing_ann.title != ann.title or
                        existing_ann.description != ann.description):
                    suffix = f" ({ann.author})" if ann.author else " (imported)"
                    merged_ann = Annotation(
                        point=ann.point,
                        title=ann.title + suffix,
                        description=ann.description,
                        category=ann.category or existing_ann.category,
                        measurement_mm=ann.measurement_mm or existing_ann.measurement_mm,
                        confidence=ann.confidence or existing_ann.confidence,
                        author=ann.author or existing_ann.author,
                        timestamp=max(ann.timestamp, existing_ann.timestamp)
                        if ann.timestamp and existing_ann.timestamp
                        else ann.timestamp or existing_ann.timestamp,
                        attached_photos=list(set(existing_ann.attached_photos + ann.attached_photos)),
                    )
                    # Replace in-place
                    for i, e in enumerate(merged.annotations):
                        if self._point_key(e.point) == key:
                            merged.annotations[i] = merged_ann
                            break
                    warnings.append(
                        f"Merged annotation at ({ann.point[0]:.3f}, {ann.point[1]:.3f}, "
                        f"{ann.point[2]:.3f}): conflicting data resolved with author suffix"
                    )
                else:
                    # Same content — prefer newer timestamp
                    if ann.timestamp and (not existing_ann.timestamp or
                                          ann.timestamp > existing_ann.timestamp):
                        for i, e in enumerate(merged.annotations):
                            if self._point_key(e.point) == key:
                                e.timestamp = ann.timestamp
                                e.author = ann.author or e.author
                                break
            else:
                merged.annotations.append(ann)

        return merged, warnings

    def compute_checksum(self, mesh_path: Path) -> str:
        """Compute SHA-256 hex digest of a mesh file."""
        sha = hashlib.sha256()
        with open(mesh_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha.update(chunk)
        return f"sha256:{sha.hexdigest()}"
```

- [ ] Run tests: `pytest lithicore/tests/test_annotations.py::TestAnnotation -v` → PASS
- [ ] Commit:

```bash
git add lithicore/tests/test_annotations.py lithicore/src/lithicore/_annotations.py
git commit -m "feat: add annotation data model (Annotation, AnnotationSet, JSON serialization)"
```

---

### Task 2: AnnotationSet — JSON round-trip + merge tests

**Files:**
- Modify: `lithicore/tests/test_annotations.py`

- [ ] Add these tests to `test_annotations.py`:

```python
class TestAnnotationSet:
    """AnnotationSet JSON round-trip and merge."""

    def test_round_trip_json(self):
        ann = Annotation(point=(1.0, 2.0, 3.0), title="Test")
        ann_set = AnnotationSet(
            artefact_label="Flake_42",
            author="mark",
            created="2026-05-27T12:00:00",
            annotations=[ann],
        )
        json_str = ann_set.to_json()
        restored = AnnotationSet.from_json(json_str)
        assert restored.artefact_label == "Flake_42"
        assert len(restored.annotations) == 1
        assert restored.annotations[0].point == (1.0, 2.0, 3.0)
        assert restored.annotations[0].title == "Test"

    def test_merge_disjoint_positions(self):
        a1 = Annotation(point=(1.0, 2.0, 3.0), title="A")
        a2 = Annotation(point=(4.0, 5.0, 6.0), title="B")
        base = AnnotationSet(artefact_label="test", annotations=[a1])
        incoming = AnnotationSet(artefact_label="test", annotations=[a2])
        merged, warnings = base.merge(incoming)
        assert len(merged.annotations) == 2
        assert warnings == []

    def test_merge_same_position(self):
        a1 = Annotation(point=(1.0, 2.0, 3.0), title="Scar A",
                        author="mark", timestamp="2026-05-27T12:00:00")
        a2 = Annotation(point=(1.0, 2.0, 3.0), title="Scar A",
                        author="anna", timestamp="2026-05-28T12:00:00")
        base = AnnotationSet(artefact_label="test", annotations=[a1])
        incoming = AnnotationSet(artefact_label="test", annotations=[a2])
        merged, warnings = base.merge(incoming)
        assert len(merged.annotations) == 1
        # Should keep the newer timestamp
        assert merged.annotations[0].timestamp == "2026-05-28T12:00:00"

    def test_merge_conflict(self):
        a1 = Annotation(point=(1.0, 2.0, 3.0), title="Scar A")
        a2 = Annotation(point=(1.0, 2.0, 3.0), title="Scar B (different opinion)",
                        author="anna")
        base = AnnotationSet(artefact_label="test", annotations=[a1])
        incoming = AnnotationSet(artefact_label="test", annotations=[a2])
        merged, warnings = base.merge(incoming)
        assert len(merged.annotations) == 1
        assert len(warnings) >= 1
        assert "(anna)" in merged.annotations[0].title or "(imported)" in merged.annotations[0].title

    def test_merge_into_empty(self):
        ann = Annotation(point=(1.0, 2.0, 3.0), title="Only")
        empty = AnnotationSet()
        incoming = AnnotationSet(annotations=[ann])
        merged, warnings = empty.merge(incoming)
        assert len(merged.annotations) == 1
        assert warnings == []

    def test_round_trip_empty_set(self):
        s = AnnotationSet()
        restored = AnnotationSet.from_json(s.to_json())
        assert restored.annotations == []

    def test_round_trip_with_sub_annotations(self):
        sub = Annotation(point=(4.0, 5.0, 6.0), title="Sub")
        main = Annotation(point=(1.0, 2.0, 3.0), title="Main",
                          sub_annotations=[sub])
        s = AnnotationSet(annotations=[main])
        restored = AnnotationSet.from_json(s.to_json())
        assert len(restored.annotations) == 1
        assert len(restored.annotations[0].sub_annotations) == 1
        assert restored.annotations[0].sub_annotations[0].title == "Sub"
```

- [ ] Run: `pytest lithicore/tests/test_annotations.py -v` → PASS
- [ ] Commit:

```bash
git add lithicore/tests/test_annotations.py lithicore/src/lithicore/_annotations.py
git commit -m "feat: AnnotationSet JSON round-trip and merge logic"
```

---

### Task 3: Wire into `lithicore.__init__`

**Files:**
- Modify: `lithicore/src/lithicore/__init__.py`

- [ ] Add import and export:

```python
    from lithicore._annotations import (
        Annotation,
        AnnotationSet,
    )

    __all__ = [
        # ... existing entries ...
        "Annotation", "AnnotationSet",
        # ... rest of existing ...
    ]
```

- [ ] Verify import works: `python -c "from lithicore import Annotation, AnnotationSet"`
- [ ] Commit:

```bash
git add lithicore/src/lithicore/__init__.py
git commit -m "feat: export Annotation and AnnotationSet from lithicore"
```

---

### Task 4: Viewer Display Modes — `Viewer3D`

**Files:**
- Modify: `lithicope/src/lithicope/_viewer_3d.py`

- [ ] Add annotation state and display methods to `Viewer3D` class:

Add to `__init__` after existing instance variables:

```python
        # Annotation state
        self._annotation_mode_enabled: bool = False
        self._annotation_display_mode: str = "pin_label"  # pin_label | pin_only | numbered
        self._annotation_actors: list = []
        self._annotation_place_callback: Optional[callable] = None
        self._annotation_select_callback: Optional[callable] = None
        self._annotations_data: list = []  # list of Annotation objects for display
```

Add these methods:

```python
    # ── Annotation display ──────────────────────────────────

    def set_annotation_display_mode(self, mode: str) -> None:
        """Set annotation visual style: 'pin_label', 'pin_only', or 'numbered'."""
        self._annotation_display_mode = mode
        self.refresh_annotations()

    def refresh_annotations(self, annotations: Optional[list] = None) -> None:
        """Refresh displayed annotation pins/labels from a list of Annotation objects."""
        if not HAS_PYVISTAQT or self._pv_mesh is None:
            return

        if annotations is not None:
            self._annotations_data = annotations

        # Remove old annotation actors
        for actor in self._annotation_actors:
            self.plotter.remove_actor(actor, render=False)
        self._annotation_actors.clear()

        if not self._annotations_data:
            self.plotter.render()
            return

        # Category colour palette
        CATEGORY_COLORS = {
            "scar":     (0.9, 0.2, 0.2),
            "ridge":    (0.2, 0.6, 0.9),
            "notch":    (0.2, 0.8, 0.3),
            "cortex":   (0.9, 0.6, 0.1),
            "flake":    (0.6, 0.2, 0.8),
            "breakage": (0.9, 0.4, 0.6),
        }
        default_color = (0.5, 0.5, 0.5)

        for i, ann in enumerate(self._annotations_data):
            pt = ann.point
            colour = CATEGORY_COLORS.get(ann.category.lower(), default_color)
            sphere_radius = max(0.5, self._pv_mesh.length * 0.008)

            sphere = pv.Sphere(radius=sphere_radius, center=[pt[0], pt[1], pt[2]])
            actor = self.plotter.add_mesh(
                sphere, color=colour, smooth_shading=True, opacity=0.9,
            )
            self._annotation_actors.append(actor)

            if self._annotation_display_mode == "pin_label":
                # Always show label
                label_actor = self.plotter.add_point_labels(
                    np.array([[pt[0], pt[1], pt[2]]]),
                    [ann.title],
                    point_size=0.01,
                    font_size=10,
                    text_color="black",
                    shape="rect",
                    fill_shape=False,
                )
                self._annotation_actors.append(label_actor)

            elif self._annotation_display_mode == "numbered":
                # Numbered labels
                label_actor = self.plotter.add_point_labels(
                    np.array([[pt[0], pt[1], pt[2]]]),
                    [f"{i + 1}"],
                    point_size=0.01,
                    font_size=12,
                    text_color="black",
                    shape="rect",
                    fill_shape=False,
                )
                self._annotation_actors.append(label_actor)

            # Pin-only mode: no labels (shown on hover via picking)

        self.plotter.render()

    def clear_annotations(self) -> None:
        """Remove all annotation overlays."""
        for actor in self._annotation_actors:
            self.plotter.remove_actor(actor, render=False)
        self._annotation_actors.clear()
        self._annotations_data = []
        self.plotter.render()

    def enable_annotation_placement_mode(self, callback: callable) -> None:
        """Enter click-to-place annotation mode.

        The callback receives (x, y, z) tuple of the clicked mesh point.
        """
        if not HAS_PYVISTAQT or self._pv_mesh is None:
            return

        self._annotation_mode_enabled = True
        self._annotation_place_callback = callback

        def _on_pick(picked_point):
            if picked_point is None:
                return
            if self._annotation_place_callback:
                self._annotation_place_callback(
                    float(picked_point[0]),
                    float(picked_point[1]),
                    float(picked_point[2]),
                )

        self.plotter.enable_point_picking(
            callback=_on_pick,
            show_message="Click on the mesh to place an annotation",
            use_mesh=True,
            pickable=True,
        )

    def disable_annotation_placement_mode(self) -> None:
        """Exit annotation placement mode."""
        if not HAS_PYVISTAQT:
            return
        self.plotter.disable_picking()
        self._annotation_mode_enabled = False
        self._annotation_place_callback = None

    def focus_on_point(self, point: tuple[float, float, float]) -> None:
        """Move camera to focus on a specific 3D point."""
        if not HAS_PYVISTAQT:
            return
        self.plotter.camera_position = [
            [point[0], point[1], point[2] + self._pv_mesh.length * 0.5],
            [point[0], point[1], point[2]],
            [0, 0, 1],
        ]
        self.plotter.reset_camera()
        self.plotter.render()
```

- [ ] Run existing viewer tests (if any) — verify nothing broken:

```bash
python -m pytest lithicope/tests/ -v
```

- [ ] Commit:

```bash
git add lithicope/src/lithicope/_viewer_3d.py
git commit -m "feat: add annotation display modes and placement to Viewer3D"
```

---

### Task 5: Annotation Panel Widget — `_annotation_panel.py`

**Files:**
- Create: `lithicope/src/lithicope/_annotation_panel.py`

This is the largest new file. Full implementation:

```python
"""_annotation_panel.py — Side panel for managing 3D mesh annotations.

exports: AnnotationPanel(QWidget)
used_by: MainWindow right-side tab widget
rules:   No direct lithicore imports; operates on Annotation/AnnotationSet objects.
         All file I/O is user-initiated (import/export buttons, never auto-save).
agent:   deepseek-v4-flash | 2026-05-27 | Initial implementation
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from lithicore._annotations import Annotation, AnnotationSet


CATEGORY_COLORS = {
    "scar":     "#e63333",
    "ridge":    "#3399e6",
    "notch":    "#33cc44",
    "cortex":   "#e69933",
    "flake":    "#9933cc",
    "breakage": "#e66699",
    "other":    "#888888",
}


class AnnotationPanel(QWidget):
    """Side panel for viewing, editing, and managing mesh annotations."""

    annotation_selected = pyqtSignal(object)   # Annotation
    annotation_added = pyqtSignal(object)       # Annotation
    annotation_deleted = pyqtSignal(object)     # Annotation
    focus_requested = pyqtSignal(tuple)         # (x, y, z)
    placement_mode_requested = pyqtSignal()
    capture_photo_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._annotations: list[Annotation] = []
        self._selected_index: Optional[int] = None
        self._annotation_dir: Path = Path()
        self._current_set: Optional[AnnotationSet] = None

        self._build_ui()
        self._update_empty_state()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Header
        header_row = QHBoxLayout()
        self._count_label = QLabel("Annotations")
        self._count_label.setStyleSheet("font-size: 13pt; font-weight: bold;")
        header_row.addWidget(self._count_label)
        self._count_badge = QLabel("0")
        self._count_badge.setStyleSheet(
            "background-color: #4472C4; color: white; border-radius: 8px; "
            "padding: 2px 8px; font-weight: bold;"
        )
        header_row.addWidget(self._count_badge)
        header_row.addStretch()

        # Display mode selector
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Pin + Label", "Pin Only", "Numbered"])
        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)
        header_row.addWidget(QLabel("Show:"))
        header_row.addWidget(self._mode_combo)

        layout.addLayout(header_row)

        # Toolbar
        toolbar = QHBoxLayout()
        self._add_btn = QPushButton("+ Add")
        self._add_btn.clicked.connect(self._on_add_clicked)
        toolbar.addWidget(self._add_btn)
        self._export_btn = QPushButton("Export")
        self._export_btn.clicked.connect(self._on_export)
        toolbar.addWidget(self._export_btn)
        self._import_btn = QPushButton("Import")
        self._import_btn.clicked.connect(self._on_import)
        toolbar.addWidget(self._import_btn)
        self._merge_btn = QPushButton("Merge")
        self._merge_btn.clicked.connect(self._on_merge)
        toolbar.addWidget(self._merge_btn)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        # Category filter
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filter:"))
        self._filter_combo = QComboBox()
        self._filter_combo.addItems(["All", "scar", "ridge", "notch", "cortex", "flake", "breakage", "other"])
        self._filter_combo.currentTextChanged.connect(self._on_filter_changed)
        filter_row.addWidget(self._filter_combo)
        filter_row.addStretch()
        layout.addLayout(filter_row)

        # Annotation list
        self._list_widget = QListWidget()
        self._list_widget.currentRowChanged.connect(self._on_list_selection_changed)
        self._list_widget.setAlternatingRowColors(True)
        layout.addWidget(self._list_widget, stretch=2)

        # Edit form
        self._edit_group = QGroupBox("Edit Annotation")
        edit_form = QFormLayout(self._edit_group)

        self._edit_title = QLineEdit()
        edit_form.addRow("Title:", self._edit_title)

        self._edit_category = QComboBox()
        self._edit_category.addItems(["", "scar", "ridge", "notch", "cortex", "flake", "breakage", "other"])
        edit_form.addRow("Category:", self._edit_category)

        self._edit_measurement = QDoubleSpinBox()
        self._edit_measurement.setRange(0.0, 99999.0)
        self._edit_measurement.setDecimals(2)
        self._edit_measurement.setSuffix(" mm")
        edit_form.addRow("Measurement:", self._edit_measurement)

        self._edit_confidence = QDoubleSpinBox()
        self._edit_confidence.setRange(0.0, 1.0)
        self._edit_confidence.setDecimals(2)
        self._edit_confidence.setSingleStep(0.05)
        edit_form.addRow("Confidence:", self._edit_confidence)

        self._edit_author = QLineEdit()
        edit_form.addRow("Author:", self._edit_author)

        self._edit_description = QTextEdit()
        self._edit_description.setMaximumHeight(80)
        edit_form.addRow("Description:", self._edit_description)

        # Photo area
        photo_row = QHBoxLayout()
        self._photo_label = QLabel("No photos")
        self._photo_label.setStyleSheet("color: #888; font-style: italic;")
        photo_row.addWidget(self._photo_label)
        self._capture_btn = QPushButton("Capture View")
        self._capture_btn.clicked.connect(self._on_capture_photo)
        photo_row.addWidget(self._capture_btn)
        self._attach_btn = QPushButton("+ Attach")
        self._attach_btn.clicked.connect(self._on_attach_photo)
        photo_row.addWidget(self._attach_btn)
        edit_form.addRow("Photos:", photo_row)

        # Point display
        self._point_label = QLabel("")
        self._point_label.setStyleSheet("color: #666; font-family: monospace; font-size: 9pt;")
        edit_form.addRow("Point:", self._point_label)

        # Edit buttons
        edit_btn_row = QHBoxLayout()
        self._save_btn = QPushButton("Save")
        self._save_btn.clicked.connect(self._on_save_edit)
        edit_btn_row.addWidget(self._save_btn)
        self._delete_btn = QPushButton("Delete")
        self._delete_btn.clicked.connect(self._on_delete_annotation)
        self._delete_btn.setStyleSheet("color: #cc3333;")
        edit_btn_row.addWidget(self._delete_btn)
        self._focus_btn = QPushButton("Focus")
        self._focus_btn.clicked.connect(self._on_focus_annotation)
        edit_btn_row.addWidget(self._focus_btn)
        edit_btn_row.addStretch()
        edit_form.addRow(edit_btn_row)

        layout.addWidget(self._edit_group, stretch=1)

        self._edit_group.setVisible(False)

    # ── Public API ──

    def set_annotations(self, annotations: list[Annotation]) -> None:
        """Replace all annotations and refresh the display."""
        self._annotations = list(annotations)
        self._refresh_list()

    def get_annotations(self) -> list[Annotation]:
        """Return the current list of annotations."""
        return list(self._annotations)

    def add_annotation(self, point: tuple[float, float, float]) -> None:
        """Add a new annotation at a 3D point and select it for editing."""
        ann = Annotation(
            point=point,
            title=f"Annotation {len(self._annotations) + 1}",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._annotations.append(ann)
        self._refresh_list()
        # Select the new annotation
        self._list_widget.setCurrentRow(len(self._annotations) - 1)
        self.annotation_added.emit(ann)

    def set_annotation_dir(self, directory: Path) -> None:
        """Set the working directory for photo files."""
        self._annotation_dir = directory

    # ── Display mode ──

    def get_display_mode(self) -> str:
        """Get current display mode string for the viewer."""
        mapping = {"Pin + Label": "pin_label", "Pin Only": "pin_only", "Numbered": "numbered"}
        return mapping.get(self._mode_combo.currentText(), "pin_label")

    # ── Internal helpers ──

    def _refresh_list(self) -> None:
        self._list_widget.blockSignals(True)
        self._list_widget.clear()

        filter_text = self._filter_combo.currentText().lower()
        for i, ann in enumerate(self._annotations):
            if filter_text != "all" and ann.category.lower() != filter_text:
                continue
            colour = CATEGORY_COLORS.get(ann.category.lower(), "#888888")
            preview = ann.description[:60] + "..." if len(ann.description) > 60 else ann.description
            display_text = f"{ann.title}"
            if preview:
                display_text += f"\n  {preview}"
            item = QListWidgetItem(display_text)
            item.setData(Qt.ItemDataRole.UserRole, i)  # original index
            self._list_widget.addItem(item)

        self._list_widget.blockSignals(False)
        self._count_badge.setText(str(len(self._annotations)))
        self._update_empty_state()

    def _update_empty_state(self) -> None:
        has_items = len(self._annotations) > 0
        self._export_btn.setEnabled(has_items)
        self._edit_group.setVisible(self._selected_index is not None)
        self._list_widget.setVisible(has_items or self._filter_combo.currentText() != "All")

    def _get_filtered_index(self, list_row: int) -> Optional[int]:
        """Map list widget row to original annotations list index."""
        item = self._list_widget.item(list_row)
        if item is None:
            return None
        return item.data(Qt.ItemDataRole.UserRole)

    # ── Slots ──

    def _on_mode_changed(self, text: str) -> None:
        # Signal to viewer — handled by MainWindow
        pass

    def _on_filter_changed(self, text: str) -> None:
        self._refresh_list()

    def _on_add_clicked(self) -> None:
        self.placement_mode_requested.emit()

    def _on_list_selection_changed(self, row: int) -> None:
        idx = self._get_filtered_index(row)
        if idx is None:
            self._selected_index = None
            self._edit_group.setVisible(False)
            return

        self._selected_index = idx
        ann = self._annotations[idx]

        # Populate edit form
        self._edit_title.setText(ann.title)
        cat_idx = self._edit_category.findText(ann.category)
        self._edit_category.setCurrentIndex(max(0, cat_idx))
        self._edit_measurement.setValue(ann.measurement_mm)
        self._edit_confidence.setValue(ann.confidence)
        self._edit_author.setText(ann.author)
        self._edit_description.setPlainText(ann.description)
        self._point_label.setText(f"({ann.point[0]:.3f}, {ann.point[1]:.3f}, {ann.point[2]:.3f})")

        # Photo display
        if ann.attached_photos:
            self._photo_label.setText(f"{len(ann.attached_photos)} photo(s)")
        else:
            self._photo_label.setText("No photos")

        self._edit_group.setVisible(True)
        self.annotation_selected.emit(ann)
        self.focus_requested.emit(ann.point)

    def _on_save_edit(self) -> None:
        if self._selected_index is None:
            return
        ann = self._annotations[self._selected_index]
        ann.title = self._edit_title.text() or ann.title
        ann.category = self._edit_category.currentText()
        ann.measurement_mm = self._edit_measurement.value()
        ann.confidence = self._edit_confidence.value()
        ann.author = self._edit_author.text()
        ann.description = self._edit_description.toPlainText()
        ann.timestamp = datetime.now(timezone.utc).isoformat()
        self._refresh_list()

    def _on_delete_annotation(self) -> None:
        if self._selected_index is None:
            return
        ann = self._annotations.pop(self._selected_index)
        self._selected_index = None
        self._refresh_list()
        self.annotation_deleted.emit(ann)

    def _on_focus_annotation(self) -> None:
        if self._selected_index is None:
            return
        ann = self._annotations[self._selected_index]
        self.focus_requested.emit(ann.point)

    def _on_capture_photo(self) -> None:
        self.capture_photo_requested.emit()

    def _on_attach_photo(self) -> None:
        if self._selected_index is None:
            return
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Attach Photo", "",
            "Images (*.png *.jpg *.jpeg *.tiff *.tif)",
        )
        if not path_str:
            return
        ann = self._annotations[self._selected_index]
        ann.attached_photos.append(path_str)
        self._photo_label.setText(f"{len(ann.attached_photos)} photo(s)")

    def _on_export(self) -> None:
        if not self._annotations:
            return
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Export Annotations", "",
            "JSON (*.json)",
        )
        if not path_str:
            return
        ann_set = AnnotationSet(
            artefact_label=os.path.basename(str(self._annotation_dir)),
            author=self._edit_author.text() or "unknown",
            created=datetime.now(timezone.utc).isoformat(),
            annotations=self._annotations,
        )
        with open(path_str, "w") as f:
            f.write(ann_set.to_json())

    def _on_import(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Import Annotations", "",
            "JSON (*.json)",
        )
        if not path_str:
            return
        with open(path_str) as f:
            data = f.read()
        try:
            ann_set = AnnotationSet.from_json(data)
        except Exception as exc:
            QMessageBox.warning(self, "Import Error", f"Failed to parse: {exc}")
            return

        self._annotations = ann_set.annotations
        self._refresh_list()
        self.annotation_added.emit(None)  # bulk update

    def _on_merge(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Merge Annotation Set", "",
            "JSON (*.json)",
        )
        if not path_str:
            return
        with open(path_str) as f:
            data = f.read()
        try:
            incoming = AnnotationSet.from_json(data)
        except Exception as exc:
            QMessageBox.warning(self, "Merge Error", f"Failed to parse: {exc}")
            return

        current_set = AnnotationSet(
            artefact_label=os.path.basename(str(self._annotation_dir)),
            annotations=self._annotations,
        )
        merged, warnings = current_set.merge(incoming)
        self._annotations = merged.annotations
        self._refresh_list()

        if warnings:
            QMessageBox.information(
                self, "Merge Complete",
                f"Merged {len(incoming.annotations)} annotations.\n"
                + "\n".join(warnings[:5]),
            )

    def add_captured_photo(self, photo_path: str) -> None:
        """Attach a captured screenshot to the currently selected annotation."""
        if self._selected_index is None:
            return
        ann = self._annotations[self._selected_index]
        ann.attached_photos.append(photo_path)
        self._photo_label.setText(f"{len(ann.attached_photos)} photo(s)")
```

- [ ] Create the file above
- [ ] No unit tests needed for pure GUI — tested via integration with main window

---

### Task 6: Main Window Wiring

**Files:**
- Modify: `lithicope/src/lithicope/_main_window.py`

Changes needed:

1. Import `AnnotationPanel` near the top
2. Add `AnnotationPanel` as a tab alongside `ResultsPanel` in the right splitter
3. Add "Annotations" menu to the Tools menu
4. Wire signals between viewer and annotation panel
5. Handle photo capture integration
6. Set annotation dir when a mesh is loaded

- [ ] **Add import:**

```python
from lithicope._annotation_panel import AnnotationPanel
```

- [ ] **Add annotation panel** to `_init_ui()` after `self.results_panel`:

```python
        # Right: tabbed results + annotations
        self._right_tabs = QTabWidget()
        self._right_tabs.addTab(self.results_panel, "Results")
        self._annotation_panel = AnnotationPanel()
        self._right_tabs.addTab(self._annotation_panel, "Annotations")
        splitter.addWidget(self._right_tabs)
```

Replace `splitter.addWidget(self.results_panel)` with `splitter.addWidget(self._right_tabs)`.

- [ ] **Wire signals** after viewer and annotation panel init:

```python
        self._annotation_panel.focus_requested.connect(self.viewer.focus_on_point)
        self._annotation_panel.placement_mode_requested.connect(self._on_annotation_placement)
        self._annotation_panel.capture_photo_requested.connect(self._on_annotation_capture_photo)
```

- [ ] **Add annotation mode management:**

```python
    def _on_annotation_placement(self) -> None:
        """Enter annotation placement mode."""
        self.viewer.enable_annotation_placement_mode(
            callback=self._on_annotation_placed
        )
        self.status.showMessage("Click on the mesh to place an annotation")

    def _on_annotation_placed(self, x: float, y: float, z: float) -> None:
        """Called when user clicks a point on the mesh in placement mode."""
        self.viewer.disable_annotation_placement_mode()
        self._annotation_panel.add_annotation((x, y, z))
        self._sync_annotation_viewer()
        self.status.showMessage(f"Annotation placed at ({x:.1f}, {y:.1f}, {z:.1f})")

    def _sync_annotation_viewer(self) -> None:
        """Sync annotations from panel to viewer."""
        annotations = self._annotation_panel.get_annotations()
        mode = self._annotation_panel.get_display_mode()
        self.viewer._annotation_display_mode = mode
        self.viewer.refresh_annotations(annotations)
```

- [ ] **Connect display mode changes** (in `_init_ui` after annotation panel):

```python
        self._annotation_panel.annotation_added.connect(
            lambda _: self._sync_annotation_viewer()
        )
        self._annotation_panel.annotation_deleted.connect(
            lambda _: self._sync_annotation_viewer()
        )
```

- [ ] **Set annotation dir** when mesh is loaded (in `_process_single`):

```python
        if hasattr(self, '_annotation_panel'):
            self._annotation_panel.set_annotation_dir(mesh_path.parent)
```

- [ ] **Add menu items** in `_init_menu` → Tools menu:

```python
        # Annotations submenu
        tools_menu.addSeparator()
        ann_menu = tools_menu.addMenu("&Annotations")
        add_ann_action = QAction("&Add Annotation", self)
        add_ann_action.setShortcut("Ctrl+Shift+A")
        add_ann_action.triggered.connect(self._on_annotation_placement)
        ann_menu.addAction(add_ann_action)
        import_ann_action = QAction("&Import from JSON...", self)
        import_ann_action.triggered.connect(self._annotation_panel._on_import)
        ann_menu.addAction(import_ann_action)
        export_ann_action = QAction("&Export to JSON...", self)
        export_ann_action.setShortcut("Ctrl+S")
        export_ann_action.triggered.connect(self._annotation_panel._on_export)
        ann_menu.addAction(export_ann_action)
        merge_ann_action = QAction("&Merge Annotation Set...", self)
        merge_ann_action.triggered.connect(self._annotation_panel._on_merge)
        ann_menu.addAction(merge_ann_action)
```

- [ ] **Add photo capture handler:**

```python
    def _on_annotation_capture_photo(self) -> None:
        """Capture the current 3D view and attach to the selected annotation."""
        if not hasattr(self.viewer, 'plotter'):
            return
        screenshot = self.viewer.plotter.screenshot(return_img=True)
        # Save to annotation directory
        ann_dir = getattr(self._annotation_panel, '_annotation_dir', Path.home())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        photo_path = ann_dir / f"annotation_capture_{timestamp}.png"
        # Convert RGBA to BGR for cv2
        cv2.imwrite(str(photo_path), cv2.cvtColor(screenshot, cv2.COLOR_RGBA2BGR))
        self._annotation_panel.add_captured_photo(str(photo_path))
        self.status.showMessage(f"Photo captured: {photo_path.name}")
```

- [ ] Run tests: `python -m pytest lithicore/tests/ lithicope/tests/ -v` → all pass
- [ ] Commit:

```bash
git add lithicope/src/lithicope/_main_window.py lithicope/src/lithicope/_annotation_panel.py
git commit -m "feat: wire annotation panel and viewer into main window"
```

---

## Self-Review Checklist

- [ ] **Spec coverage:** Every spec section has a corresponding task:
  - Section 1 (Data model) → Task 1, 2
  - Section 2 (Viewer display modes) → Task 4
  - Section 3 (Annotation panel) → Task 5
  - Section 4 (Multi-user merge) → Task 2 (merge logic in `_annotations.py`), Task 5 (merge button in panel)
  - Section 5 (JSON format) → Task 2 (serialization round-trip)
  - Section 6 (Main window wiring) → Task 6
  - Section 7 (Photo capture) → Task 5, 6
  - Section 8 (Testing) → Task 1, 2, 6

- [ ] **No placeholders:** All code blocks are complete implementations, not TODOs.
- [ ] **Type consistency:** `Annotation.point` is `tuple[float, float, float]` everywhere. `AnnotationSet.merge()` returns `tuple[AnnotationSet, list[str]]` consistent across spec and plan.
- [ ] **Testing strategy:** Unit tests for data model, integration for GUI.
