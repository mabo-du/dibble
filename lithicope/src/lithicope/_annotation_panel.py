"""_annotation_panel.py — Side panel for managing 3D mesh annotations.

exports: AnnotationPanel(QWidget)
used_by: MainWindow right-side tab widget
rules:   No direct lithicore imports; operates on Annotation/AnnotationSet objects.
         All file I/O is user-initiated (import/export buttons, never auto-save).
agent:   deepseek-v4-flash | 2026-05-27 | Initial implementation
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
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
