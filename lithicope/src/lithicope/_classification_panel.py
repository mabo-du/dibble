"""_classification_panel.py — Side panel for lithic typology classification.

exports: ClassificationPanel(QWidget)
used_by: MainWindow right-side tab widget
rules:   Operates on ClassificationResult objects. No direct sklearn imports.
agent:   deepseek-v4-flash | 2026-05-27 | Initial implementation
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from lithicore import (
    ClassificationResult,
    ClassifierModel,
    LithicFeatureVector,
    extract_features,
    extract_diagnostic_coordinates,
)

TRADITION_ITEMS: list[tuple[str, str | None]] = [
    ("Auto", None),
    ("OAP (Europe)", "OAP"),
    ("COADS (Ohio)", "COADS"),
    ("Levantine (Israel)", "Levantine"),
    ("Experimental (Cores)", "Experimental"),
]


class ClassificationPanel(QWidget):
    """Panel for running and displaying lithic typology classification."""

    classification_computed = pyqtSignal(object)  # ClassificationResult
    diagnostic_overlay_requested = pyqtSignal(dict)  # coordinate dict
    auto_classify_changed = pyqtSignal(bool)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._current_result: Optional[ClassificationResult] = None
        self._current_mesh = None
        self._models: dict[str, ClassifierModel] = {}
        self._tradition_models: dict[str, ClassifierModel] = {}
        self._correction_timer = QTimer()
        self._correction_timer.setInterval(500)
        self._correction_timer.setSingleShot(True)
        self._correction_timer.timeout.connect(self._on_debounced_retrain)
        self._build_ui()
        self._load_models()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Header
        header = QLabel("Lithic Typology Classification")
        header.setStyleSheet("font-size: 13pt; font-weight: bold;")
        layout.addWidget(header)

        # Typology selector
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Typology:"))
        self._typology_combo = QComboBox()
        self._typology_combo.addItems(
            ["Basic Morphological", "Bordes Typology", "Technological", "Custom"]
        )
        self._typology_combo.currentTextChanged.connect(self._on_typology_changed)
        type_row.addWidget(self._typology_combo)
        type_row.addStretch()
        layout.addLayout(type_row)

        # Tradition selector
        trad_row = QHBoxLayout()
        trad_row.addWidget(QLabel("Tradition:"))
        self._tradition_combo = QComboBox()
        for display, _data in TRADITION_ITEMS:
            self._tradition_combo.addItem(display)
        self._tradition_combo.currentTextChanged.connect(self._on_tradition_changed)
        trad_row.addWidget(self._tradition_combo)
        trad_row.addStretch()
        layout.addLayout(trad_row)

        # Auto-classify toggle
        self._auto_check = QCheckBox("Auto-classify on load")
        self._auto_check.setChecked(False)
        self._auto_check.toggled.connect(self.auto_classify_changed.emit)
        layout.addWidget(self._auto_check)

        # Predict button
        self._classify_btn = QPushButton("Classify Artefact")
        self._classify_btn.setStyleSheet(
            "font-size: 12pt; padding: 8px;"
        )
        self._classify_btn.clicked.connect(self._on_classify)
        layout.addWidget(self._classify_btn)

        # Result card
        self._result_group = QGroupBox("Classification Result")
        result_layout = QVBoxLayout(self._result_group)

        self._label_display = QLabel("")
        self._label_display.setStyleSheet("font-size: 16pt; font-weight: bold;")
        result_layout.addWidget(self._label_display)

        self._confidence_display = QLabel("")
        result_layout.addWidget(self._confidence_display)

        self._features_list = QListWidget()
        self._features_list.setMaximumHeight(150)
        result_layout.addWidget(self._features_list)

        self._alternatives_label = QLabel("")
        self._alternatives_label.setStyleSheet("color: #666;")
        result_layout.addWidget(self._alternatives_label)

        layout.addWidget(self._result_group)
        self._result_group.setVisible(False)

        # Correction area
        correct_row = QHBoxLayout()
        correct_row.addWidget(QLabel("Correct?"))
        self._correct_combo = QComboBox()
        correct_row.addWidget(self._correct_combo)
        self._correct_btn = QPushButton("Submit Correction")
        self._correct_btn.clicked.connect(self._on_correct)
        correct_row.addWidget(self._correct_btn)
        correct_row.addStretch()
        layout.addLayout(correct_row)

        # Overlay toggle
        self._overlay_check = QCheckBox("Show diagnostic overlays on mesh")
        self._overlay_check.setChecked(True)
        layout.addWidget(self._overlay_check)

        # Save / Load custom model
        model_row = QHBoxLayout()
        self._save_model_btn = QPushButton("Save Custom Model")
        self._save_model_btn.clicked.connect(self._on_save_model)
        model_row.addWidget(self._save_model_btn)
        self._load_model_btn = QPushButton("Load Custom Model")
        self._load_model_btn.clicked.connect(self._on_load_model)
        model_row.addWidget(self._load_model_btn)
        layout.addLayout(model_row)

        layout.addStretch()

    def _load_models(self) -> None:
        """Load pre-trained models and tradition routers on startup."""
        for name in ["basic", "bordes", "technological"]:
            try:
                self._models[name] = ClassifierModel.load_pre_trained(name)
            except FileNotFoundError:
                pass
            # Load tradition router for this typology (pre-load with OAP)
            try:
                self._tradition_models[name] = ClassifierModel.load_pre_trained(
                    name, tradition="OAP"
                )
            except FileNotFoundError:
                pass

    def set_mesh(self, mesh) -> None:
        """Set the current mesh for classification."""
        self._current_mesh = mesh
        if self._auto_check.isChecked():
            self._on_classify()

    # ── Typology selection ──

    def _on_typology_changed(self, text: str) -> None:
        """Handle typology dropdown change. Re-classify if auto mode."""
        self._result_group.setVisible(False)
        self._populate_correction_combo()
        if self._auto_check.isChecked() and self._current_mesh is not None:
            self._on_classify()

    def _on_tradition_changed(self, text: str) -> None:
        """Handle tradition dropdown change. Re-classify if auto mode."""
        self._populate_correction_combo()
        if self._auto_check.isChecked() and self._current_mesh is not None:
            self._on_classify()

    def _populate_correction_combo(self) -> None:
        """Fill correction combo with classes from the current model / tradition."""
        self._correct_combo.clear()
        tradition_val = self._get_tradition_value()
        if tradition_val is not None:
            # Use tradition router's classes for the selected tradition
            router_model = self._get_tradition_router()
            if router_model is not None and router_model._router is not None:
                classes = router_model._router.tradition_classes.get(
                    tradition_val, []
                )
                for cls_name in sorted(classes):
                    self._correct_combo.addItem(cls_name)
                return
        model = self._get_current_model()
        if model is not None and model.is_loaded():
            for cls_name in sorted(model._classes):
                self._correct_combo.addItem(cls_name)

    def _get_current_model(self) -> Optional[ClassifierModel]:
        """Get the model for the currently selected typology."""
        typology_map = {
            "Basic Morphological": "basic",
            "Bordes Typology": "bordes",
            "Technological": "technological",
            "Custom": "custom",
        }
        key = typology_map.get(self._typology_combo.currentText(), "basic")
        return self._models.get(key)

    def _get_tradition_value(self) -> Optional[str]:
        """Get the internal tradition identifier from the combo, or None for Auto."""
        for display, data in TRADITION_ITEMS:
            if display == self._tradition_combo.currentText():
                return data
        return None

    def _get_tradition_router(self) -> Optional[ClassifierModel]:
        """Get the tradition-router model for the current typology, if loaded."""
        typology_map = {
            "Basic Morphological": "basic",
            "Bordes Typology": "bordes",
            "Technological": "technological",
        }
        key = typology_map.get(self._typology_combo.currentText())
        if key is None:
            return None
        return self._tradition_models.get(key)

    # ── Classification ──

    def _on_classify(self) -> None:
        """Run classification on the current mesh."""
        if self._current_mesh is None:
            self._label_display.setText("No mesh loaded — open a mesh first")
            self._result_group.setVisible(True)
            return

        tradition_val = self._get_tradition_value()

        if tradition_val is not None:
            # Use tradition-router model
            model = self._get_tradition_router()
            if model is None or model._router is None:
                self._label_display.setText(
                    "Tradition model not available for this typology"
                )
                self._confidence_display.setText(
                    "Pre-trained tradition models may need to be generated first. "
                    "Run: python -m lithicore.data.generate_training_data"
                )
                self._result_group.setVisible(True)
                return
        else:
            model = self._get_current_model()

        if model is None or not model.is_loaded():
            self._label_display.setText(
                "Model not available — try a different typology"
            )
            self._confidence_display.setText(
                "Pre-trained models may need to be generated first. "
                "Run: python -m lithicore.data.generate_training_data"
            )
            self._result_group.setVisible(True)
            return

        fv = extract_features(self._current_mesh)
        result = model.predict(fv, tradition=tradition_val)
        self._current_result = result
        self._show_result(result)

    def _show_result(self, result: ClassificationResult) -> None:
        """Display a ClassificationResult in the panel."""
        self._result_group.setVisible(True)

        self._label_display.setText(f"🏷 {result.label}")

        colour = (
            "green" if result.confidence >= 0.8
            else "orange" if result.confidence >= 0.6
            else "red"
        )
        self._confidence_display.setText(
            f"Confidence: <span style='color:{colour}; font-weight:bold;'>"
            f"{result.confidence:.0%}</span>"
        )

        # Feature importances
        self._features_list.clear()
        for f in result.top_features:
            status = "✓" if f.passed else "✗"
            self._features_list.addItem(
                f"  {f.name}: {f.value:.2f} ({f.contribution_pct:.0%}) {status}"
            )

        # Alternatives
        if result.alternatives:
            alt_text = "Also possible: " + ", ".join(
                f"{label} ({conf:.0%})"
                for label, conf in result.alternatives
            )
            self._alternatives_label.setText(alt_text)
        else:
            self._alternatives_label.setText("")

        self._populate_correction_combo()
        self._emit_overlays()

        self.classification_computed.emit(result)

    def _emit_overlays(self) -> None:
        """Emit diagnostic overlay coordinates if enabled."""
        if self._overlay_check.isChecked() and self._current_mesh is not None:
            coords = extract_diagnostic_coordinates(self._current_mesh)
            self.diagnostic_overlay_requested.emit(coords)

    # ── Correction / active learning ──

    def _on_correct(self) -> None:
        """Submit a correction for active learning."""
        if self._current_result is None or self._current_mesh is None:
            return
        correct_label = self._correct_combo.currentText()
        if not correct_label:
            return

        if self._get_tradition_value() is not None:
            model = self._get_tradition_router()
        else:
            model = self._get_current_model()
        if model is None:
            return

        fv = extract_features(self._current_mesh)
        count = model.queue_correction(fv, correct_label)
        self._label_display.setText(f"Corrected to: {correct_label}")
        self._correction_timer.start()

    def _on_debounced_retrain(self) -> None:
        """Check retrain threshold after correction debounce."""
        if self._get_tradition_value() is not None:
            model = self._get_tradition_router()
        else:
            model = self._get_current_model()
        if model is not None:
            retrained = model.retrain_if_ready(threshold=10)
            if retrained:
                self._label_display.setText("Model retrained from corrections")

    # ── Model persistence ──

    def _on_save_model(self) -> None:
        """Save the current custom model to a .joblib file."""
        model = self._get_current_model()
        if model is None or not model.is_loaded():
            QMessageBox.information(
                self, "No Model", "No custom model loaded to save. "
                "Train a custom model first via Train Custom Typology."
            )
            return
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Save Custom Model", "",
            "Joblib Model (*.joblib)",
        )
        if not path_str:
            return
        model.save(Path(path_str))
        QMessageBox.information(
            self, "Model Saved",
            f"Custom model saved with {len(model._classes)} classes."
        )

    def _on_load_model(self) -> None:
        """Load a custom model from a .joblib file."""
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Load Custom Model", "",
            "Joblib Model (*.joblib)",
        )
        if not path_str:
            return
        try:
            model = ClassifierModel(typology_name="custom", model_path=Path(path_str))
            self._models["custom"] = model
            self._typology_combo.setCurrentText("Custom")
            self._populate_correction_combo()
            QMessageBox.information(
                self, "Model Loaded",
                f"Custom model loaded with {len(model._classes)} classes: "
                f"{', '.join(model._classes)}"
            )
        except Exception as exc:
            QMessageBox.warning(
                self, "Load Error", f"Failed to load model:\n{exc}"
            )

    def has_model(self) -> bool:
        """Check if the current typology / tradition has a loaded model."""
        if self._get_tradition_value() is not None:
            model = self._get_tradition_router()
        else:
            model = self._get_current_model()
        return model is not None and model.is_loaded()

    def get_overlay_enabled(self) -> bool:
        """Check if diagnostic overlays should be shown."""
        return self._overlay_check.isChecked()

    def get_auto_classify(self) -> bool:
        """Check if auto-classify is enabled."""
        return self._auto_check.isChecked()
