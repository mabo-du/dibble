"""_import_dialog.py — Import dialog with mode selection and advanced options.

exports: ImportDialog(QDialog)
used_by: MainWindow file open and batch operations
rules:   Three import modes: Single, Batch-auto, Batch-manual.
         Advanced section with skip auto-repair / skip auto-validation toggles.
         Single and Batch-manual modes show advanced options; auto hides them.
agent:   deepseek-v4-flash | 2026-05-26 | Initial implementation
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QRadioButton,
    QCheckBox, QDialogButtonBox, QGroupBox, QWidget,
)
from lithicore._models import MeasurementConfig


class ImportDialog(QDialog):
    """Import configuration dialog.

    Allows the user to choose import mode and configure
    mesh repair and validation settings.
    """

    def __init__(self, parent: Optional[QWidget] = None, mode: str = "single") -> None:
        super().__init__(parent)
        self.setWindowTitle("Import Mesh")
        self.setMinimumWidth(450)

        self._mode = mode
        self._config = MeasurementConfig()

        layout = QVBoxLayout(self)

        # Mode selection
        mode_group = QGroupBox("Import Mode")
        mode_layout = QVBoxLayout(mode_group)

        self._radio_single = QRadioButton("Single artefact")
        self._radio_batch_auto = QRadioButton("Batch — auto  (auto-orient)")
        self._radio_batch_review = QRadioButton("Batch — review  (review flags)")
        self._radio_batch_manual = QRadioButton("Batch — manual  (orient each)")

        if mode == "single":
            self._radio_single.setChecked(True)
        elif mode == "batch_auto":
            self._radio_batch_auto.setChecked(True)
        elif mode == "batch_review":
            self._radio_batch_review.setChecked(True)
        elif mode == "batch_manual":
            self._radio_batch_manual.setChecked(True)

        mode_layout.addWidget(self._radio_single)
        mode_layout.addWidget(self._radio_batch_auto)
        mode_layout.addWidget(self._radio_batch_review)
        mode_layout.addWidget(self._radio_batch_manual)
        layout.addWidget(mode_group)

        # Advanced options (visible for single + batch-manual)
        self._advanced_group = QGroupBox("Advanced")
        advanced_layout = QVBoxLayout(self._advanced_group)

        self._skip_repair = QCheckBox("Skip auto-repair")
        self._skip_validation = QCheckBox("Skip auto-validation")
        advanced_layout.addWidget(self._skip_repair)
        advanced_layout.addWidget(self._skip_validation)
        self._advanced_group.setVisible(mode in ("single", "batch_manual"))
        layout.addWidget(self._advanced_group)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Connect radio buttons to toggle advanced visibility
        self._radio_single.toggled.connect(self._on_mode_changed)
        self._radio_batch_manual.toggled.connect(self._on_mode_changed)

    def _on_mode_changed(self) -> None:
        """Show advanced panel for single/manual modes only."""
        show_advanced = self._radio_single.isChecked() or self._radio_batch_manual.isChecked()
        self._advanced_group.setVisible(show_advanced)

    def get_config(self) -> MeasurementConfig:
        """Return a MeasurementConfig based on dialog selections."""
        return MeasurementConfig(
            repair_mesh=not self._skip_repair.isChecked(),
        )

    def get_mode(self) -> str:
        """Return the selected import mode."""
        if self._radio_single.isChecked():
            return "single"
        elif self._radio_batch_auto.isChecked():
            return "batch_auto"
        elif self._radio_batch_review.isChecked():
            return "batch_review"
        else:
            return "batch_manual"
