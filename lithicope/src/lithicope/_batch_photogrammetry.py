"""_batch_photogrammetry.py — Batch queue dialog for photogrammetry.

exports: BatchPhotogrammetryDialog(QDialog)
used_by: MainWindow -> File -> New Batch Photogrammetry
rules:   Sequential processing. Each artefact = one sub-folder.
         Results saved to output_folder/<label>/<label>.ply.
agent:   deepseek-v4-flash | 2026-05-26 | Initial implementation
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lithicore._photogrammetry import (
    PhotogrammetryConfig,
    run_pipeline,
)


class BatchPhotogrammetryDialog(QDialog):
    """Batch photogrammetry queue."""

    STATUS_QUEUED = "Queued"
    STATUS_RUNNING = "Running"
    STATUS_COMPLETE = "Complete"
    STATUS_FAILED = "Failed"

    def __init__(
        self,
        artefacts_dir: Path,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._artefacts_dir = artefacts_dir
        self._artefacts: list[dict] = []
        self._current_index: int = -1
        self._cancelled: bool = False

        self.setWindowTitle("Batch Photogrammetry")
        self.setMinimumWidth(600)
        self.setModal(True)

        layout = QVBoxLayout(self)

        # Table
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["", "Artefact", "Photos", "Status"])
        self._table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._table)

        # Output folder
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output folder:"))
        self._output_edit = QLabel(str(artefacts_dir / "results"))
        out_row.addWidget(self._output_edit)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_output)
        out_row.addWidget(browse_btn)
        layout.addLayout(out_row)

        # Preset
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Settings:"))
        self._preset_combo = QComboBox()
        self._preset_combo.addItems(["Default", "High Quality", "Fast"])
        preset_row.addWidget(self._preset_combo)
        preset_row.addStretch()
        layout.addLayout(preset_row)

        # Buttons
        btn_row = QHBoxLayout()
        self._add_btn = QPushButton("Add Artefacts...")
        self._add_btn.clicked.connect(self._add_artefacts)
        btn_row.addWidget(self._add_btn)
        self._start_btn = QPushButton("Start Batch")
        self._start_btn.clicked.connect(self._start_batch)
        self._start_btn.setEnabled(False)
        btn_row.addWidget(self._start_btn)
        btn_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        self._scan_artefacts()
        self._start_btn.setEnabled(len(self._artefacts) > 0)

    def _scan_artefacts(self) -> None:
        if not self._artefacts_dir.is_dir():
            return
        for child in sorted(self._artefacts_dir.iterdir()):
            if not child.is_dir():
                continue
            photos = [
                p for p in child.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
            if len(photos) >= 3:
                self._artefacts.append({
                    "path": child,
                    "label": child.name,
                    "photo_count": len(photos),
                    "status": self.STATUS_QUEUED,
                })
        self._refresh_table()

    def _add_artefacts(self) -> None:
        dir_str = QFileDialog.getExistingDirectory(self, "Select Artefact Folder")
        if not dir_str:
            return
        path = Path(dir_str)
        photos = [p for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if len(photos) >= 3:
            self._artefacts.append({
                "path": path,
                "label": path.name,
                "photo_count": len(photos),
                "status": self.STATUS_QUEUED,
            })
            self._refresh_table()
            self._start_btn.setEnabled(True)

    def _refresh_table(self) -> None:
        self._table.setRowCount(len(self._artefacts))
        for i, art in enumerate(self._artefacts):
            cb = QTableWidgetItem("")
            cb.setFlags(cb.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            cb.setCheckState(Qt.CheckState.Checked)
            self._table.setItem(i, 0, cb)
            self._table.setItem(i, 1, QTableWidgetItem(art["label"]))
            self._table.setItem(i, 2, QTableWidgetItem(str(art["photo_count"])))
            self._table.setItem(i, 3, QTableWidgetItem(art["status"]))
        self._table.resizeColumnsToContents()

    def _browse_output(self) -> None:
        dir_str = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if dir_str:
            self._output_edit.setText(dir_str)

    def _start_batch(self) -> None:
        self._start_btn.setEnabled(False)
        self._add_btn.setEnabled(False)
        self._current_index = -1
        self._process_next()

    def _process_next(self) -> None:
        self._current_index += 1
        if self._cancelled or self._current_index >= len(self._artefacts):
            self._start_btn.setText("Batch Complete")
            return

        art = self._artefacts[self._current_index]
        item = self._table.item(self._current_index, 0)
        if item and item.checkState() != Qt.CheckState.Checked:
            self._process_next()
            return

        art["status"] = self.STATUS_RUNNING
        self._refresh_table()

        output_dir = Path(self._output_edit.text()) / art["label"]
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{art['label']}.ply"

        quality_map = {"Default": "high", "High Quality": "high", "Fast": "low"}
        quality = quality_map.get(self._preset_combo.currentText(), "high")

        config = PhotogrammetryConfig(
            photo_folder=art["path"],
            output_path=out_path,
            artefact_label=art["label"],
            quality=quality,
            mode="default",
        )
        self._current_config = config
        QTimer.singleShot(100, self._run_current)

    def _run_current(self) -> None:
        try:
            run_pipeline(self._current_config)
            self._artefacts[self._current_index]["status"] = self.STATUS_COMPLETE
        except Exception as exc:
            self._artefacts[self._current_index]["status"] = f"Failed: {str(exc)[:50]}"
        self._refresh_table()
        QTimer.singleShot(200, self._process_next)

    def closeEvent(self, event) -> None:  # type: ignore
        self._cancelled = True
        super().closeEvent(event)
