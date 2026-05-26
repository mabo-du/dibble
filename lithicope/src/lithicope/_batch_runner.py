"""_batch_runner.py — Batch processing runner with progress feedback.

exports: BatchRunner(QDialog)
used_by: MainWindow batch import
rules:   Runs batch_process in a separate thread to keep UI responsive.
         Shows progress bar and per-file status updates.
         Opens results panel when complete.
agent:   deepseek-v4-flash | 2026-05-26 | Initial implementation
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar,
    QPushButton, QTextEdit, QMessageBox, QWidget,
)
from lithicore._models import MeasurementConfig, ArtefactResult
from lithicore._batch import batch_process


class BatchWorker(QThread):
    """Background worker for batch processing."""

    progress = pyqtSignal(int, int, str)  # current, total, filename
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, directory: Path, config: MeasurementConfig) -> None:
        super().__init__()
        self._directory = directory
        self._config = config

    def run(self) -> None:
        try:
            results = batch_process(self._directory, self._config)
            self.finished.emit(results)
        except Exception as exc:
            self.error.emit(str(exc))


class BatchRunner(QDialog):
    """Dialog showing batch processing progress."""

    def __init__(
        self,
        directory: Path,
        config: MeasurementConfig,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Batch Processing")
        self.setMinimumWidth(500)
        self.setModal(True)

        self._results: List[ArtefactResult] = []

        layout = QVBoxLayout(self)

        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        layout.addWidget(self._progress_bar)

        # Status text
        self._status_text = QTextEdit()
        self._status_text.setReadOnly(True)
        self._status_text.setMaximumHeight(200)
        layout.addWidget(self._status_text)

        # Buttons
        btn_layout = QHBoxLayout()
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._on_cancel)
        btn_layout.addWidget(self._cancel_btn)

        self._close_btn = QPushButton("Close")
        self._close_btn.clicked.connect(self.accept)
        self._close_btn.setEnabled(False)
        btn_layout.addWidget(self._close_btn)
        layout.addLayout(btn_layout)

        # Start worker
        self._worker = BatchWorker(directory, config)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, current: int, total: int, filename: str) -> None:
        percent = int((current / total) * 100) if total > 0 else 0
        self._progress_bar.setValue(percent)
        self._status_text.append(f"[{current}/{total}] Processing {filename}...")

    def _on_finished(self, results: List[ArtefactResult]) -> None:
        self._results = results
        self._progress_bar.setValue(100)
        success = sum(1 for r in results if r.measurements)
        self._status_text.append(f"\nComplete: {len(results)} files processed ({success} with measurements)")
        self._cancel_btn.setEnabled(False)
        self._close_btn.setEnabled(True)

    def _on_error(self, error_msg: str) -> None:
        self._status_text.append(f"\nError: {error_msg}")
        self._cancel_btn.setEnabled(False)
        self._close_btn.setEnabled(True)

    def _on_cancel(self) -> None:
        self._worker.terminate()
        self._status_text.append("\nCancelled by user")
        self._cancel_btn.setEnabled(False)
        self._close_btn.setEnabled(True)

    def get_results(self) -> List[ArtefactResult]:
        return self._results
