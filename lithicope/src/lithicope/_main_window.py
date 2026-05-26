"""_main_window.py — Main application window for lithicope.

exports: MainWindow(QMainWindow)
used_by: main.py entry point
rules:   Single-window layout: 3D viewer (60%) left, measurements panel (40%) right.
         Menu bar: File, Edit, Tools, Help.
         Status bar shows current artefact and batch progress.
agent:   deepseek-v4-flash | 2026-05-26 | Initial implementation
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QMainWindow, QMenuBar, QStatusBar, QSplitter, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QMessageBox,
)
from PyQt6.QtGui import QAction

from lithicope._viewer_3d import Viewer3D
from lithicope._import_dialog import ImportDialog
from lithicope._results_panel import ResultsPanel
from lithicope._batch_runner import BatchRunner


class MainWindow(QMainWindow):
    """Primary application window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Lithic Analysis Platform")
        self.setMinimumSize(1200, 800)

        self._current_mesh_path: Optional[Path] = None
        self._current_results: Optional[list] = None
        self._batch_results: List = []

        self._init_ui()
        self._init_menu()
        self._init_status_bar()

    def _init_ui(self) -> None:
        """Build the main layout: viewer left, results right."""
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: 3D viewer
        self.viewer = Viewer3D()
        splitter.addWidget(self.viewer)

        # Right: results panel
        self.results_panel = ResultsPanel()
        self.results_panel.export_requested.connect(self._on_export)
        splitter.addWidget(self.results_panel)

        splitter.setSizes([720, 480])
        self.setCentralWidget(splitter)

    def _init_menu(self) -> None:
        menu = self.menuBar()

        # File menu
        file_menu = menu.addMenu("&File")
        open_action = QAction("&Open Mesh...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._on_open)
        file_menu.addAction(open_action)

        batch_action = QAction("&Batch Import...", self)
        batch_action.setShortcut("Ctrl+B")
        batch_action.triggered.connect(self._on_batch)
        file_menu.addAction(batch_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Tools menu
        tools_menu = menu.addMenu("&Tools")
        export_action = QAction("&Export CSV...", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(lambda: self._on_export("csv"))
        tools_menu.addAction(export_action)

        # Help menu
        help_menu = menu.addMenu("&Help")
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    def _init_status_bar(self) -> None:
        self.status = self.statusBar()
        self.status.showMessage("Ready")

    def _on_open(self) -> None:
        """Open a single mesh file."""
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Open Mesh", "",
            "Mesh Files (*.ply *.obj *.stl);;All Files (*)",
        )
        if not path_str:
            return
        path = Path(path_str)

        dialog = ImportDialog(self, mode="single")
        if dialog.exec() == ImportDialog.DialogCode.Accepted:
            config = dialog.get_config()
            self._process_single(path, config)

    def _on_batch(self) -> None:
        """Open batch import dialog."""
        dir_str = QFileDialog.getExistingDirectory(
            self, "Select Mesh Directory"
        )
        if not dir_str:
            return
        directory = Path(dir_str)

        dialog = ImportDialog(self, mode="batch_auto")
        if dialog.exec() == ImportDialog.DialogCode.Accepted:
            config = dialog.get_config()
            self._run_batch(directory, config)

    def _on_export(self, fmt: str = "csv") -> None:
        """Export current results."""
        if self._current_results is None:
            QMessageBox.information(self, "No Data", "No measurements to export.")
            return
        self.results_panel.export_results(self._current_results, fmt)

    def _process_single(self, path: Path, config) -> None:
        """Load, orient, measure a single mesh."""
        self.status.showMessage(f"Loading {path.name}...")
        try:
            import trimesh
            mesh = trimesh.load(str(path), force="mesh")
            from lithicore._validation import validate_mesh, repair_mesh
            from lithicore._orientation import orient_auto
            from lithicore._metrics import extract_metrics
            from lithicore._platform_angle import platform_angles
            from lithicore._edge_detection import detect_edges
            from lithicore._models import MeasurementConfig

            quality = validate_mesh(mesh)
            if config.repair_mesh:
                _, mesh = repair_mesh(mesh)

            oriented, _ = orient_auto(mesh, config)
            measurements = extract_metrics(oriented, config)
            epa, ipa = platform_angles(oriented, config)
            if epa:
                measurements.append(epa)
            if ipa:
                measurements.append(ipa)
            edge_vertices, _ = detect_edges(oriented, config)

            self._current_mesh_path = path
            self._current_results = measurements
            self.viewer.display_mesh(oriented, edge_vertices)
            self.results_panel.show_measurements(measurements, path.name, quality.grade)
            self.status.showMessage(f"Loaded: {path.name}")
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to process mesh:\n{exc}")
            self.status.showMessage("Error loading mesh")

    def _run_batch(self, directory: Path, config) -> None:
        """Run batch processing in the GUI."""
        from lithicore._batch import batch_process
        runner = BatchRunner(directory, config, self)
        runner.exec()

    def _on_about(self) -> None:
        QMessageBox.about(
            self,
            "About Lithic Analysis Platform",
            "Lithic 3D Morphological Analyzer v0.1\n\n"
            "An open-source desktop application for automated\n"
            "3D lithic artefact measurement and analysis.\n\n"
            "Built with lithicore + PyQt6 + Open3D",
        )
