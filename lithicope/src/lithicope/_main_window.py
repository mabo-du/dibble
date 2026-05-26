"""_main_window.py — Main application window for lithicope.

exports: MainWindow(QMainWindow)
used_by: main.py entry point
rules:   Single-window layout: 3D viewer (60%) left, measurements panel (40%) right.
         Menu bar: File, Edit, Tools, Help.
         Status bar shows current artefact and batch progress.
agent:   deepseek-v4-flash | 2026-05-26 | Initial implementation
agent:   deepseek-v4-flash | 2026-05-26 | Added Publication Figure menu item and _on_publication_figure handler
         message: "imports lithicore._figure inside method — lazy import to avoid startup dependency"
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QMainWindow, QMenuBar, QStatusBar, QSplitter, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QMessageBox, QSlider,
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
        self._compare_mesh_path: Optional[Path] = None
        self._in_comparison_mode: bool = False

        self._init_ui()
        self._init_menu()
        self._init_status_bar()

    def _init_ui(self) -> None:
        """Build the main layout: viewer left, results right."""
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: 3D viewer
        self.viewer = Viewer3D()
        splitter.addWidget(self.viewer)

        # Center: comparison controls (hidden by default)
        self._compare_widget = QWidget()
        compare_layout = QVBoxLayout(self._compare_widget)
        compare_layout.setContentsMargins(5, 5, 5, 5)
        self._compare_widget.setMaximumWidth(60)
        self._compare_label = QLabel("Opacity")
        self._compare_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._opacity_slider = QSlider(Qt.Orientation.Vertical)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(50)
        self._opacity_slider.valueChanged.connect(self._on_opacity_changed)
        compare_layout.addWidget(self._compare_label)
        compare_layout.addWidget(self._opacity_slider)
        compare_layout.addStretch()
        self._compare_widget.setVisible(False)
        splitter.addWidget(self._compare_widget)

        # Right: results panel
        self.results_panel = ResultsPanel()
        self.results_panel.export_requested.connect(self._on_export)
        splitter.addWidget(self.results_panel)

        splitter.setSizes([660, 60, 480])
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

        # Publication figure
        fig_action = QAction("&Publication Figure...", self)
        fig_action.triggered.connect(self._on_publication_figure)
        tools_menu.addAction(fig_action)

        # Comparison mode
        tools_menu.addSeparator()
        compare_action = QAction("&Compare with Another Mesh...", self)
        compare_action.setShortcut("Ctrl+D")
        compare_action.triggered.connect(self._on_compare)
        tools_menu.addAction(compare_action)

        clear_compare_action = QAction("&Clear Comparison", self)
        clear_compare_action.triggered.connect(self._on_clear_comparison)
        tools_menu.addAction(clear_compare_action)

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

    def _on_publication_figure(self) -> None:
        """Export a publication figure from the current mesh."""
        if self._current_mesh_path is None:
            QMessageBox.information(self, "No Mesh", "Load a mesh first.")
            return

        path_str, _ = QFileDialog.getSaveFileName(
            self, "Save Publication Figure", "figure.svg",
            "SVG Files (*.svg);;PDF Files (*.pdf)",
        )
        if not path_str:
            return
        path = Path(path_str)

        from lithicore._figure import FigureConfig, generate_figure

        config = FigureConfig(
            artefact_label=self._current_mesh_path.stem,
        )

        try:
            svg = generate_figure(self.viewer._mesh, self.viewer.plotter, config)
            path.write_text(svg)
            QMessageBox.information(self, "Exported", f"Figure saved to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to generate figure:\n{exc}")

    def _on_compare(self) -> None:
        """Compare current mesh with another mesh file."""
        if self._current_mesh_path is None:
            QMessageBox.information(self, "No Mesh", "Load a mesh first, then compare.")
            return

        path_str, _ = QFileDialog.getOpenFileName(
            self, "Select Mesh to Compare Against", "",
            "Mesh Files (*.ply *.obj *.stl);;All Files (*)",
        )
        if not path_str:
            return
        compare_path = Path(path_str)

        self.status.showMessage(f"Loading {compare_path.name} for comparison...")
        try:
            import trimesh
            from lithicore._orientation import orient_auto
            from lithicore._edge_detection import detect_edges
            from lithicore._comparison import compare_meshes
            from lithicore._models import MeasurementConfig

            mesh_a = trimesh.load(str(self._current_mesh_path), force="mesh")
            mesh_b = trimesh.load(str(compare_path), force="mesh")

            config = MeasurementConfig()
            oriented_a, _ = orient_auto(mesh_a, config)
            oriented_b, _ = orient_auto(mesh_b, config)
            edges_a, _ = detect_edges(oriented_a, config)

            # Show overlay
            self.viewer.display_comparison(oriented_a, oriented_b, edges_a)
            self._compare_widget.setVisible(True)
            self._opacity_slider.setValue(50)

            # Compute metrics
            result = compare_meshes(oriented_a, oriented_b)
            self._compare_mesh_path = compare_path
            self._in_comparison_mode = True

            # Build comparison results display
            comp_results = [
                ("hausdorff_distance_mm", f"{result.hausdorff_distance_mm} mm"),
                ("centroid_distance_mm", f"{result.centroid_distance_mm} mm"),
                ("volume_difference_mm3", f"{result.volume_difference_mm3} mm³"),
                ("surface_area_difference_mm2", f"{result.surface_area_difference_mm2} mm²"),
                ("length_diff_mm", f"{result.length_diff_mm} mm"),
                ("width_diff_mm", f"{result.width_diff_mm} mm"),
                ("thickness_diff_mm", f"{result.thickness_diff_mm} mm"),
            ]
            from lithicore._models import MeasurementResult
            results_list = [
                MeasurementResult(name=n, value=float(v.split()[0]), unit=v.split()[1] if len(v.split()) > 1 else "", confidence=0.9)
                for n, v in comp_results
            ]

            self.results_panel.show_measurements(
                results_list,
                f"{self._current_mesh_path.stem} vs {compare_path.stem}",
                "pass",
            )
            self.status.showMessage(f"Comparison: {self._current_mesh_path.stem} vs {compare_path.stem}")
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Comparison failed:\n{exc}")
            self.status.showMessage("Comparison failed")

    def _on_clear_comparison(self) -> None:
        """Remove comparison overlay and restore single mesh view."""
        self.viewer.clear_comparison()
        self._compare_widget.setVisible(False)
        self._in_comparison_mode = False
        self._compare_mesh_path = None
        self.status.showMessage("Comparison cleared")

    def _on_opacity_changed(self, value: int) -> None:
        """Adjust overlay mesh opacity."""
        opacity = value / 100.0
        self.viewer.set_overlay_opacity(opacity)
