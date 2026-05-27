"""_main_window.py — Main application window for lithicope.

exports: MainWindow(QMainWindow)
used_by: main.py entry point
rules:   Single-window layout: 3D viewer (60%) left, measurements panel (40%) right.
         Menu bar: File, Edit, Tools, Help.
         Status bar shows current artefact and batch progress.
agent:   deepseek-v4-flash | 2026-05-26 | Initial implementation
agent:   deepseek-v4-flash | 2026-05-26 | Added Publication Figure menu item and _on_publication_figure handler
          message: "imports lithicore._figure inside method — lazy import to avoid startup dependency"
agent:   deepseek-v4-flash | 2026-05-27 | Task 6: wired annotation panel into main window — tabbed results/annotations, placement handlers, menu items, capture photo
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QMainWindow, QMenuBar, QStatusBar, QSplitter, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QMessageBox, QSlider, QTabWidget,
)
from PyQt6.QtGui import QAction

from lithicore._models import MeshGrade
from lithicope._viewer_3d import Viewer3D
from lithicope._import_dialog import ImportDialog
from lithicope._results_panel import ResultsPanel
from lithicope._batch_runner import BatchRunner
from lithicope._photogrammetry_dialog import PhotogrammetryDialog
from lithicope._batch_photogrammetry import BatchPhotogrammetryDialog
from lithicope._annotation_panel import AnnotationPanel
from lithicope._classification_panel import ClassificationPanel
from lithicope._assistant_panel import AssistantPanel

import cv2
from datetime import datetime


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
        self._landmark_manager = None
        self._in_landmark_mode: bool = False

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

        # Right: tabbed results + annotations
        self.results_panel = ResultsPanel()
        self.results_panel.export_requested.connect(self._on_export)
        self._right_tabs = QTabWidget()
        self._right_tabs.addTab(self.results_panel, "Results")
        self._annotation_panel = AnnotationPanel()
        self._right_tabs.addTab(self._annotation_panel, "Annotations")
        self._classification_panel = ClassificationPanel()
        self._right_tabs.addTab(self._classification_panel, "Classification")
        self._assistant_panel = AssistantPanel()
        self._right_tabs.addTab(self._assistant_panel, "Assistant")
        splitter.addWidget(self._right_tabs)

        splitter.setSizes([660, 60, 480])
        self.setCentralWidget(splitter)

        # Wire annotation signals
        self._annotation_panel.focus_requested.connect(self.viewer.focus_on_point)
        self._annotation_panel.placement_mode_requested.connect(self._on_annotation_placement)
        self._annotation_panel.capture_photo_requested.connect(self._on_annotation_capture_photo)
        self._annotation_panel.annotation_added.connect(lambda _: self._sync_annotation_viewer())
        self._annotation_panel.annotation_deleted.connect(lambda _: self._sync_annotation_viewer())

        # Wire classification signals
        self._classification_panel.diagnostic_overlay_requested.connect(self._on_diagnostic_overlay)
        self._classification_panel.auto_classify_changed.connect(self._on_auto_classify_toggle)

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

        # New from Photos (Default photogrammetry)
        photos_action = QAction("&New from Photos...", self)
        photos_action.setShortcut("Ctrl+P")
        photos_action.triggered.connect(self._on_new_from_photos)
        file_menu.addAction(photos_action)

        # New Batch Photogrammetry
        batch_photo_action = QAction("&New Batch Photogrammetry...", self)
        batch_photo_action.setShortcut("Ctrl+Shift+P")
        batch_photo_action.triggered.connect(self._on_batch_photogrammetry)
        file_menu.addAction(batch_photo_action)

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

        # Photogrammetry submenu
        tools_menu.addSeparator()
        photo_submenu = tools_menu.addMenu("&Photogrammetry")
        guided_action = QAction("&Guided...", self)
        guided_action.triggered.connect(self._on_photogrammetry_guided)
        photo_submenu.addAction(guided_action)
        expert_action = QAction("&Expert...", self)
        expert_action.triggered.connect(self._on_photogrammetry_expert)
        photo_submenu.addAction(expert_action)
        tools_menu.addSeparator()

        # Comparison mode
        compare_action = QAction("&Compare with Another Mesh...", self)
        compare_action.setShortcut("Ctrl+D")
        compare_action.triggered.connect(self._on_compare)
        tools_menu.addAction(compare_action)

        clear_compare_action = QAction("&Clear Comparison", self)
        clear_compare_action.triggered.connect(self._on_clear_comparison)
        tools_menu.addAction(clear_compare_action)

        # Landmark placement
        tools_menu.addSeparator()
        lm_action = QAction("&Place Landmarks...", self)
        lm_action.setShortcut("Ctrl+L")
        lm_action.triggered.connect(self._on_place_landmarks)
        tools_menu.addAction(lm_action)

        undo_lm_action = QAction("&Undo Last Landmark", self)
        undo_lm_action.setShortcut("Ctrl+Z")
        undo_lm_action.triggered.connect(self._on_undo_landmark)
        tools_menu.addAction(undo_lm_action)

        clear_lm_action = QAction("C&lear All Landmarks", self)
        clear_lm_action.triggered.connect(self._on_clear_landmarks)
        tools_menu.addAction(clear_lm_action)

        # Scar detection
        tools_menu.addSeparator()
        scar_action = QAction("&Detect Flake Scars", self)
        scar_action.triggered.connect(self._on_detect_scars)
        tools_menu.addAction(scar_action)

        clear_scar_action = QAction("&Clear Scar Overlay", self)
        clear_scar_action.triggered.connect(self._on_clear_scars)
        tools_menu.addAction(clear_scar_action)

        # Annotations
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

        # Classification submenu
        tools_menu.addSeparator()
        class_menu = tools_menu.addMenu("&Classification")
        classify_action = QAction("&Classify Artefact", self)
        classify_action.setShortcut("Ctrl+Shift+C")
        classify_action.triggered.connect(self._classification_panel._on_classify)
        class_menu.addAction(classify_action)

        # Assistant submenu
        tools_menu.addSeparator()
        asst_menu = tools_menu.addMenu("&Assistant")
        open_asst_action = QAction("&Open Assistant", self)
        open_asst_action.setShortcut("Ctrl+Shift+Q")
        open_asst_action.triggered.connect(self._on_open_assistant)
        asst_menu.addAction(open_asst_action)

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

    def _on_new_from_photos(self) -> None:
        """Default mode photogrammetry: pick a folder, run pipeline."""
        dir_str = QFileDialog.getExistingDirectory(self, "Select Photo Folder")
        if not dir_str:
            return
        photo_folder = Path(dir_str)
        default_label = photo_folder.stem

        out_str, _ = QFileDialog.getSaveFileName(
            self, "Save Mesh As",
            str(photo_folder / f"{default_label}.ply"),
            "PLY Mesh (*.ply);;OBJ Mesh (*.obj);;STL Mesh (*.stl)",
        )
        if not out_str:
            return

        dialog = PhotogrammetryDialog(
            self,
            mode="default",
            photo_folder=photo_folder,
            output_path=Path(out_str),
            artefact_label=default_label,
        )
        dialog.exec()

    def _on_batch_photogrammetry(self) -> None:
        """Open the batch photogrammetry dialog."""
        dir_str = QFileDialog.getExistingDirectory(self, "Select Artefacts Folder")
        if not dir_str:
            return
        dialog = BatchPhotogrammetryDialog(Path(dir_str), self)
        dialog.exec()

    def _on_photogrammetry_guided(self) -> None:
        """Open photogrammetry dialog in guided mode."""
        dir_str = QFileDialog.getExistingDirectory(self, "Select Photo Folder")
        if not dir_str:
            return
        photo_folder = Path(dir_str)
        default_label = photo_folder.stem

        out_str, _ = QFileDialog.getSaveFileName(
            self, "Save Mesh As",
            str(photo_folder / f"{default_label}.ply"),
            "PLY Mesh (*.ply);;OBJ Mesh (*.obj);;STL Mesh (*.stl)",
        )
        if not out_str:
            return

        dialog = PhotogrammetryDialog(
            self,
            mode="guided",
            photo_folder=photo_folder,
            output_path=Path(out_str),
            artefact_label=default_label,
        )
        dialog.exec()

    def _on_photogrammetry_expert(self) -> None:
        """Open photogrammetry dialog in expert mode."""
        dir_str = QFileDialog.getExistingDirectory(self, "Select Photo Folder")
        if not dir_str:
            return
        photo_folder = Path(dir_str)
        default_label = photo_folder.stem

        out_str, _ = QFileDialog.getSaveFileName(
            self, "Save Mesh As",
            str(photo_folder / f"{default_label}.ply"),
            "PLY Mesh (*.ply);;OBJ Mesh (*.obj);;STL Mesh (*.stl)",
        )
        if not out_str:
            return

        dialog = PhotogrammetryDialog(
            self,
            mode="expert",
            photo_folder=photo_folder,
            output_path=Path(out_str),
            artefact_label=default_label,
        )
        dialog.exec()

    def _on_export(self, fmt: str = "csv") -> None:
        """Export current results."""
        if self._current_results is None:
            QMessageBox.information(self, "No Data", "No measurements to export.")
            return
        self.results_panel.export_results(self._current_results, fmt)

    def _process_single(self, path: Path, config) -> None:
        """Load, orient, measure a single mesh."""
        # Clear previous state
        self.viewer.clear_diagnostic_overlay()
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
            # Build collection DataFrame for AI assistant
            if self._current_results is not None:
                import pandas as pd
                row = {}
                for m in self._current_results:
                    row[m.name] = m.value
                row["artefact_label"] = path.stem
                # Add typology if classified
                df_class = pd.DataFrame([row])
                self._assistant_panel.set_collection(df_class)
            self._current_mesh = oriented
            self.viewer.display_mesh(oriented, edge_vertices)
            # Feed mesh to classification panel for potential auto-classify
            if self._current_mesh is not None:
                self._classification_panel.set_mesh(self._current_mesh)
            self.results_panel.show_measurements(measurements, path.name, quality.grade)
            # Set annotation working directory
            if hasattr(self, '_annotation_panel'):
                self._annotation_panel.set_annotation_dir(path.parent)
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
            from lithicore._models import MeasurementConfig, MeshGrade

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
                MeshGrade.PASS,
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

    # ── Landmark methods ────────────────────────────────────────

    def _on_place_landmarks(self) -> None:
        """Enter landmark placement mode."""
        if self._current_mesh_path is None:
            QMessageBox.information(self, "No Mesh", "Load a mesh first.")
            return

        from lithicore._landmarks import LandmarkManager, LANDMARK_SCHEMES

        if self._landmark_manager is None:
            self._landmark_manager = LandmarkManager(LANDMARK_SCHEMES["flake_13"])

        if not self._in_landmark_mode:
            self._in_landmark_mode = True
            self.viewer.enable_landmark_mode(self._on_landmark_placed)
            self.status.showMessage(
                f"Landmark mode: click on mesh to place landmarks "
                f"({self._landmark_manager.remaining} remaining)"
            )
        else:
            QMessageBox.information(
                self, "Already Placing",
                "Already in landmark mode. Click on the mesh to place points, "
                "or use Undo Last Landmark to remove the most recent."
            )

    def _on_landmark_placed(self, x: float, y: float, z: float) -> None:
        """Callback when user clicks on the mesh in landmark mode."""
        if self._landmark_manager is None:
            return

        import numpy as np
        lm = self._landmark_manager.place_landmark(np.array([x, y, z]))
        self.viewer.refresh_landmarks(self._landmark_manager.landmarks)

        # Update results panel with current landmarks
        from lithicore._models import MeasurementResult
        results = [
            MeasurementResult(
                name=f"LM {i+1}: {lm.name}",
                value=0.0,
                unit="",
                confidence=1.0,
            )
            for i, lm in enumerate(self._landmark_manager.landmarks)
        ]
        self.results_panel.show_measurements(
            results,
            f"{self._current_mesh_path.stem} — Landmarks ({self._landmark_manager.remaining} left)",
            MeshGrade.PASS,
        )

        if self._landmark_manager.is_complete:
            self.status.showMessage(f"All {len(self._landmark_manager.landmarks)} landmarks placed! Export to MorphoJ.")
            self.viewer.disable_landmark_mode()
            self._in_landmark_mode = False
        else:
            self.status.showMessage(
                f"Placed {lm.name} — {self._landmark_manager.remaining} remaining"
            )

    def _on_undo_landmark(self) -> None:
        """Remove the most recently placed landmark."""
        if self._landmark_manager is None or not self._landmark_manager.landmarks:
            QMessageBox.information(self, "No Landmarks", "No landmarks to undo.")
            return

        removed = self._landmark_manager.remove_last()
        if removed:
            self.viewer.refresh_landmarks(self._landmark_manager.landmarks)
            self.status.showMessage(f"Removed {removed.name} — {self._landmark_manager.remaining} remaining")

    def _on_clear_landmarks(self) -> None:
        """Clear all placed landmarks."""
        if self._landmark_manager is None:
            return
        self._landmark_manager.clear()
        self.viewer.clear_landmarks()
        self._in_landmark_mode = False
        self.results_panel.show_measurements([], "No landmarks", MeshGrade.PASS)
        self.status.showMessage("All landmarks cleared")

    # ── Scar detection ─────────────────────────────────────────

    def _on_detect_scars(self) -> None:
        """Run flake scar detection on the current mesh."""
        if self._current_mesh_path is None:
            QMessageBox.information(self, "No Mesh", "Load a mesh first.")
            return

        self.status.showMessage("Detecting flake scars...")
        try:
            from lithicore._scar_detection import detect_scars, ScarConfig
            from lithicore._models import MeasurementResult
            import trimesh

            mesh = trimesh.load(str(self._current_mesh_path), force="mesh")
            config = ScarConfig()
            result = detect_scars(mesh, config)

            # Show coloured overlay
            self.viewer.show_scar_overlay(result.face_labels)

            # Build results
            scar_msgs = []
            if result.scar_count > 0:
                for scar in result.scars:
                    scar_msgs.append(
                        f"Scar {scar.index + 1}: {scar.area_mm2:.2f} mm², "
                        f"depth={scar.max_depth_mm:.2f}"
                    )
                results = [
                    MeasurementResult(name="Scar Count", value=float(result.scar_count), unit="", confidence=0.9),
                    MeasurementResult(name="Total Scar Area", value=result.total_scar_area_mm2, unit="mm²", confidence=0.85),
                    MeasurementResult(name="Scar Density", value=result.scar_density, unit="", confidence=0.85),
                ]
                self.results_panel.show_measurements(
                    results,
                    f"{self._current_mesh_path.stem} — {result.scar_count} scars",
                    MeshGrade.PASS,
                )
            else:
                self.results_panel.show_measurements(
                    [MeasurementResult(name="Scars Detected", value=0.0, unit="", confidence=1.0)],
                    f"{self._current_mesh_path.stem} — No scars found",
                    MeshGrade.PASS,
                )

            self.status.showMessage(f"Scar detection complete: {result.scar_count} scars")
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Scar detection failed:\n{exc}")
            self.status.showMessage("Scar detection failed")

    def _on_clear_scars(self) -> None:
        """Remove scar overlay and restore default mesh appearance."""
        self.viewer.clear_scar_overlay()
        self.status.showMessage("Scar overlay cleared")

    # ── Annotation handlers ──

    def _on_annotation_placement(self) -> None:
        """Enter annotation placement mode in the viewer."""
        self._right_tabs.setCurrentIndex(1)  # Switch to Annotations tab
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
        """Sync annotations from panel to viewer display."""
        annotations = self._annotation_panel.get_annotations()
        mode = self._annotation_panel.get_display_mode()
        self.viewer._annotation_display_mode = mode
        self.viewer.refresh_annotations(annotations)

    # ── Classification handlers ──

    def _on_diagnostic_overlay(self, coords: dict) -> None:
        """Show diagnostic overlays on the viewer."""
        self.viewer.show_diagnostic_overlay(coords)

    def _on_auto_classify_toggle(self, enabled: bool) -> None:
        """Handle auto-classify toggle."""
        self.status.showMessage(f"Auto-classify: {'ON' if enabled else 'OFF'}")

    def _on_annotation_capture_photo(self) -> None:
        """Capture the current 3D view and attach to the selected annotation."""
        if not hasattr(self.viewer, 'plotter'):
            return
        screenshot = self.viewer.plotter.screenshot(return_img=True)
        ann_dir = getattr(self._annotation_panel, '_annotation_dir', Path.home())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        photo_path = ann_dir / f"annotation_capture_{timestamp}.png"
        cv2.imwrite(str(photo_path), cv2.cvtColor(screenshot, cv2.COLOR_RGBA2BGR))
        self._annotation_panel.add_captured_photo(str(photo_path))
        self.status.showMessage(f"Photo captured: {photo_path.name}")

    def _on_open_assistant(self) -> None:
        """Switch to the Assistant tab."""
        self._right_tabs.setCurrentIndex(
            self._right_tabs.indexOf(self._assistant_panel)
        )
