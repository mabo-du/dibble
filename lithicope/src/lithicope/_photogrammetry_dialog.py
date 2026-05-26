"""_photogrammetry_dialog.py — GUI dialog for the photogrammetry pipeline.

exports: PhotogrammetryDialog(QDialog)
used_by: MainWindow menu actions
rules:   Three stacked pages: config -> progress -> result.
         Runs pipeline in a QThread to keep UI responsive.
         Follows same QThread pattern as BatchRunner.
agent:   deepseek-v4-flash | 2026-05-26 | Full dialog implementation
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from lithicore._photogrammetry import (
    PhotogrammetryConfig,
    PhotogrammetryResult,
    PhotogrammetryError,
    run_pipeline,
    colmap_available,
)


class PhotogrammetryWorker(QThread):
    """Background worker for the photogrammetry pipeline."""

    progress = pyqtSignal(str, float, str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, config: PhotogrammetryConfig) -> None:
        super().__init__()
        self._config = config
        self._cancelled = False

    def run(self) -> None:
        try:
            def cb(stage: str, progress: float, message: str) -> None:
                if self._cancelled:
                    from lithicore._photogrammetry import PhotogrammetryCancelledError
                    raise PhotogrammetryCancelledError("Cancelled by user")
                self.progress.emit(stage, progress, message)

            result = run_pipeline(self._config, progress_cb=cb)
            self.finished.emit(result)
        except PhotogrammetryError as exc:
            self.error.emit(str(exc))
        except Exception as exc:
            self.error.emit(f"Unexpected error: {exc}")

    def cancel(self) -> None:
        self._cancelled = True


class PhotogrammetryDialog(QDialog):
    """Multi-page dialog for photogrammetry pipeline."""

    STAGES = [
        "validation", "feature_extraction", "feature_matching",
        "sparse_reconstruction", "dense_undistortion", "dense_stereo",
        "dense_fusion", "scale_detection", "cleaning", "meshing", "decimation", "output",
    ]

    def __init__(
        self,
        parent: Optional[QWidget],
        mode: str = "default",
        photo_folder: Optional[Path] = None,
        output_path: Optional[Path] = None,
        artefact_label: str = "",
    ) -> None:
        super().__init__(parent)
        self._mode = mode
        self._result: Optional[PhotogrammetryResult] = None
        self._worker: Optional[PhotogrammetryWorker] = None

        self.setWindowTitle(f"Photogrammetry \u2014 {mode.capitalize()}")
        self.setMinimumWidth(520)
        self.setModal(True)

        # Store initial values
        self._photo_folder = photo_folder
        self._output_path = output_path
        self._artefact_label = artefact_label

        layout = QVBoxLayout(self)
        self._stack = QStackedWidget()
        layout.addWidget(self._stack)

        self._config_page = self._build_config_page()
        self._progress_page = self._build_progress_page()
        self._result_page = self._build_result_page()

        self._stack.addWidget(self._config_page)
        self._stack.addWidget(self._progress_page)
        self._stack.addWidget(self._result_page)

        if not colmap_available():
            self._show_colmap_warning()

    def _build_config_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        # Photo folder
        folder_row = QHBoxLayout()
        folder_row.addWidget(QLabel("Photos folder:"))
        self._folder_edit = QLineEdit()
        if self._photo_folder:
            self._folder_edit.setText(str(self._photo_folder))
        folder_row.addWidget(self._folder_edit)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_folder)
        folder_row.addWidget(browse_btn)
        layout.addLayout(folder_row)

        # Artefact label
        label_row = QHBoxLayout()
        label_row.addWidget(QLabel("Artefact label:"))
        self._label_edit = QLineEdit(self._artefact_label or "")
        label_row.addWidget(self._label_edit)
        layout.addLayout(label_row)

        # Quality radios
        quality_row = QHBoxLayout()
        quality_row.addWidget(QLabel("Mesh quality:"))
        self._quality_high = QRadioButton("High")
        self._quality_high.setChecked(True)
        self._quality_med = QRadioButton("Medium")
        self._quality_low = QRadioButton("Low")
        quality_row.addWidget(self._quality_high)
        quality_row.addWidget(self._quality_med)
        quality_row.addWidget(self._quality_low)
        quality_row.addStretch()
        layout.addLayout(quality_row)

        # Output path
        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("Output file:"))
        self._output_edit = QLineEdit()
        if self._output_path:
            self._output_edit.setText(str(self._output_path))
        output_row.addWidget(self._output_edit)
        save_btn = QPushButton("Save As...")
        save_btn.clicked.connect(self._browse_output)
        output_row.addWidget(save_btn)
        layout.addLayout(output_row)

        # Guided / Expert extras
        if self._mode in ("guided", "expert"):
            self._build_guided_settings(layout)
        if self._mode == "expert":
            self._build_expert_settings(layout)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._process_btn = QPushButton("Process")
        self._process_btn.clicked.connect(self._start_pipeline)
        self._process_btn.setDefault(True)
        btn_row.addWidget(self._process_btn)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        return page

    def _build_guided_settings(self, layout: QVBoxLayout) -> None:
        photo_group = QGroupBox("Photo settings")
        photo_form = QFormLayout(photo_group)
        self._camera_combo = QComboBox()
        self._camera_combo.addItems(["Auto-detect", "Smartphone", "DSLR"])
        photo_form.addRow("Camera:", self._camera_combo)
        self._scale_combo = QComboBox()
        self._scale_combo.addItems(["None", "3 cm", "5 cm", "10 cm"])
        photo_form.addRow("Scale reference:", self._scale_combo)
        layout.addWidget(photo_group)

        cleanup_group = QGroupBox("Cleanup")
        cleanup_layout = QVBoxLayout(cleanup_group)
        self._crop_check = QCheckBox("Auto-crop background")
        self._crop_check.setChecked(True)
        cleanup_layout.addWidget(self._crop_check)
        self._holes_check = QCheckBox("Fill holes")
        self._holes_check.setChecked(True)
        cleanup_layout.addWidget(self._holes_check)
        noise_row = QHBoxLayout()
        noise_row.addWidget(QLabel("Noise reduction:"))
        self._noise_combo = QComboBox()
        self._noise_combo.addItems(["Low", "Medium", "High"])
        self._noise_combo.setCurrentText("Medium")
        noise_row.addWidget(self._noise_combo)
        noise_row.addStretch()
        cleanup_layout.addLayout(noise_row)
        layout.addWidget(cleanup_group)

    def _build_expert_settings(self, layout: QVBoxLayout) -> None:
        expert_group = QGroupBox("COLMAP settings")
        expert_form = QFormLayout(expert_group)
        self._feature_combo = QComboBox()
        self._feature_combo.addItems(["SIFT"])
        expert_form.addRow("Feature type:", self._feature_combo)
        self._match_combo = QComboBox()
        self._match_combo.addItems(["Exhaustive", "Sequential", "Vocab Tree"])
        expert_form.addRow("Matching:", self._match_combo)
        self._dense_combo = QComboBox()
        self._dense_combo.addItems(["Low", "Medium", "High", "Extreme"])
        self._dense_combo.setCurrentText("Extreme")
        expert_form.addRow("Dense quality:", self._dense_combo)
        self._mesh_combo = QComboBox()
        self._mesh_combo.addItems(["Poisson", "Delaunay"])
        expert_form.addRow("Meshing:", self._mesh_combo)
        self._max_vertices_edit = QLineEdit("500000")
        expert_form.addRow("Max vertices:", self._max_vertices_edit)
        self._keep_temp_check = QCheckBox("Keep temporary files")
        expert_form.addRow(self._keep_temp_check)
        self._crop_margin_combo = QComboBox()
        self._crop_margin_combo.addItems(["1.0x", "1.5x", "2.0x", "3.0x"])
        self._crop_margin_combo.setCurrentText("1.5x")
        expert_form.addRow("Crop margin:", self._crop_margin_combo)
        layout.addWidget(expert_group)

    def _build_progress_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        title = QLabel("Processing...")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(title)

        self._stage_widgets: dict[str, tuple[QLabel, QProgressBar]] = {}
        for stage in self.STAGES:
            row = QHBoxLayout()
            label = QLabel(f"\u25cb {stage.replace('_', ' ').title()}")
            label.setMinimumWidth(250)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setMaximumWidth(200)
            row.addWidget(label)
            row.addWidget(bar)
            row.addStretch()
            layout.addLayout(row)
            self._stage_widgets[stage] = (label, bar)

        layout.addStretch()
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._cancel_pipeline_btn = QPushButton("Cancel")
        self._cancel_pipeline_btn.clicked.connect(self._cancel_pipeline)
        btn_row.addWidget(self._cancel_pipeline_btn)
        layout.addLayout(btn_row)
        return page

    def _build_result_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        title = QLabel("Photogrammetry Complete")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(title)
        self._result_details = QLabel("")
        layout.addWidget(self._result_details)
        layout.addStretch()
        btn_row = QHBoxLayout()
        self._open_viewer_btn = QPushButton("Open in Viewer")
        self._open_viewer_btn.clicked.connect(self._open_in_viewer)
        btn_row.addWidget(self._open_viewer_btn)
        self._save_as_btn = QPushButton("Save Mesh As...")
        self._save_as_btn.clicked.connect(self._save_mesh_as)
        btn_row.addWidget(self._save_as_btn)
        self._scale_btn = QPushButton("Set Scale Manually...")
        self._scale_btn.clicked.connect(self._open_scale_mode)
        btn_row.addWidget(self._scale_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)
        return page

    # ── Actions ──

    def _browse_folder(self) -> None:
        dir_str = QFileDialog.getExistingDirectory(self, "Select Photo Folder")
        if dir_str:
            self._folder_edit.setText(dir_str)

    def _browse_output(self) -> None:
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Save Mesh As", "",
            "PLY Mesh (*.ply);;OBJ Mesh (*.obj);;STL Mesh (*.stl)",
        )
        if path_str:
            self._output_edit.setText(path_str)

    def _get_config(self) -> PhotogrammetryConfig:
        photo_folder = Path(self._folder_edit.text())
        output_path = Path(self._output_edit.text())
        label = self._label_edit.text() or photo_folder.stem
        quality = "high"
        if self._quality_low.isChecked():
            quality = "low"
        elif self._quality_med.isChecked():
            quality = "medium"

        config = PhotogrammetryConfig(
            photo_folder=photo_folder,
            output_path=output_path,
            artefact_label=label,
            quality=quality,
            mode=self._mode,
        )

        if self._mode in ("guided", "expert"):
            config.auto_crop_background = self._crop_check.isChecked()
            config.fill_holes = self._holes_check.isChecked()

        if self._mode == "expert":
            config.colmap_feature_type = self._feature_combo.currentText().lower()
            config.colmap_matching_strategy = (
                self._match_combo.currentText().lower().replace(" ", "_")
            )
            config.colmap_dense_quality = self._dense_combo.currentText().lower()
            config.colmap_meshing = self._mesh_combo.currentText().lower()
            try:
                config.max_vertices = int(self._max_vertices_edit.text())
            except ValueError:
                pass
            config.cleanup_temp = not self._keep_temp_check.isChecked()

        return config

    def _start_pipeline(self) -> None:
        config = self._get_config()
        self._stack.setCurrentIndex(1)

        for stage, (label, bar) in self._stage_widgets.items():
            label.setText(f"\u25cb {stage.replace('_', ' ').title()}")
            bar.setValue(0)

        self._worker = PhotogrammetryWorker(config)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_pipeline_finished)
        self._worker.error.connect(self._on_pipeline_error)
        self._worker.start()

    def _on_progress(self, stage: str, progress: float, message: str) -> None:
        if stage in self._stage_widgets:
            label, bar = self._stage_widgets[stage]
            label.setText(f"\u25cf {stage.replace('_', ' ').title()}")
            bar.setValue(int(progress * 100))

    def _on_pipeline_finished(self, result: object) -> None:
        pr = result  # type: PhotogrammetryResult
        self._result = pr

        for stage, (label, bar) in self._stage_widgets.items():
            label.setText(f"\u2713 {stage.replace('_', ' ').title()}")
            bar.setValue(100)

        details = (
            f"Artefact: {pr.artefact_label}\n"
            f"Photos:   {pr.camera_count}\n"
            f"Mesh:     {pr.face_count:,} faces, {pr.vertex_count:,} vertices\n"
            f"Time:     {pr.processing_time_s:.0f}s"
        )
        if pr.warnings:
            details += "\n\nWarnings:\n" + "\n".join(f"  \u2022 {w}" for w in pr.warnings)

        self._result_details.setText(details)
        self._stack.setCurrentIndex(2)

        # Show scale button only if scale was not detected
        has_scale_warning = any("No scale reference" in w for w in pr.warnings)
        self._scale_btn.setVisible(has_scale_warning)

    def _on_pipeline_error(self, error_msg: str) -> None:
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Photogrammetry Error", error_msg)
        self._stack.setCurrentIndex(0)

    def _cancel_pipeline(self) -> None:
        if self._worker:
            self._worker.cancel()
        self._cancel_pipeline_btn.setEnabled(False)
        self._cancel_pipeline_btn.setText("Cancelling...")

    def closeEvent(self, event) -> None:  # type: ignore
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(3000)
        super().closeEvent(event)

    def _open_in_viewer(self) -> None:
        if self._result:
            from lithicore._models import MeasurementConfig
            parent = self.parent()
            while parent is not None:
                if hasattr(parent, '_process_single'):
                    parent._process_single(
                        self._result.mesh_path,
                        MeasurementConfig(),
                    )
                    break
                parent = parent.parent()
            self.accept()

    def _save_mesh_as(self) -> None:
        if not self._result:
            return
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Save Mesh As", str(self._result.mesh_path),
            "PLY Mesh (*.ply);;OBJ Mesh (*.obj);;STL Mesh (*.stl)",
        )
        if path_str:
            import shutil
            shutil.copy2(self._result.mesh_path, path_str)

    def _open_scale_mode(self) -> None:
        """Open mesh in viewer in scale measurement mode."""
        if self._result:
            parent = self.parent()
            while parent is not None:
                if hasattr(parent, 'viewer'):
                    viewer = parent.viewer
                    viewer.enable_scale_mode(self._on_scale_complete)
                    break
                parent = parent.parent()
            self.accept()

    def _on_scale_complete(self, ax: float, ay: float, az: float,
                           bx: float, by: float, bz: float) -> None:
        """Callback when user has picked two points in scale mode."""
        from PyQt6.QtWidgets import QInputDialog, QMessageBox
        import numpy as np
        import trimesh

        # Calculate distance between points in arbitrary units
        dist_arb = float(np.linalg.norm([bx - ax, by - ay, bz - az]))
        if dist_arb <= 0:
            QMessageBox.warning(None, "Scale Error",
                                "Points are too close together. Try again.")
            return

        # Ask user for real-world distance
        mm, ok = QInputDialog.getDouble(
            None, "Set Scale",
            f"Distance between points: {dist_arb:.2f} arbitrary units\n"
            "Enter real-world distance in millimetres:",
            decimals=2, min=0.01, max=10000.0,
        )
        if not ok or mm <= 0:
            return

        scale_factor = mm / dist_arb

        # Find the main window and apply scale to the mesh
        parent = self.parent()
        while parent is not None:
            if hasattr(parent, 'viewer') and hasattr(parent, '_mesh'):
                mesh = parent._mesh
                if mesh is not None:
                    from lithicore._scale_detection import apply_scale_to_mesh
                    scaled = apply_scale_to_mesh(mesh, scale_factor)
                    parent._mesh = scaled
                    parent.viewer.display_mesh(scaled)
                    QMessageBox.information(
                        None, "Scale Set",
                        f"Scale factor: {scale_factor:.4f}\n"
                        f"Distance set to {mm:.1f} mm",
                    )
                break
            parent = parent.parent()

    def _show_colmap_warning(self) -> None:
        from PyQt6.QtWidgets import QMessageBox
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle("COLMAP Not Found")
        msg.setText(
            "COLMAP is required for photogrammetry but was not found.\n\n"
            "Install it:\n"
            "  macOS:    brew install colmap\n"
            "  Linux:    sudo apt install colmap\n"
            "  Conda:    conda install -c conda-forge colmap\n\n"
            "Then restart this dialog."
        )
        msg.open()
