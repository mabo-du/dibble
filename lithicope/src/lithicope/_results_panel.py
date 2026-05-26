"""_results_panel.py — Measurement results display and export panel.

exports: ResultsPanel(QWidget)
used_by: MainWindow right panel
rules:   Table shows measurement name, value, unit, confidence.
         Export buttons for CSV, JSON, MorphoJ, PDF.
         Uses pyqtSignal to request export from main window.
agent:   deepseek-v4-flash | 2026-05-26 | Initial implementation
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog,
    QGroupBox, QMessageBox, QComboBox,
)
from PyQt6.QtGui import QFont

from lithicore._models import MeasurementResult, MeshGrade


class ResultsPanel(QWidget):
    """Right-side panel showing measurements and export controls."""

    export_requested = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._current_measurements: List[MeasurementResult] = []
        self._current_label: str = ""
        self._current_grade: MeshGrade = MeshGrade.PASS

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Artefact info
        self._info_label = QLabel("No artefact loaded")
        self._info_label.setWordWrap(True)
        font = QFont()
        font.setPointSize(11)
        font.setBold(True)
        self._info_label.setFont(font)
        layout.addWidget(self._info_label)

        # Quality badge
        self._quality_label = QLabel("")
        layout.addWidget(self._quality_label)

        # Measurements table
        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Measurement", "Value", "Confidence"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self._table)

        layout.addStretch()

        # Export section
        export_group = QGroupBox("Export")
        export_layout = QHBoxLayout(export_group)

        self._export_combo = QComboBox()
        self._export_combo.addItems(["CSV", "JSON", "MorphoJ", "PDF"])
        export_layout.addWidget(self._export_combo)

        export_btn = QPushButton("Export")
        export_btn.clicked.connect(self._on_export_clicked)
        export_layout.addWidget(export_btn)

        layout.addWidget(export_group)

    def show_measurements(
        self,
        measurements: List[MeasurementResult],
        label: str,
        grade: MeshGrade,
    ) -> None:
        """Populate the table with measurement data."""
        self._current_measurements = measurements
        self._current_label = label
        self._current_grade = grade

        self._info_label.setText(f"Artefact: {label}")

        quality_text = f"Quality: {grade.value}"
        if grade == MeshGrade.PASS:
            self._quality_label.setStyleSheet("color: green;")
        elif grade == MeshGrade.WARN:
            self._quality_label.setStyleSheet("color: orange;")
        else:
            self._quality_label.setStyleSheet("color: red;")
        self._quality_label.setText(quality_text)

        self._table.setRowCount(len(measurements))
        for i, m in enumerate(measurements):
            name_item = QTableWidgetItem(m.name.replace("_", " ").title())
            value_item = QTableWidgetItem(f"{m.value} {m.unit}")
            conf_item = QTableWidgetItem(f"{m.confidence:.0%}")
            self._table.setItem(i, 0, name_item)
            self._table.setItem(i, 1, value_item)
            self._table.setItem(i, 2, conf_item)

    def _on_export_clicked(self) -> None:
        """Handle export button click."""
        fmt = self._export_combo.currentText().lower()
        self.export_requested.emit(fmt)

    def export_results(self, measurements: List[MeasurementResult], fmt: str) -> None:
        """Export measurement results to a file."""
        if not measurements:
            QMessageBox.information(self, "No Data", "No measurements to export.")
            return

        ext_map = {"csv": "csv", "json": "json", "morphoj": "txt", "pdf": "pdf"}
        ext = ext_map.get(fmt, "csv")
        default_name = f"{self._current_label}_measurements.{ext}"

        path_str, _ = QFileDialog.getSaveFileName(
            self, f"Export as {fmt.upper()}", default_name,
            f"{fmt.upper()} Files (*.{ext})",
        )
        if not path_str:
            return
        path = Path(path_str)

        if fmt == "csv":
            self._export_csv(path, measurements)
        elif fmt == "json":
            self._export_json(path, measurements)
        elif fmt == "morphoj":
            self._export_morphoj(path, measurements)
        elif fmt == "pdf":
            self._export_pdf(path, measurements)

        QMessageBox.information(self, "Exported", f"Results saved to:\n{path}")

    def _export_csv(self, path: Path, measurements: List[MeasurementResult]) -> None:
        """Export as CSV."""
        import csv
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["measurement", "value", "unit", "confidence"])
            for m in measurements:
                writer.writerow([m.name, m.value, m.unit, m.confidence])

    def _export_json(self, path: Path, measurements: List[MeasurementResult]) -> None:
        """Export as JSON."""
        import json
        data = {
            "artefact": self._current_label,
            "quality": self._current_grade.value,
            "measurements": [
                {"name": m.name, "value": m.value, "unit": m.unit, "confidence": m.confidence}
                for m in measurements
            ],
        }
        path.write_text(json.dumps(data, indent=2))

    def _export_morphoj(self, path: Path, measurements: List[MeasurementResult]) -> None:
        """Export in MorphoJ-compatible landmark format."""
        lines = [f"# MorphoJ export — {self._current_label}"]
        lines.append(f"# Generated by Lithic Analysis Platform v0.1")
        lines.append(f"# Measurements: {len(measurements)}")
        lines.append("")
        lines.append("LRMM 3D")  # Landmark data marker
        lines.append(f"{self._current_label}")
        lines.append("1")  # Single specimen
        lines.append(f"{len(measurements)}")  # Number of landmarks
        for i, m in enumerate(measurements):
            lines.append(f"{i+1} {m.value:.3f} {0.0:.3f} {0.0:.3f}")
        path.write_text("\n".join(lines))

    def _export_pdf(self, path: Path, measurements: List[MeasurementResult]) -> None:
        """Export as PDF report."""
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer,
        )
        from reportlab.lib.styles import getSampleStyleSheet

        doc = SimpleDocTemplate(str(path), pagesize=A4)
        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph(f"Lithic Analysis Report", styles["Title"]))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(f"Artefact: {self._current_label}", styles["Heading2"]))
        elements.append(Paragraph(f"Quality: {self._current_grade.value}", styles["Normal"]))
        elements.append(Spacer(1, 12))

        table_data = [["Measurement", "Value", "Confidence"]]
        for m in measurements:
            table_data.append([
                m.name.replace("_", " ").title(),
                f"{m.value} {m.unit}",
                f"{m.confidence:.0%}",
            ])

        t = Table(table_data, colWidths=[150, 100, 80])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 20))
        elements.append(Paragraph("Generated by Lithic Analysis Platform v0.1", styles["Italic"]))

        doc.build(elements)
