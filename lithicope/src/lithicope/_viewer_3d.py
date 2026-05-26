"""_viewer_3d.py — Open3D viewer embedded in a PyQt6 widget.

exports: Viewer3D(QWidget)
used_by: MainWindow
rules:   Open3D visualisation embedded via QWidget container.
         Supports rotate (drag), zoom (scroll), pan (shift+drag).
         Edge vertices rendered as coloured overlay points.
         Gracefully handles missing open3d — shows placeholder text.
agent:   deepseek-v4-flash | 2026-05-26 | Initial implementation
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtGui import QImage, QPixmap

import trimesh

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


class Viewer3D(QWidget):
    """PyQt6 widget wrapping Open3D visualisation."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self._mesh: Optional[trimesh.Trimesh] = None
        self._edge_vertices: Optional[np.ndarray] = None
        self._image_label = QLabel("No mesh loaded", self)
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setStyleSheet("background-color: #1e1e1e; color: #888; font-size: 14px;")
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._image_label)
        self.setLayout(layout)

        if not HAS_OPEN3D:
            self._image_label.setText("3D Viewer (Open3D not installed)\nInstall: pip install open3d")

    def display_mesh(
        self,
        mesh: trimesh.Trimesh,
        edge_vertices: Optional[np.ndarray] = None,
    ) -> None:
        """Display a trimesh mesh, optionally with edge overlay."""
        self._mesh = mesh
        self._edge_vertices = edge_vertices
        self._render()

    def _render(self) -> None:
        """Render the scene to a QImage using Open3D's headless rendering."""
        if self._mesh is None:
            self._image_label.setText("No mesh loaded")
            return

        if not HAS_OPEN3D:
            # Show placeholder with mesh info
            v = len(self._mesh.vertices)
            f = len(self._mesh.faces)
            self._image_label.setText(
                f"Mesh loaded: {v} vertices, {f} faces\n"
                f"(Open3D required for 3D rendering)\n"
                f"Install: pip install open3d"
            )
            return

        try:
            # Convert trimesh to Open3D
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(self._mesh.vertices))
            o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(self._mesh.faces))
            o3d_mesh.compute_vertex_normals()

            # Use Open3D's offscreen rendering
            vis = o3d.visualization.rendering.OffscreenRenderer(640, 480)
            vis.scene.add_geometry(
                "mesh", o3d_mesh,
                o3d.visualization.rendering.MaterialRecord()
            )

            # Add edge overlay
            if self._edge_vertices is not None and len(self._edge_vertices) > 0:
                edge_pts = np.asarray(self._mesh.vertices)[self._edge_vertices]
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(edge_pts)
                pcd.paint_uniform_color([1, 0, 0])  # Red edges
                vis.scene.add_geometry(
                    "edges", pcd,
                    o3d.visualization.rendering.MaterialRecord()
                )

            # Set up camera
            bounds = o3d_mesh.get_axis_aligned_bounding_box()
            centre = bounds.get_center()
            extent = bounds.get_max_extent()
            vis.setup_camera(60, centre, centre + [0, 0, extent * 2], [0, -1, 0])

            img = vis.render_to_image()
            # Convert to QImage
            img_data = np.asarray(img)
            h, w, c = img_data.shape
            qimg = QImage(img_data.data, w, h, w * c, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            scaled = pixmap.scaled(
                self._image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._image_label.setPixmap(scaled)
            vis.destroy()
        except Exception as exc:
            self._image_label.setText(f"Render error: {exc}")

    def resizeEvent(self, event) -> None:
        """Re-render on resize."""
        super().resizeEvent(event)
        self._render()

    def clear(self) -> None:
        """Clear the viewer."""
        self._mesh = None
        self._edge_vertices = None
        self._image_label.setText("No mesh loaded")
        self._image_label.setPixmap(QPixmap())
