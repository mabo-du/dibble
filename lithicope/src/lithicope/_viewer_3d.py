"""_viewer_3d.py — 3D mesh viewer using PyVista embedded in a PyQt6 widget.

exports: Viewer3D(QWidget)
used_by: MainWindow
rules:   Uses PyVista (VTK) for interactive 3D rendering.
         Supports rotate (drag), zoom (scroll), pan (right-drag/ctrl+shift).
         Edge vertices rendered as coloured overlay points.
         Embedded directly as an interactive QWidget (not screenshot-based).
agent:   deepseek-v4-flash | 2026-05-26 | Switched to PyVista backend
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel

import pyvista as pv
import trimesh

try:
    from pyvistaqt import QtInteractor
    HAS_PYVISTAQT = True
except Exception:
    HAS_PYVISTAQT = False


class Viewer3D(QWidget):
    """PyQt6 widget wrapping a PyVista interactive 3D viewport.

    Supports single mesh display and dual-mesh comparison overlay.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(400, 300)

        self._mesh: Optional[trimesh.Trimesh] = None
        self._mesh_b: Optional[trimesh.Trimesh] = None
        self._main_mesh_actor: Optional[tuple] = None
        self._compare_mesh_actor: Optional[tuple] = None
        self._edge_actor: Optional[tuple] = None
        self._placeholder: Optional[QLabel] = None

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        if HAS_PYVISTAQT:
            self.plotter = QtInteractor(self)
            self.plotter.set_background("white")
            layout.addWidget(self.plotter)
        else:
            self._placeholder = QLabel(
                "3D Viewer unavailable\n"
                "(pyvistaqt not available — "
                "install display dependencies)")
            self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._placeholder.setStyleSheet(
                "background-color: #1e1e1e; color: #888; font-size: 14px;")
            layout.addWidget(self._placeholder)

        self.setLayout(layout)

    def _mesh_to_polydata(self, mesh: trimesh.Trimesh) -> pv.PolyData:
        """Convert a trimesh mesh to PyVista PolyData."""
        vertices = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces, dtype=int)
        n_vertices = faces.shape[1]
        pyvista_faces = np.column_stack(
            [np.full(len(faces), n_vertices, dtype=int), faces]
        ).ravel()
        return pv.PolyData(vertices, pyvista_faces)

    def display_mesh(
        self,
        mesh: trimesh.Trimesh,
        edge_vertices: Optional[np.ndarray] = None,
    ) -> None:
        """Display a trimesh mesh, optionally with edge overlay.
        Clears any comparison overlay."""
        if not HAS_PYVISTAQT:
            return

        self.clear_comparison()
        self._clear_scene()

        self._mesh = mesh
        pv_mesh = self._mesh_to_polydata(mesh)

        self._main_mesh_actor = self.plotter.add_mesh(
            pv_mesh,
            color="lightgray",
            show_edges=False,
            smooth_shading=True,
            lighting=True,
            opacity=1.0,
        )

        if edge_vertices is not None and len(edge_vertices) > 0:
            edge_pts = np.asarray(mesh.vertices)[edge_vertices]
            if len(edge_pts) > 0:
                cloud = pv.PolyData(edge_pts)
                self._edge_actor = self.plotter.add_points(
                    cloud,
                    color="red",
                    point_size=5.0,
                    render_points_as_spheres=True,
                )

        self._reset_camera(pv_mesh)

    def display_comparison(
        self,
        mesh_a: trimesh.Trimesh,
        mesh_b: trimesh.Trimesh,
        edge_vertices_a: Optional[np.ndarray] = None,
    ) -> None:
        """Display two meshes overlaid for shape comparison.
        Mesh A in gray, Mesh B in translucent blue."""
        if not HAS_PYVISTAQT:
            return

        self._clear_scene()

        self._mesh = mesh_a
        self._mesh_b = mesh_b
        pv_a = self._mesh_to_polydata(mesh_a)
        pv_b = self._mesh_to_polydata(mesh_b)

        # Mesh A — solid gray
        self._main_mesh_actor = self.plotter.add_mesh(
            pv_a,
            color="lightgray",
            show_edges=False,
            smooth_shading=True,
            lighting=True,
            opacity=1.0,
        )

        # Mesh B — translucent orange/red for contrast
        self._compare_mesh_actor = self.plotter.add_mesh(
            pv_b,
            color="#e67e22",
            show_edges=False,
            smooth_shading=True,
            lighting=True,
            opacity=0.5,
        )

        # Edge overlay for Mesh A (optional)
        if edge_vertices_a is not None and len(edge_vertices_a) > 0:
            edge_pts = np.asarray(mesh_a.vertices)[edge_vertices_a]
            if len(edge_pts) > 0:
                cloud = pv.PolyData(edge_pts)
                self._edge_actor = self.plotter.add_points(
                    cloud,
                    color="red",
                    point_size=5.0,
                    render_points_as_spheres=True,
                )

        self._reset_camera(pv_a)

    def set_overlay_opacity(self, value: float) -> None:
        """Set opacity of the comparison overlay mesh (0.0–1.0)."""
        if self._compare_mesh_actor is not None:
            self._compare_mesh_actor.GetProperty().SetOpacity(value)
            self.plotter.render()

    def _reset_camera(self, pv_mesh: pv.PolyData) -> None:
        """Reset camera to a nice initial view."""
        centre = pv_mesh.center
        vertices = np.asarray(pv_mesh.points)
        extent = np.max(vertices.ptp(axis=0)) if len(vertices) > 0 else 10.0
        camera_dist = extent * 2.0 if extent > 0 else 10.0
        self.plotter.camera_position = [
            centre + [0, camera_dist * 0.7, camera_dist * 0.7],
            centre,
            [0, 0, 1],
        ]
        self.plotter.reset_camera()
        self.plotter.render()

    def _clear_scene(self) -> None:
        """Remove all actors from the scene."""
        if not HAS_PYVISTAQT:
            return
        for actor in [self._main_mesh_actor, self._compare_mesh_actor, self._edge_actor]:
            if actor is not None:
                self.plotter.remove_actor(actor, render=False)
        self._main_mesh_actor = None
        self._compare_mesh_actor = None
        self._edge_actor = None
        self._mesh = None
        self._mesh_b = None

    def clear_comparison(self) -> None:
        """Remove only the comparison overlay mesh, keep the main mesh."""
        if not HAS_PYVISTAQT:
            return
        if self._compare_mesh_actor is not None:
            self.plotter.remove_actor(self._compare_mesh_actor, render=False)
            self._compare_mesh_actor = None
        self._mesh_b = None
        if self._main_mesh_actor is not None:
            self.plotter.render()

    def clear(self) -> None:
        """Clear the viewer entirely."""
        if not HAS_PYVISTAQT:
            return
        self._clear_scene()
        self.plotter.clear()
        self.plotter.render()
