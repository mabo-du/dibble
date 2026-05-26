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
    Also supports interactive 3D landmark placement via point picking
    and scar overlay visualisation.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(400, 300)

        self._mesh: Optional[trimesh.Trimesh] = None
        self._pv_mesh: Optional[pv.PolyData] = None
        self._mesh_b: Optional[trimesh.Trimesh] = None
        self._main_mesh_actor: Optional[tuple] = None
        self._compare_mesh_actor: Optional[tuple] = None
        self._edge_actor: Optional[tuple] = None
        self._landmark_actors: list = []
        self._landmark_callback = None
        self._scale_points: list = []
        self._scale_actors: list = []
        self._scale_callback = None
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
        self._pv_mesh = pv_mesh

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
        self._pv_mesh = pv_a

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

    # ── Landmark placement ─────────────────────────────────────

    def enable_landmark_mode(self, callback) -> None:
        """Enable click-to-place landmark mode.

        The callback receives (x, y, z) tuple of the clicked mesh point.
        """
        if not HAS_PYVISTAQT or self._pv_mesh is None:
            return

        self._landmark_callback = callback

        def _on_pick(picked_point):
            """Handle mesh click — extract 3D coordinate from the picked point."""
            if picked_point is None:
                return
            # picked_point is a 3D numpy array of the clicked location
            if self._landmark_callback:
                self._landmark_callback(
                    float(picked_point[0]),
                    float(picked_point[1]),
                    float(picked_point[2]),
                )

        self.plotter.enable_point_picking(
            callback=_on_pick,
            show_message="Click on the mesh to place a landmark",
            use_mesh=True,
            pickable=True,
        )

    def disable_landmark_mode(self) -> None:
        """Disable landmark placement mode."""
        if not HAS_PYVISTAQT:
            return
        # Disable picker by clearing the interaction style
        self.plotter.disable_picking()
        self._landmark_callback = None

    def refresh_landmarks(self, landmarks: list) -> None:
        """Refresh displayed landmark spheres from a list of Landmark objects."""
        # Remove old landmark actors
        for actor in self._landmark_actors:
            self.plotter.remove_actor(actor, render=False)
        self._landmark_actors.clear()

        # Add new landmark spheres
        from lithicore._models import Landmark
        for i, lm in enumerate(landmarks):
            sphere = pv.Sphere(radius=max(1.0, self._pv_mesh.length * 0.01),
                               center=[lm.x, lm.y, lm.z])
            actor = self.plotter.add_mesh(
                sphere,
                color="red",
                smooth_shading=True,
                opacity=0.9,
            )
            # Add label
            label_actor = self.plotter.add_point_labels(
                np.array([[lm.x, lm.y, lm.z]]),
                [f"{i + 1}: {lm.name}"],
                point_size=0.01,
                font_size=10,
                text_color="black",
                shape="rect",
                fill_shape=False,
            )
            self._landmark_actors.append(actor)
            self._landmark_actors.append(label_actor)

        self.plotter.render()

    def clear_landmarks(self) -> None:
        """Remove all landmark actors."""
        for actor in self._landmark_actors:
            self.plotter.remove_actor(actor, render=False)
        self._landmark_actors.clear()
        self.disable_landmark_mode()
        self.plotter.render()

    # ── Scale measurement mode ─────────────────────────────────

    def enable_scale_mode(self, complete_callback) -> None:
        """Enable click-two-points scale measurement mode.

        First click: place green sphere at Point A.
        Second click: place red sphere at Point B, draw a line,
                      emit (point_a, point_b) to the callback.

        The callback receives (ax, ay, az, bx, by, bz) as floats.
        """
        if not HAS_PYVISTAQT or self._pv_mesh is None:
            return

        self._scale_points = []
        self._scale_actors = []
        self._scale_callback = complete_callback

        def _on_pick(picked_point):
            if picked_point is None:
                return
            pt = (float(picked_point[0]), float(picked_point[1]), float(picked_point[2]))
            self._scale_points.append(pt)

            # Draw sphere at the clicked point
            sphere = pv.Sphere(
                radius=max(0.5, self._pv_mesh.length * 0.008),
                center=[pt[0], pt[1], pt[2]],
            )
            color = "green" if len(self._scale_points) == 1 else "red"
            actor = self.plotter.add_mesh(sphere, color=color, smooth_shading=True)
            self._scale_actors.append(actor)

            if len(self._scale_points) == 1:
                # Show message for second click
                self.plotter.add_text(
                    "Now click a second point with known distance",
                    position="lower_edge",
                    font_size=12,
                    color="green",
                )
                self.plotter.render()
            elif len(self._scale_points) == 2:
                # Draw line between the two points
                a, b = self._scale_points[0], self._scale_points[1]
                line = pv.Line(a, b)
                line_actor = self.plotter.add_mesh(
                    line, color="yellow", line_width=3, lighting=False
                )
                self._scale_actors.append(line_actor)
                self.plotter.render()

                # Disable picking and fire callback
                self.plotter.disable_picking()
                # Remove the helper text
                # Can't easily remove specific text, will be cleared on next render
                self._scale_callback(a[0], a[1], a[2], b[0], b[1], b[2])

        self.plotter.enable_point_picking(
            callback=_on_pick,
            show_message="Click the first reference point on the mesh",
            use_mesh=True,
            pickable=True,
        )

    def disable_scale_mode(self) -> None:
        """Remove scale measurement overlays and disable picking."""
        if not HAS_PYVISTAQT:
            return
        for actor in self._scale_actors:
            self.plotter.remove_actor(actor, render=False)
        self._scale_actors.clear()
        self._scale_points = []
        self._scale_callback = None
        self.plotter.disable_picking()
        self.plotter.render()

    # ── Scar overlay ───────────────────────────────────────────

    def show_scar_overlay(self, face_labels: np.ndarray) -> None:
        """Colour mesh faces by scar label for visualisation.

        Each scar gets a distinct colour from a preset palette.
        Non-scar faces remain light gray.
        """
        if not HAS_PYVISTAQT or self._pv_mesh is None:
            return

        # Generate colours for each scar
        scar_ids = np.unique(face_labels[face_labels >= 0])
        palette = [
            (0.9, 0.2, 0.2),  # Red
            (0.2, 0.6, 0.9),  # Blue
            (0.2, 0.8, 0.3),  # Green
            (0.9, 0.6, 0.1),  # Orange
            (0.6, 0.2, 0.8),  # Purple
            (0.9, 0.4, 0.6),  # Pink
            (0.3, 0.7, 0.7),  # Teal
            (0.7, 0.5, 0.2),  # Brown
        ]

        # Build per-face colour array
        n_faces = len(face_labels)
        face_colors = np.ones((n_faces, 3), dtype=float) * 0.85  # light gray base

        for i, sid in enumerate(scar_ids):
            scar_faces = np.where(face_labels == sid)[0]
            colour = palette[i % len(palette)]
            face_colors[scar_faces] = colour

        # Apply to the mesh
        self._pv_mesh.face_data["scar_labels"] = face_labels
        if self._main_mesh_actor is not None:
            self.plotter.remove_actor(self._main_mesh_actor, render=False)

        self._main_mesh_actor = self.plotter.add_mesh(
            self._pv_mesh,
            scalars=face_colors,
            rgb=True,
            show_edges=True,
            edge_color="black",
            line_width=0.5,
            smooth_shading=True,
            lighting=True,
        )
        self.plotter.render()

    def clear_scar_overlay(self) -> None:
        """Restore default mesh appearance."""
        if not HAS_PYVISTAQT or self._pv_mesh is None:
            return
        # Re-add mesh with default appearance
        if self._main_mesh_actor is not None:
            self.plotter.remove_actor(self._main_mesh_actor, render=False)
        self._main_mesh_actor = self.plotter.add_mesh(
            self._pv_mesh,
            color="lightgray",
            show_edges=False,
            smooth_shading=True,
            lighting=True,
            opacity=1.0,
        )
        self.plotter.render()

    # ── Scene management ────────────────────────────────────────

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
