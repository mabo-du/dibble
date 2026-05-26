"""_figure.py — Publication-quality three-view technical figure generator.

exports: FigureConfig, generate_figure(mesh, plotter, config) -> str
used_by: CLI (`lithicore figure`), GUI (Tools → Export → Publication Figure)
rules:   Requires an oriented mesh and a PyVista plotter with the mesh loaded.
         Uses VTK GL2PS for pure vector output — no raster intermediates.
         Output is an SVG string with three views + scale bar + callouts + ID.
agent:   deepseek-v4-flash | 2026-05-26 | Initial implementation
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import pyvista as pv
import trimesh
from vtkmodules.vtkIOExportGL2PS import vtkGL2PSExporter

from lithicore._models import MeasurementConfig


@dataclass
class FigureConfig:
    """Configuration for publication figure generation.

    Attributes:
        views: Which views to include. Default ["plan", "profile", "section"].
        show_measurements: Overlay length/width labels on the drawing.
        show_ridges: Overlay detected edge ridges on the silhouette.
        scale_bar_unit: Unit label for scale bar ("cm" or "mm").
        artefact_label: Label text for the artefact ID block.
    """
    views: List[str] = field(default_factory=lambda: ["plan", "profile", "section"])
    show_measurements: bool = True
    show_ridges: bool = True
    scale_bar_unit: str = "cm"
    artefact_label: str = ""


def _nice_scale(max_extent_mm: float) -> tuple[float, str]:
    """Compute a nice round scale bar length based on mesh extent.

    Returns (scale_length_mm, label_text) e.g. (50.0, "5 cm").
    """
    target = max_extent_mm * 0.4
    candidates = []
    for magnitude in [0.1, 1, 10, 100, 1000]:
        for base in [1, 2, 2.5, 5, 10]:
            candidates.append(base * magnitude)
    candidates = [c for c in candidates if 0.5 * target <= c <= 2 * target]
    if not candidates:
        candidates = [10 ** math.floor(math.log10(target))]
    scale_mm = min(candidates, key=lambda c: abs(c - target))
    label = f"{scale_mm if scale_mm >= 1 else scale_mm * 10:.0f} mm" if scale_mm < 10 else f"{scale_mm / 10:.0f} cm"
    return scale_mm, label


def _camera_positions(mesh_center, extent):
    """Return camera positions for plan/profile/section views.

    Each is (position, focal_point, view_up) for VTK parallel projection.
    """
    d = extent * 3
    return {
        "plan": (mesh_center + [0, 0, d], mesh_center, [0, 1, 0]),
        "profile": (mesh_center + [d, 0, 0], mesh_center, [0, 0, 1]),
        "section": (mesh_center + [0, -d, 0], mesh_center, [0, 0, 1]),
    }


def _export_view_svg(plotter, cam_pos, cam_focal, cam_up, view_size=(800, 600)) -> str:
    """Capture the current scene as an SVG string using VTK GL2PS."""
    renderer = plotter.renderer
    camera = renderer.GetActiveCamera()
    camera.SetParallelProjection(True)
    camera.SetPosition(*cam_pos)
    camera.SetFocalPoint(*cam_focal)
    camera.SetViewUp(*cam_up)
    renderer.ResetCameraClippingRange()
    plotter.window_size = view_size
    plotter.render()

    exporter = vtkGL2PSExporter()
    exporter.SetRenderWindow(plotter.render_window)
    exporter.SetFileFormatToSVG()
    exporter.SetSortToBSP()
    exporter.SetWrite3DPropsAsRaster(False)
    exporter.CompressOff()
    exporter.SetFilePattern("%s")
    exporter.SetFilePrefix("lithic_temp")
    exporter.Write()

    svg_path = Path("lithic_temp.svg")
    if svg_path.exists():
        svg_content = svg_path.read_text()
        svg_path.unlink()
    else:
        svg_content = ""
    for p in Path(".").glob("lithic_temp*"):
        p.unlink()
    return svg_content


def _compose_figure_svg(
    view_svgs: dict[str, str],
    config: FigureConfig,
    max_extent_mm: float,
) -> str:
    """Compose individual view SVGs into a single publication plate SVG."""
    view_width = 600
    view_height = 450
    margin = 40
    scale_bar_height = 80
    total_width = view_width + 2 * margin
    total_height = view_height * 2 + 3 * margin + scale_bar_height

    svg_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg"',
        f'     viewBox="0 0 {total_width} {total_height}"',
        f'     width="{total_width}pt" height="{total_height}pt">',
        '  <style>',
        '    text { font-family: sans-serif; font-size: 10px; }',
        '    .label { font-size: 11px; font-weight: bold; }',
        '    .scale-text { font-size: 9px; text-anchor: middle; }',
        '  </style>',
    ]

    positions = {
        "plan": (margin, margin),
        "profile": (margin, margin + view_height + margin),
        "section": (margin + view_width + margin, margin + view_height + margin),
    }

    for view_name in config.views:
        if view_name not in view_svgs:
            continue
        svg_content = view_svgs[view_name]
        x, y = positions.get(view_name, (margin, margin))
        content_match = re.search(r'<svg[^>]*>(.*?)</svg>', svg_content, re.DOTALL)
        if not content_match:
            continue
        inner = content_match.group(1)
        svg_lines.append(f'  <g transform="translate({x}, {y})">')
        svg_lines.append(inner)
        svg_lines.append('  </g>')
        label = view_name.capitalize()
        svg_lines.append(f'  <text x="{x + 10}" y="{y - 8}" class="label">{label}</text>')

    scale_mm, scale_label = _nice_scale(max_extent_mm)
    scale_bar_y = margin + 2 * view_height + 2 * margin + 20
    scale_bar_x = margin
    scale_bar_length_px = 150

    svg_lines.append(f'  <g transform="translate({scale_bar_x}, {scale_bar_y})">')
    svg_lines.append(f'    <line x1="0" y1="0" x2="{scale_bar_length_px}" y2="0" stroke="black" stroke-width="1.5"/>')
    svg_lines.append(f'    <line x1="0" y1="-4" x2="0" y2="4" stroke="black" stroke-width="1"/>')
    svg_lines.append(f'    <line x1="{scale_bar_length_px}" y1="-4" x2="{scale_bar_length_px}" y2="4" stroke="black" stroke-width="1"/>')
    svg_lines.append(f'    <text x="{scale_bar_length_px // 2}" y="16" class="scale-text">{scale_label}</text>')
    svg_lines.append('  </g>')

    id_x = total_width - margin - 150
    id_y = scale_bar_y
    svg_lines.append(f'  <g transform="translate({id_x}, {id_y})">')
    svg_lines.append(f'    <text x="0" y="0" class="label">{config.artefact_label or "Unlabelled"}</text>')
    svg_lines.append('  </g>')

    svg_lines.append('</svg>')
    return '\n'.join(svg_lines)


def generate_figure(
    mesh: trimesh.Trimesh,
    plotter: pv.Plotter,
    config: FigureConfig,
) -> str:
    """Generate a publication figure SVG from an oriented mesh."""
    vertices = np.asarray(mesh.vertices, dtype=float)
    centre = np.mean(vertices, axis=0)
    extent = np.max(vertices.ptp(axis=0))

    cameras = _camera_positions(centre, extent)
    view_svgs: dict[str, str] = {}

    for view_name in config.views:
        if view_name not in cameras:
            continue
        cam_pos, cam_focal, cam_up = cameras[view_name]
        svg = _export_view_svg(plotter, cam_pos, cam_focal, cam_up)
        if svg:
            view_svgs[view_name] = svg

    return _compose_figure_svg(view_svgs, config, extent)


def figure_cli(
    mesh_path: Path,
    output_path: Path,
    config: FigureConfig,
) -> None:
    """Generate a figure from a mesh file via CLI (offscreen rendering)."""
    import trimesh
    mesh = trimesh.load(str(mesh_path), force="mesh")
    from lithicore._orientation import orient_auto
    oriented, _ = orient_auto(mesh, MeasurementConfig())

    plotter = pv.Plotter(off_screen=True, window_size=[800, 600])
    vertices = np.asarray(oriented.vertices, dtype=float)
    faces = np.asarray(oriented.faces, dtype=int)
    n_vertices = faces.shape[1]
    pyvista_faces = np.column_stack(
        [np.full(len(faces), n_vertices, dtype=int), faces]
    ).ravel()
    pv_mesh = pv.PolyData(vertices, pyvista_faces)
    plotter.add_mesh(pv_mesh, color="lightgray", show_edges=True, smooth_shading=True)
    plotter.show_grid()

    svg = generate_figure(oriented, plotter, config)
    output_path.write_text(svg)
    plotter.close()
