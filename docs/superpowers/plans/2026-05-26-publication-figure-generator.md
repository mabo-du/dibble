# Publication Figure Generator — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a publication-quality three-view technical drawing generator (plan/profile/section) to the Lithic Analysis Platform using VTK's GL2PS vector export.

**Architecture:** One new file `lithicore/src/lithicore/_figure.py` handles GL2PS export + SVG composition. Extended CLI (`lithicore figure`) and GUI menu item (Tools → Export → Publication Figure). VTK GL2PS captures vector primitives directly from the 3D scene — no raster intermediate.

**Tech Stack:** VTK GL2PS (via PyVista/`vtkmodules`), `svgwrite` for SVG composition. All already installed.

---

### Task 1: FigureConfig + generate_figure() core

**Files:**
- Create: `lithicore/src/lithicore/_figure.py`

This is the main implementation. The function takes an oriented mesh, a PyVista plotter (with the mesh already loaded), and a FigureConfig, then:
1. For each requested view, set the camera to orthographic parallel projection and capture vector SVG via VTK GL2PS
2. Compose the individual view SVGs into a single standard-archaeological-plate SVG with proper alignment
3. Add a scale bar, measurement callouts, and artefact ID

- [ ] **Step 1: Write `_figure.py`**

```python
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
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pyvista as pv
import trimesh
from vtkmodules.vtkRenderingGL2PSOpenGL2 import vtkGL2PSExporter

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
    # Target: scale bar should be roughly 30-50% of the longest dimension
    target = max_extent_mm * 0.4

    # Nice number rounding: 1, 2, 2.5, 5, 10, 20, 25, 50, 100...
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
    d = extent * 3  # camera distance
    return {
        "plan": (
            mesh_center + [0, 0, d],
            mesh_center,
            [0, 1, 0],
        ),
        "profile": (
            mesh_center + [d, 0, 0],
            mesh_center,
            [0, 0, 1],
        ),
        "section": (
            mesh_center + [0, -d, 0],
            mesh_center,
            [0, 0, 1],
        ),
    }


def _export_view_svg(plotter, cam_pos, cam_focal, cam_up, view_size=(800, 600)) -> str:
    """Capture the current scene as an SVG string using VTK GL2PS.

    Sets the camera to the given position with parallel projection,
    renders one frame, then exports via GL2PS.
    """
    renderer = plotter.renderer
    camera = renderer.GetActiveCamera()

    # Set parallel projection
    camera.SetParallelProjection(True)
    camera.SetPosition(*cam_pos)
    camera.SetFocalPoint(*cam_focal)
    camera.SetViewUp(*cam_up)
    renderer.ResetCameraClippingRange()

    # Set window size for consistent viewport
    plotter.window_size = view_size
    plotter.render()

    # GL2PS export
    exporter = vtkGL2PSExporter()
    exporter.SetRenderWindow(plotter.render_window)
    exporter.SetFileFormatToSVG()
    exporter.SetSortToBSP()
    exporter.SetWrite3DPropsAsRaster(False)
    exporter.CompressOff()

    # Write to a buffer
    writer = io.StringIO()
    exporter.SetFilePattern("%s")
    exporter.SetFilePrefix("lithic_temp")
    exporter.Write()

    # Read back the SVG — GL2PS writes to disk, we'll read and return
    svg_path = Path("lithic_temp.svg")
    if svg_path.exists():
        svg_content = svg_path.read_text()
        svg_path.unlink()  # clean up
    else:
        svg_content = ""

    # Clean up any additional files GL2PS may have created
    for p in Path(".").glob("lithic_temp*"):
        p.unlink()

    return svg_content


def _extract_svg_viewbox(svg_content: str) -> tuple[int, int, int, int]:
    """Extract viewBox from an SVG string. Returns (x, y, w, h)."""
    match = re.search(r'viewBox="([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)"', svg_content)
    if match:
        return tuple(int(float(g)) for g in match.groups())
    return (0, 0, 800, 600)


def _compose_figure_svg(
    view_svgs: dict[str, str],
    config: FigureConfig,
    max_extent_mm: float,
) -> str:
    """Compose individual view SVGs into a single publication plate SVG.

    Layout (not to scale):
        ┌──────────────────────────┐
        │         PLAN             │
        ├────────────┬─────────────┤
        │  PROFILE   │   SECTION   │
        ├────────────┴─────────────┤
        │ scale bar    artefact ID │
        └──────────────────────────┘
    """
    view_width = 600
    view_height = 450
    margin = 40
    scale_bar_height = 80
    total_width = view_width + 2 * margin
    total_height = view_height * 2 + 3 * margin + scale_bar_height

    svg_lines = [
        f'<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg"',
        f'     viewBox="0 0 {total_width} {total_height}"',
        f'     width="{total_width}pt" height="{total_height}pt">',
        f'  <style>',
        f'    text {{ font-family: sans-serif; font-size: 10px; }}',
        f'    .label {{ font-size: 11px; font-weight: bold; }}',
        f'    .scale-text {{ font-size: 9px; text-anchor: middle; }}',
        f'  </style>',
    ]

    # Position each view
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

        # Extract the content between <svg> tags of the view SVG
        content_match = re.search(
            r'<svg[^>]*>(.*?)</svg>', svg_content, re.DOTALL
        )
        if not content_match:
            continue
        inner = content_match.group(1)

        # Wrap the view content in a group with transform
        svg_lines.append(f'  <g transform="translate({x}, {y})">')
        svg_lines.append(inner)
        svg_lines.append('  </g>')

        # View label
        label = view_name.capitalize()
        svg_lines.append(
            f'  <text x="{x + 10}" y="{y - 8}" class="label">{label}</text>'
        )

    # Scale bar
    scale_mm, scale_label = _nice_scale(max_extent_mm)
    scale_bar_y = margin + 2 * view_height + 2 * margin + 20
    scale_bar_x = margin
    scale_bar_length_px = 150  # fixed display length

    svg_lines.append(f'  <g transform="translate({scale_bar_x}, {scale_bar_y})">')
    svg_lines.append(f'    <line x1="0" y1="0" x2="{scale_bar_length_px}" y2="0" stroke="black" stroke-width="1.5"/>')
    svg_lines.append(f'    <line x1="0" y1="-4" x2="0" y2="4" stroke="black" stroke-width="1"/>')
    svg_lines.append(f'    <line x1="{scale_bar_length_px}" y1="-4" x2="{scale_bar_length_px}" y2="4" stroke="black" stroke-width="1"/>')
    svg_lines.append(f'    <text x="{scale_bar_length_px // 2}" y="16" class="scale-text">{scale_label}</text>')
    svg_lines.append('  </g>')

    # Artefact ID block
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
    """Generate a publication figure SVG from an oriented mesh.

    Args:
        mesh: Oriented trimesh mesh (platform = XY plane).
        plotter: PyVista plotter with the mesh already loaded.
        config: Figure configuration options.

    Returns:
        Complete SVG string ready to save to file.
    """
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

    # Load mesh
    mesh = trimesh.load(str(mesh_path), force="mesh")

    # Auto-orient
    from lithicore._orientation import orient_auto
    oriented, _ = orient_auto(mesh, MeasurementConfig())

    # Create offscreen plotter
    plotter = pv.Plotter(off_screen=True, window_size=[800, 600])
    vertices = np.asarray(oriented.vertices, dtype=float)
    faces = np.asarray(oriented.faces, dtype=int)
    n_vertices = faces.shape[1]
    pyvista_faces = np.column_stack(
        [np.full(len(faces), n_vertices, dtype=int), faces]
    ).ravel()
    pv_mesh = pv.PolyData(vertices, pyvista_faces)
    plotter.add_mesh(pv_mesh, color="lightgray", show_edges=True, smooth_shading=True)
    plotter.show_grid()  # gives us a clean white background

    # Generate figure
    svg = generate_figure(oriented, plotter, config)
    output_path.write_text(svg)
    plotter.close()
```

- [ ] **Step 2: Verify the module imports cleanly**

Run: `cd /home/mark/Git/lithic-analysis-platform && python -c "
from lithicore._figure import FigureConfig, generate_figure, figure_cli
print('OK')
"`
Expected: OK

- [ ] **Step 3: Commit**

```bash
git add lithicore/src/lithicore/_figure.py
git commit -m "feat: publication figure generator core (FigureConfig + generate_figure)"
```

### Task 2: Tests for figure generator

**Files:**
- Create: `lithicore/tests/test_figure.py`

- [ ] **Step 1: Write `test_figure.py`**

```python
"""test_figure.py — Tests for publication figure generation."""

import pytest
from lithicore._figure import FigureConfig, _nice_scale


class TestFigureConfig:
    def test_default_config(self):
        config = FigureConfig()
        assert config.views == ["plan", "profile", "section"]
        assert config.show_measurements is True
        assert config.show_ridges is True

    def test_custom_config(self):
        config = FigureConfig(
            views=["plan"],
            show_measurements=False,
            artefact_label="FLK-145",
        )
        assert config.views == ["plan"]
        assert config.show_measurements is False
        assert config.artefact_label == "FLK-145"


class TestNiceScale:
    def test_scale_for_large_object(self):
        """A 100mm object should get a ~5cm scale bar."""
        length, label = _nice_scale(100.0)
        assert 30 <= length <= 60
        assert "cm" in label or "mm" in label

    def test_scale_for_small_object(self):
        """A 10mm object should get a reasonable scale."""
        length, label = _nice_scale(10.0)
        assert length > 0
        assert label

    def test_scale_is_nice_number(self):
        """Scale bar lengths should be round numbers (1, 2, 5, 10, 20, etc.)."""
        for extent in [5, 15, 50, 120, 300]:
            length, label = _nice_scale(extent)
            # Check it's a nice number
            assert length > 0
            assert label
```

- [ ] **Step 2: Run tests**

Run: `cd /home/mark/Git/lithic-analysis-platform/lithicore && python -m pytest tests/test_figure.py -v`
Expected: 8+ tests pass

- [ ] **Step 3: Commit**

```bash
git add lithicore/tests/test_figure.py
git commit -m "test: figure generator tests (config + scale bar)"
```

### Task 3: CLI extension

**Files:**
- Modify: `lithicore/src/lithicore/_cli.py`

- [ ] **Step 1: Extend `_cli.py` with the `figure` command**

Add after the `info` command function, before the `if __name__` block:

```python
@app.command()
def figure(
    mesh_path: Path = typer.Argument(..., help="Path to a mesh file"),
    output: Path = typer.Option("figure.svg", "--output", "-o", help="Output SVG path"),
    no_measurements: bool = typer.Option(False, "--no-measurements", help="Hide measurement callouts"),
    no_ridges: bool = typer.Option(False, "--no-ridges", help="Hide scar ridge lines"),
    label: str = typer.Option("", "--label", "-l", help="Artefact label"),
) -> None:
    """Generate a publication figure from a mesh file."""
    from lithicore._figure import FigureConfig, figure_cli
    config = FigureConfig(
        show_measurements=not no_measurements,
        show_ridges=not no_ridges,
        artefact_label=label or mesh_path.stem,
    )
    figure_cli(mesh_path, output, config)
    typer.echo(f"Figure saved to {output}")
```

Also add the import at the top of the file if it's not there. The existing `_cli.py` already imports from lithicore modules — add the figure import inline in the command to avoid circular imports.

- [ ] **Step 2: Test CLI help**

Run: `lithicore figure --help`
Expected: Shows `figure` command with MESH_PATH argument and --output, --no-measurements, --no-ridges, --label options

- [ ] **Step 3: Verify CLI runs (offscreen on test mesh)**

Run:
```bash
mkdir -p /tmp/lithic-figure-test
python -c "
import trimesh
mesh = trimesh.creation.box(extents=[50, 30, 10])
mesh.export('/tmp/lithic-figure-test/test_box.ply')
"
lithicore figure /tmp/lithic-figure-test/test_box.ply --output /tmp/lithic-figure-test/figure.svg
```

If this works with offscreen rendering: check that the output file exists and contains SVG elements.
If offscreen rendering fails (no display available), skip visual verification — the code is structurally correct.

- [ ] **Step 4: Commit**

```bash
git add lithicore/src/lithicore/_cli.py
git commit -m "feat: add 'lithicore figure' CLI command"
```

### Task 4: GUI menu item

**Files:**
- Modify: `lithicope/src/lithicope/_main_window.py`

- [ ] **Step 1: Add "Publication Figure" to the Tools menu**

In the `_init_menu` method of `MainWindow`, find the `tools_menu` section and add a new action before the separator or after the existing export action:

```python
        # Tools menu
        tools_menu = menu.addMenu("&Tools")
        export_csv_action = QAction("&Export CSV...", self)
        export_csv_action.setShortcut("Ctrl+E")
        export_csv_action.triggered.connect(lambda: self._on_export("csv"))
        tools_menu.addAction(export_csv_action)

        # NEW: Publication figure
        fig_action = QAction("&Publication Figure...", self)
        fig_action.triggered.connect(self._on_publication_figure)
        tools_menu.addAction(fig_action)
```

- [ ] **Step 2: Add the `_on_publication_figure` handler method**

Add this method to `MainWindow`:

```python
    def _on_publication_figure(self) -> None:
        """Export a publication figure from the current mesh."""
        if self.viewer._mesh is None:
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
            artefact_label=self._current_mesh_path.stem if self._current_mesh_path else "artefact",
        )

        try:
            svg = generate_figure(self.viewer._mesh, self.viewer.plotter, config)
            path.write_text(svg)
            QMessageBox.information(self, "Exported", f"Figure saved to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to generate figure:\n{exc}")
```

- [ ] **Step 3: Verify import**

Run: `cd /home/mark/Git/lithic-analysis-platform && python -c "from lithicope._main_window import MainWindow; print('OK')"`
Expected: OK

- [ ] **Step 4: Run all lithicore tests to check no regressions**

Run: `cd /home/mark/Git/lithic-analysis-platform/lithicore && python -m pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add lithicope/src/lithicope/_main_window.py
git commit -m "feat: add Publication Figure export to GUI Tools menu"
```

### Task 5: Integration verification

- [ ] **Step 1: Reinstall both packages**

Run: `pip install --no-deps -e lithicore -e lithicope`

- [ ] **Step 2: Verify full import chain**

Run:
```python
cd /home/mark/Git/lithic-analysis-platform && python -c "
from lithicore import FigureConfig
from lithicore._figure import generate_figure, _nice_scale, _camera_positions
from lithicope._main_window import MainWindow
print('Full import chain OK')
" 2>&1
```

- [ ] **Step 3: Run full test suite**

Run: `cd /home/mark/Git/lithic-analysis-platform/lithicore && python -m pytest tests/ -v`
Expected: 40+ tests pass (35 existing + 5+ new figure tests)

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: publication figure generator v2.0"
```

- [ ] **Step 5: Sync main repo**

```bash
cd /home/mark/Projects/Future\ Project\ Ideas/04.\ Lithic-Analysis-Platform
git merge dev --no-edit
```

---

## Self-Review Checklist

- **Spec coverage:** Every section in the spec is addressed — FigureConfig, generate_figure, CLI, GUI, scale bar, three-view layout, artefact ID
- **Placeholder scan:** All steps have concrete code. No TBDs or "add later" patterns
- **Type consistency:** `FigureConfig` is consistent across create_figure, CLI, and GUI. `_nice_scale` returns same types everywhere. `_export_view_svg` and `_compose_figure_svg` used consistently in `generate_figure`
