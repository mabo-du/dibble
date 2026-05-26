# Publication Figure Generator — Design Specification

**Date:** 2026-05-26
**Status:** Approved design, pending implementation

## 1. Overview

Generate standardised three-view archaeological technical drawings (plan, profile, section) from an oriented 3D lithic mesh, using VTK's GL2PS vector export pipeline. Output is publication-quality SVG with scale bar, measurement callouts, and artefact ID block.

## 2. Architecture

### 2.1 Module locations

```
lithicore/src/lithicore/
├── _figure.py              ← NEW

lithicope/src/lithicope/
├── _main_window.py         ← EXTENDED (menu item)
```

### 2.2 Key constraint

The figure generator lives in `lithicore` (pure logic + VTK export), not `lithicope`, so it's accessible from both the GUI and the CLI. The VTK GL2PS call needs an OpenGL context — the GUI provides this via the existing PyVista viewer; the CLI uses offscreen rendering (VTK's `vtkOffscreenGL2PSExporter` works headlessly on Linux with Mesa).

## 3. Data flow

```
User orients mesh (auto or manual)
    ↓
Export → Publication Figure
    ↓
FigureConfig(views=["plan","profile","section"], show_measurements=True, ...)
    ↓
For each view:
    └─ Set PyVista camera to orthographic projection
    └─ VTK GL2PS exporter captures scene as SVG vector string
    └─ Return raw SVG view
    ↓
Compose three SVGs into standard archaeological plate layout
    ↓
Add scale bar (nice-number tick marks + unit label)
Add measurement callouts (length/width labels at standard positions)
Add artefact ID block (label + date)
    ↓
Single SVG output → file or bytes
```

## 4. FigureConfig

```python
@dataclass
class FigureConfig:
    """Configuration for publication figure generation."""
    views: list[str] = field(default_factory=lambda: ["plan", "profile", "section"])
    show_measurements: bool = True
    show_ridges: bool = True
    scale_bar_unit: str = "cm"  # "cm" or "mm"
    output_format: str = "svg"
    artefact_label: str = ""
```

## 5. Figure layout

```
┌──────────────────────────────────────────────┐
│                  PLAN VIEW                    │
│         ┌────────────────────────┐            │
│         │  dorsal silhouette     │            │
│         │  + internal ridges     │            │
│         └────────────────────────┘            │
├────────────────┬─────────────────────────────┤
│   PROFILE      │          SECTION             │
│  ┌──────────┐  │        ┌──────────┐          │
│  │ lateral   │  │        │ distal   │          │
│  │ silhouette│  │        │ silhouet │          │
│  └──────────┘  │        └──────────┘          │
├────────────────┴─────────────────────────────┤
│  ═══ 5 cm ═══              ID: FLK-145       │
└──────────────────────────────────────────────┘
```

- Views use equal scaling so the artefact appears at the same magnification across all three
- Scale bar computed from mesh bounding box extent with nice-number rounding
- Callouts placed at standard anatomical positions (max length, max width, platform angle)

## 6. CLI

```bash
lithicore figure input.ply --output figure.svg
lithicore figure input.ply --output figure.pdf --no-measurements --ridges
```

## 7. GUI

New menu item: **Tools → Export → Publication Figure**

Opens a dialog with:
- View toggles: ☑ Plan ☑ Profile ☑ Section
- ☑ Show measurements | ☑ Show ridges
- Scale bar unit: [cm ▼]
- [Generate] → file save dialog → SVG saved

## 8. Testing

- Unit test: `FigureConfig` defaults and custom values
- Integration test: Generate a figure from `oriented_prism` fixture; verify output SVG contains expected elements (scale bar, view groups, artefact ID)
- Visual check: Open generated SVG in a browser and confirm three-view layout

## 9. Technical notes

- VTK GL2PS: `vtkGL2PSExporter` with `SetFileFormatToSVG()`, `SetWrite3DPropsAsRaster(False)`, `SetSortToBSP()`
- Camera positions: Plan `(0, 0, +Z)`, Profile `(+X, 0, 0)`, Section `(0, +Y, 0)` with parallel projection
- SVG composition uses `svgwrite` library for precise SVG layout (scale bar, annotations, view positioning)
- Mesh visual properties: white background, grey mesh with black edges, red edge overlay points become red ridge lines in vector output
- The CLI offscreen path uses VTK's `vtkOffscreenGL2PSExporter` or simply sets `DISPLAY` to a virtual X framebuffer (Xvfb)

---

*Specification approved by user on 2026-05-26. Ready for implementation planning.*
