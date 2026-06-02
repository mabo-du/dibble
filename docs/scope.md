# PROJECT 7 — Lithic 3D Morphological Analyzer

> **Historical planning document.** This was the original v1 MVP scope written before any code existed. The current app (v4.x) far exceeds these goals — see [README](../README.md) for up-to-date feature documentation.

## Overview

A desktop application that allows archaeologists and anthropologists to load a 3D scan of a stone tool (lithic artefact) and automatically extract standardised morphological measurements — edge angles, platform dimensions, flake scar counts and orientations, maximum length/width/thickness, mass estimation — without writing any code. Currently this work requires R (Lithics3D package) or Python (PyLithics), which excludes most lithic analysts. This tool provides a GUI wrapper around these algorithms.

## Target users

- Lithic analysts and knapping technology specialists
- Palaeolithic archaeologists studying hominin tool use
- Zooarchaeologists studying bone-tool modification
- Graduate students in prehistoric archaeology
- Museum collections researchers measuring comparative assemblages

## MVP scope (v1)

- Import 3D meshes in OBJ, PLY, and STL formats
- 3D interactive viewer with rotation, zoom, pan
- Semi-automatic orientation tool: define a reference plane (platform surface), the app computes the tool's orientation
- Basic metric extraction: maximum length, width, thickness, surface area, volume estimate
- Platform angle measurement: measure the angle between platform and dorsal face
- Edge identification: highlight detected edges on the mesh
- Export measurements to CSV (one row per artefact)
- Batch processing: load a folder of meshes, extract metrics for all

## Feature roadmap (v2+)

- Automated edge angle calculation along the full edge profile
- Flake scar detection and counting using curvature analysis
- Dorsal scar pattern classification (unidirectional, bidirectional, centripetal)
- Cortex area percentage calculation
- 3D landmark placement for geometric morphometrics
- Integration with MorphoJ for geometric morphometric analysis
- Comparison mode: overlay two artefacts to compare shape
- Classification against reference assemblages
- Publication figure generator: standardised three-view technical drawings

## Tech stack recommendation

| Layer | Choice | Rationale |
|---|---|---|
| Language | Python | Open3D, trimesh are best 3D mesh libraries |
| GUI | PyQt6 | Rich 3D widget support via OpenGL |
| 3D rendering | Open3D or VTK | Open3D is more pythonic; VTK more powerful |
| Mesh processing | trimesh + Open3D | Complementary: trimesh for boolean ops, Open3D for visualisation |
| Numerical | NumPy + SciPy | Vector geometry calculations |
| Export | CSV via pandas, PDF via ReportLab | |

## Architecture notes

- The **orientation step** is critical and must come before metric extraction. Once the platform surface is defined, the tool's coordinate system is fixed and all measurements (length along reduction axis, platform angle, etc.) are computed in that coordinate space.
- **Edge detection** from a 3D mesh uses dihedral angle thresholding — edges are where the angle between adjacent faces exceeds a threshold. Allow users to adjust this threshold.
- Separate the **measurement algorithms** from the GUI completely. Each algorithm (edge angle, volume, platform angle) is a pure function taking mesh + parameters → float/array. This allows unit testing and future CLI access.
- **Batch mode** should generate a single CSV with one row per artefact and one column per measurement, ready for statistical analysis in R or Python.
- Embed a **landmark guide** — an illustrated reference showing exactly what each measurement corresponds to (referencing Andrefsky or Clarkson measurement protocols).

## Core data model

```
Artefact
  id, file_path, label, raw_mesh (reference), oriented_mesh (reference)
  import_date, notes

Measurement
  artefact_id, measurement_name, value, unit, method, computed_at

Landmark
  artefact_id, name, x, y, z (in oriented coordinate space)

BatchRun
  id, input_folder, output_csv_path, parameters (JSON), run_date
```

## Existing resources to leverage

- **Lithics3D R package** — reference algorithms: https://github.com/cornelmpop/Lithics3D
- **PyLithics** — Python reference: https://zenodo.org/records/5898149
- **Open3D** — 3D processing library: http://www.open3d.org
- **trimesh** — Python mesh library: https://trimesh.org
- **Andrefsky 2005** "Lithics: Macroscopic Approaches to Analysis" — standard measurement protocol reference
- **Clarkson 2008** — platform angle and scar measurement standards

## Technical risks

- **Mesh quality variance** — user-supplied meshes from photogrammetry or low-quality scanners may have holes, noise, or non-manifold geometry. Build a mesh validation and cleaning step before any analysis.
- **Semi-automatic orientation** — fully automatic orientation is an unsolved research problem. v1 requires user guidance (click three points on the platform). Be upfront about this.
- **Edge angle accuracy** — accuracy depends heavily on mesh resolution. Low-resolution meshes give imprecise results. Document minimum recommended mesh resolution.

---

## Deep Research Prompt — Project 7

> I am building an open-source 3D lithic (stone tool) morphological analysis application. I need technical and domain research:
>
> 1. **Standard lithic measurement protocols**: What are the standardised metric measurement protocols for lithic artefacts? Cover the Andrefsky protocol, Clarkson's platform measurements, and any ISO or inter-laboratory standards. For each measurement (maximum length, width, thickness, platform width, platform thickness, EPA, IPA, exterior platform angle), describe exactly how it is taken and what anatomical reference points define it.
>
> 2. **Existing software analysis**: Provide detailed technical analysis of Lithics3D (R), PyLithics (Python), and any other software for 3D lithic analysis. What algorithms do they implement? What are their input formats and limitations? Are their measurement outputs validated against hand-caliper measurements?
>
> 3. **3D mesh edge detection**: What algorithms detect edges in 3D triangle meshes? Describe dihedral angle thresholding, ridge detection via shape index, and Gaussian curvature approaches. Which Python/C++ libraries implement these? How are they applied specifically to detect retouched edges on lithic artefacts?
>
> 4. **Platform orientation algorithms**: How can the platform surface of a flake be automatically detected in a 3D mesh? What surface normal analysis approaches have been proposed? How do researchers currently orient lithics in 3D space for standardised measurement?
>
> 5. **Geometric morphometrics for lithics**: What landmark-based and outline-based geometric morphometric approaches are used for lithic shape analysis? What software (MorphoJ, geomorph R package) performs these analyses? What landmark schemes have been published for different tool types (handaxes, blades, flakes)?
>
> 6. **Community needs**: Search archaeological computing blogs, open-archaeo.info, and Reddit for what lithic analysts most want from digital analysis tools. What measurements do they most commonly need to extract?

---
---
