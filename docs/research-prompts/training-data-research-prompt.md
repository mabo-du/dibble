---
mode: subagent
---

# Deep Research: Lithic 3D Mesh Training Data Sources

## Goal

Find downloadable 3D mesh datasets (PLY/OBJ/STL) of lithic artefacts (chipped stone tools, debitage, cores) suitable for training a morphometric classifier. We already have the 4-volume Open Aurignacian Project (~2,010 meshes from Fumane, Castelcivita, Cala, and Bombrini). We need **more — ideally 5,000+ additional labelled meshes** from different sites, periods, and typologies to improve classifier accuracy (currently 82% CV on 5 classes).

## Context

We're building Dibble, an open-source lithic analysis platform. The classifier uses 22 morphometric features (length, width, thickness, elongation, edge angles, etc.) extracted from oriented 3D meshes. Labels are archaeological typologies (Class: Core/Tool, Blank: Blade/Bladelet/Flake, Technology, etc.).

## What to Find

### 1. Zenodo Records (highest priority)
The Open Aurignacian Project by Armando Falcucci has 4 volumes. Are there more volumes? Also search for:
- "Open Aurignacian" projects
- 3D scans of lithic artefacts on Zenodo
- Other archaeological 3D scanning datasets
- Check Falcucci's other repos for dataset links
- Search Zenodo for: `3D meshes lithic`, `Aurignacian 3D`, `Palaeolithic 3D scan`, `lithic artefact mesh`

### 2. Museum Collections
Which museums offer downloadable 3D mesh files for lithic artefacts?
- British Museum 3D scans
- Smithsonian 3D digitisation
- Sketchfab archaeological collections (check if downloadable)
- MorphoSource (morphological data repository)
- Open Context
- tDAR (Digital Archaeological Record)

### 3. Academic Repositories
- Figshare archaeological datasets
- Dryad digital repository
- OSF (Open Science Framework) archaeology projects
- Dataverse archaeological collections

### 4. Research Compendiums
Papers that published 3D mesh datasets alongside them:
- Geometric morphometric studies of lithics
- 3D scanning methodology papers
- Machine learning on lithic artefacts
- Use Google Scholar and search "3D mesh" + "lithic" + "dataset" + ".ply"

### 5. Specific Sources to Investigate
- Anyone who has cited Falcucci's Open Aurignacian papers may have similar data
- The PaleoHub network
- "Lithic 3D database" initiatives
- European Research Council (ERC) projects on lithic technology

## Constraints

- Must be **downloadable** (not just viewable online). Actual mesh files (.ply, .obj, .stl, .off, .glb).
- **Labelled** with at least basic typology (Class/Blank, or similar).
- Ideally **oriented** consistently or orientable.
- Millimetre-scale resolution.
- Open access or CC-licensed preferred.
- Meshes should be complete artefacts (not fragments/refits unless labelled).

## Output Format

For each source found, provide:
1. **Name** and URL
2. **Number of meshes** available
3. **Typology labels** available (what metadata exists?)
4. **File format** (PLY, OBJ, etc.)
5. **Download method** (direct link, Zenodo API, requires registration, etc.)
6. **Licensing**
7. **Estimated download size**

Rank by: number of meshes × label quality × ease of access.

## Current Baseline

| Volume | Site | Meshes | Labels | Source |
|--------|------|--------|--------|--------|
| Vol 1 | Grotta di Fumane | 948 | Class, Blank, Technology | Zenodo 15382869 |
| Vol 2 | Grotta di Castelcivita | 538 | Class, Blank, Technology | Zenodo 15383157 |
| Vol 3 | Grotta della Cala | 420 | Class, Blank, Technology | Zenodo 15383121 |
| Vol 4 | Riparo Bombrini | 110 | Class, Blank, Technology | Zenodo 15383190 |
| **Total** | | **2,016** | | |
