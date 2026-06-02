# Deep Research Prompt — Lithic 3D Mesh Training Data Expansion

## Context

We are building an open-source lithic (stone tool) 3D morphological classifier called **Dibble**. It currently has **3,412 labelled 3D meshes** (PLY/STL/OBJ formats) from 5 continents, and uses a 22-dimensional morphometric feature vector with a RandomForest classifier.

The classifier achieves **~81% 5-fold cross-validation accuracy** across 10 typology classes (Basic system). The limiting factor is not total data volume, but severe **class imbalance**:

| Class | Current Samples | Status |
|-------|----------------|--------|
| Biface | 1,018 | Sufficient |
| Core | 751 | Sufficient |
| Bladelet | 592 | Sufficient |
| Blade | 401 | Sufficient |
| Flake | 262 | Adequate |
| Experimental Core | 254 | Adequate |
| **Retouched Flake** | **50** | **Severely lacking** |
| **Unmodified Flake** | **50** | **Severely lacking** |
| **Unmodified Cobble** | **30** | **Severely lacking** |
| **Tool** | **4** | **Near-zero (will merge)** |

We need to source **200–500 additional labelled 3D meshes** specifically for the under-represented classes above (Retouched Flake, Unmodified Flake, Unmodified Cobble) to push accuracy toward 85%+.

## Existing data sources (already integrated — do not re-list)

These have already been downloaded and processed:

| Source | Meshes | Origin |
|--------|--------|--------|
| Open Aurignacian Project (Vols 1-4) | ~2,010 | Italy |
| Levantine Acheulean Handaxes | 526 | Israel/Palestine |
| COADS (Central Ohio Arch. Digitization Survey) | 514 | Ohio, USA |
| Lombao Experimental Cores | 254 | Spain |
| Morales Experimental Retouch | 100 | Spain |
| Selden Paleoindian Collections | ~30 | Texas, USA |

## Search parameters

### Primary targets (highest priority)

1. **Retouched flakes and tools** — any collection with 3D scans of retouched/utilized flakes, scrapers, denticulates, notched pieces, or composite tools. Look for datasets specifically labelled by modification type (retouched, utilised, backed, etc.)

2. **Unmodified flakes / debitage** — 3D scans of complete flakes, flake fragments, shatter, or angular debris from knapping experiments or archaeological assemblages. Experimental knapping sequences are ideal because they have ground-truth labels.

3. **Cobbles / unmodified cores** — 3D scans of river cobbles, manuports, hammerstones, or unmodified raw material nodules. Natural clast shape baselines for comparison.

### Secondary targets

4. **Any 3D lithic repository with >100 labelled meshes** that we haven't already indexed. Prioritize:
   - Open-access repositories (Zenodo, MorphoSource, Figshare, OSF, OpenArchaeo)
   - Collections with CSV metadata including typological classifications
   - PLY, STL, or OBJ formats (not mesh formats that require conversion)
   - Per-artefact download (not ZIP archives that commingle multiple assemblages)

5. **Experimental archaeology collections** — knapping experiments with known reduction sequences, raw material types, and knapping techniques. These provide ground-truth labels that archaeological assemblages lack.

### Excluded

- Nubian Levallois Database (OSF download broken, no files in repository)
- Biśnik Cave (Zenodo record exists but no publicly accessible mesh files)
- Gahagan Bifaces (Zenodo record contains R code only, no meshes)

## Output requirements

For each identified source, provide:

1. **Repository name** and DOI/URL
2. **Estimated number of 3D meshes** available
3. **Mesh format(s)** (PLY, STL, OBJ, WRL, GLB, etc.)
4. **Label type(s)** — what classification scheme do they use? Can it map to: Retouched Flake, Unmodified Flake, Unmodified Cobble, Tool, Flake?
5. **Download method** — direct Zenodo API link, OSF storage link, MorphoSource batch download, Figshare article
6. **File size estimate** — total download size in MB/GB
7. **Access restrictions** — open access, embargoed, requires account, requires institutional login
8. **Confidence** — high/medium/low based on whether we can verify that 3D meshes are actually present and downloadable

## Format preference

Support for PLY, STL, or OBJ is preferred (our pipeline reads these directly). WRL/VRML and GLB are acceptable with format conversion. We cannot use RAR, ZIP (without inspection), or proprietary formats.

## Success criteria

The research is successful if it identifies **3+ new datasets** that collectively provide **200+ additional meshes** for the under-represented classes, with verified direct download URLs and clear typological labels.
