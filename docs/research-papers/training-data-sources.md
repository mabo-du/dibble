# 3D Lithic Mesh Repositories for Classifier Training

## Summary
Deep research identified **5,000+ additional labelled 3D lithic meshes** across 8 major open-access repositories. Integrating these could push classifier accuracy well beyond the current 82% CV.

## Prioritised Sources

### Tier 1: High-Volume Macro-Assemblages

| # | Source | Meshes | Labels | Format | Access |
|---|--------|--------|--------|--------|--------|
| 1 | **COADS** (Central Ohio Arch. Digitization Survey) | 600+ | Projectile point types (Adena, Hi-Lo, Snyder's), raw material | PLY, GLB | Zenodo (bsu_aal) + Sketchfab |
| 2 | **Later Acheulean Handaxes** (Muller & Grosman) | 686 | Handaxes, site provenance (Ma'ayan Barukh, Revadim, etc.) | WRL, VRML | Zenodo 10.5281/zenodo.14534846 |
| 3 | **Nubian Levallois Database** (Hallinan & Cascalheira) | ~200 | Levallois cores, Middle Stone Age | PLY | OSF 10.17605/OSF.IO/SJ8ZV |
| 4 | **Lombao Experimental Cores** | 290 | Reduction stages, knapping strategy | PLY | Zenodo 10.5281/zenodo.2585423 |
| 5 | **Morales Retouch Flakes** | ~34 | Pre/post retouch, experimental flint | STL | Zenodo 10.5281/zenodo.1405047 |

### Tier 2: Specialised & Regional

| # | Source | Meshes | Labels | Format | Access |
|---|--------|--------|--------|--------|--------|
| 6 | **Selden Paleoindian Collections** (Gault/Blackwater Draw) | 50-100 | Clovis points, Paleoindian bifaces | PLY, STL | SFA ScholarWorks / Zenodo |
| 7 | **Gahagan Bifaces** (Caddo) | 20-30 | Gahagan bifaces, Caddo culture | PLY | Zenodo 10.5281/zenodo.3457943 |
| 8 | **Biśnik Cave** (Kobyłka et al.) | Variable | Middle Palaeolithic, Levallois | 3D formats | Zenodo 10.5281/zenodo.19186143 |
| 9 | **Figshare Aggregate** | 100+ | Subspheroids, handaxes, unifacial tools | PLY, OBJ | Multiple DOIs |
| 10 | **Linsel Scar Topology** | Variable | Annotated scar networks, directed graphs | PLY, Graph | Zenodo 10.5281/zenodo.10477448 |
| 11 | **ReViBE Refits** | 9 refit sets | Refit sequences, core reduction | OBJ, MTL | CORA Dataverse 10.34810/data924 |

## Key Technical Notes
- **WRL/VRML** (Levantine handaxes) need format conversion to PLY
- **COADS** has individual Zenodo records per artifact — needs batch download
- Many datasets have machine-readable CSV metadata with Class/Blank/Technology labels
- Total estimated download: ~25-30 GB across all sources

## Impact on Classifier
- Adds **bifacial geometries** (handaxes, projectile points) — entirely absent from current training
- Adds **prepared core technologies** (Nubian Levallois)
- Adds **experimental reduction sequences** (ground truth for volumetric decay)
- Adds **edge modification data** (pre/post retouch pairs)
- Total training corpus could reach **7,000+ meshes** from <50 to >500,000 years BP

Sources: Deep Research report generated 2026-05-28.
