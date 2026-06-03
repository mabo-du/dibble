# Persistent Homology — Research Scope

## What it is

Persistent Homology (PH) is a Topological Data Analysis (TDA) technique that
extracts multi-scale topological features from 3D point clouds or meshes. It
tracks the "birth" and "death" of topological structures (connected components,
loops, voids) as a scale parameter increases, producing persistence diagrams
that serve as rotation-invariant shape descriptors.

## Why it might help with lithic classification

The 22 current morphometric features capture macro-morphology (length, width,
angles, ratios) but miss **micro-topography** — the fine-scale surface texture
patterns created by flake scars, ridge intersections, and retouch. PH is
specifically designed to capture these because:

- **1-dimensional features (loops)** map to concavities, notches, and scar
  boundaries — key for distinguishing denticulates from scrapers
- **Short-persistence bars** (historically discarded as noise) actually encode
  surface texture intensity — directly relevant to flake scar density and
  retouch intensity
- **Rotation invariance** means the features are orientation-independent,
  avoiding the PCA alignment sensitivity of current features

## Expected impact

| Source | Estimate | Basis |
|--------|----------|-------|
| Deep Research Paper 1 | +2 to +4pp | PH for micro-topography is the highest-ceiling feature expansion |
| Deep Research Paper 2 | +1 to +3pp | More conservative — PH adds orthogonal information to existing features |
| Best guess | **+1.5 to +3pp** | Realistic given 3,415 samples and label noise ceiling |

## Implementation approach

1. **Mesh preprocessing**: For each artefact in the training set, extract surface
   vertices as a 3D point cloud. Sub-sample to ~5,000 points for computational
   tractability.

2. **PH computation** (via GUDHI or ripser):
   - Compute Vietoris-Rips persistence on the point cloud
   - Extract persistence diagrams for dimensions 0, 1, and 2
   - Vectorise diagrams using persistence images or persistence landscapes
   - Result: ~50-100 new topological features per artefact

3. **Dimensionality reduction**: Apply PCA or UMAP to compress PH features to
   5-15 components. Concatenate with existing 22 core + 10 interaction features.

4. **Retrain**: Train classifier on 37-47 features. Evaluate via 5-fold CV.

## Dependencies and cost

| Dependency | Purpose | Cost |
|------------|---------|------|
| `gudhi` | PH computation (fast, C++ backend) | `pip install gudhi` |
| `ripser` | Alternative PH (pure Python, slower) | `pip install ripser` |
| `persim` | Persistence diagram vectorisation | `pip install persim` |
| `scipy.spatial` | KDTree for mesh sampling | Already installed |
| Disk space | ~500 MB for per-artefact persistence diagrams | Temporary |

## Estimated effort

| Phase | Time | Description |
|-------|------|-------------|
| 1. Prototype | 2-3 days | PH computation + vectorisation on 10 test meshes |
| 2. Batch compute | 2-3 days | Run on all 3,415 training meshes |
| 3. Integration | 1-2 days | Add to feature extraction pipeline, retrain, benchmark |
| 4. Optimisation | 2-3 days | Tune PH parameters (filtration type, subsample size, vectorisation) |
| **Total** | **~10 days** | |

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| PH adds no predictive value | Medium | Test on small subset first; abort if no signal in first 100 artefacts |
| Computational cost too high | Low-Medium | GUDHI is efficient; 5K-point Vietoris-Rips takes ~2s per mesh |
| Non-watertight meshes break PH | Medium | Use vertex cloud (no mesh topology needed) instead of mesh filtration |
| Feature explosion → overfitting | Low | PCA/UMAP compression before concatenation |
| GUDHI installation issues | Medium | Falls back to ripser (pure Python) |

## Decision point

Before full investment, run a **quick signal test**:
1. Take 100 artefacts (50 from one class, 50 from another)
2. Compute PH features via ripser
3. Train a simple 2-class classifier
4. If CV accuracy > 70%, proceed to full implementation
