# Deep Research Prompt: Alternative Routes to Classifier Accuracy Without New Training Data

## Context

We built a lithic (stone tool) classifier in Python (scikit-learn RandomForest) that
predicts typological classes from 22 morphometric mesh features. Current accuracy:

| Typology | Classes | Accuracy | Notes |
|----------|---------|----------|-------|
| Basic | 9 | 84.8% | Broad morphological types |
| Bordes | 9 | 84.8% | Morphology-based (same mapping) |
| Technological | 8 | 73.6% | Reduction stages |

### The constraint

We cannot obtain additional training data. The training set is fixed at ~3,415
real-world 3D artefact scans from five published sources. Classes are imbalanced
(minority classes have ~50-100 samples, majority ~400-600). The 22 features are
morphometric measurements (length, width, thickness, edge angles, scar counts,
curvature, symmetry, etc.) computed from 3D mesh geometry.

We want to push accuracy higher — ideally above 90% for at least one typology —
using only the data and features we already have.

### What we've already tried

- Edge-angle feature was silently zero due to a column-name bug — fixing it gave
  +3.2pp (81.6% → 84.8%)
- Merged a 4-sample class ("Tool") into a related class ("Retouched Flake")
- Reduced n_estimators from 500→200 with <0.2pp loss (to stay under GH 100MB limit)
- Benchmark runs stratified 5-fold CV with class-balanced predictions

---

## Research Prompt

We need fresh perspectives. A team of six domain experts will each approach this
from a different angle. After each expert reports, one final integrative synthesis
will reconcile their findings.

---

### Expert 1: Traditional ML Optimiser

The data is what it is — make the model work harder with it.

- Hyperparameter search: what's the actual optimal RandomForest config for ~3,415
  samples × 22 features with imbalanced classes? Grid/Random/Bayesian search over
  max_depth, min_samples_split, min_samples_leaf, max_features, criterion, etc.
- Feature selection: which of the 22 features are actually predictive? RF
  importance, mutual information, Boruta, recursive feature elimination. Can we
  improve by removing noisy/diagnostic features?
- Ensemble stacking: can a meta-learner (LogisticRegression, GradientBoosting)
  combine predictions from multiple base RandomForest models trained on different
  feature subsets or bootstrap samples? Does soft/hard voting across different
  tree configs beat a single forest?
- Calibration: are the confidence scores well-calibrated? If not, can Platt
  scaling or isotonic regression improve the reliability (even if accuracy doesn't
  change, better uncertainty estimates are valuable)?
- Cross-validation scheme: is 5-fold stratified the right choice? Would grouped
  CV (by dataset-of-origin) give a more honest estimate of generalisation to novel
  assemblages?

**Deliverable:** Concrete recommendations with expected accuracy deltas (+/- X pp),
ordered by effort-to-impact ratio.

---

### Expert 2: Data Augmentation & Synthetic Data Specialist

You can't get more real data — so manufacture it. What transformations of existing
3D meshes preserve the typological label while generating new valid samples?

- Mesh perturbation: small-scale random vertex displacements, smoothing,
  simplification/decimation at different rates. How much noise can you inject
  before the label flips?
- Rotation augmentation: does alignment sensitivity mean small orientation offsets
  produce useful variation?
- Geometric morphometric interpolation: can we create plausible "intermediate"
  artefacts by interpolating between two labelled examples of the same type?
  (Between-class interpolation for the Technological typology, which is inherently
  a continuum?)
- Partial occlusion simulation: masking regions of the mesh (mimicking breakage,
  soil coverage, incomplete excavation)
- 2D→3D tricks: could random renders of the 3D mesh back-projected into
  photogrammetry-style noise produce useful training images? (Only relevant if
  we had a 2D-based pipeline — but worth flagging.)

**Key constraint:** Augmentations must produce meshes that pass mesh validation
(watertight, non-degenerate faces, etc.) so feature extraction succeeds.

**Deliverable:** Catalogue of viable augmentations with expected yield (n synthetic
samples per real sample) and technical feasibility (what code infrastructure is
needed).

---

### Expert 3: Feature Engineering & Dimensionality Reduction Expert

The 22 existing features are sensible hand-crafted morphometrics. But there might
be features hiding in the data the human experts missed.

- Feature interactions: pairwise products, ratios, differences between existing
  features. The elongation (L/W) and flatness (W/T) are already ratios — what
  about L×W, L×T, or three-way interactions?
- Derived features from existing: log transforms, polynomial expansions, RBF
  expansions on key features
- Higher-order shape descriptors:
  - **Spherical harmonics** (SH) — fit a spherical harmonic decomposition to the
    mesh surface; SH coefficients are powerful global shape descriptors. Only
    works on star-shaped meshes. Estimated cost: moderate.
  - **Persistent homology** (topological data analysis) — compute persistence
    diagrams for the mesh filtration (lower-star, Vietoris-Rips on vertex
    cloud); barcode statistics (total persistence, birth-death range, number of
    significant features at each dimension) as features. Gains reported in
    shape classification literature.
  - **Heat kernel signature** (HKS) — multi-scale local descriptor aggregated to
    a global histogram. Common in 3D shape retrieval. Mitigates non-rigid
    deformation sensitivity.
  - **3D Zernike moments** — rotation-invariant shape descriptors from voxelised
    mesh. Proven in protein surface classification.
- Clustering-derived features: run unsupervised clustering (k-means, DBSCAN) on
  the feature space; use cluster membership/distance-to-cluster-centre as
  additional features for the final classifier (semi-supervised trick).
- Feature hashing / random projections: map 22 → 256 with fixed random projection;
  sometimes linear separability improves in higher dimension.
- Dimensionality reduction THEN classify: does PCA → RandomForest beat full-space
  RF? What about t-SNE or UMAP components as features?

**Deliverable:** Ranked list of candidate features by (a) expected impact on
accuracy (low/medium/high) and (b) implementation complexity (low/medium/high).
Include any that are well-known in 3D shape analysis or archaeological
morphometrics but under-explored in lithic classification.

---

### Expert 4: Alternative Classifier Architectures Expert

What if RandomForest isn't the right tool?

- **Gradient Boosting** (XGBoost/LightGBM/CatBoost): often outperforms RF on
  tabular data with <10k samples. Sparse gradient updates might help with
  imbalanced classes. What learning rate / tree count works for this scale?
- **Support Vector Machine** (RBF kernel): SVM with careful C/gamma tuning often
  excels at small-to-medium tabular problems. How does it compare to RF?
- **Multi-layer Perceptron** (small — 1-2 hidden layers): can it learn non-linear
  feature interactions the RF's axis-aligned splits might miss? With 3,415
  samples, a tiny net (32→16→9) shouldn't overfit badly with dropout.
- **k-Nearest Neighbours**: simple baseline — does the feature space already have
  good local structure? A weighted k-NN with learned metric (LMNN/LFDA) might
  capture what the typology system is actually doing.
- **Feature → type → subtype cascades**: instead of one 9-class classifier,
  train a hierarchy: first level distinguishes broad groups (flakes vs cores vs
  bifaces), second level does fine-grained within-group. This might help if
  some features are good for coarse but not fine distinctions.
- **One-vs-one vs one-vs-rest**: does the multi-class strategy matter for RF?
  (Probably not, RF is inherently multi-class — but worth checking).
- **Cost-sensitive learning**: class_weight='balanced_subsample' already used.
  What about sample weights derived from the dataset-of-origin? Artefacts from
  certain collections might be noisier.
- **Deep learning on raw mesh**: if we could voxelise the mesh and train a 3D CNN
  (like VoxNet, 3D ResNet), or use a point-cloud network (PointNet++), could it
  learn features the hand-crafted ones miss? This is high-risk because of our
  small dataset, but transfer learning from ShapeNet/ModelNet might help.

**Deliverable:** For each alternative architecture, give: expected accuracy delta
(± range), training time, risk of overfitting, and implementation complexity.
Include explicit "do not try" recommendations for approaches that would clearly
fail at this sample size.

---

### Expert 5: Semi-Supervised & Self-Supervised Learning Expert

The 3,415 labelled samples are the labelled set. But what about unlabelled
artefacts?

Wait — the constraint says no additional data. But consider:

- **Self-training / pseudo-labelling**: train an initial model, use it to label
  the lowest-confidence predictions, add them to the training set with low weight,
  retrain. This can create a "virtuous cycle" if the model's high-confidence
  predictions are largely correct.
- **Co-training**: if 22 features can be split into two independent views
  (e.g., size-based features vs shape-based features), train two classifiers
  independently and use their agreement to pseudo-label.
- **Consistency regularisation**: perturb the input (add small noise to features)
  and penalise the model for making different predictions. This is standard in
  semi-supervised learning (FixMatch, UDA) and doesn't need unlabelled data —
  it just prevents overfitting to decision boundaries that aren't locally smooth.
- **Contrastive pre-training**: use all 3,415 samples (ignore labels) to learn an
  embedding where similar artefacts are nearby. Train a SimCLR-style contrastive
  model on augmented feature vectors, then fine-tune a linear classifier on top.
  This is compute-heavy but could capture latent structure the RF misses.
- **Information maximisation**: use entropy minimisation (Grandvalet & Bengio) to
  push decision boundaries away from high-density regions of feature space.
- **Leave-one-dataset-out**: train on 4 datasets, test on the 5th — this directly
  measures generalisation to novel lithic traditions. If some datasets are very
  different, dataset-specific biases may inflate CV accuracy.
- **Multi-task learning**: train a single model to predict both Basic typology AND
  Technology typology simultaneously. The shared representation might improve
  both tasks (especially the weaker Technological one). Train a multi-output RF
  or a small neural net with two output heads.

**Deliverable:** Which semi-supervised/self-supervised approaches work at the
3,415-sample scale, with concrete implementation steps. Flag approaches that need
significantly more data and are off the table.

---

### Expert 6: Archaeological & Domain Epistemology Expert

The problem may not be the classifier — it may be the typology system itself.

- **Typology consistency audit**: are the labels in the training set reliable?
  The data comes from five published sources, each with their own typological
  tradition. Does "Side Scraper" mean the same thing in OAP (Italy) as in COADS
  (Ohio)? Disagreement between annotators / source datasets would limit accuracy
  regardless of model quality.
- **Latent typologies**: cluster the feature space and see what clusters emerge.
  If they don't match the assigned typology labels, the typology may not be
  recoverable from geometry alone — surface attributes (colour, material, raw
  material type) or technological trace evidence might carry more information.
- **Ordinal regression**: if the Technological typology is "core reduction
  stages" (e.g., Early Core → Middle Core → Late Core → Exhausted), this is
  inherently ordinal. Treating it as ordinal regression (not multi-class
  classification) could improve accuracy — the model is penalised less for
  predicting "Middle Core" when the true label is "Late Core" than for predicting
  "Flake".
- **Confusion analysis**: which pairs of classes are most confused? For Basic
  typology, which are the most confused class pairs? Is it always the same pair?
  If so, can we merge or redefine those classes? The Tool→Retouched Flake merge
  was productive — where else might that work?
- **Representativeness**: the training set heavily over-represents European
  Aurignacian material (OAP: 2,418 of 3,415 samples). The model may be very good
  at European Upper Palaeolithic artefacts but poor at anything else. CV accuracy
  may overstate real-world performance. Dataset-stratified CV would reveal this.
- **Consult archaeologist**: is 85% accuracy clinically useful? In lithic
  analysis, inter-annotator agreement between human experts is often 70-80%.
  Perhaps the goal shouldn't be 95% — perhaps it should be "as good as or better
  than a human expert." That changes the framing and the evaluation metric.

**Deliverable:** Recommendations that could change the target variable (merge
classes, ordinal encoding, human-baseline comparison) alongside model-centric
changes. Include a proposed human-expert labelling study to benchmark current
performance.

---

## Synthesis Task

After all six experts have reported:

1. **Produce a ranked roadmap** of all recommended actions, ordered by
   expected-impact ÷ implementation-effort. Estimate % accuracy gain for
   Basic, Bordes, and Technological typologies.

2. **Identify quick wins** (< 1 day to implement) and differentiate from
   research projects (weeks to implement).

3. **Find the hidden gems** — recommendations that appear in multiple expert
   reports (these tend to be the most robust ideas).

4. **Flag contradictions** — where experts disagree. Resolve or explain the
   trade-off.

5. **Give a "stop doing" list** — things that sound promising but won't work
   at 3,415 samples × 22 features, or that would waste implementation time.

6. **Final summary table:**

| Approach | Est. impact (pp) | Effort | Risk | Priority |
|----------|------------------|--------|------|----------|
| ... | +0.5 to +1.5 | 2 days | Low | 1 |
| ... | ... | ... | ... | ... |

---

## Success Criteria

The synthesis should answer these concrete questions:

1. **Can we reach 90%+ on Basic typology with existing data?** (Yes / No /
   Probably — if we do X, Y, Z)
2. **Can we push Technological from 73.6% to 80%?** (Always struggled — here's
   why, and here's the best path.)
3. **What is the hard ceiling for this feature set?** (Given 22 morphometric
   features on ~3.4k imbalanced samples — is there an information-theoretic
   bound, or can we estimate it?)
4. **Should we try deep learning on raw meshes or not?** (Under 3 constraints:
   small dataset, no GPU guarantee for users, desire for interpretability.)
5. **Is the typology system the bottleneck, not the model?** (Evidence for and
   against.)

---

## Technical Appendix (for reference)

### Current model

```python
# scikit-learn 1.6+
RandomForestClassifier(
    n_estimators=200,
    criterion='gini',
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=1,  # OOM safety: one core
)
```

### Current features (22)

| Feature | Description |
|---------|-------------|
| length_mm | Maximum length |
| width_mm | Maximum width |
| thickness_mm | Maximum thickness |
| surface_area_mm2 | Surface area |
| volume_mm3 | Volume (watertight) |
| elongation | L/W |
| flatness | W/T |
| compactness | Vol / L³ |
| epa_deg | Exterior platform angle |
| ipa_deg | Interior platform angle |
| edge_angle_mean_deg | Mean dihedral angle |
| edge_angle_std_deg | Std of dihedral angles |
| edge_angle_skewness | Skewness |
| edge_angle_kurtosis | Kurtosis |
| scar_count | Flake scars detected |
| curvature_index | Vertex normal deviation |
| cross_section_profile | 0=flat, 1=triangular, 2=round |
| symmetry_score | 0-1 bilateral symmetry |
| dorsal_ridge_count | Parallel linear ridges |
| surface_roughness | Face/projected area |
| platform_thickness_mm | Platform thickness |
| cortex_percentage | % cortical surface |

### Training data composition

| Source | Origin | Artefacts | Typology coverage |
|--------|--------|-----------|-------------------|
| OAP (Open Aurignacian) | Italy | 2,418 | Full |
| Levantine Acheulean | Israel/Palestine | 526 | Biface-heavy |
| COADS | Ohio, USA | 614 | Projectile-point-heavy |
| Lombao Cores | Spain | 254 | Core-heavy |
| Morales Retouch | Spain | 100 | Retouch experiments |

### Class distribution (Basic typology, 9 classes)

Class | Count
------|------
Blade | ~600
Bladelet | ~150
Flake | ~500
Biface | ~500
Core | ~400
Scraper | ~400
Notch/Denticulate | ~300
Composite | ~200
Retouched Flake | ~300
(Original "Tool" class — 4 samples — was merged into Retouched Flake)

### Evaluation protocol

- Stratified 5-fold cross-validation (preserves class proportions per fold)
- Metric: macro-averaged accuracy (each class weighted equally, not by
  frequency)
- No hyperparameter tuning on CV — current accuracy is the default RF config
  (except 200 trees, balanced_subsample)
- Benchmark script: `lithicore/data/run_benchmark.py`
- Output: HTML report with confusion matrices, per-class metrics, config
