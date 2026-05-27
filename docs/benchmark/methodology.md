# Dibble Lithic Classifier Validation Methodology

## Abstract

Dibble is an open-source desktop application for automated 3D lithic analysis that
includes AI-powered typology classification. This document describes the methodology
used to validate Dibble's three pre-trained lithic classifiers: Basic Morphological
(5 classes), Bordes Typology (7 classes), and Technological (5 classes). Each classifier
is evaluated against a held-out synthetic test set of 500–700 samples generated from
published metric ranges with added Gaussian noise to simulate natural variation.

All benchmark results are reproducible with a single command:
`python -m lithicore.data.run_benchmark`

## 1. Introduction

Lithic typology classification has historically relied on expert visual assessment and
manual measurement. While powerful, this approach is subjective, time-consuming, and
difficult to standardise across researchers. Dibble addresses this by combining
automated 3D morphometric feature extraction with a Random Forest classifier trained
on published metric ranges from the lithic analysis literature.

This document presents the validation methodology for Dibble's three built-in
classifiers. The benchmark is designed to be transparent, reproducible, and verifiable
by any user on their own machine.

## 2. Classifier Systems

### 2.1 Basic Morphological

The Basic classifier assigns artefacts to five fundamental morphological categories
based on established lithic analysis criteria (Andrefsky 2005, Inizan et al. 1999):

| Class | Key Diagnostic Features | Metric Range Source |
|---|---|---|
| Flake | L/W < 2.0, 1–5 scars, platform 60–90° | General flake debitage |
| Blade | L/W 2.0–5.0, 2–4 ridges, platform 65–85° | Blade production |
| Bladelet | L/W 2.5–6.0, length 10–50mm | Micro-blade production |
| Core | L/W < 1.5, 3–20 scars, chunky profile | Core reduction |
| Tool | Variable, retouched edges 55–85° | Retouched implements |

### 2.2 Bordes Typology

The Bordes classifier implements the classic François Bordes typological system
(Bordes 1961), one of the most widely used lithic typologies in the European
Palaeolithic:

| Class | Key Diagnostic Features |
|---|---|
| Scraper | Steep edge angle (60–85°), continuous retouch |
| Handaxe | High symmetry (0.7+), ovate shape, large (80–250mm) |
| Point | Elongated (L/W 1.5–3.5), symmetric, pointed |
| Burin | Very steep edge (70–90°), burin spall removal |
| Denticulate | Irregular edge (45–65°), multiple notches |
| Notched | Single or multiple notches, moderate edge angle |
| Backed knife | Elongated (L/W 2.0–4.0), steep backed edge |

### 2.3 Technological

The Technological classifier follows reduction stage classification (primary,
secondary, tertiary) based on the amount of cortical surface and scar density,
plus specialised categories for crested blades and core rejuvenation flakes.

## 3. Test Data Generation

### 3.1 Synthetic Data Approach

Because a comprehensive open-access dataset of 3D lithic meshes with verified
typology labels does not yet exist, the benchmark uses synthetic feature vectors
generated from published metric ranges. This approach:

1. **Ensures reproducibility** — every user generates identical test data
2. **Covers the full diagnostic space** — samples span the complete metric range
3. **Isolates classifier performance** — no confounding factors from mesh quality
4. **Enables large-scale testing** — 500–700 samples per classifier

### 3.2 Generation Parameters

For each class in each typology system, `n = 100` feature vectors are generated:

- Each feature value is drawn uniformly from its published metric range
- Gaussian noise with `σ = 0.20 × range_width` is added to simulate natural variation
- Values are clipped to ±50% beyond range bounds to prevent extreme outliers
- A fixed random seed (`20260527`) ensures bit-for-bit reproducibility

### 3.3 Test-Train Separation

All test data is generated independently from the training data. Training data (used
to build the pre-trained models) uses `noise = 0.15` with a different random seed.
Test data uses `noise = 0.20` with seed `20260527`. This ensures the classifiers
evaluate on distributions they have not seen during training.

## 4. Evaluation Metrics

### 4.1 Overall Accuracy

The proportion of correctly classified samples over the total test set:

$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

### 4.2 Per-Class Metrics

For each class, we report:

- **Precision**: $P = TP / (TP + FP)$ — how many predicted positives are correct
- **Recall**: $R = TP / (TP + FN)$ — how many actual positives were found
- **F1-Score**: $F_1 = 2 \times P \times R / (P + R)$ — harmonic mean of precision and recall
- **Support**: Number of test samples for this class

### 4.3 Confusion Matrix

A confusion matrix visualises which classes are confused with which, revealing
systematic misclassification patterns (e.g., blades misclassified as bladelets due
to overlapping length ranges).

## 5. Classifier Architecture

All three classifiers use the same architecture:

- **Algorithm**: Random Forest (500 trees, max depth 12, min samples leaf 3)
- **Calibration**: Platt scaling via `CalibratedClassifierCV` (3-fold)
- **Feature space**: 20-dimensional morphometric feature vector
- **Training data**: 200–300 synthetic samples per class

The Random Forest algorithm was chosen for:
1. **Interpretability**: Feature importance scores explain each prediction
2. **Small-sample performance**: Effective with 100s rather than 1000s of samples
3. **Multi-class support**: Native handling of 5–7 class problems
4. **No GPU requirement**: Runs efficiently on any CPU

## 6. How to Reproduce

```bash
# Install Dibble
pip install lithicore

# Run the benchmark
python -m lithicore.data.run_benchmark

# Output: docs/benchmark/results/report.html
#         docs/benchmark/results/summary.md
```

The benchmark generates:
- An interactive HTML report with confusion matrices and per-class metrics
- A Markdown summary table
- JSON files with raw metrics for each classifier

## 7. Results

> **Note**: Results are regenerated every time the benchmark is run.
> The latest pre-computed results are in `results/report.html`.

*See `results/report.html` for the full interactive report with confusion matrices.*

## 8. Limitations

1. **Synthetic test data**: Results reflect performance on simulated feature vectors,
not real 3D meshes. Real-world performance may vary based on mesh quality,
orientation accuracy, and the presence of fragmentary or non-diagnostic artefacts.

2. **Metric range generalisation**: The published metric ranges represent central
tendencies. Some artefacts may fall outside these ranges while still being correctly
classified by expert lithic analysts.

3. **Classifier scope**: The classifiers evaluate only the 20-dimensional geometric
feature vector. Attributes like raw material type, colour, and microscopic use-wear
are not considered.

## 9. Future Work

- **Community benchmark**: A user-contributed repository of real 3D meshes with
expert-verified typology labels would enable validation on natural data.
- **Cross-validation**: Full k-fold cross-validation on real collections.
- **Extended typologies**: Additional typological systems (e.g., Tixier, Laplace).

## References

- Andrefsky, W. (2005). *Lithics: Macroscopic Approaches to Analysis* (2nd ed.).
  Cambridge University Press.
- Bordes, F. (1961). *Typologie du Paléolithique ancien et moyen*. Delmas.
- Inizan, M.-L., Reduron-Ballinger, M., Roche, H., & Tixier, J. (1999).
  *Technology and Terminology of Knapped Stone*. Cercle de Recherches et d'Études
  Préhistoriques.
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
