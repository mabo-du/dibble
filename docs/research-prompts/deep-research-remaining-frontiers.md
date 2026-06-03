# Deep Research Prompt: Three Remaining Frontiers for Lithic Classifier Accuracy

## Context

We built a lithic (stone tool) classifier in Python (scikit-learn RandomForest) that predicts typological classes from 32 morphometric features (22 core + 10 interaction features) extracted from 3D mesh geometry.

### Current performance

| Typology | CV Accuracy | Classes |
|----------|-------------|---------|
| Basic / Bordes | 86.1% | 8 |
| Technological | 74.1% | 8 |

Over the past several sessions we've exhausted model-centric improvements:
- Bladelet→Blade merge: +0.8pp
- Hyperparameter tuning (max_features=0.3): +0.3pp
- Feature interactions (10 derived shape indices): +0.5pp
- Ordinal regression (LogisticAT and cumulative-link RF): failed to beat flat RF
- Ensemble stacking (RF + ExtraTrees): negligible gain
- Persistent Homology signal test (GUDHI, 12 summary stats + 20 histogram bins): PH at 41.3% vs baseline 81.2%; combined no improvement

### The critical finding: Leave-One-Dataset-Out CV

When we hold out each source dataset and train on the remaining four, accuracy **collapses from 86% to 6-12%** on non-OAP data:

| Held-out group | CV Accuracy | Note |
|----------------|-------------|------|
| OAP (Europe) | N/A | Missing classes when OAP excluded |
| COADS (Ohio) | 6.1% | Projectile points — completely different morphology |
| Levantine Acheulean | 10.6% | Handaxes — morphologically distinct |
| Lombao (Spain) | N/A | Experimental cores — unique classes |
| Morales (Spain) | N/A | Experimental retouch — unique classes |

The model is an OAP specialist. Training data is 71% European Aurignacian (OAP). The remaining 29% comes from four other traditions with genuinely different morphology for the same typological labels.

### Training data

3,415 artefacts, 5 source datasets, 22 features (now 32 with interactions):

| Source | Origin | Samples | Typology coverage |
|--------|--------|---------|-------------------|
| OAP (Open Aurignacian) | Italy | 2,010 | Full (Blade, Bladelet, Core, Flake, Scraper, etc.) |
| Levantine Acheulean | Israel/Palestine | 526 | Handaxes only (all → "Biface"/"Handaxe") |
| COADS | Ohio, USA | 492 | Projectile points (all → "Biface"/"Handaxe") |
| Lombao Cores | Spain | 284 | Experimental cores only |
| Morales Retouch | Spain | 100 | Retouch experiments |

### What we need

Explore three remaining frontiers and produce an integrated, implementable roadmap.

---

## Topic 1: Rebalancing to Reduce OAP Dominance

The OAP dataset dominates at 71% of training data. The LOGO CV shows the model can't generalise. We need concrete strategies.

### Questions to answer

1. **Downsampling**: If we downsample OAP to match the next-largest dataset (say 526 samples, matching Levantine), what's the expected accuracy within OAP vs cross-dataset? What's the optimal downsampling ratio?

2. **Inverse frequency weighting**: What class_weights or sample_weights would best balance dataset representation? How does sklearn's sample_weight interact with class_weight='balanced'? Should we weight by dataset (not class)?

3. **Dataset-specific models**: Is the best approach to train separate models per tradition and let the user select the relevant one? Would this be more honest and more accurate than one-size-fits-all?

4. **Multi-dataset training**: Beyond downsampling, are there techniques like domain adaptation, domain-adversarial training, or dataset-conditional normalisation that could help a single model generalise across assemblages?

5. **Cross-dataset verification**: What's the best protocol for honestly measuring cross-dataset performance? The 86% CV is clearly inflated by OAP dominance.

### Deliverable

- Concrete rebalancing strategy with expected accuracy trade-offs
- Code-friendly approach (prefer sklearn-native, keep CPU-compatible)
- Recommended evaluation protocol for honest cross-dataset reporting

---

## Topic 2: Redefining the Typology System

The Deep Research papers both concluded that the **typology system itself (not the model, not the features) is the primary bottleneck**. Human experts disagree on 15-25% of classifications. Different source datasets apply the same labels to different morphologies.

### Questions to answer

1. **Data-driven typology refinement**: Cluster the 32-dimensional feature space (unsupervised). What natural groupings emerge? How do they compare to the existing typology labels? Are there classes that clearly don't correspond to geometric reality?

2. **Diagnosing label inconsistency**: The five source datasets each use their own typological tradition. "Side Scraper" from Fumane Cave (Italy) may not match "Side Scraper" from COADS (Ohio). How can we quantify and correct this inter-source label drift?

3. **Beyond discrete types**: Are there approaches from continuous morphospace analysis (e.g., geometric morphometrics, PCA of shape variables, or landmark-based methods) that would be more faithful to the data than discrete categorical typology?

4. **Ordinal and hierarchical alternatives**:
   - For Technological typology (core reduction stages), ordinal approaches failed with linear and cumulative-link methods. Are there kernel-based ordinal regression methods that might work?
   - For Basic/Bordes, would a hierarchical taxonomy (first: flake vs core vs biface; then: sub-classification within each) improve overall accuracy?

5. **Tradition-specific label mappings**: Should we maintain separate label mappings per source dataset (e.g., OAP "Blade" ≠ Levantine "Blade") and let the user specify which tradition they're working in?

### Deliverable

- Concrete recommendations for typology modifications (what to merge, what to reorder, what to split)
- Code implementation approach for any new label mappings
- Expected accuracy impact of each change

---

## Topic 3: Expanded PH with Full Persistence Images

Our signal test used GUDHI with crude summary statistics (12 values: mean birth/lifetime/std/sum/max per dimension). A proper implementation uses **persistence images** — 2D histograms of (birth, persistence) pairs convolved with a Gaussian kernel — producing 50-100 dimensional features.

### Questions to answer

1. **Full persistence image pipeline**: What's the optimal implementation? Steps should include:
   - Mesh normalisation (scale to unit bounding box, centre at origin)
   - Filtration type (Vietoris-Rips vs Alpha complex vs mesh-based lower-star)
   - Subsample strategy (how many vertices? Farthest-point sampling vs random?)
   - Persistence image resolution and kernel bandwidth
   - Dimensionality reduction (PCA/UMAP on the image vectors)

2. **Which of our artefacts benefit most from PH?**: Handaxes have smooth faceted surfaces; blades have scar ridges; cores have complex void structure. Should PH features be computed selectively? Should different PH parameters be used for different artefact types?

3. **Integration with existing pipeline**: The 32 existing features are fast (0.1s per artefact). PH takes ~2s per artefact with GUDHI. Is there a caching strategy that makes this tolerable for users processing new artefacts?

4. **Archaeological literature**: Are there published studies using persistence images or persistent homology specifically on lithic artefacts or similar archaeological 3D objects? What parameters did they use?

5. **Quick signal test redesign**: Our test showed 41.3% PH-only accuracy vs 81.2% baseline, with no improvement when combined. Did our crude approach miss the signal? What would a proper signal test look like with full persistence images?

### Deliverable

- Complete, implementable PH pipeline specification (parameter choices, code structure)
- Expected accuracy impact range (realistic: +0.5-1pp? Optimistic: +1-3pp?)
- Decision criteria: at what point should we abort PH if results don't materialise?
- Code integration plan with existing feature extraction pipeline

---

## Synthesis

After exploring all three topics, produce an **integrated execution roadmap** answering:

1. **What should we do first, second, third?** Order by expected-impact ÷ implementation-effort.

2. **What's the realistic new ceiling for Basic/Bordes and Technological accuracy** after all three interventions?

3. **Is the 90% target on Basic/Bordes reachable**, or should we publicly recalibrate expectations to ~86-88% and focus on honesty (LOGO CV reporting) instead?

4. **What's the simplest thing that actually works?** If we had only one week, what single intervention from the three topics would give the most improvement?

5. **What should we NOT do?** — Specific approaches that sound promising but won't work given our constraints (3,415 samples, CPU-only deployment, sklearn ecosystem, <100MB models).
