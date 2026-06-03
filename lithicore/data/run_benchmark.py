"""run_benchmark.py — Real-data classifier validation benchmark.

Loads the actual 3,312-artefact training matrix, runs all three pre-trained
classifiers, and produces an interactive HTML validation report with confusion
matrices, per-class precision/recall/F1, and cross-validation accuracy.

OOM-safe design:
- Training matrix is small (3,312 x 22 floats — ~73 KiB + CSV overhead)
- Only one model loaded at a time (48-50 MB each)
- Model reference is deleted + gc.collect() between typologies

Usage:
    # Via CLI (recommended):
    lithicore benchmark

    # Direct:
    python -m lithicore.data.run_benchmark
"""

from __future__ import annotations

import csv
import gc
import json
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

# ── Path setup (works for both `python -m` and direct execution) ──
_script_dir = Path(__file__).resolve().parent
_src_dir = _script_dir.parent.parent / "lithicore" / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from lithicore._classification import ClassifierModel as _ClassifierModel  # noqa: E402
from lithicore._models import LithicFeatureVector  # noqa: E402 — used in load_matrix()

PROJECT_ROOT = _script_dir.parent.parent
MATRIX_PATH = PROJECT_ROOT / "lithicore" / "data" / "training" / "processed" / "training_matrix.csv"
MODELS_DIR = PROJECT_ROOT / "lithicore" / "data" / "models"
BENCHMARK_DIR = PROJECT_ROOT / "docs" / "benchmark"
RESULTS_DIR = BENCHMARK_DIR / "results"

TYPOLOGY_MODELS = [
    ("basic", "typology_basic.joblib", "Basic Morphological"),
    ("bordes", "typology_bordes.joblib", "Bordes Typology"),
    ("technological", "typology_technological.joblib", "Technological"),
]


# ── Training data loading (mirrors retrain.py helpers) ──

def load_matrix(path: Path) -> tuple[list[LithicFeatureVector], list[dict]]:
    """Load training matrix and return (feature_vectors, metadata_rows).

    Streams the CSV row-by-row — never holds more than one row + the full
    features list in memory (~1.5 MB total for 3,312 artefacts).
    """
    feature_vectors: list[LithicFeatureVector] = []
    rows: list[dict] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames) if reader.fieldnames else []
        feature_names = [c for c in fieldnames if c not in (
            "artefact_id", "typology", "dataset", "source_csv"
        )]
        for row in reader:
            fv = LithicFeatureVector()
            for name in feature_names:
                val = row.get(name, "0")
                try:
                    setattr(fv, name, float(val))
                except (ValueError, AttributeError):
                    pass
            fv.scar_count = int(float(row.get("scar_count", "0")))
            fv.dorsal_ridge_count = int(float(row.get("dorsal_ridge_count", "0")))
            feature_vectors.append(fv)
            rows.append(row)
    return feature_vectors, rows


def load_metadata_lookup(csv_path: Path) -> dict[str, dict]:
    """Load metadata CSV and return {ID: row} lookup."""
    lookup: dict[str, dict] = {}
    if not csv_path.exists():
        return lookup
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            aid = row.get("ID", "").strip()
            if aid:
                lookup[aid] = row
    return lookup


def get_labels(rows: list[dict], system: str, lookup: dict) -> list[str]:
    """Map artefact IDs to labels for a given typology system.

    This is an exact copy of retrain.py's label resolution logic so that
    benchmark predictions are evaluated against the same ground truth used
    during training.
    """
    labels: list[str] = []
    for row in rows:
        aid = row["artefact_id"]
        ds = row.get("dataset", "")
        csv_typology = row.get("typology", "")

        if "Levantine_Acheulean" in ds or "COADS" in ds:
            if system == "technological":
                labels.append("Handaxe")
            else:
                labels.append("Biface")
            continue

        meta = lookup.get(aid, {})
        cls_val = meta.get("Class", "").strip()
        blank_val = meta.get("Blank", "").strip()

        if system == "basic":
            # Bladelet merged into Blade (see retrain.py for rationale)
            if cls_val in ("Core", "Core-Tool"):
                labels.append("Core")
            elif cls_val == "Tool":
                if blank_val in ("Blade", "Bladelet"):
                    labels.append("Blade")
                elif blank_val == "Flake":
                    labels.append("Flake")
                else:
                    # Merge generic "Tool" into Retouched Flake (4 samples not viable)
                    labels.append("Retouched Flake")
            elif blank_val in ("Blade", "Bladelet"):
                labels.append("Blade")
            elif blank_val == "Flake":
                labels.append("Flake")
            elif csv_typology and system == "basic":
                labels.append(csv_typology)
            else:
                labels.append("Other")

        elif system == "technological":
            tech = meta.get("Technology", "").strip()
            core = meta.get("Core_classification", "").strip()
            if tech:
                labels.append(tech)
            elif core:
                labels.append(core)
            elif cls_val == "Core":
                labels.append("Core reduction")
            elif cls_val == "Tool":
                labels.append("Tool production")
            else:
                labels.append("Unknown")

        elif system == "bordes":
            # Bladelet merged into Blade (see retrain.py for rationale)
            if cls_val in ("Core", "Core-Tool"):
                labels.append("Core")
            elif blank_val in ("Blade", "Bladelet"):
                labels.append("Blade")
            elif blank_val == "Flake":
                labels.append("Flake")
            elif cls_val == "Tool":
                labels.append("Retouched Flake")
            elif csv_typology:
                labels.append(csv_typology)
            else:
                labels.append("Other")
        else:
            labels.append(csv_typology or "Unknown")
    return labels


# ── Benchmark helpers ──

def _compute_metrics(
    y_true: list[str],
    y_pred: list[str],
    classes: list[str],
) -> dict:
    """Compute accuracy, per-class precision/recall/F1, and confusion matrix.

    Pure numpy — no sklearn dependency for the hot path.
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix,
        classification_report,
    )

    y_t = np.array(y_true)
    y_p = np.array(y_pred)

    accuracy = float(accuracy_score(y_t, y_p))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_t, y_p, labels=classes, zero_division=0,
    )
    cm = confusion_matrix(y_t, y_p, labels=classes)

    per_class = [
        {
            "class": cls,
            "precision": round(float(p), 3),
            "recall": round(float(r), 3),
            "f1": round(float(f), 3),
            "support": int(s),
        }
        for cls, p, r, f, s in zip(classes, precision, recall, f1, support)
    ]

    return {
        "accuracy": round(accuracy, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "classes": classes,
        "n_samples": len(y_t),
        "n_classes": len(classes),
        "classification_report": classification_report(
            y_t, y_p, labels=classes, zero_division=0,
        ),
    }


def _crossval_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
) -> tuple[float, float]:
    """Run cross-validation and return (mean_accuracy, std_accuracy).

    Uses a bare RandomForest (no calibration) to avoid the ~2x overhead of
    CalibratedClassifierCV's internal refit. Gives an unbiased accuracy
    estimate — calibration affects probability scores, not hard predictions.

    OOM-safe: single-threaded (n_jobs=1), one fold at a time via manual loop.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    n_classes = len(set(y.tolist()))
    cv_k = min(cv_folds, min(Counter(y.tolist()).values()))
    if cv_k < 2:
        return 0.0, 0.0

    skf = StratifiedKFold(n_splits=cv_k, shuffle=True, random_state=42)
    scores: list[float] = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=min(20, max(12, n_classes * 2)),
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        scores.append(float(accuracy_score(y_test, y_pred)))

        # Free fold — OOM safety
        del rf, X_train, X_test, y_train, y_test, y_pred

    mean_cv = float(np.mean(scores))
    std_cv = float(np.std(scores))
    return mean_cv, std_cv


# ── HTML report generator ──

def _generate_html(results: list[dict], cv_results: list[dict]) -> str:
    """Generate an interactive HTML validation report."""
    all_pass = all(r.get("accuracy", 0) >= 0.60 for r in results if "error" not in r)
    overall_status = "PASS" if all_pass else "REVIEW"

    html_parts = [f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Dibble Classifier Validation Report — Real Data</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         max-width: 960px; margin: 0 auto; padding: 20px; background: #f8f9fa; color: #333; }}
  h1 {{ color: #1a1a2e; border-bottom: 3px solid #4472C4; padding-bottom: 8px; }}
  h2 {{ color: #2d3436; margin-top: 32px; }}
  .status {{ display: inline-block; padding: 6px 16px; border-radius: 4px;
             font-weight: bold; font-size: 14px; }}
  .pass {{ background: #d4edda; color: #155724; }}
  .review {{ background: #fff3cd; color: #856404; }}
  .summary {{ background: white; border-radius: 8px; padding: 16px; margin: 16px 0;
              box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
  th, td {{ border: 1px solid #dee2e6; padding: 8px 12px; text-align: left; }}
  th {{ background: #4472C4; color: white; }}
  tr:nth-child(even) {{ background: #f2f2f2; }}
  .cm-table td {{ text-align: center; font-family: monospace; min-width: 60px; }}
  .cm-header {{ background: #e8e8e8 !important; font-weight: bold; }}
  .cm-diagonal {{ background: #d4edda !important; }}
  .metrics {{ display: flex; gap: 16px; flex-wrap: wrap; }}
  .metric-card {{ background: white; border-radius: 8px; padding: 16px; flex: 1;
                  min-width: 200px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .metric-value {{ font-size: 28px; font-weight: bold; color: #4472C4; }}
  .metric-label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
  pre {{ background: #f5f5f5; padding: 12px; border-radius: 4px; overflow-x: auto;
        font-size: 12px; }}
  .note {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 8px 12px;
          margin: 12px 0; border-radius: 4px; }}
  .footer {{ margin-top: 40px; padding: 12px; text-align: center;
             color: #888; font-size: 12px; border-top: 1px solid #dee2e6; }}
</style>
</head>
<body>
<h1>Dibble Classifier Validation Report</h1>
<p>Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
<p class="status {'pass' if all_pass else 'review'}">{overall_status}</p>
<p>Validated against the real <b>{results[0]['n_samples'] if results else 0}</b>-artefact
training matrix ({results[0]['n_features'] if results else 0} features).
Results include pre-trained model accuracy (full training set) and
5-fold cross-validation accuracy (unbiased estimate).</p>
<div class="note">
  <strong>Note:</strong> The pre-trained accuracy is an optimistic upper bound
  (model tested on data it trained on). The cross-validation accuracy is the
  honest estimate of real-world performance.
</div>
"""]

    for i, result in enumerate(results):
        if "error" in result:
            html_parts.append(
                f"<h2>{result.get('display_name', 'Unknown')}</h2>"
                f"<p>Error: {result['error']}</p>"
            )
            continue

        cv = cv_results[i] if i < len(cv_results) else {}

        html_parts.append(f"""
<h2>{result['display_name']} — {result['n_classes']} classes, {result['n_samples']} samples</h2>
<div class="summary">
<div class="metrics">
  <div class="metric-card">
    <div class="metric-value">{result['accuracy']:.1%}</div>
    <div class="metric-label">Pre-trained Accuracy</div>
  </div>
  <div class="metric-card" style="border-left: 3px solid #ffc107;">
    <div class="metric-value">{cv.get('cv_mean', 0):.1%} ± {cv.get('cv_std', 0):.1%}</div>
    <div class="metric-label">Cross-Validation (5-fold)</div>
  </div>
  <div class="metric-card">
    <div class="metric-value">{result['n_classes']}</div>
    <div class="metric-label">Classes</div>
  </div>
  <div class="metric-card">
    <div class="metric-value">{result['n_samples']}</div>
    <div class="metric-label">Samples</div>
  </div>
</div>

<h3>Per-Class Metrics (pre-trained model)</h3>
<table>
<tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr>
""")
        for pc in result["per_class"]:
            f1_colour = "green" if pc["f1"] >= 0.8 else "orange" if pc["f1"] >= 0.6 else "red"
            html_parts.append(f"""
<tr>
  <td><b>{pc['class']}</b></td>
  <td>{pc['precision']:.3f}</td>
  <td>{pc['recall']:.3f}</td>
  <td style="color:{f1_colour};font-weight:bold;">{pc['f1']:.3f}</td>
  <td>{pc['support']}</td>
</tr>""")

        html_parts.append("""
</table>

<h3>Confusion Matrix (pre-trained model)</h3>
<table class="cm-table">
<tr><td class="cm-header">True \\ Pred</td>""")
        for cls in result["classes"]:
            html_parts.append(f"<td class='cm-header'>{cls}</td>")
        html_parts.append("</tr>")

        cm = result["confusion_matrix"]
        for i_row, cls in enumerate(result["classes"]):
            html_parts.append(f"<tr><td class='cm-header'>{cls}</td>")
            for j, val in enumerate(cm[i_row]):
                diag_class = "cm-diagonal" if i_row == j else ""
                html_parts.append(f"<td class='{diag_class}'>{val}</td>")
            html_parts.append("</tr>")

        html_parts.append("""
</table>

<h3>Classification Report</h3>
<pre>""" + result["classification_report"] + "</pre>")

    html_parts.append(f"""
<div class="footer">
  Dibble Classifier Validation — <a href="https://github.com/mabo-du/dibble">github.com/mabo-du/dibble</a><br>
  Data: {results[0]['n_samples'] if results else 0} real-world artefacts<br>
  Reproduce: <code>lithicore benchmark</code>
</div>
</body>
</html>""")

    return "\n".join(html_parts)


# ── Main ──

def main() -> None:
    """Run all benchmarks and generate the HTML report."""
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Dibble Classifier Validation Benchmark (Real Data)")
    print("=" * 60)

    # ── Phase 1: Load training data ──
    print("\nLoading training matrix...")
    t0 = time.time()
    feature_vectors, rows = load_matrix(MATRIX_PATH)
    X_core = np.array([fv.to_array() for fv in feature_vectors])
    from lithicore._classification import compute_interactions
    X_inter = np.array([compute_interactions(row) for row in X_core])
    X = np.concatenate([X_core, X_inter], axis=1)
    # Also load PH features if available
    from lithicore._ph_features import CACHE_DIR, load_ph_matrix
    all_aids = [r['artefact_id'] for r in rows]
    X_ph, ph_valid_idx = load_ph_matrix(all_aids, cache_dir=CACHE_DIR)
    if X_ph is not None and len(ph_valid_idx) > 0:
        n_ph = X_ph.shape[1]
        X_ph_full = np.zeros((len(X), n_ph), dtype=float)
        for i_ph, i_orig in enumerate(ph_valid_idx):
            if i_orig < len(X_ph_full):
                X_ph_full[i_orig] = X_ph[i_ph]
        X = np.concatenate([X, X_ph_full], axis=1)
        print(f"  PH features loaded: +{n_ph} dims ({X.shape[1]} total)")
    del X_ph, X_ph_full, all_aids
    print(f"  {len(feature_vectors)} artefacts, {X.shape[1]} features ({time.time()-t0:.1f}s)")

    # Free feature_vectors list — we only need X and rows from here
    del feature_vectors
    gc.collect()

    # Load metadata lookups
    print("Loading metadata lookups...")
    t0 = time.time()
    # Metadata CSVs are at the canonical raw data path (may be a symlink
    # to /data/dibble-training/raw, or the absolute path on other machines).
    raw_candidates = [
        MODELS_DIR.parent / "training" / "raw",          # symlink in repo
        Path("/data/dibble-training/raw"),               # canonical path
    ]
    raw_dir: Path | None = None
    for candidate in raw_candidates:
        d = candidate.resolve() if candidate.is_symlink() else candidate
        if d.is_dir() and any(d.glob("*_metadata.csv")):
            raw_dir = d
            break

    lookups: dict[str, dict] = {}
    if raw_dir is not None:
        for csv_path in sorted(raw_dir.glob("*_metadata.csv")):
            lookups[csv_path.stem] = load_metadata_lookup(csv_path)
            print(f"  {csv_path.stem}: {len(lookups[csv_path.stem])} records")
        for csv_path in sorted(raw_dir.glob("*_Dataset.csv")):
            lookups[csv_path.stem] = load_metadata_lookup(csv_path)
            print(f"  {csv_path.stem}: {len(lookups[csv_path.stem])} records")
    else:
        print("  WARNING: No metadata CSVs found. Labels will use CSV-level typology only.")
    master_lookup: dict[str, dict] = {}
    for lu in lookups.values():
        master_lookup.update(lu)
    print(f"  Total unique IDs: {len(master_lookup)} ({time.time()-t0:.1f}s)")

    # Free individual lookups — keep only master
    del lookups
    gc.collect()

    # ── Phase 2: Benchmark each typology ──
    results: list[dict] = []
    cv_results: list[dict] = []

    for sys_name, model_file, display_name in TYPOLOGY_MODELS:
        print(f"\n{'─'*60}")
        print(f"  {display_name} ({sys_name})")
        print(f"{'─'*60}")

        # Resolve labels
        labels = get_labels(rows, sys_name, master_lookup)
        classes = sorted(set(labels))
        print(f"  Labels: {len(labels)} artefacts, {len(classes)} classes")

        # ── Phase 2a: Pre-trained model evaluation ──
        model_path = MODELS_DIR / model_file
        if not model_path.exists():
            print(f"  Model not found: {model_path}. Run retrain.py first.")
            results.append({
                "name": sys_name,
                "display_name": display_name,
                "error": "Model not found",
            })
            cv_results.append({})
            continue

        print(f"  Loading model ({model_path.stat().st_size / 1e6:.0f} MB)...")
        t1 = time.time()
        model = _ClassifierModel.load_pre_trained(sys_name)
        print(f"  Loaded in {time.time()-t1:.1f}s")

        # Predict on full training set (batch prediction via raw model)
        # Using _model.predict directly is ~100x faster than calling the
        # per-artefact predict() method on 3,312 artefacts.
        print("  Predicting on full training set (batch)...")
        t2 = time.time()
        probs = model._model.predict_proba(X)  # (n_samples, n_classes)
        y_pred = [model._classes[int(np.argmax(p))] for p in probs]
        train_time = time.time() - t2

        # Compute metrics
        result_metrics = _compute_metrics(labels, y_pred, classes)
        result_metrics["name"] = sys_name
        result_metrics["display_name"] = display_name
        result_metrics["n_features"] = X.shape[1]
        results.append(result_metrics)

        print(f"  Pre-trained accuracy: {result_metrics['accuracy']:.1%} "
              f"({result_metrics['n_samples']} samples, {train_time:.1f}s)")

        # Free model — OOM safety
        del model, y_pred
        gc.collect()

        # ── Phase 2b: Cross-validation accuracy estimate ──
        print("  Cross-validating (5-fold)...")
        t3 = time.time()
        try:
            cv_mean, cv_std = _crossval_accuracy(X, np.array(labels))
            cv_time = time.time() - t3
            print(f"  CV accuracy: {cv_mean:.1%} ± {cv_std:.1%} ({cv_time:.1f}s)")
            cv_results.append({"cv_mean": cv_mean, "cv_std": cv_std})
        except Exception as exc:
            print(f"  CV error: {exc}")
            cv_results.append({})

        # Save per-typology metrics JSON
        (RESULTS_DIR / f"{sys_name}_metrics.json").write_text(
            json.dumps(result_metrics, indent=2, default=str)
        )

        gc.collect()

    # Free remaining bulk data
    del X, rows, master_lookup
    gc.collect()

    # ── Phase 3: Tradition-specific evaluation ──
    print(f"\n{'─'*60}")
    print("  Tradition-Specific Evaluation")
    print(f"{'─'*60}")

    def dataset_group(name: str) -> str:
        nl = name.lower()
        if any(x in nl for x in ['fumane', 'castelcivita', 'cala', 'bombrini', 'edgeangle']):
            return 'OAP'
        if 'levantine' in nl: return 'Levantine'
        if 'coads' in nl: return 'COADS'
        if 'lombao' in nl or 'morales' in nl: return 'Experimental'
        return 'Other'

    # Reload X and rows (they were freed above)
    feature_vectors, rows = load_matrix(MATRIX_PATH)
    X_core = np.array([fv.to_array() for fv in feature_vectors])
    X_inter = np.array([compute_interactions(row) for row in X_core])
    X = np.concatenate([X_core, X_inter], axis=1)
    # Also load PH features
    all_aids = [r['artefact_id'] for r in rows]
    X_ph, ph_valid_idx = load_ph_matrix(all_aids, cache_dir=CACHE_DIR)
    if X_ph is not None and len(ph_valid_idx) > 0:
        n_ph = X_ph.shape[1]
        X_ph_full = np.zeros((len(X), n_ph), dtype=float)
        for i_ph, i_orig in enumerate(ph_valid_idx):
            if i_orig < len(X_ph_full):
                X_ph_full[i_orig] = X_ph[i_ph]
        X = np.concatenate([X, X_ph_full], axis=1)
    del feature_vectors, X_ph, X_ph_full, all_aids; gc.collect()

    # Reload metadata
    master_lookup = {}
    for candidate in raw_candidates:
        d = candidate.resolve() if candidate.is_symlink() else candidate
        if d.is_dir() and any(d.glob("*_metadata.csv")):
            for csv_path in sorted(d.glob("*_metadata.csv")) + sorted(d.glob("*_Dataset.csv")):
                with open(csv_path) as f:
                    for r in csv.DictReader(f):
                        aid = r.get('ID', '').strip()
                        if aid: master_lookup[aid] = r
            break

    for sys_name, _, display_name in TYPOLOGY_MODELS:
        router_path = MODELS_DIR / f"typology_{sys_name}_traditions.joblib"
        if not router_path.exists():
            print(f"\n  {display_name}: tradition router not found (run retrain.py)")
            continue

        print(f"\n  {display_name} — Per-Tradition Accuracy:")
        print(f"  {'Tradition':<20} {'Samples':>8} {'Classes':>8} {'Accuracy':>10}")
        print(f"  {'-'*48}")

        labels = get_labels(rows, sys_name, master_lookup)
        router = joblib.load(str(router_path))

        for trad_name in sorted(router.traditions):
            # Get this tradition's data
            trad_mask = [dataset_group(r.get('dataset', '')) == trad_name for r in rows]
            if not any(trad_mask):
                continue
            X_trad = X[trad_mask]
            y_trad = np.array([labels[i] for i, m in enumerate(trad_mask) if m])
            classes_trad = sorted(set(y_trad))

            # Predict using tradition-specific model (trained on 32 features, not 47)
            X_trad_32 = X_trad[:, :32]  # Slice to core 32 features
            trad_model = router.models[trad_name]
            if hasattr(trad_model, 'predict'):
                y_pred = trad_model.predict(X_trad_32)
            else:
                y_pred = np.full(len(y_trad), trad_model._classes[0])

            acc = float(np.mean(y_pred == y_trad))
            print(f"  {trad_name:<20} {len(y_trad):>8} {len(classes_trad):>8} {acc:>10.1%}")

        # Also test the combined model on each tradition for comparison
        model_path = MODELS_DIR / f"typology_{sys_name}.joblib"
        if model_path.exists():
            combined = _ClassifierModel.load_pre_trained(sys_name)
            print(f"\n  Combined model reference:")
            for trad_name in sorted(router.traditions):
                trad_mask = [dataset_group(r.get('dataset', '')) == trad_name for r in rows]
                if not any(trad_mask):
                    continue
                X_trad = X[trad_mask]
                y_trad = np.array([labels[i] for i, m in enumerate(trad_mask) if m])
                X_trad_32 = X_trad[:, :32]  # slice to 32 features
                try:
                    probs = combined._model.predict_proba(X_trad)
                    y_pred = [combined._classes[int(np.argmax(p))] for p in probs]
                    acc = float(np.mean(y_pred == y_trad))
                    print(f"    {trad_name:<18} combined model: {acc:.1%}")
                except Exception as e:
                    print(f"    {trad_name:<18} combined model error: {e}")
            del combined; gc.collect()

    # ── Phase 4: Generate report ──
    print(f"\n{'─'*60}")
    print("  Generating report...")

    # Save benchmark config
    config = {
        "benchmark_date": __import__("datetime").datetime.now().isoformat(),
        "classifiers": [t[0] for t in TYPOLOGY_MODELS],
        "n_artefacts": results[0]["n_samples"] if results else 0,
        "n_features": results[0].get("n_features", 22) if results else 22,
    }
    (RESULTS_DIR / "config.json").write_text(json.dumps(config, indent=2))

    # Generate HTML
    html = _generate_html(results, cv_results)
    report_path = RESULTS_DIR / "report.html"
    report_path.write_text(html)
    print(f"  Report saved: {report_path}")

    # Generate summary markdown
    summary_lines = ["# Classifier Validation Summary (Real Data)\n\n"]
    summary_lines.append(
        "| Typology | Pre-trained Acc. | CV Accuracy | Classes | Samples |\n"
    )
    summary_lines.append(
        "|----------|-----------------|-------------|---------|---------|\n"
    )
    for r, cv in zip(results, cv_results):
        if "error" in r:
            summary_lines.append(f"| {r['display_name']} | ERROR | — | — | — |\n")
            continue
        acc = f"{r['accuracy']:.1%}"
        cv_acc = f"{cv.get('cv_mean', 0):.1%} ± {cv.get('cv_std', 0):.1%}" if cv else "—"
        summary_lines.append(
            f"| {r['display_name']} | {acc} | {cv_acc} "
            f"| {r['n_classes']} | {r['n_samples']} |\n"
        )
    summary_lines.append(
        f"\n_Data: {config['n_artefacts']} real-world artefacts, "
        f"{config['n_features']} features._\n"
    )
    (RESULTS_DIR / "summary.md").write_text("".join(summary_lines))
    print(f"  Summary saved: {RESULTS_DIR / 'summary.md'}")

    print(f"\n{'='*60}")
    print("  Benchmark complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
