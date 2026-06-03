"""retrain.py — Retrain classifier models from the training matrix.

Reads the training_matrix.csv, maps labels to three typology systems
(Basic, Bordes, Technological), trains models, and saves them.

Usage:
    python3 lithicore/data/training/retrain.py
"""

import csv
import sys
import time
import joblib
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "lithicore" / "src"))

from lithicore._classification import train_model
from lithicore._models import LithicFeatureVector

MATRIX_PATH = Path(__file__).resolve().parent / "processed" / "training_matrix.csv"
MODELS_DIR = PROJECT_ROOT / "lithicore" / "data" / "models"


def load_matrix(path: Path) -> tuple[list[LithicFeatureVector], list[dict]]:
    """Load training matrix and return (feature_vectors, metadata_rows)."""
    feature_vectors = []
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        # Feature column names (everything after the 4 metadata columns)
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
            # Handle int fields
            fv.scar_count = int(float(row.get("scar_count", "0")))
            fv.dorsal_ridge_count = int(float(row.get("dorsal_ridge_count", "0")))
            
            feature_vectors.append(fv)
            rows.append(row)

    return feature_vectors, rows


def load_metadata_lookup(csv_path: Path) -> dict[str, dict]:
    """Load metadata CSV and return {ID: row} lookup."""
    lookup = {}
    if not csv_path.exists():
        return lookup
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            aid = row.get("ID", "").strip()
            if aid:
                lookup[aid] = row
    return lookup


def get_labels(rows: list[dict], system: str, lookup: dict) -> list[str]:
    """Map artefact IDs to labels for a given typology system."""
    labels = []
    for row in rows:
        aid = row["artefact_id"]
        ds = row.get("dataset", "")

        # Handaxe/biface datasets get their own label
        # Use CSV-level typology as fallback for datasets without metadata
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
            # Basic: Core, Blade, Flake, Retouched Flake, etc.
            # Note: Bladelet is merged into Blade — the Bladelet↔Blade boundary
            # is an arbitrary threshold on a continuous length distribution, and
            # the 22 cross-confusions between them vanish on merge.
            if cls_val in ("Core", "Core-Tool"):
                labels.append("Core")
            elif cls_val == "Tool":
                if blank_val == "Blade" or blank_val == "Bladelet":
                    labels.append("Blade")
                elif blank_val == "Flake":
                    labels.append("Flake")
                else:
                    # Generic "Tool" without specific blank type — merge into
                    # Retouched Flake. The basic/bordes distinction between
                    # "tool" and "retouched flake" is one of degree, not kind,
                    # and 4 samples is too few for a viable ML class.
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
            # Technological: based on Technology + Core_classification columns
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
            # Bordes typology: based on Blank + Class
            # Bladelet merged into Blade (see basic system for rationale)
            if cls_val in ("Core", "Core-Tool"):
                labels.append("Core")
            elif blank_val in ("Blade", "Bladelet"):
                labels.append("Blade")
            elif blank_val == "Flake":
                labels.append("Flake")
            elif cls_val == "Tool":
                # Merge generic "Tool" into Retouched Flake (see basic system)
                labels.append("Retouched Flake")
            elif csv_typology and system != "technological":
                labels.append(csv_typology)
            else:
                labels.append("Other")
        else:
            if csv_typology:
                labels.append(csv_typology)
            else:
                labels.append("Unknown")
    return labels


def evaluate(X: np.ndarray, y: np.ndarray, model) -> dict:
    """Cross-validate and return metrics.

    For standard sklearn-compatible models, uses cross_val_score.
    For hierarchical/ordinal pipelines, uses a manual train/test split
    since the pipeline doesn't support sklearn's clone() protocol.
    """
    from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
    from sklearn.metrics import accuracy_score
    
    n_classes = len(set(y))
    cv = min(5, n_classes) if n_classes >= 2 else 2
    
    is_ordinal = hasattr(model._model, "ord_model")
    is_hierarchical = hasattr(model._model, "root_rf")
    
    if is_ordinal or is_hierarchical:
        # Manual train/test split for non-standard pipelines
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y,
            )
        except ValueError:
            # Stratify fails if a class has only 1 sample in the split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42,
            )
        
        model_type = "hierarchical" if is_hierarchical else "ordinal"
        try:
            model._model.fit(X_train, y_train)
        except Exception:
            # Model may already be fitted — skip refit
            pass
        y_pred = model._model.predict(X_test)
        train_pred = model._model.predict(X_train)
        return {
            "cv_mean": float(accuracy_score(y_test, y_pred)),
            "cv_std": 0.0,
            "train_acc": float(accuracy_score(y_train, train_pred)),
            "n_samples": len(y),
            "n_classes": n_classes,
            "cv_folds": 1,
            "note": f"Single train/test split ({model_type} pipeline not CV-compatible)",
        }
    else:
        scores = cross_val_score(model._model, X, y, cv=cv, scoring="accuracy")
        model._model.fit(X, y)
        y_pred = model._model.predict(X)
        
        return {
            "cv_mean": float(scores.mean()),
            "cv_std": float(scores.std()),
            "train_acc": float(accuracy_score(y, y_pred)),
            "n_samples": len(y),
            "n_classes": n_classes,
            "cv_folds": cv,
        }


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading training matrix...")
    feature_vectors, rows = load_matrix(MATRIX_PATH)
    X = np.array([fv.to_array() for fv in feature_vectors])
    print(f"  {len(feature_vectors)} artifacts, {X.shape[1]} features")

    # Load metadata lookups
    print("\nLoading metadata lookups...")
    raw_candidates = [
        Path("/data/dibble-training/raw"),
        MODELS_DIR.parent / "training" / "raw",
    ]
    raw_dir: Path | None = None
    for candidate in raw_candidates:
        d = candidate.resolve() if candidate.is_symlink() else candidate
        if d.is_dir() and any(d.glob("*_metadata.csv")):
            raw_dir = d
            break
    if raw_dir is None:
        print("  WARNING: No raw data directory found. Labels will be limited.")
        lookups = {}
    else:
        lookups = {}
        for csv_path in sorted(raw_dir.glob("*_metadata.csv")):
            name = csv_path.stem
            lookups[name] = load_metadata_lookup(csv_path)
            print(f"  {name}: {len(lookups[name])} records")
        for csv_path in sorted(raw_dir.glob("*_Dataset.csv")):
            name = csv_path.stem
            lookups[name] = load_metadata_lookup(csv_path)
            print(f"  {name}: {len(lookups[name])} records")

    # Combine lookups into a single dict (first match wins)
    master_lookup = {}
    for lu in lookups.values():
        master_lookup.update(lu)
    print(f"  Total unique IDs in metadata: {len(master_lookup)}")

    # Train each typology system
    systems = [
        ("basic", "typology_basic.joblib"),
        ("bordes", "typology_bordes.joblib"),
        ("technological", "typology_technological.joblib"),
    ]

    for sys_name, model_file in systems:
        print(f"\n{'='*60}")
        print(f"  Training: {sys_name.title()} Typology")
        print(f"{'='*60}")

        labels = get_labels(rows, sys_name, master_lookup)
        classes = sorted(set(labels))
        print(f"  Classes: {classes}")
        print(f"  Distribution:")
        from collections import Counter
        for cls, cnt in sorted(Counter(labels).items()):
            print(f"    {cls}: {cnt}")

        # Filter out low-frequency classes (< 3 samples)
        min_samples = 3
        valid = [(fv, lbl) for fv, lbl in zip(feature_vectors, labels) 
                 if labels.count(lbl) >= min_samples]
        if len(valid) < len(feature_vectors):
            dropped = len(feature_vectors) - len(valid)
            print(f"  Dropped {dropped} samples from rare classes (<{min_samples})")

        if len(valid) < 10:
            print(f"  WARNING: Only {len(valid)} samples. Skipping.")
            continue

        fv_list, lbl_list = zip(*valid)
        X_core = np.array([fv.to_array() for fv in fv_list])
        from lithicore._classification import compute_interactions
        X_inter = np.array([compute_interactions(row) for row in X_core])
        X_sub = np.concatenate([X_core, X_inter], axis=1)

        # ── Dataset-based sample weights ──
        # Counteract OAP dominance (71% of data) by weighting each sample
        # inversely proportional to its source dataset size.
        # Papers (both Deep Research) agree this is the single highest-impact
        # intervention for honest cross-dataset generalization.
        def dataset_group(name: str) -> str:
            nl = name.lower()
            if any(x in nl for x in ['fumane', 'castelcivita', 'cala', 'bombrini', 'edgeangle']):
                return 'OAP'
            if 'levantine' in nl: return 'Levantine'
            if 'coads' in nl: return 'COADS'
            if 'lombao' in nl: return 'Lombao'
            if 'morales' in nl: return 'Morales'
            return 'Other'

        # Map the valid rows to their dataset groups
        # (the valid filter may have dropped some rows, so we need valid_rows)
        valid_rows = [rows[i] for i, _ in enumerate(zip(feature_vectors, labels))
                      if labels.count(labels[i]) >= min_samples][:len(fv_list)]
        ds_groups = [dataset_group(r.get('dataset', '')) for r in valid_rows]
        from collections import Counter
        group_counts = Counter(ds_groups)
        total = len(ds_groups)
        # weight = total / group_count — datasets with fewer samples get higher weight
        ds_weights = np.array([total / max(group_counts[g], 1) for g in ds_groups])
        # Normalise so mean weight = 1.0 (prevents extreme values from destabilising RF)
        ds_weights = ds_weights / np.mean(ds_weights)
        n_unique = len(set(ds_groups))
        if n_unique > 1:
            print(f"  Dataset weights ({n_unique} groups): "
                  f"{', '.join(f'{g}={group_counts[g]}' for g in sorted(group_counts))}")
            print(f"  Weight range: {ds_weights.min():.2f} – {ds_weights.max():.2f} (mean=1.0)")
        else:
            ds_weights = None

        # Train
        t0 = time.time()

        if sys_name in ("basic", "bordes"):
            # ── Hierarchical cascade for Basic/Bordes ──
            from types import SimpleNamespace
            from lithicore._classification import HierarchicalClassifier
            hier = HierarchicalClassifier(n_features=X_sub.shape[1])
            hier.fit(X_sub, np.array(lbl_list), sample_weight=ds_weights)
            model = SimpleNamespace(
                _model=hier, _classes=hier._all_classes,
                typology_name=sys_name,
            )
        else:
            model = train_model(
                list(fv_list), list(lbl_list),
                typology_name=sys_name, sample_weight=ds_weights,
            )

        train_time = time.time() - t0

        # Evaluate
        metrics = evaluate(X_sub, np.array(lbl_list), model)

        print(f"\n  Results:")
        print(f"    Training time: {train_time:.1f}s")
        print(f"    CV accuracy: {metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")
        print(f"    Train accuracy: {metrics['train_acc']:.3f}")
        print(f"    Samples: {metrics['n_samples']}, Classes: {metrics['n_classes']}")
        print(f"    CV folds: {metrics['cv_folds']}")

        # Save model
        model_path = MODELS_DIR / model_file
        joblib.dump(model, model_path)
        size_mb = model_path.stat().st_size / 1e6
        print(f"\n  Model saved: {model_path} ({size_mb:.1f} MB)")

    # ── Phase 2: Tradition-specific models ──
    # The LOGO CV analysis confirmed that a single classifier cannot generalise
    # across assemblages. Train separate models per tradition and wrap them
    # in a TraditionRouter for user-context-aware classification.
    print(f"\n{'='*60}")
    print(f"  Training tradition-specific models")
    print(f"{'='*60}")

    from types import SimpleNamespace
    from lithicore._classification import HierarchicalClassifier, TraditionRouter, SingleClassPredictor

    def dataset_group(name: str) -> str:
        nl = name.lower()
        if any(x in nl for x in ['fumane', 'castelcivita', 'cala', 'bombrini', 'edgeangle']):
            return 'OAP'
        if 'levantine' in nl: return 'Levantine'
        if 'coads' in nl: return 'COADS'
        if 'lombao' in nl or 'morales' in nl: return 'Experimental'
        return 'Other'

    # Group rows by tradition
    tradition_rows: dict[str, list[dict]] = {}
    for row in rows:
        trad = dataset_group(row.get('dataset', ''))
        tradition_rows.setdefault(trad, []).append(row)

    for sys_name, sys_label in [('basic', 'Basic'), ('bordes', 'Bordes'), ('technological', 'Technological')]:
        print(f"\n  --- {sys_label} ---")
        router = TraditionRouter()

        for trad_name in sorted(tradition_rows.keys()):
            trad_rows = tradition_rows[trad_name]
            if len(trad_rows) < 20:
                print(f"    {trad_name}: {len(trad_rows)} samples (skipped, <20)")
                continue

            # Get labels for this tradition's rows
            trad_labels = get_labels(trad_rows, sys_name, master_lookup)
            trad_classes = sorted(set(trad_labels))
            if len(trad_classes) < 1:
                continue

            # Get feature vectors
            trad_fvs = []
            for r in trad_rows:
                fv = LithicFeatureVector()
                for name in LithicFeatureVector.FEATURE_NAMES:
                    val = r.get(name, '0')
                    try: setattr(fv, name, float(val))
                    except: pass
                fv.scar_count = int(float(r.get('scar_count', '0')))
                fv.dorsal_ridge_count = int(float(r.get('dorsal_ridge_count', '0')))
                trad_fvs.append(fv)

            X_trad_core = np.array([fv.to_array() for fv in trad_fvs])
            X_trad_inter = np.array([compute_interactions(row) for row in X_trad_core])
            X_trad = np.concatenate([X_trad_core, X_trad_inter], axis=1)
            y_trad = np.array(trad_labels)

            if len(trad_classes) == 1:
                # Single-class tradition: trivial predictor
                single_class = trad_classes[0]
                router.add_model(trad_name, SingleClassPredictor(single_class), trad_classes)
                print(f"    {trad_name}: {len(trad_rows)} samples, 1 class ({single_class}) — trivial")
            elif sys_name in ("basic", "bordes") and len(trad_classes) >= 3:
                # Hierarchical for multi-class traditions
                trad_hier = HierarchicalClassifier(n_features=X_trad.shape[1])
                trad_hier.fit(X_trad, y_trad)
                router.add_model(trad_name, trad_hier, trad_classes)
                # Evaluate
                from sklearn.model_selection import train_test_split
                X_tr, X_te, y_tr, y_te = train_test_split(X_trad, y_trad, test_size=0.2, random_state=42)
                trad_hier.fit(X_tr, y_tr)
                acc = float(np.mean(trad_hier.predict(X_te) == y_te))
                print(f"    {trad_name}: {len(trad_rows)} samples, {len(trad_classes)} classes, "
                      f"hierarchical, holdout acc: {acc:.1%}")
            else:
                # Simple RF for small traditions
                trad_rf = train_model(trad_fvs, trad_labels, typology_name=f"{sys_name}_{trad_name}")
                router.add_model(trad_name, trad_rf._model, trad_classes)
                print(f"    {trad_name}: {len(trad_rows)} samples, {len(trad_classes)} classes, flat RF")

        # Save router
        router_path = MODELS_DIR / f"typology_{sys_name}_traditions.joblib"
        joblib.dump(router, router_path)
        size_mb = router_path.stat().st_size / 1e6
        print(f"  Router saved: {router_path} ({size_mb:.1f} MB)")

    print(f"\n{'='*60}")
    print(f"  All models retrained!")
    print(f"  Combined models: typology_basic/bordes/technological.joblib")
    print(f"  Tradition routers: typology_basic/bordes/technological_traditions.joblib")


if __name__ == "__main__":
    main()
