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
    For ordinal hierarchical models (Technological), uses a manual train/test
    split since the pipeline doesn't support sklearn's clone() protocol.
    """
    from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
    from sklearn.metrics import accuracy_score
    
    n_classes = len(set(y))
    cv = min(5, n_classes) if n_classes >= 2 else 2
    
    is_ordinal = hasattr(model._model, "ord_model")
    
    if is_ordinal:
        # Manual train/test split for ordinal pipeline
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y,
        )
        model._model.fit(X_train, y_train)
        y_pred = model._model.predict(X_test)
        train_pred = model._model.predict(X_train)
        return {
            "cv_mean": float(accuracy_score(y_test, y_pred)),
            "cv_std": 0.0,
            "train_acc": float(accuracy_score(y_train, train_pred)),
            "n_samples": len(y),
            "n_classes": n_classes,
            "cv_folds": 1,
            "note": "Single train/test split (ordinal pipeline not CV-compatible)",
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

        # Train
        t0 = time.time()

        if sys_name == "technological":
            # ── Flat RF for Technological (ordinal approaches tested but
            #     underperformed flat RF on this feature set) ──
            #
            # Tested alternatives and their issues:
            #   LogisticAT (linear ordinal): collapsed → all predictions to Optimal
            #   Cumulative-link RF (non-linear ordinal): Optimal recall 93%→67%
            #
            # The flat RF achieves 73.6% CV. Ordinal regression on non-linear
            # morphometric features needs a dedicated kernel or metric-learning
            # approach to be viable — filed as future research.
            model = train_model(list(fv_list), list(lbl_list), typology_name=sys_name)

        else:
            model = train_model(list(fv_list), list(lbl_list), typology_name=sys_name)

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

    print(f"\n{'='*60}")
    print(f"  All models retrained!")


if __name__ == "__main__":
    main()
