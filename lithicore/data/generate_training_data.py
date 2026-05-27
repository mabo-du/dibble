"""generate_training_data.py — Generate synthetic training data for lithic classifiers.

Generates feature vectors from published metric ranges for each lithic typology
system (basic, Bordes, technological), with added Gaussian noise to simulate
natural variation.

Output: .joblib files in data/models/

Usage:
    python -m lithicore.data.generate_training_data
"""

from pathlib import Path
import numpy as np
from lithicore._models import LithicFeatureVector


MODEL_DIR = Path(__file__).resolve().parent / "models"

# ── Basic typology: Flake, Blade, Bladelet, Core, Tool ──

BASIC_RANGES = {
    "Flake": {
        "length_mm": (15, 80), "width_mm": (10, 60), "thickness_mm": (3, 20),
        "elongation": (0.8, 1.8), "flatness": (2.0, 5.0),
        "scar_count": (1, 5),
        "platform_angle_deg": (60, 90),
        "edge_angle_mean_deg": (40, 70),
        "dorsal_ridge_count": (0, 1),
        "curvature_index": (0.1, 0.5),
        "symmetry_score": (0.3, 0.7),
        "com_z_ratio": (0.2, 0.4),
    },
    "Blade": {
        "length_mm": (50, 250), "width_mm": (10, 40), "thickness_mm": (2, 10),
        "elongation": (2.0, 5.0), "flatness": (3.0, 8.0),
        "scar_count": (2, 6),
        "platform_angle_deg": (65, 85),
        "edge_angle_mean_deg": (50, 75),
        "dorsal_ridge_count": (2, 4),
        "curvature_index": (0.05, 0.3),
        "symmetry_score": (0.5, 0.9),
        "com_z_ratio": (0.15, 0.35),
    },
    "Bladelet": {
        "length_mm": (10, 50), "width_mm": (3, 12), "thickness_mm": (1, 4),
        "elongation": (2.5, 6.0), "flatness": (3.0, 7.0),
        "scar_count": (1, 3),
        "platform_angle_deg": (60, 80),
        "edge_angle_mean_deg": (40, 65),
        "dorsal_ridge_count": (1, 2),
        "curvature_index": (0.05, 0.25),
        "symmetry_score": (0.4, 0.8),
        "com_z_ratio": (0.2, 0.4),
    },
    "Core": {
        "length_mm": (30, 150), "width_mm": (25, 100), "thickness_mm": (15, 80),
        "elongation": (0.5, 1.5), "flatness": (1.0, 2.5),
        "scar_count": (3, 20),
        "platform_angle_deg": (70, 90),
        "edge_angle_mean_deg": (60, 85),
        "dorsal_ridge_count": (0, 2),
        "curvature_index": (0.2, 0.6),
        "symmetry_score": (0.2, 0.5),
        "com_z_ratio": (0.3, 0.7),
    },
    "Tool": {
        "length_mm": (20, 120), "width_mm": (15, 70), "thickness_mm": (5, 30),
        "elongation": (0.8, 2.5), "flatness": (2.0, 4.0),
        "scar_count": (2, 8),
        "platform_angle_deg": (50, 80),
        "edge_angle_mean_deg": (55, 85),
        "dorsal_ridge_count": (0, 2),
        "curvature_index": (0.1, 0.4),
        "symmetry_score": (0.4, 0.8),
        "com_z_ratio": (0.2, 0.5),
    },
}

# ── Bordes typology ──

BORDES_RANGES = {
    "Scraper": {
        "length_mm": (30, 100), "width_mm": (20, 60), "thickness_mm": (5, 20),
        "elongation": (0.8, 2.0), "edge_angle_mean_deg": (60, 85),
        "edge_angle_std_deg": (5, 15),  # continuous retouch → consistent edge
        "scar_count": (3, 10), "dorsal_ridge_count": (0, 2),
        "symmetry_score": (0.3, 0.6),
    },
    "Handaxe": {
        "length_mm": (80, 250), "width_mm": (50, 120), "thickness_mm": (20, 60),
        "elongation": (1.2, 2.2), "edge_angle_mean_deg": (50, 75),
        "edge_angle_std_deg": (8, 20),
        "scar_count": (5, 20), "dorsal_ridge_count": (1, 3),
        "symmetry_score": (0.7, 0.95),
    },
    "Point": {
        "length_mm": (30, 100), "width_mm": (10, 35), "thickness_mm": (3, 12),
        "elongation": (1.5, 3.5), "edge_angle_mean_deg": (55, 80),
        "edge_angle_std_deg": (5, 15),
        "scar_count": (2, 6), "dorsal_ridge_count": (1, 3),
        "symmetry_score": (0.6, 0.9),
    },
    "Burin": {
        "length_mm": (20, 80), "width_mm": (10, 30), "thickness_mm": (4, 15),
        "elongation": (1.0, 3.0), "edge_angle_mean_deg": (70, 90),
        "edge_angle_std_deg": (3, 10),  # burin spall → very regular edge
        "scar_count": (1, 4), "dorsal_ridge_count": (0, 1),
        "symmetry_score": (0.3, 0.6),
    },
    "Denticulate": {
        "length_mm": (20, 70), "width_mm": (15, 45), "thickness_mm": (4, 15),
        "elongation": (0.8, 2.0), "edge_angle_mean_deg": (45, 65),
        "edge_angle_std_deg": (15, 30),  # serrated → highly irregular edge
        "scar_count": (3, 8), "dorsal_ridge_count": (0, 1),
        "symmetry_score": (0.3, 0.6),
    },
    "Notched": {
        "length_mm": (20, 70), "width_mm": (15, 45), "thickness_mm": (4, 15),
        "elongation": (0.8, 2.0), "edge_angle_mean_deg": (50, 70),
        "edge_angle_std_deg": (10, 22),  # one/two notches → moderately irregular
        "scar_count": (2, 5), "dorsal_ridge_count": (0, 1),
        "symmetry_score": (0.3, 0.6),
    },
    "Backed knife": {
        "length_mm": (40, 150), "width_mm": (15, 40), "thickness_mm": (3, 12),
        "elongation": (2.0, 4.0), "edge_angle_mean_deg": (60, 80),
        "edge_angle_std_deg": (5, 15),
        "scar_count": (2, 5), "dorsal_ridge_count": (1, 2),
        "symmetry_score": (0.4, 0.7),
    },
}

# ── Technological typology ──

TECH_RANGES = {
    "Primary": {
        "length_mm": (30, 120), "width_mm": (20, 80), "thickness_mm": (8, 30),
        "elongation": (0.8, 2.0), "scar_count": (0, 1),
        "surface_roughness": (0.7, 1.0),
    },
    "Secondary": {
        "length_mm": (20, 100), "width_mm": (15, 60), "thickness_mm": (5, 25),
        "elongation": (0.8, 2.0), "scar_count": (1, 3),
        "surface_roughness": (0.5, 0.9),
    },
    "Tertiary": {
        "length_mm": (15, 80), "width_mm": (10, 50), "thickness_mm": (3, 15),
        "elongation": (0.8, 2.5), "scar_count": (2, 5),
        "surface_roughness": (0.3, 0.7),
    },
    "Crested blade": {
        "length_mm": (40, 150), "width_mm": (8, 25), "thickness_mm": (3, 10),
        "elongation": (3.0, 6.0), "scar_count": (2, 4),
        "dorsal_ridge_count": (2, 4),
        "surface_roughness": (0.3, 0.6),
    },
    "Core rejuvenation": {
        "length_mm": (15, 60), "width_mm": (10, 40), "thickness_mm": (5, 20),
        "elongation": (0.8, 2.0), "scar_count": (1, 3),
        "platform_angle_deg": (70, 90),
        "surface_roughness": (0.5, 0.8),
    },
}


def generate_samples(ranges: dict, n_per_class: int = 200, noise: float = 0.15):
    """Generate synthetic feature vectors with Gaussian noise."""
    features = []
    labels = []
    rng = np.random.default_rng(42)

    for label, params in ranges.items():
        for _ in range(n_per_class):
            vec = {}
            for key, (lo, hi) in params.items():
                val = lo + rng.random() * (hi - lo)
                val += rng.normal(0, (hi - lo) * noise)
                val = max(lo * 0.5, min(hi * 1.5, val))
                vec[key] = round(float(val), 4)

            for name in LithicFeatureVector.FEATURE_NAMES:
                if name not in vec:
                    vec[name] = 0.0

            fv = LithicFeatureVector(**{k: vec.get(k, 0.0) for k in LithicFeatureVector.FEATURE_NAMES})
            features.append(fv)
            labels.append(label)

    return features, labels


def train_and_save(name: str, ranges: dict, n_per_class: int = 200) -> Path:
    """Generate training data, train model, save to file."""
    from lithicore._classification import train_model

    features, labels = generate_samples(ranges, n_per_class=n_per_class)
    model = train_model(features, labels, typology_name=name)
    path = MODEL_DIR / f"typology_{name}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    print(f"  Saved {name} model ({len(ranges)} classes, {n_per_class * len(ranges)} samples) -> {path}")
    return path


if __name__ == "__main__":
    print("Generating pre-trained lithic classifier models...")
    train_and_save("basic", BASIC_RANGES, n_per_class=300)
    train_and_save("bordes", BORDES_RANGES, n_per_class=300)
    train_and_save("technological", TECH_RANGES, n_per_class=300)
    print("Done.")
