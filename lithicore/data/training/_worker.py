"""Worker for process_safe.py — processes one mesh in a subprocess.

Each mesh gets its own Python process. Exits cleanly, freeing all memory.
Now uses the standard extract_features from _classification.py (scar detection
was fixed to use fast vertex_defects instead of hanging curvature measures).
"""
import csv
import io
import sys
from pathlib import Path

import trimesh

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "lithicore" / "src"))

from lithicore._classification import extract_features


if __name__ == "__main__":
    mesh_path, artefact_id, typology, dataset, csv_name = sys.argv[1:6]

    mesh = trimesh.load(mesh_path, force="mesh")

    # Fix inverted normals: if volume is negative the faces are wound inward
    if mesh.is_watertight and mesh.volume < 0:
        mesh.fix_normals()

    fv = extract_features(mesh)

    fieldnames = ["artefact_id", "typology", "dataset", "source_csv"] + fv.FEATURE_NAMES
    row = {
        "artefact_id": artefact_id,
        "typology": typology,
        "dataset": dataset,
        "source_csv": csv_name,
    }
    for name in fv.FEATURE_NAMES:
        row[name] = getattr(fv, name)

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writerow(row)
    print(buf.getvalue().strip())
