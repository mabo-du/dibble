"""_batch.py — Batch processing for lithicore measurement pipeline.

exports: batch_process(directory, config) -> list[ArtefactResult]
used_by: CLI entry point, batch runner UI
rules:   Iterates all .ply, .obj, .stl files in directory.
         Each artefact: validate -> repair (if configured) -> orient -> measure.
         Returns ArtefactResult list with one entry per file.
agent:   deepseek-v4-flash | 2026-05-26 | Initial implementation
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import trimesh
from lithicore._models import MeasurementConfig, ArtefactResult
from lithicore._validation import validate_mesh, repair_mesh
from lithicore._orientation import orient_auto
from lithicore._metrics import extract_metrics
from lithicore._platform_angle import platform_angles
from lithicore._edge_detection import detect_edges

_SUPPORTED_EXTENSIONS = {".ply", ".obj", ".stl"}


def batch_process(
    directory: Path,
    config: MeasurementConfig,
) -> List[ArtefactResult]:
    """Process all supported mesh files in a directory.

    For each file: validate, optionally repair, auto-orient,
    extract metrics + platform angles, return ArtefactResult.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    mesh_files = sorted(
        [f for f in directory.iterdir()
         if f.suffix.lower() in _SUPPORTED_EXTENSIONS]
    )

    results: List[ArtefactResult] = []
    for filepath in mesh_files:
        try:
            mesh = trimesh.load(str(filepath), force="mesh")
        except Exception as exc:
            results.append(ArtefactResult(
                file_path=filepath,
                label=filepath.stem,
                measurements=[],
                landmarks=[],
                warnings=[f"Failed to load: {exc}"],
            ))
            continue

        # Validate
        quality = validate_mesh(mesh)
        warnings = list(quality.warnings)

        # Repair if configured
        if config.repair_mesh:
            _, mesh = repair_mesh(mesh)

        # Orient
        try:
            oriented, _ = orient_auto(mesh, config)
        except Exception as exc:
            results.append(ArtefactResult(
                file_path=filepath,
                label=filepath.stem,
                measurements=[],
                landmarks=[],
                warnings=[f"Orientation failed: {exc}"],
            ))
            continue

        # Extract metrics
        all_measurements = extract_metrics(oriented, config)
        epa, ipa = platform_angles(oriented, config)
        if epa:
            all_measurements.append(epa)
        if ipa:
            all_measurements.append(ipa)

        # Edge detection (for visualisation, not exported as measurement yet)
        detect_edges(oriented, config)

        results.append(ArtefactResult(
            file_path=filepath,
            label=filepath.stem,
            measurements=all_measurements,
            landmarks=[],
            warnings=warnings,
        ))

    return results
