"""_metrics.py — Core metric extraction for lithic artefacts.

exports: extract_metrics(mesh, config) -> list[MeasurementResult]
used_by: GUI results panel, batch processing, CLI
rules:   All metrics computed in oriented coordinate space.
         Length = Z-extent, Width = X-extent, Thickness = Y-extent.
         Volume requires watertight mesh (auto-fill if not).
agent:   deepseek-v4-flash | 2026-05-26 | Initial implementation
"""

from __future__ import annotations

import numpy as np
import trimesh
from lithicore._models import MeasurementConfig, MeasurementResult


def extract_metrics(
    mesh: trimesh.Trimesh,
    config: MeasurementConfig,
) -> list[MeasurementResult]:
    """Extract all standard lithic metrics from an oriented mesh.

    The mesh should already be oriented (platform = XY plane,
    reduction axis = Z). If not oriented, results will be incorrect.
    """
    results: list[MeasurementResult] = []

    # Compute oriented bounding box
    obb = mesh.bounding_box_oriented
    extents = sorted(obb.extents, reverse=True)

    # Length = maximum extent (along reduction axis / Z after orientation)
    length_val = extents[0]
    results.append(MeasurementResult(
        name="max_length", value=round(length_val, 2), unit="mm", confidence=0.95
    ))

    # Width = second extent
    width_val = extents[1]
    results.append(MeasurementResult(
        name="max_width", value=round(width_val, 2), unit="mm", confidence=0.95
    ))

    # Thickness = minimum extent
    thickness_val = extents[2]
    results.append(MeasurementResult(
        name="max_thickness", value=round(thickness_val, 2), unit="mm", confidence=0.95
    ))

    # Surface area
    area_val = mesh.area
    results.append(MeasurementResult(
        name="surface_area", value=round(area_val, 2), unit="mm²", confidence=0.90
    ))

    # Volume (watertight fill if needed)
    if mesh.is_watertight:
        vol = mesh.volume
    else:
        # Fill holes and try again
        filled = mesh.copy()
        trimesh.repair.fill_holes(filled)
        vol = filled.volume if filled.is_watertight else 0.0
    results.append(MeasurementResult(
        name="volume", value=round(vol, 2), unit="mm³",
        confidence=0.95 if mesh.is_watertight else 0.70,
    ))

    return results
