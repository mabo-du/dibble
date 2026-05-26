"""_platform_angle.py — Platform angle extraction (EPA, IPA).

exports: platform_angles(mesh, config) -> tuple[MeasurementResult | None, MeasurementResult | None]
used_by: GUI results panel, batch processing
rules:   EPA = angle between platform plane and dorsal surface.
         IPA = angle between platform plane and ventral surface.
         Platform normal always points +Z (into the artefact) in oriented space.
         Adjacent non-platform faces at proximal end are classified by Z-hemisphere.
agent:   deepseek-v4-flash | 2026-05-26 | Initial implementation
"""

from __future__ import annotations

import numpy as np
import trimesh
from lithicore._models import MeasurementConfig, MeasurementResult

# Fraction of mesh Z-extent from proximal end to search for the platform.
_PLATFORM_SEARCH_FRACTION = 0.1


def platform_angles(
    mesh: trimesh.Trimesh,
    config: MeasurementConfig,
) -> tuple[MeasurementResult | None, MeasurementResult | None]:
    """Compute Exterior and Interior Platform Angles.

    In oriented space the platform sits at the proximal (minimum Z) end.
    The platform normal points +Z (into the artefact, toward the distal tip).

    EPA = acute angle between the platform plane and adjacent dorsal  (exterior) face normals.
    IPA = acute angle between the platform plane and adjacent ventral  (interior / bulbar) face normals.
    """
    z_min = mesh.bounds[0, 2]
    z_max = mesh.bounds[1, 2]
    z_range = z_max - z_min

    # --- 1. Gather faces near the proximal end --------------------------------
    proximal_z_threshold = z_min + z_range * _PLATFORM_SEARCH_FRACTION
    proximal_face_indices = np.where(
        np.any(mesh.vertices[mesh.faces][:, :, 2] < proximal_z_threshold, axis=1)
    )[0]

    if len(proximal_face_indices) < 3:
        return None, None

    proximal_normals = mesh.face_normals[proximal_face_indices]

    # --- 2. Identify platform faces (those lying in the XY plane) ------------
    abs_z_proximal = np.abs(proximal_normals[:, 2])

    if np.max(abs_z_proximal) > 0.7:
        # Strong Z-component →  face lies in (or close to) the XY plane.
        platform_face_indices = proximal_face_indices[abs_z_proximal > 0.7]
    else:
        # Fallback: the top-N proximal faces with the strongest Z-component.
        top_k = min(10, len(proximal_face_indices))
        top_idx = np.argsort(abs_z_proximal)[-top_k:]
        platform_face_indices = proximal_face_indices[top_idx]

    # --- 3. Platform normal --------------------------------------------------
    # In oriented space the platform is the XY plane and its interior-pointing
    # normal is always +Z (from the platform toward the distal end).
    platform_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    # --- 4. Adjacent (non-platform) proximal faces ---------------------------
    platform_set = set(platform_face_indices.tolist())
    non_platform_mask = np.array(
        [fi not in platform_set for fi in proximal_face_indices], dtype=bool
    )
    non_platform_indices = proximal_face_indices[non_platform_mask]

    if len(non_platform_indices) == 0:
        return None, None

    adjacent_normals = mesh.face_normals[non_platform_indices]

    # --- 5. Classify adjacent faces by Z-hemisphere --------------------------
    # Dorsal  (exterior) → negative-Z component (outside the flake).
    # Ventral (interior) → positive-Z component (toward the bulb).
    # Faces with exactly zero Z contribute to both.
    zero_mask = adjacent_normals[:, 2] == 0.0
    dorsal_normals = adjacent_normals[(adjacent_normals[:, 2] < 0.0) | zero_mask]
    ventral_normals = adjacent_normals[(adjacent_normals[:, 2] > 0.0) | zero_mask]

    # --- 6. Compute angles ---------------------------------------------------
    def _mean_acute_angle(normal_a: np.ndarray, normals_b: np.ndarray) -> float | None:
        """Mean acute angle between a single normal and a set of normals."""
        if len(normals_b) == 0:
            return None
        dots = np.clip(np.dot(normals_b, normal_a), -1.0, 1.0)
        acute_angles = np.degrees(np.arccos(np.abs(dots)))
        return float(np.mean(acute_angles))

    epa_value = _mean_acute_angle(platform_normal, dorsal_normals)
    ipa_value = _mean_acute_angle(platform_normal, ventral_normals)

    # --- 7. Build result objects ---------------------------------------------
    epa = (
        MeasurementResult(
            name="exterior_platform_angle",
            value=round(epa_value, 1),
            unit="°",
            confidence=0.85,
        )
        if epa_value is not None
        else None
    )

    ipa = (
        MeasurementResult(
            name="interior_platform_angle",
            value=round(ipa_value, 1),
            unit="°",
            confidence=0.85,
        )
        if ipa_value is not None
        else None
    )

    return epa, ipa
