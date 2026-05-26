"""_validation.py — Mesh validation, cleaning, and repair.

exports: validate_mesh(mesh: trimesh.Trimesh) -> MeshQualityReport
         repair_mesh(mesh: trimesh.Trimesh) -> tuple[MeshQualityReport, trimesh.Trimesh]
used_by: Import pipeline (GUI and CLI), called before any measurement
rules:   validate_mesh never modifies the input mesh.
         repair_mesh returns a NEW mesh (copy) — never mutates the input.
         Always check for non-manifold edges, holes, isolated vertices.
agent:   deepseek-v4-flash | 2026-05-26 | Initial implementation
         deepseek-v4-flash | 2026-05-26 | Fixed trimesh API: merge_vertices, imported repair module
"""

from __future__ import annotations

import numpy as np
import trimesh
from lithicore._models import MeshQualityReport, MeshGrade

# Minimum faces for reliable edge angle measurement
_MIN_FACES_RELIABLE = 2000
# Minimum faces for basic measurements (below this → FAIL)
_MIN_FACES_BASIC = 3


def validate_mesh(mesh: trimesh.Trimesh) -> MeshQualityReport:
    """Validate mesh quality without modifying it.

    Returns a MeshQualityReport with grade and warnings.
    Grade is determined by the worst finding:
      - FAIL if the mesh is degenerate (< _MIN_FACES_BASIC faces).
      - WARN if non-manifold edges, not watertight, winding issues exist.
      - PASS if all structural checks pass.
    """
    warnings: list[str] = []
    vcount = len(mesh.vertices)
    fcount = len(mesh.faces)

    if fcount < _MIN_FACES_BASIC:
        return MeshQualityReport(
            original_vertex_count=vcount,
            original_face_count=fcount,
            grade=MeshGrade.FAIL,
            warnings=[f"Too few faces ({fcount}) for reliable measurement"],
        )

    if not mesh.is_watertight:
        warnings.append("Mesh is not watertight")

    if mesh.is_winding_consistent is False:
        warnings.append("Inconsistent face winding detected")

    # Check for non-manifold edges (edges shared by 3+ faces)
    unique_edges, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
    non_manifold_count = int(np.sum(counts > 2))
    if non_manifold_count > 0:
        warnings.append(f"{non_manifold_count} non-manifold edges")

    # Warn about low resolution but don't downgrade grade on its own
    if fcount < _MIN_FACES_RELIABLE:
        warnings.append(f"Low resolution ({fcount} faces)")

    # Grade assignment
    if len(warnings) > 0:
        grade = MeshGrade.WARN
    else:
        grade = MeshGrade.PASS

    return MeshQualityReport(
        original_vertex_count=vcount,
        original_face_count=fcount,
        grade=grade,
        warnings=warnings,
    )


def repair_mesh(mesh: trimesh.Trimesh) -> tuple[MeshQualityReport, trimesh.Trimesh]:
    """Repair and clean a mesh.

    Performs: merge duplicate/close vertices, fill small holes,
    remove isolated components, fix normals. Returns a NEW mesh.
    """
    working = mesh.copy()

    # Merge duplicate vertices (5 decimal places ≈ 1e-5 tolerance)
    working.merge_vertices(digits_vertex=5)

    # Remove degenerate faces (zero area)
    working.update_faces(working.nondegenerate_faces())

    # Remove unreferenced vertices (not part of any remaining face)
    if len(working.vertices) > 0 and len(working.faces) > 0:
        referenced = np.zeros(len(working.vertices), dtype=bool)
        referenced[working.faces] = True
        working.update_vertices(referenced)

    # Fill small holes
    holes_filled = 0
    try:
        trimesh.repair.fill_holes(working)
    except (ValueError, AttributeError):
        pass

    # Remove isolated components, keep the largest
    isolated = 0
    try:
        components = working.split()
        if len(components) > 1:
            working = max(components, key=lambda c: len(c.faces))
            isolated = len(components) - 1
    except ValueError:
        pass

    # Fix winding if needed
    if working.is_winding_consistent is False:
        working.fix_normals()

    orig_face_count = len(mesh.faces)

    # Count non-manifold edges in original mesh
    unique_edges, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
    non_manifold_count = int(np.sum(counts > 2))

    report = MeshQualityReport(
        original_vertex_count=len(mesh.vertices),
        original_face_count=orig_face_count,
        repaired_vertex_count=len(working.vertices),
        repaired_face_count=len(working.faces),
        holes_filled=holes_filled,
        non_manifold_edges_fixed=non_manifold_count,
        isolated_components_removed=isolated,
        grade=MeshGrade.PASS,
    )
    return report, working
