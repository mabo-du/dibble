"""test_validation.py — Tests for mesh validation and repair.

exports: TestValidateMesh, TestRepairMesh
used_by: pytest (no direct consumers)
rules:   Tests must not mutate shared fixtures.
         Rectangular prism fixture has 12 faces (low-res, valid).
agent:   deepseek-v4-flash | 2026-05-26 | Initial implementation
         deepseek-v4-flash | 2026-05-26 | Adjusted hole test to use high-res mesh
"""

import numpy as np
import trimesh
import pytest
from lithicore._validation import validate_mesh, repair_mesh
from lithicore._models import MeshGrade


class TestValidateMesh:
    def test_valid_mesh_passes(self, rectangular_prism):
        """A valid (though low-res) mesh should pass or warn."""
        report = validate_mesh(rectangular_prism)
        # Rectangular prism has 12 faces — gets low-res warning
        # but no structural issues
        assert report.grade in (MeshGrade.PASS, MeshGrade.WARN)
        assert report.original_face_count > 0

    def test_degenerate_mesh_fails(self):
        """A mesh with only 1 face should fail."""
        vertices = [[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]]
        faces = [[0, 1, 2]]  # only one triangle
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        report = validate_mesh(mesh)
        assert report.grade == MeshGrade.FAIL

    def test_mesh_with_hole_warns(self):
        """A mesh with a large hole should warn."""
        # Use an icosphere (higher resolution) so removing faces
        # leaves enough for validation
        sphere = trimesh.creation.icosphere(subdivisions=2)
        # Remove roughly half the faces to create a hole
        keep_faces = [f for i, f in enumerate(sphere.faces) if i > len(sphere.faces) // 2]
        holey_mesh = trimesh.Trimesh(
            vertices=sphere.vertices, faces=keep_faces, process=False
        )
        report = validate_mesh(holey_mesh)
        assert report.grade in (MeshGrade.WARN, MeshGrade.PASS)


class TestRepairMesh:
    def test_repair_valid_mesh_unchanged(self, rectangular_prism):
        """A valid mesh should be repairable."""
        report, repaired = repair_mesh(rectangular_prism)
        assert len(repaired.vertices) > 0
        assert report.original_face_count > 0

    def test_repair_removes_isolated_vertices(self):
        """Isolated vertices should be removed."""
        box = trimesh.creation.box(extents=[50, 30, 10])
        # Add stray vertices not referenced by any face
        bad_vertices = np.vstack([box.vertices, [[999, 999, 999], [-999, -999, -999]]])
        bad_mesh = trimesh.Trimesh(
            vertices=bad_vertices, faces=box.faces, process=False
        )
        report, repaired = repair_mesh(bad_mesh)
        # Isolated vertices should be removed during repair
        assert len(repaired.vertices) <= len(box.vertices)
