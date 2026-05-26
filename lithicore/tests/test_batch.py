"""test_batch.py — Tests for batch processing."""

import csv
import json
import trimesh
import pytest
from pathlib import Path
from lithicore._batch import batch_process
from lithicore._models import MeasurementConfig


class TestBatchProcess:
    def test_batch_processes_all_files(self, mesh_dir_with_various):
        """Batch should find and process all supported mesh files."""
        config = MeasurementConfig(repair_mesh=True)
        results = batch_process(mesh_dir_with_various, config)
        assert len(results) == 3  # cube.ply, sphere.obj, cone.stl

    def test_batch_empty_directory(self, tmp_path):
        """An empty directory should return no results."""
        config = MeasurementConfig()
        results = batch_process(tmp_path, config)
        assert len(results) == 0

    def test_batch_each_result_has_measurements(self, mesh_dir_with_various):
        """Each artefact result should contain measurements."""
        config = MeasurementConfig()
        results = batch_process(mesh_dir_with_various, config)
        for result in results:
            assert len(result.measurements) > 0
            assert result.file_path.exists()
