"""lithicore — 3D lithic artefact morphological measurement library.

exports: orient_auto(mesh, config) -> tuple[trimesh.Trimesh, np.ndarray]
         orient_manual(mesh, points, config) -> tuple[trimesh.Trimesh, np.ndarray]
         extract_metrics(mesh, config) -> list[MeasurementResult]
         detect_edges(mesh, config) -> np.ndarray
         platform_angles(mesh, config) -> tuple[MeasurementResult, MeasurementResult]
         validate_mesh(mesh) -> MeshQualityReport
         repair_mesh(mesh) -> trimesh.Trimesh
         batch_process(directory, config) -> list[ArtefactResult]
used_by: lithicope GUI, CLI users
rules:   No GUI imports. Every public function takes a mesh + config and returns typed results.
agent:   deepseek-v4-flash | 2026-05-26 | Initial scaffolding
"""

# _models.py is created in Task 4 (Phase 2). Import restored there.
# pylint: disable=unused-import
try:
    from lithicore._models import (
        MeasurementConfig,
        MeasurementResult,
        ArtefactResult,
        Landmark,
        MeshQualityReport,
    )
    from lithicore._orientation import orient_auto, orient_manual

    __all__ = [
        "MeasurementConfig", "MeasurementResult", "ArtefactResult", "Landmark",
        "MeshQualityReport",
        "orient_auto", "orient_manual",
    ]
except ImportError:
    # Forward reference — _models.py will exist after Task 4
    MeasurementConfig = None  # type: ignore
    MeasurementResult = None  # type: ignore
    ArtefactResult = None  # type: ignore
    Landmark = None  # type: ignore
    MeshQualityReport = None  # type: ignore
    orient_auto = None  # type: ignore
    orient_manual = None  # type: ignore

    __all__: list[str] = []
