"""lithicore — 3D lithic artefact morphological measurement library.

exports: orient_auto(mesh, config) -> tuple[trimesh.Trimesh, np.ndarray]
         orient_manual(mesh, points, config) -> tuple[trimesh.Trimesh, np.ndarray]
         extract_metrics(mesh, config) -> list[MeasurementResult]
         detect_edges(mesh, config) -> np.ndarray
         platform_angles(mesh, config) -> tuple[MeasurementResult, MeasurementResult]
         validate_mesh(mesh) -> MeshQualityReport
         repair_mesh(mesh) -> trimesh.Trimesh
         batch_process(directory, config) -> list[ArtefactResult]
          PhotogrammetryConfig, PhotogrammetryResult, PhotogrammetryError,
            ColmapNotFoundError, ColmapStageError, InsufficientPhotosError,
            PhotogrammetryCancelledError, colmap_available, run_pipeline
          ScaleResult, detect_scale_aruco, apply_scale_to_mesh
used_by: lithicope GUI, CLI users
rules:   No GUI imports. Every public function takes a mesh + config and returns typed results.
agent:   deepseek-v4-flash | 2026-05-26 | Initial scaffolding
         deepseek-v4-flash | 2026-05-26 | All modules wired — full public API exported
         deepseek-v4-flash | 2026-05-26 | Added FigureConfig + generate_figure export
         deepseek-v4-pro | 2026-06-12 | Added AssistantResult to __all__ and imports
"""

# pylint: disable=unused-import
try:
    from lithicore._models import (
        MeasurementConfig,
        MeasurementResult,
        ArtefactResult,
        Landmark,
        MeshQualityReport,
        MeshGrade,
        ClassificationResult,
        FeatureImportance,
        LithicFeatureVector,
        AssistantResult,
    )
    from lithicore._orientation import orient_auto, orient_manual
    from lithicore._metrics import extract_metrics
    from lithicore._edge_detection import detect_edges
    from lithicore._platform_angle import platform_angles
    from lithicore._validation import validate_mesh, repair_mesh
    from lithicore._batch import batch_process
    from lithicore._figure import FigureConfig, generate_figure
    from lithicore._comparison import compare_meshes, ComparisonResult
    from lithicore._photogrammetry import (
        PhotogrammetryConfig,
        PhotogrammetryResult,
        PhotogrammetryError,
        ColmapNotFoundError,
        ColmapStageError,
        InsufficientPhotosError,
        PhotogrammetryCancelledError,
        run_pipeline,
        colmap_available,
        clean_point_cloud,
        ProgressCallback,
    )
    from lithicore._scale_detection import (
        ScaleResult,
        detect_scale_aruco,
        apply_scale_to_mesh,
    )
    from lithicore._photo_preprocessing import (
        PreprocessingConfig,
        PreprocessingResult,
        preprocess_photos,
        compute_laplacian_variance,
        compute_blur_scores,
    )
    from lithicore._annotations import (
        Annotation,
        AnnotationSet,
    )
    from lithicore._classification import (
        ClassifierModel,
        extract_features,
        train_model,
        extract_diagnostic_coordinates,
    )
    from lithicore._assistant import (
        AssistantEngine,
    )

    __all__ = [
        "MeasurementConfig", "MeasurementResult", "ArtefactResult", "Landmark",
        "MeshQualityReport", "MeshGrade",
        "orient_auto", "orient_manual",
        "extract_metrics", "detect_edges", "platform_angles",
        "validate_mesh", "repair_mesh", "batch_process",
        "FigureConfig", "generate_figure",
        "compare_meshes", "ComparisonResult",
        "PhotogrammetryConfig", "PhotogrammetryResult",
        "PhotogrammetryError", "ColmapNotFoundError", "ColmapStageError",
        "InsufficientPhotosError", "PhotogrammetryCancelledError",
        "run_pipeline", "colmap_available", "clean_point_cloud",
        "ProgressCallback",
        "ScaleResult", "detect_scale_aruco", "apply_scale_to_mesh",
        "PreprocessingConfig", "PreprocessingResult",
        "preprocess_photos", "compute_laplacian_variance", "compute_blur_scores",
        "Annotation", "AnnotationSet",
        "ClassificationResult", "FeatureImportance", "LithicFeatureVector",
        "ClassifierModel", "extract_features", "train_model",
        "extract_diagnostic_coordinates",
        "AssistantEngine", "AssistantResult",
    ]
except ImportError as _exc:
    # Forward reference — all modules exist since Phases 2-4
    raise ImportError(
        f"lithicore module import failed: {_exc}. "
        "Try: pip install --no-deps -e lithicore"
    ) from _exc
