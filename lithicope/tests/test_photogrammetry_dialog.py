"""Tests for photogrammetry dialog."""
import pytest
try:
    from PyQt6.QtWidgets import QApplication, QDialog
    from lithicope._photogrammetry_dialog import PhotogrammetryDialog
    HAS_QT = True
except (ImportError, RuntimeError):
    HAS_QT = False


@pytest.fixture
def tmp_photos(tmp_path):
    """Create a temporary folder with test photos."""
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    for i in range(5):
        (photo_dir / f"img_{i:03d}.jpg").write_text("fake")
    return photo_dir


@pytest.mark.skipif(not HAS_QT, reason="PyQt6 not available")
def test_dialog_creates_in_default_mode(qapp, qtbot, tmp_path, tmp_photos):
    """Dialog should construct in default mode."""
    output_path = tmp_path / "result.ply"
    dialog = PhotogrammetryDialog(
        None,
        mode="default",
        photo_folder=tmp_photos,
        output_path=output_path,
        artefact_label="test-artefact",
    )
    qtbot.addWidget(dialog)
    assert dialog.windowTitle() != ""
    dialog.close()


@pytest.mark.skipif(not HAS_QT, reason="PyQt6 not available")
def test_dialog_creates_in_guided_mode(qapp, qtbot, tmp_path, tmp_photos):
    output_path = tmp_path / "result.ply"
    dialog = PhotogrammetryDialog(
        None,
        mode="guided",
        photo_folder=tmp_photos,
        output_path=output_path,
        artefact_label="test-artefact",
    )
    qtbot.addWidget(dialog)
    assert dialog.windowTitle() != ""
    dialog.close()


@pytest.mark.skipif(not HAS_QT, reason="PyQt6 not available")
def test_dialog_creates_in_expert_mode(qapp, qtbot, tmp_path, tmp_photos):
    output_path = tmp_path / "result.ply"
    dialog = PhotogrammetryDialog(
        None,
        mode="expert",
        photo_folder=tmp_photos,
        output_path=output_path,
        artefact_label="test-artefact",
    )
    qtbot.addWidget(dialog)
    assert dialog.windowTitle() != ""
    dialog.close()


@pytest.mark.skipif(not HAS_QT, reason="PyQt6 not available")
def test_batch_dialog_construction(qapp, qtbot, tmp_path):
    """Batch dialog should construct without error."""
    from lithicope._batch_photogrammetry import BatchPhotogrammetryDialog
    artefacts_dir = tmp_path / "artefacts"
    artefacts_dir.mkdir()
    for label in ["FLK-001", "FLK-002"]:
        (artefacts_dir / label).mkdir()
        for i in range(3):
            (artefacts_dir / label / f"img_{i:03d}.jpg").write_text("fake")
    dialog = BatchPhotogrammetryDialog(artefacts_dir, None)
    qtbot.addWidget(dialog)
    assert dialog.windowTitle() != ""
    dialog.close()
