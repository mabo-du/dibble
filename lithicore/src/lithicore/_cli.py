"""_cli.py — Command-line interface for lithicore.

exports: app (typer.Typer)
used_by: Users running `lithicore batch --input ...`
rules:   Typer-based CLI. Subcommands: batch, info, figure, photogrammetry.
agent:   deepseek-v4-flash | 2026-05-26 | Initial implementation
agent:   deepseek-v4-flash | 2026-05-26 | Added photogrammetry CLI subcommand
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
import pandas as pd
import json
from lithicore._models import MeasurementConfig
from lithicore._batch import batch_process

app = typer.Typer(
    name="lithicore",
    help="3D lithic artefact morphological analysis toolkit",
)


@app.command()
def batch(
    input: Path = typer.Argument(..., help="Directory containing mesh files"),
    output: Path = typer.Option("results.csv", "--output", "-o", help="Output CSV path"),
    format: str = typer.Option("csv", "--format", "-f", help="Output format: csv, json"),
    repair: bool = typer.Option(True, "--repair/--no-repair", help="Auto-repair meshes"),
    edge_threshold: float = typer.Option(50.0, "--edge-threshold", help="Edge detection angle threshold"),
) -> None:
    """Batch process all meshes in a directory."""
    config = MeasurementConfig(repair_mesh=repair, edge_threshold_degrees=edge_threshold)
    results = batch_process(input, config)

    if not results:
        typer.echo(f"No supported mesh files found in {input}")
        raise typer.Exit()

    # Build rows
    rows = []
    for artefact in results:
        row = {"file": artefact.file_path.name, "label": artefact.label}
        for m in artefact.measurements:
            row[m.name] = m.value
        row["warnings"] = "; ".join(artefact.warnings)
        rows.append(row)

    output_path = Path(output)

    if format.lower() == "json":
        output_path.write_text(json.dumps(rows, indent=2))
        typer.echo(f"Wrote {len(rows)} results to {output_path}")
    else:
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        typer.echo(f"Wrote {len(rows)} results to {output_path}")


@app.command()
def info(
    mesh_path: Path = typer.Argument(..., help="Path to a mesh file"),
) -> None:
    """Display information about a single mesh file."""
    import trimesh
    mesh = trimesh.load(str(mesh_path), force="mesh")
    typer.echo(f"File:     {mesh_path.name}")
    typer.echo(f"Vertices: {len(mesh.vertices)}")
    typer.echo(f"Faces:    {len(mesh.faces)}")
    typer.echo(f"Watertight: {mesh.is_watertight}")
    typer.echo(f"Area:     {mesh.area:.2f} mm²")
    if mesh.is_watertight:
        typer.echo(f"Volume:   {mesh.volume:.2f} mm³")


@app.command()
def figure(
    mesh_path: Path = typer.Argument(..., help="Path to a mesh file"),
    output: Path = typer.Option("figure.svg", "--output", "-o", help="Output SVG path"),
    no_measurements: bool = typer.Option(False, "--no-measurements", help="Hide measurement callouts"),
    no_ridges: bool = typer.Option(False, "--no-ridges", help="Hide scar ridge lines"),
    label: str = typer.Option("", "--label", "-l", help="Artefact label"),
) -> None:
    """Generate a publication figure from a mesh file."""
    from lithicore._figure import FigureConfig, figure_cli

    config = FigureConfig(
        show_measurements=not no_measurements,
        show_ridges=not no_ridges,
        artefact_label=label or mesh_path.stem,
    )
    figure_cli(mesh_path, output, config)
    typer.echo(f"Figure saved to {output}")


@app.command()
def photogrammetry(
    photo_folder: Path = typer.Argument(..., help="Folder containing photos (jpg/png/tiff)"),
    output: Path = typer.Option("mesh.ply", "--output", "-o", help="Output mesh path"),
    label: str = typer.Option("", "--label", "-l", help="Artefact label"),
    quality: str = typer.Option("high", "--quality", "-q", help="Mesh quality: low, medium, high"),
    colmap_feature_type: str = typer.Option("sift", "--colmap-feature-type", help="COLMAP feature type"),
    colmap_matching: str = typer.Option("exhaustive", "--colmap-matching", help="Matching strategy"),
    dense_quality: str = typer.Option("extreme", "--dense-quality", help="Dense reconstruction quality"),
    batch: bool = typer.Option(False, "--batch", help="Batch mode: each sub-folder is one artefact"),
    batch_output: Optional[Path] = typer.Option(None, "--batch-output", help="Output directory for batch results"),
) -> None:
    """Run photogrammetry pipeline: photos → 3D mesh via COLMAP."""
    from lithicore._photogrammetry import (
        PhotogrammetryConfig,
        run_pipeline,
    )

    if batch:
        output_dir = batch_output or photo_folder / "results"
        output_dir.mkdir(parents=True, exist_ok=True)

        artefact_folders = sorted(
            [d for d in photo_folder.iterdir() if d.is_dir()]
        )
        if not artefact_folders:
            typer.echo(f"No sub-folders found in {photo_folder}")
            raise typer.Exit()

        typer.echo(f"Found {len(artefact_folders)} artefacts for batch processing")

        for artefact_dir in artefact_folders:
            label_used = artefact_dir.name
            out_path = output_dir / label_used / f"{label_used}.ply"
            typer.echo(f"\nProcessing {label_used} ({artefact_dir})...")

            cfg = PhotogrammetryConfig(
                photo_folder=artefact_dir,
                output_path=out_path,
                artefact_label=label_used,
                quality=quality,
                mode="default",
            )

            def cli_progress(stage: str, progress: float, message: str) -> None:
                if progress == 0.0:
                    typer.echo(f"  {stage}...")
                elif progress == 1.0:
                    typer.echo(f"  \u2713 {stage}")

            try:
                result = run_pipeline(cfg, progress_cb=cli_progress)
                typer.echo(f"  \u2713 Complete: {result.face_count} faces in {result.processing_time_s:.0f}s")
            except Exception as exc:
                typer.echo(f"  \u2717 Failed: {exc}", err=True)

        typer.echo(f"\nBatch complete. Results in {output_dir}")
    else:
        mode = "expert" if any([
            colmap_feature_type != "sift",
            colmap_matching != "exhaustive",
            dense_quality != "extreme",
        ]) else "default"

        cfg = PhotogrammetryConfig(
            photo_folder=photo_folder,
            output_path=output,
            artefact_label=label or photo_folder.stem,
            quality=quality,
            mode=mode,
            colmap_feature_type=colmap_feature_type,
            colmap_matching_strategy=colmap_matching,
            colmap_dense_quality=dense_quality,
        )

        def cli_progress(stage: str, progress: float, message: str) -> None:
            if progress == 0.0:
                typer.echo(f"\u23f3 {stage}...")
            elif progress == 1.0:
                typer.echo(f"\u2705 {stage}")

        try:
            result = run_pipeline(cfg, progress_cb=cli_progress)
            typer.echo(f"\n\u2705 Photogrammetry complete!")
            typer.echo(f"   Artefact: {result.artefact_label}")
            typer.echo(f"   Photos:   {result.camera_count}")
            typer.echo(f"   Faces:    {result.face_count}")
            typer.echo(f"   Time:     {result.processing_time_s:.0f}s")
            typer.echo(f"   Output:   {result.mesh_path}")
            if result.warnings:
                for w in result.warnings:
                    typer.echo(f"   \u26a0 {w}")
        except Exception as exc:
            typer.echo(f"\u274c Photogrammetry failed: {exc}", err=True)
            raise typer.Exit(code=1) from exc


@app.command()
def benchmark(
    output: Path = typer.Option(
        None, "--output", "-o",
        help="Output directory for benchmark report (default: docs/benchmark/results)",
    ),
) -> None:
    """Run the classifier validation benchmark and generate a report.

    Tests all pre-trained lithic classifiers against held-out synthetic data
    and generates an interactive HTML validation report with confusion matrices,
    per-class metrics, and accuracy scores.
    """
    typer.echo("Running Dibble classifier validation benchmark...")
    import subprocess
    import sys
    benchmark_script = Path(__file__).resolve().parent.parent.parent / "data" / "run_benchmark.py"
    result = subprocess.run([sys.executable, str(benchmark_script)], capture_output=False)
    if result.returncode != 0:
        raise typer.Exit(code=1)
    typer.echo("\nBenchmark complete. Open docs/benchmark/results/report.html to view results.")


if __name__ == "__main__":
    app()
