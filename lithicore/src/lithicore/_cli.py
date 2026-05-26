"""_cli.py — Command-line interface for lithicore.

exports: app (typer.Typer)
used_by: Users running `lithicore batch --input ...`
rules:   Typer-based CLI. Subcommands: batch, info.
agent:   deepseek-v4-flash | 2026-05-26 | Initial implementation
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


if __name__ == "__main__":
    app()
