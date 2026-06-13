from __future__ import annotations

from pathlib import Path

import typer
from rich import print

from model_port.common.config import load_yaml

app = typer.Typer(help="Validate and submit a vendor model manifest.")


@app.command()
def main(
    manifest: Path = typer.Option(..., "--manifest", "-m", help="Model manifest YAML."),
    dry_run: bool = False,
) -> None:
    data = load_yaml(manifest)
    required = ["model", "training", "evaluation", "deployment"]
    missing = [k for k in required if k not in data]
    if missing:
        raise typer.BadParameter(f"Missing sections: {missing}")

    model = data["model"]
    for key in ["name", "version", "vendor", "task", "runtime", "artifact_uri"]:
        if key not in model:
            raise typer.BadParameter(f"model.{key} is required")

    print("[green]Manifest validation passed[/green]")
    print({"model": model["name"], "version": model["version"], "vendor": model["vendor"]})
    if dry_run:
        print("[yellow]Dry run: not persisted[/yellow]")


if __name__ == "__main__":
    app()
