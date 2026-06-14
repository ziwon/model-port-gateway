from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich import print

from model_port.pipelines.runtime_compare import markdown_table, runtime_row

app = typer.Typer(help="Compare runtime evaluation reports.")


def _load_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise typer.BadParameter(f"Report does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@app.command()
def main(
    reports: list[Path] = typer.Argument(..., help="Evaluation report JSON files."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Optional markdown output path."),
) -> None:
    rows = [runtime_row(_load_report(path)) for path in reports]
    table = markdown_table(rows)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(table + "\n", encoding="utf-8")
        print(f"[model-port] wrote runtime comparison to {output}")
    print(table)


if __name__ == "__main__":
    app()
