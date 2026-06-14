from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich import print

from model_port.common.config import load_train_config
from model_port.pipelines.evaluate_classifier import _load_classes
from model_port.pipelines.train_classifier import _make_model

app = typer.Typer(help="Export an edge classifier artifact to ONNX.")


def _require_export_deps() -> dict[str, Any]:
    try:
        import torch
        from torchvision import models
    except ImportError as exc:
        raise typer.BadParameter(
            "ONNX export requires torch and torchvision. Install/rebuild the train extra with "
            "`pip install -e '.[train,dev,api]'` or `docker compose build trainer`."
        ) from exc
    return {"torch": torch, "models": models}


def _dir_size_mb(path: Path) -> float:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) / (1024**2)


@app.command()
def main(
    config: Path = typer.Option(..., "--config", help="Classifier train config YAML."),
    model_dir: Path = typer.Option(..., "--model-dir", help="Trained classifier artifact dir."),
    output: Path = typer.Option(..., "--output", help="Output ONNX file path."),
    opset: int = typer.Option(17, "--opset", help="ONNX opset version."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate inputs only."),
) -> None:
    cfg = load_train_config(config)
    if not model_dir.exists():
        raise typer.BadParameter(f"Model dir does not exist: {model_dir}")
    weights_path = model_dir / "model.pt"
    if not weights_path.exists():
        raise typer.BadParameter(f"Missing classifier weights: {weights_path}")

    classes = _load_classes(model_dir)
    image_size = int(cfg.dataset.get("image_size", 224))
    print(f"[model-port] classifier={cfg.training.model_name}")
    print(f"[model-port] version={cfg.training.version}")
    print(f"[model-port] source_model_dir={model_dir}")
    print(f"[model-port] output={output}")
    print(f"[model-port] opset={opset}")
    if dry_run:
        return

    deps = _require_export_deps()
    torch = deps["torch"]
    model = _make_model(
        {"models": deps["models"], "torch": torch},
        len(classes),
        pretrained=bool(cfg.training.get("pretrained", False)),
        freeze_backbone=bool(cfg.training.get("freeze_backbone", False)),
    )
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    output.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        model,
        dummy_input,
        output,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"},
        },
        dynamo=False,
    )
    (output.parent / "classes.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")
    metadata = {
        "vendor": cfg.vendor,
        "model_name": cfg.training.model_name,
        "version": cfg.training.version,
        "runtime": "onnx",
        "source_model_dir": str(model_dir),
        "onnx_path": str(output),
        "opset": opset,
        "image_size": image_size,
        "num_classes": len(classes),
        "onnx_size_mb": output.stat().st_size / (1024**2),
        "artifact_size_mb": _dir_size_mb(output.parent),
    }
    (output.parent / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[model-port] onnx_size_mb={metadata['onnx_size_mb']:.4f}")
    print(f"[model-port] wrote ONNX artifact to {output}")


if __name__ == "__main__":
    app()
