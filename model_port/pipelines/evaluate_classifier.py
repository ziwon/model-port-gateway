from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import typer
from rich import print

from model_port.common.config import dump_yaml, load_train_config
from model_port.common.images import load_image
from model_port.common.quality import (
    evaluate_all_quality_gates,
    evaluate_quality_gate,
    quality_gate_config,
)
from model_port.common.tracking import wandb_enabled
from model_port.pipelines.eval_wandb import quality_gate_table_rows, wandb_summary
from model_port.pipelines.train_classifier import _make_model, _read_jsonl, _transforms

app = typer.Typer(help="Evaluate a small edge-friendly image classifier.")


def _require_eval_deps() -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise typer.BadParameter(
            "Classifier evaluation dependencies are missing. Install/rebuild the train extra with "
            "`pip install -e '.[train,dev,api]'` or `docker compose build trainer`."
        ) from exc

    return {"torch": torch}


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _dir_size_mb(path: Path) -> float:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) / (1024**2)


def _distribution_distance(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    return sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys) / 2


def _distribution(labels: list[str]) -> dict[str, float]:
    counts = Counter(labels)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {label: count / total for label, count in counts.items()}


def _load_classes(model_dir: Path) -> list[str]:
    classes_path = model_dir / "classes.json"
    if not classes_path.exists():
        raise typer.BadParameter(f"Missing classes file: {classes_path}")
    return json.loads(classes_path.read_text(encoding="utf-8"))


def _load_model(cfg: Any, model_dir: Path, classes: list[str], deps: dict[str, Any]) -> tuple[Any, Any]:
    torch = deps["torch"]
    try:
        from torchvision import models, transforms  # noqa: F401
    except ImportError as exc:
        raise typer.BadParameter(
            "torchvision is required for classifier evaluation. Rebuild/install train extra."
        ) from exc
    weights_path = model_dir / "model.pt"
    if not weights_path.exists():
        raise typer.BadParameter(f"Missing classifier weights: {weights_path}")
    model = _make_model(
        {"models": models, "torch": torch},
        len(classes),
        pretrained=bool(cfg.training.get("pretrained", False)),
        freeze_backbone=bool(cfg.training.get("freeze_backbone", False)),
    )
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    transform = _transforms({"transforms": transforms}, int(cfg.dataset.get("image_size", 224)))
    return model, transform


def _evaluate_rows(
    cfg: Any,
    model_dir: Path,
    rows: list[dict[str, Any]],
    dataset_dir: Path,
    deps: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[float], list[str | None]]:
    torch = deps["torch"]
    classes = _load_classes(model_dir)
    model, transform = _load_model(cfg, model_dir, classes, deps)
    device = next(model.parameters()).device
    predictions: list[dict[str, Any]] = []
    latencies: list[float] = []
    failures: list[str | None] = []
    softmax = torch.nn.Softmax(dim=1)

    for row in rows:
        try:
            image = load_image(row["image_path"], dataset_dir)
            tensor = transform(image).unsqueeze(0).to(device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.no_grad():
                probabilities = softmax(model(tensor))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - started) * 1000
            confidence, predicted_id = probabilities.max(dim=1)
            prediction = classes[int(predicted_id.item())]
            predictions.append({
                "image_path": row["image_path"],
                "ground_truth": str(row["label"]),
                "prediction": prediction,
                "confidence": float(confidence.item()),
                "latency_ms": latency_ms,
                "passed": prediction == str(row["label"]),
            })
            latencies.append(latency_ms)
            failures.append(None)
        except Exception as exc:  # noqa: BLE001 - keep evaluation going per sample.
            predictions.append({
                "image_path": row.get("image_path", ""),
                "ground_truth": str(row.get("label", "")),
                "prediction": "",
                "confidence": 0.0,
                "latency_ms": 0.0,
                "passed": False,
            })
            latencies.append(0.0)
            failures.append(type(exc).__name__)

    return predictions, latencies, failures


def _build_report(
    cfg: Any,
    dataset: Path,
    model_dir: Path,
    rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    latencies: list[float],
    failures: list[str | None],
    quality_profile: str | None,
    version: str | None,
) -> dict[str, Any]:
    successful_latencies = [latency for latency, failure in zip(latencies, failures, strict=False) if failure is None]
    runtime_sec = sum(successful_latencies) / 1000
    correct = sum(1 for prediction in predictions if prediction["passed"])
    failure_rate = sum(failure is not None for failure in failures) / max(len(failures), 1)
    ground_truths = [prediction["ground_truth"] for prediction in predictions]
    predicted_labels = [prediction["prediction"] for prediction in predictions if prediction["prediction"]]
    drift_score = round(
        _distribution_distance(_distribution(ground_truths), _distribution(predicted_labels)),
        4,
    )
    metrics = {
        "accuracy": correct / max(len(predictions), 1),
        "p50_latency_ms": _percentile(successful_latencies, 0.50),
        "p95_latency_ms": _percentile(successful_latencies, 0.95),
        "runtime_sec": runtime_sec,
        "samples_per_second": len(predictions) / runtime_sec if runtime_sec > 0 else 0.0,
        "steps_per_second": len(predictions) / runtime_sec if runtime_sec > 0 else 0.0,
        "failure_rate": failure_rate,
        "drift_score": drift_score,
        "model_size_mb": _dir_size_mb(model_dir),
    }
    gate = quality_gate_config(cfg.validation.model_dump(), quality_profile)
    quality_gate = evaluate_quality_gate(metrics, gate)
    profile_results = evaluate_all_quality_gates(metrics, cfg.validation.model_dump())
    return {
        "model_name": cfg.training.model_name,
        "version": version or cfg.training.version,
        "vendor": cfg.vendor,
        "dataset": cfg.dataset.name or dataset.stem,
        "num_samples": len(predictions),
        "task": "image-classification",
        "runtime": "torchvision",
        "base_model": cfg.base_model,
        "model_dir": str(model_dir),
        "metrics": metrics,
        "quality_gate": quality_gate,
        "quality_gates": profile_results,
        "quality_gate_profiles": profile_results,
        "inference": {
            "image_size": int(cfg.dataset.get("image_size", 224)),
            "batch_size": int(cfg.inference.get("batch_size", 1)),
        },
        "class_distribution": {
            "ground_truth": _distribution(ground_truths),
            "prediction": _distribution(predicted_labels),
        },
    }


def _log_wandb(
    cfg: Any,
    report: dict[str, Any],
    predictions: list[dict[str, Any]],
    dataset_dir: Path,
) -> None:
    if not wandb_enabled():
        return
    try:
        import wandb
    except ImportError:
        print("[yellow]W&B is not installed; skipping W&B classifier logging.[/yellow]")
        return

    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", cfg.wandb.project),
        name=f"{cfg.vendor}-{report['model_name']}-{report['version']}-eval",
        job_type="evaluate-classifier",
        config={
            "vendor": cfg.vendor,
            "model_name": report["model_name"],
            "version": report["version"],
            "dataset": report["dataset"],
            "runtime": "torchvision",
        },
    )
    table = wandb.Table(columns=[
        "image",
        "ground_truth",
        "prediction",
        "confidence",
        "latency_ms",
        "passed",
    ])
    for row in predictions:
        image_path = dataset_dir / str(row["image_path"])
        table.add_data(
            wandb.Image(str(image_path)) if image_path.exists() else None,
            row["ground_truth"],
            row["prediction"],
            row["confidence"],
            row["latency_ms"],
            row["passed"],
        )

    quality_table = wandb.Table(columns=["metric", "value", "threshold", "passed", "profile"])
    for row in quality_gate_table_rows(report):
        quality_table.add_data(
            row["metric"],
            row["value"],
            row["threshold"],
            row["passed"],
            row["profile"],
        )

    metrics = report["metrics"]
    wandb.log({"eval/predictions": table})
    wandb.log({"eval/quality_gate_table": quality_table})
    wandb.log({
        "eval/accuracy": metrics["accuracy"],
        "eval/p50_latency_ms": metrics["p50_latency_ms"],
        "eval/p95_latency_ms": metrics["p95_latency_ms"],
        "eval/runtime": metrics["runtime_sec"],
        "eval/samples_per_second": metrics["samples_per_second"],
        "eval/steps_per_second": metrics["steps_per_second"],
        "eval/model_size_mb": metrics["model_size_mb"],
        "eval/failure_rate": metrics["failure_rate"],
        "drift/score": metrics["drift_score"],
        "quality_gate/passed": int(report["quality_gate"]["passed"]),
        "quality_gate/profile": report["quality_gate"]["profile"],
    })
    wandb.summary.update(wandb_summary(report))
    run.finish()


@app.command()
def main(
    config: Path = typer.Option(..., "--config", help="Classifier train config YAML."),
    model_dir: Path = typer.Option(..., "--model-dir", help="Trained classifier artifact dir."),
    dataset: Path = typer.Option(..., "--dataset", help="Evaluation JSONL dataset."),
    output: Path = typer.Option(..., "--output", help="Output eval report JSON."),
    quality_profile: str | None = typer.Option(
        None,
        "--quality-profile",
        help="Quality gate profile to apply, e.g. edge-target.",
    ),
    version: str | None = typer.Option(None, "--version", help="Override model version."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate inputs only."),
) -> None:
    cfg = load_train_config(config)
    if not model_dir.exists():
        raise typer.BadParameter(f"Model dir does not exist: {model_dir}")
    if not dataset.exists():
        raise typer.BadParameter(f"Dataset does not exist: {dataset}")
    rows = _read_jsonl(dataset)
    if cfg.dataset.max_samples:
        rows = rows[: int(cfg.dataset.max_samples)]
    print(f"[model-port] classifier={cfg.training.model_name}")
    print(f"[model-port] version={version or cfg.training.version}")
    print(f"[model-port] dataset={dataset}")
    print(f"[model-port] model_dir={model_dir}")
    print(f"[model-port] quality_profile={quality_profile or cfg.validation.active_profile}")
    if dry_run:
        return

    deps = _require_eval_deps()
    dataset_dir = Path(cfg.dataset.get("root_dir", dataset.parent))
    predictions, latencies, failures = _evaluate_rows(cfg, model_dir, rows, dataset_dir, deps)
    report = _build_report(
        cfg,
        dataset,
        model_dir,
        rows,
        predictions,
        latencies,
        failures,
        quality_profile,
        version,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    dump_yaml(report, output.with_suffix(".yaml"))
    _log_wandb(cfg, report, predictions, dataset_dir)
    print(f"[model-port] accuracy={report['metrics']['accuracy']:.4f}")
    print(f"[model-port] p95_latency_ms={report['metrics']['p95_latency_ms']:.4f}")
    print(f"[model-port] quality_gate_passed={report['quality_gate']['passed']}")
    print(f"[model-port] wrote eval report to {output}")


if __name__ == "__main__":
    app()
