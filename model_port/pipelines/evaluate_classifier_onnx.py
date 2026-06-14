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
from model_port.pipelines.evaluate_classifier import _percentile
from model_port.pipelines.train_classifier import _read_jsonl, _transforms

app = typer.Typer(help="Evaluate an ONNX edge classifier artifact.")


def _require_onnx_deps() -> dict[str, Any]:
    try:
        import numpy as np
        import onnxruntime as ort
        from torchvision import transforms
    except ImportError as exc:
        raise typer.BadParameter(
            "ONNX evaluation requires onnxruntime, numpy, and torchvision. "
            "Install/rebuild the train extra with `pip install -e '.[train,dev,api]'` "
            "or `docker compose build trainer`."
        ) from exc
    return {"np": np, "ort": ort, "transforms": transforms}


def _load_classes(onnx_dir: Path) -> list[str]:
    classes_path = onnx_dir / "classes.json"
    if not classes_path.exists():
        raise typer.BadParameter(f"Missing classes file: {classes_path}")
    return json.loads(classes_path.read_text(encoding="utf-8"))


def _providers(ort: Any, runtime: str) -> list[str]:
    available = set(ort.get_available_providers())
    if runtime == "cpu":
        return ["CPUExecutionProvider"]
    if runtime == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise typer.BadParameter(
                "ONNX Runtime CUDA provider is not available in this environment. "
                f"Available providers: {sorted(available)}"
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    raise typer.BadParameter("runtime must be one of: cpu, cuda")


def _softmax(np: Any, logits: Any) -> Any:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _distribution(labels: list[str]) -> dict[str, float]:
    counts = Counter(labels)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {label: count / total for label, count in counts.items()}


def _distribution_distance(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    return sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys) / 2


def _artifact_size_mb(onnx_path: Path) -> float:
    return sum(item.stat().st_size for item in onnx_path.parent.rglob("*") if item.is_file()) / (1024**2)


def _evaluate_rows(
    cfg: Any,
    onnx_path: Path,
    rows: list[dict[str, Any]],
    dataset_dir: Path,
    runtime: str,
    deps: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[float], list[str | None], list[str]]:
    np = deps["np"]
    ort = deps["ort"]
    classes = _load_classes(onnx_path.parent)
    session = ort.InferenceSession(str(onnx_path), providers=_providers(ort, runtime))
    input_name = session.get_inputs()[0].name
    transform = _transforms({"transforms": deps["transforms"]}, int(cfg.dataset.get("image_size", 224)))

    predictions: list[dict[str, Any]] = []
    latencies: list[float] = []
    failures: list[str | None] = []
    for row in rows:
        try:
            image = load_image(row["image_path"], dataset_dir)
            tensor = transform(image).unsqueeze(0)
            inputs = {input_name: tensor.numpy()}
            started = time.perf_counter()
            logits = session.run(None, inputs)[0]
            latency_ms = (time.perf_counter() - started) * 1000
            probabilities = _softmax(np, logits)
            predicted_id = int(np.argmax(probabilities, axis=1)[0])
            confidence = float(probabilities[0, predicted_id])
            prediction = classes[predicted_id]
            predictions.append({
                "image_path": row["image_path"],
                "ground_truth": str(row["label"]),
                "prediction": prediction,
                "confidence": confidence,
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
    return predictions, latencies, failures, session.get_providers()


def _build_report(
    cfg: Any,
    dataset: Path,
    onnx_path: Path,
    predictions: list[dict[str, Any]],
    latencies: list[float],
    failures: list[str | None],
    quality_profile: str | None,
    version: str | None,
    runtime: str,
    providers: list[str],
) -> dict[str, Any]:
    successful_latencies = [
        latency for latency, failure in zip(latencies, failures, strict=False) if failure is None
    ]
    runtime_sec = sum(successful_latencies) / 1000
    correct = sum(1 for prediction in predictions if prediction["passed"])
    failure_rate = sum(failure is not None for failure in failures) / max(len(failures), 1)
    ground_truths = [prediction["ground_truth"] for prediction in predictions]
    predicted_labels = [prediction["prediction"] for prediction in predictions if prediction["prediction"]]
    drift_score = round(
        _distribution_distance(_distribution(ground_truths), _distribution(predicted_labels)),
        4,
    )
    runtime_name = f"onnxruntime-{runtime}"
    metrics = {
        "accuracy": correct / max(len(predictions), 1),
        "p50_latency_ms": _percentile(successful_latencies, 0.50),
        "p95_latency_ms": _percentile(successful_latencies, 0.95),
        "runtime_sec": runtime_sec,
        "samples_per_second": len(predictions) / runtime_sec if runtime_sec > 0 else 0.0,
        "steps_per_second": len(predictions) / runtime_sec if runtime_sec > 0 else 0.0,
        "failure_rate": failure_rate,
        "drift_score": drift_score,
        "model_size_mb": _artifact_size_mb(onnx_path),
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
        "runtime": runtime_name,
        "runtime_provider": providers[0] if providers else None,
        "providers": providers,
        "base_model": cfg.base_model,
        "model_dir": str(onnx_path.parent),
        "onnx_path": str(onnx_path),
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


def _log_wandb(cfg: Any, report: dict[str, Any]) -> None:
    if not wandb_enabled():
        return
    try:
        import wandb
    except ImportError:
        print("[yellow]W&B is not installed; skipping W&B ONNX eval logging.[/yellow]")
        return

    metrics = report["metrics"]
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", cfg.wandb.project),
        name=f"{cfg.vendor}-{report['model_name']}-{report['version']}-{report['runtime']}-eval",
        job_type="evaluate-runtime",
        config={
            "vendor": cfg.vendor,
            "model_name": report["model_name"],
            "version": report["version"],
            "dataset": report["dataset"],
            "runtime": report["runtime"],
            "providers": report["providers"],
        },
    )
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
    run.finish()


@app.command()
def main(
    config: Path = typer.Option(..., "--config", help="Classifier train config YAML."),
    onnx_path: Path = typer.Option(..., "--onnx-path", help="ONNX model path."),
    dataset: Path = typer.Option(..., "--dataset", help="Evaluation JSONL dataset."),
    output: Path = typer.Option(..., "--output", help="Output eval report JSON."),
    runtime: str = typer.Option("cpu", "--runtime", help="ONNX runtime provider: cpu or cuda."),
    quality_profile: str | None = typer.Option(None, "--quality-profile"),
    version: str | None = typer.Option(None, "--version", help="Override model version."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate inputs only."),
) -> None:
    cfg = load_train_config(config)
    if not onnx_path.exists():
        raise typer.BadParameter(f"ONNX model does not exist: {onnx_path}")
    if not dataset.exists():
        raise typer.BadParameter(f"Dataset does not exist: {dataset}")
    rows = _read_jsonl(dataset)
    if cfg.dataset.max_samples:
        rows = rows[: int(cfg.dataset.max_samples)]
    print(f"[model-port] classifier={cfg.training.model_name}")
    print(f"[model-port] version={version or cfg.training.version}")
    print(f"[model-port] runtime=onnxruntime-{runtime}")
    print(f"[model-port] dataset={dataset}")
    print(f"[model-port] onnx_path={onnx_path}")
    if dry_run:
        return

    deps = _require_onnx_deps()
    dataset_dir = Path(cfg.dataset.get("root_dir", dataset.parent))
    predictions, latencies, failures, providers = _evaluate_rows(
        cfg,
        onnx_path,
        rows,
        dataset_dir,
        runtime,
        deps,
    )
    report = _build_report(
        cfg,
        dataset,
        onnx_path,
        predictions,
        latencies,
        failures,
        quality_profile,
        version,
        runtime,
        providers,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    dump_yaml(report, output.with_suffix(".yaml"))
    _log_wandb(cfg, report)
    print(f"[model-port] accuracy={report['metrics']['accuracy']:.4f}")
    print(f"[model-port] p95_latency_ms={report['metrics']['p95_latency_ms']:.4f}")
    print(f"[model-port] quality_gate_passed={report['quality_gate']['passed']}")
    print(f"[model-port] providers={','.join(report['providers'])}")
    print(f"[model-port] wrote eval report to {output}")


if __name__ == "__main__":
    app()
