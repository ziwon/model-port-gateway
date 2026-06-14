from __future__ import annotations

import gc
import json
import os
import re
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import typer
from rich import print

from model_port.common.config import load_train_config
from model_port.common.dataset import read_caption_jsonl
from model_port.common.images import load_image
from model_port.common.metadata import model_metadata
from model_port.common.quality import (
    evaluate_all_quality_gates,
    evaluate_quality_gate,
    quality_gate_config,
)
from model_port.common.tracking import wandb_enabled, wandb_skip_message
from model_port.pipelines.eval_wandb import quality_gate_table_rows, wandb_summary

app = typer.Typer(help="Evaluate base/candidate VLM outputs and prepare drift reports.")


def _require_eval_deps() -> dict[str, Any]:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig
    except ImportError as exc:
        raise typer.BadParameter(
            "Evaluation dependencies are missing. Rebuild/install the train extra with "
            "`pip install -e '.[train,dev,api]'` or `docker compose build trainer`."
        ) from exc

    return {
        "torch": torch,
        "PeftModel": PeftModel,
        "AutoModelForMultimodalLM": AutoModelForMultimodalLM,
        "AutoProcessor": AutoProcessor,
        "BitsAndBytesConfig": BitsAndBytesConfig,
    }


def _format_prompt(processor: Any, prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def _move_to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def _load_model(
    cfg: Any,
    deps: dict[str, Any],
    model_dir: Path | None = None,
) -> tuple[Any, Any]:
    torch = deps["torch"]
    AutoModelForMultimodalLM = deps["AutoModelForMultimodalLM"]
    AutoProcessor = deps["AutoProcessor"]
    BitsAndBytesConfig = deps["BitsAndBytesConfig"]
    PeftModel = deps["PeftModel"]

    training = cfg.training
    compute_dtype = torch.bfloat16 if training.get("bf16", False) else torch.float16
    quantization_config = None
    if training.get("load_in_4bit", False):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    processor = AutoProcessor.from_pretrained(cfg.base_model)
    model = AutoModelForMultimodalLM.from_pretrained(
        cfg.base_model,
        device_map="auto",
        torch_dtype=compute_dtype,
        quantization_config=quantization_config,
    )
    model.eval()

    if model_dir is not None:
        if not model_dir.exists():
            raise typer.BadParameter(f"Fine-tuned model adapter dir does not exist: {model_dir}")
        model = PeftModel.from_pretrained(model, str(model_dir))
        model.eval()

    return processor, model


def _generation_kwargs(cfg: Any) -> dict[str, Any]:
    inference = cfg.inference
    kwargs: dict[str, Any] = {
        "max_new_tokens": int(
            inference.get("max_new_tokens", cfg.validation.get("max_new_tokens", 64))
        ),
        "do_sample": bool(inference.get("do_sample", False)),
        "num_beams": int(inference.get("num_beams", 1)),
    }
    if kwargs["do_sample"] and "temperature" in inference:
        kwargs["temperature"] = float(inference["temperature"])
    return kwargs


def _generate_predictions(
    cfg: Any,
    rows: list[dict[str, Any]],
    dataset_dir: Path,
    deps: dict[str, Any],
    model_dir: Path | None,
) -> tuple[list[str], list[float], list[str | None]]:
    torch = deps["torch"]
    processor, model = _load_model(cfg, deps, model_dir=model_dir)
    predictions: list[str] = []
    latencies: list[float] = []
    failures: list[str | None] = []
    generation_kwargs = _generation_kwargs(cfg)

    try:
        for row in rows:
            try:
                image = load_image(row["image_path"], dataset_dir)
                text = _format_prompt(processor, row["prompt"])
                batch = processor(text=text, images=[image], return_tensors="pt")
                device = next(model.parameters()).device
                batch = _move_to_device(batch, device)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.no_grad():
                    generated_ids = model.generate(**batch, **generation_kwargs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                latency_ms = (time.perf_counter() - started) * 1000

                input_len = batch["input_ids"].shape[-1]
                new_tokens = generated_ids[:, input_len:]
                prediction = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
                predictions.append(prediction)
                latencies.append(latency_ms)
                failures.append(None)
            except Exception as exc:  # noqa: BLE001 - evaluation should continue per sample.
                predictions.append("")
                latencies.append(0.0)
                failures.append(type(exc).__name__)
    finally:
        del model
        del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return predictions, latencies, failures


def _words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", text.lower())


def _keyword_distribution(texts: list[str], top_k: int = 20) -> dict[str, float]:
    counts = Counter(word for text in texts for word in _words(text))
    total = sum(counts.values())
    if total == 0:
        return {}
    return {word: count / total for word, count in counts.most_common(top_k)}


def _distribution_distance(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    return sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys) / 2


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _image_stats(rows: list[dict[str, Any]], dataset_dir: Path) -> dict[str, float]:
    widths: list[float] = []
    heights: list[float] = []
    brightness: list[float] = []
    for row in rows:
        image = load_image(row["image_path"], dataset_dir)
        widths.append(float(image.width))
        heights.append(float(image.height))
        gray = image.convert("L")
        histogram = gray.histogram()
        pixel_count = sum(histogram)
        brightness.append(
            sum(value * count for value, count in enumerate(histogram)) / pixel_count
        )

    return {
        "width_mean": statistics.fmean(widths) if widths else 0.0,
        "height_mean": statistics.fmean(heights) if heights else 0.0,
        "brightness_mean": statistics.fmean(brightness) if brightness else 0.0,
    }


def _dir_size_mb(path: Path | None) -> float:
    if path is None or not path.exists():
        return 0.0
    if path.is_file():
        return path.stat().st_size / (1024**2)
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) / (1024**2)


def _build_report(
    cfg: Any,
    dataset: Path,
    rows: list[dict[str, Any]],
    base_predictions: list[str],
    fine_tuned_predictions: list[str],
    latencies: list[float],
    failures: list[str | None],
    image_stats: dict[str, float],
    quality_profile: str | None,
    model_dir: Path | None,
    version: str | None,
) -> dict[str, Any]:
    metadata = model_metadata(cfg)
    lengths = [len(prediction.split()) for prediction in fine_tuned_predictions if prediction]
    base_lengths = [len(prediction.split()) for prediction in base_predictions if prediction]
    keyword_drift = _distribution_distance(
        _keyword_distribution(base_predictions),
        _keyword_distribution(fine_tuned_predictions),
    )
    length_mean = statistics.fmean(lengths) if lengths else 0.0
    base_length_mean = statistics.fmean(base_lengths) if base_lengths else 0.0
    length_drift = abs(length_mean - base_length_mean) / max(base_length_mean, 1.0)
    drift_score = min(1.0, round((length_drift + keyword_drift) / 2, 4))
    failure_rate = sum(failure is not None for failure in failures) / max(len(failures), 1)
    runtime_sec = sum(latencies) / 1000
    samples_per_second = len(rows) / runtime_sec if runtime_sec > 0 else 0.0

    p95_latency = _percentile(latencies, 0.95)
    metrics = {
        "caption_length_mean": round(length_mean, 4),
        "caption_length_std": round(statistics.pstdev(lengths), 4) if len(lengths) > 1 else 0.0,
        "p50_latency_ms": round(_percentile(latencies, 0.50), 4),
        "p95_latency_ms": round(p95_latency, 4),
        "runtime_sec": round(runtime_sec, 4),
        "samples_per_second": round(samples_per_second, 4),
        "steps_per_second": round(samples_per_second, 4),
        "failure_rate": round(failure_rate, 4),
        "drift_score": drift_score,
        "model_size_mb": round(_dir_size_mb(model_dir), 4),
    }
    validation = cfg.validation.model_dump(exclude_none=True)
    quality_gate = evaluate_quality_gate(metrics, quality_gate_config(validation, quality_profile))
    profile_results = evaluate_all_quality_gates(metrics, validation)

    return {
        "model_name": metadata["model_name"],
        "version": version or metadata["version"],
        "vendor": metadata["vendor"],
        "dataset": dataset.stem,
        "num_samples": len(rows),
        "inference": _generation_kwargs(cfg),
        "metrics": metrics,
        "quality_gate": quality_gate,
        "quality_gates": profile_results,
        "quality_gate_profiles": profile_results,
        "drift": {
            "baseline_caption_length_mean": round(base_length_mean, 4),
            "candidate_caption_length_mean": round(length_mean, 4),
            "keyword_distribution_distance": round(keyword_drift, 4),
            "base_keyword_distribution": _keyword_distribution(base_predictions),
            "candidate_keyword_distribution": _keyword_distribution(fine_tuned_predictions),
            "image_stats": {key: round(value, 4) for key, value in image_stats.items()},
        },
    }


def _log_wandb_table(
    cfg: Any,
    rows: list[dict[str, Any]],
    dataset_dir: Path,
    base_predictions: list[str],
    fine_tuned_predictions: list[str],
    latencies: list[float],
    failures: list[str | None],
    report: dict[str, Any],
) -> None:
    if not wandb_enabled():
        print(f"[yellow]{wandb_skip_message()}[/yellow]")
        return

    import wandb

    quality_rows = quality_gate_table_rows(report)
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", cfg.wandb.get("project", "model-port")),
        name=f"eval-{report['vendor']}-{report['model_name']}-{report['version']}",
        job_type="evaluation",
        tags=cfg.wandb.get("tags", []),
        config={
            "base_model": cfg.base_model,
            "dataset": report["dataset"],
            "model_dir": cfg.training.get("output_dir"),
        },
    )
    try:
        table = wandb.Table(
            columns=[
                "image",
                "prompt",
                "ground_truth",
                "base_prediction",
                "fine_tuned_prediction",
                "latency_ms",
                "caption_length",
                "passed",
            ]
        )
        for row, base_pred, ft_pred, latency, failure in zip(
            rows,
            base_predictions,
            fine_tuned_predictions,
            latencies,
            failures,
            strict=False,
        ):
            image_path = Path(row["image_path"])
            if not image_path.is_absolute():
                image_path = dataset_dir / image_path
            passed = failure is None and bool(ft_pred.strip())
            table.add_data(
                wandb.Image(str(image_path)),
                row["prompt"],
                row["answer"],
                base_pred,
                ft_pred,
                latency,
                len(ft_pred.split()),
                passed,
            )

        quality_table = wandb.Table(columns=["metric", "value", "threshold", "passed", "profile"])
        for row in quality_rows:
            quality_table.add_data(
                row["metric"],
                row["value"],
                row["threshold"],
                row["passed"],
                row["profile"],
            )

        wandb.log({"eval/predictions": table})
        wandb.log({"eval/quality_gate_table": quality_table})
        wandb.log({
            "eval/caption_length_mean": report["metrics"]["caption_length_mean"],
            "eval/p50_latency_ms": report["metrics"]["p50_latency_ms"],
            "eval/p95_latency_ms": report["metrics"]["p95_latency_ms"],
            "eval/runtime": report["metrics"]["runtime_sec"],
            "eval/samples_per_second": report["metrics"]["samples_per_second"],
            "eval/steps_per_second": report["metrics"]["steps_per_second"],
            "eval/failure_rate": report["metrics"]["failure_rate"],
            "drift/score": report["metrics"]["drift_score"],
            "quality_gate/passed": int(report["quality_gate"]["passed"]),
            "quality_gate/profile": report["quality_gate"]["profile"],
        })
        wandb.summary.update(wandb_summary(report))
    finally:
        run.finish()


@app.command()
def main(
    config: Path = typer.Option(..., "--config", "-c", help="Training config YAML."),
    model_dir: Path | None = typer.Option(None, "--model-dir", help="LoRA adapter directory."),
    dataset: Path | None = typer.Option(None, "--dataset", help="Evaluation JSONL dataset."),
    output: Path = typer.Option(
        Path("artifacts/eval/eval_report.json"),
        "--output",
        help="Evaluation report JSON path.",
    ),
    quality_profile: str | None = typer.Option(
        None,
        "--quality-profile",
        help="Quality gate profile to apply, e.g. cloud-sim or edge-target.",
    ),
    version: str | None = typer.Option(None, "--version", help="Override model version in report."),
    dry_run: bool = False,
) -> None:
    cfg = load_train_config(config)
    dataset_path = dataset or Path(cfg.dataset.get("local_jsonl", "data/sample_captions.jsonl"))
    candidate_dir = model_dir or Path(cfg.training.get("output_dir", ""))
    try:
        rows = read_caption_jsonl(dataset_path, cfg.dataset.get("max_samples"))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    dataset_dir = dataset_path.parent

    if dry_run:
        base_predictions = [row["answer"] for row in rows]
        fine_tuned_predictions = [row["answer"] for row in rows]
        latencies = [0.0 for _ in rows]
        failures = [None for _ in rows]
    else:
        deps = _require_eval_deps()
        print("[cyan]Running base model inference[/cyan]")
        base_predictions, _, base_failures = _generate_predictions(
            cfg,
            rows,
            dataset_dir,
            deps,
            model_dir=None,
        )
        if any(base_failures):
            print(f"[yellow]Base inference failures: {Counter(base_failures)}[/yellow]")

        print("[cyan]Running fine-tuned model inference[/cyan]")
        fine_tuned_predictions, latencies, failures = _generate_predictions(
            cfg,
            rows,
            dataset_dir,
            deps,
            model_dir=candidate_dir,
        )

    report = _build_report(
        cfg=cfg,
        dataset=dataset_path,
        rows=rows,
        base_predictions=base_predictions,
        fine_tuned_predictions=fine_tuned_predictions,
        latencies=latencies,
        failures=failures,
        image_stats=_image_stats(rows, dataset_dir),
        quality_profile=quality_profile,
        model_dir=candidate_dir,
        version=version,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not dry_run:
        _log_wandb_table(
            cfg,
            rows,
            dataset_dir,
            base_predictions,
            fine_tuned_predictions,
            latencies,
            failures,
            report,
        )

    print("[bold]Evaluation report[/bold]")
    print(report)
    print(f"[green]Wrote evaluation report to {output}[/green]")


if __name__ == "__main__":
    app()
