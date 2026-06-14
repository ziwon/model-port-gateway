from __future__ import annotations

from typing import Any


def runtime_label(report: dict[str, Any]) -> str:
    runtime = str(report.get("runtime", "unknown"))
    if runtime == "torchvision":
        return "PyTorch eager"
    if runtime == "onnxruntime-cpu":
        return "ONNX Runtime CPU"
    if runtime == "onnxruntime-cuda":
        return "ONNX Runtime CUDA"
    return runtime


def runtime_row(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report.get("metrics", {})
    gate = report.get("quality_gate", {})
    return {
        "runtime": runtime_label(report),
        "accuracy": metrics.get("accuracy"),
        "p95_latency_ms": metrics.get("p95_latency_ms"),
        "samples_per_second": metrics.get("samples_per_second"),
        "model_size_mb": metrics.get("model_size_mb"),
        "gate": "passed" if gate.get("passed") else "failed",
    }


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Runtime | Accuracy | p95 Latency | Samples/sec | Size | Gate |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['runtime']} | "
            f"{_fmt(row['accuracy'], 4)} | "
            f"{_fmt(row['p95_latency_ms'], 4)} ms | "
            f"{_fmt(row['samples_per_second'], 4)} | "
            f"{_fmt(row['model_size_mb'], 4)} MB | "
            f"{row['gate']} |"
        )
    return "\n".join(lines)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "TBD"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)
