from __future__ import annotations

from typing import Any


def wandb_summary(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report["metrics"]
    gate = report["quality_gate"]

    summary = {
        "p50_latency_ms": metrics["p50_latency_ms"],
        "p95_latency_ms": metrics["p95_latency_ms"],
        "failure_rate": metrics["failure_rate"],
        "drift_score": metrics["drift_score"],
        "quality_gate_passed": bool(gate["passed"]),
        "quality_gate_profile": gate["profile"],
        "quality_gate_reject_reason": gate.get("reject_reason"),
        "quality_gate_p95_latency_value": metrics["p95_latency_ms"],
        "quality_gate_p95_latency_threshold": gate["max_p95_latency_ms"],
        "quality_gate_p95_latency_passed": metrics["p95_latency_ms"] <= gate["max_p95_latency_ms"],
        "quality_gate_failure_rate_value": metrics["failure_rate"],
        "quality_gate_failure_rate_threshold": gate["max_failure_rate"],
        "quality_gate_failure_rate_passed": metrics["failure_rate"] <= gate["max_failure_rate"],
        "quality_gate_drift_score_value": metrics["drift_score"],
        "quality_gate_drift_score_threshold": gate["max_drift_score"],
        "quality_gate_drift_score_passed": metrics["drift_score"] <= gate["max_drift_score"],
    }
    for metric in ("runtime_sec", "samples_per_second", "steps_per_second"):
        if metric in metrics:
            summary[metric] = metrics[metric]
    if "accuracy" in metrics:
        summary["accuracy"] = metrics["accuracy"]
    if "model_size_mb" in metrics:
        summary["model_size_mb"] = metrics["model_size_mb"]
    if "min_accuracy" in gate:
        summary["quality_gate_accuracy_value"] = metrics.get("accuracy", 0.0)
        summary["quality_gate_accuracy_threshold"] = gate["min_accuracy"]
        summary["quality_gate_accuracy_passed"] = (
            metrics.get("accuracy", 0.0) >= gate["min_accuracy"]
        )
    if "max_model_size_mb" in gate:
        summary["quality_gate_model_size_value"] = metrics.get("model_size_mb", 0.0)
        summary["quality_gate_model_size_threshold"] = gate["max_model_size_mb"]
        summary["quality_gate_model_size_passed"] = (
            metrics.get("model_size_mb", 0.0) <= gate["max_model_size_mb"]
        )
    return summary


def quality_gate_table_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = report["metrics"]
    gate = report["quality_gate"]
    profile = gate["profile"]
    rows = []
    if "min_accuracy" in gate:
        rows.append({
            "metric": "accuracy",
            "value": metrics.get("accuracy", 0.0),
            "threshold": gate["min_accuracy"],
            "passed": metrics.get("accuracy", 0.0) >= gate["min_accuracy"],
            "profile": profile,
        })
    rows.extend([
        {
            "metric": "p95_latency_ms",
            "value": metrics["p95_latency_ms"],
            "threshold": gate["max_p95_latency_ms"],
            "passed": metrics["p95_latency_ms"] <= gate["max_p95_latency_ms"],
            "profile": profile,
        },
        {
            "metric": "failure_rate",
            "value": metrics["failure_rate"],
            "threshold": gate["max_failure_rate"],
            "passed": metrics["failure_rate"] <= gate["max_failure_rate"],
            "profile": profile,
        },
        {
            "metric": "drift_score",
            "value": metrics["drift_score"],
            "threshold": gate["max_drift_score"],
            "passed": metrics["drift_score"] <= gate["max_drift_score"],
            "profile": profile,
        },
    ])
    if "max_model_size_mb" in gate:
        rows.append({
            "metric": "model_size_mb",
            "value": metrics.get("model_size_mb", 0.0),
            "threshold": gate["max_model_size_mb"],
            "passed": metrics.get("model_size_mb", 0.0) <= gate["max_model_size_mb"],
            "profile": profile,
        })
    return rows
