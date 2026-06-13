from __future__ import annotations

from typing import Any


def wandb_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "p50_latency_ms": report["metrics"]["p50_latency_ms"],
        "p95_latency_ms": report["metrics"]["p95_latency_ms"],
        "failure_rate": report["metrics"]["failure_rate"],
        "drift_score": report["metrics"]["drift_score"],
        "quality_gate_passed": bool(report["quality_gate"]["passed"]),
        "quality_gate_profile": report["quality_gate"]["profile"],
    }


def quality_gate_table_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = report["metrics"]
    gate = report["quality_gate"]
    profile = gate["profile"]
    return [
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
    ]
