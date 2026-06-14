from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a deterministic demo eval report for registry alias demos."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--vendor", default="vendor-demo")
    parser.add_argument("--dataset", default="demo")
    parser.add_argument("--accuracy", type=float, required=True)
    parser.add_argument("--p95-latency-ms", type=float, required=True)
    parser.add_argument("--failure-rate", type=float, default=0.0)
    parser.add_argument("--drift-score", type=float, required=True)
    parser.add_argument("--model-size-mb", type=float, required=True)
    parser.add_argument("--min-accuracy", type=float, required=True)
    parser.add_argument("--max-p95-latency-ms", type=float, default=100.0)
    parser.add_argument("--max-failure-rate", type=float, default=0.01)
    parser.add_argument("--max-drift-score", type=float, required=True)
    parser.add_argument("--max-model-size-mb", type=float, default=100.0)
    parser.add_argument("--profile", default="edge-target")
    args = parser.parse_args()

    failures: list[str] = []
    if args.accuracy < args.min_accuracy:
        failures.append("accuracy_below_threshold")
    if args.p95_latency_ms > args.max_p95_latency_ms:
        failures.append("p95_latency_ms_exceeded")
    if args.failure_rate > args.max_failure_rate:
        failures.append("failure_rate_exceeded")
    if args.drift_score > args.max_drift_score:
        failures.append("drift_score_exceeded")
    if args.model_size_mb > args.max_model_size_mb:
        failures.append("model_size_mb_exceeded")

    report = {
        "model_name": args.model_name,
        "version": args.version,
        "vendor": args.vendor,
        "dataset": args.dataset,
        "num_samples": None,
        "metrics": {
            "accuracy": args.accuracy,
            "p50_latency_ms": None,
            "p95_latency_ms": args.p95_latency_ms,
            "runtime_sec": None,
            "samples_per_second": None,
            "steps_per_second": None,
            "failure_rate": args.failure_rate,
            "drift_score": args.drift_score,
            "model_size_mb": args.model_size_mb,
        },
        "quality_gate": {
            "profile": args.profile,
            "min_accuracy": args.min_accuracy,
            "max_p95_latency_ms": args.max_p95_latency_ms,
            "max_failure_rate": args.max_failure_rate,
            "max_drift_score": args.max_drift_score,
            "max_model_size_mb": args.max_model_size_mb,
            "passed": not failures,
            "reject_reason": failures[0] if failures else None,
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote demo eval report: {output}")


if __name__ == "__main__":
    main()
