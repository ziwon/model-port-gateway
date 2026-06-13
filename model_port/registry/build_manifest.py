from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from model_port.common.config import dump_yaml, load_yaml
from model_port.common.quality import normalize_profile_name


def main(base: Path, eval_report: Path, output: Path) -> None:
    from rich import print

    manifest = load_yaml(base)
    report = json.loads(eval_report.read_text(encoding="utf-8"))
    updated = build_manifest(manifest, report)
    dump_yaml(updated, output)
    print(f"Wrote model manifest to {output}")


def build_manifest(manifest: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(manifest)
    model = updated.setdefault("model", {})
    model["name"] = report.get("model_name", model.get("name"))
    model["version"] = str(report.get("version", model.get("version")))
    model["vendor"] = report.get("vendor", model.get("vendor"))

    metrics = report.get("metrics", {})
    gate = report.get("quality_gate", {})
    if report.get("inference"):
        updated["inference"] = report["inference"]

    updated["evaluation"] = {
        "profile": gate.get("profile"),
        "caption_length_mean": metrics.get("caption_length_mean"),
        "caption_length_std": metrics.get("caption_length_std"),
        "p50_latency_ms": metrics.get("p50_latency_ms"),
        "p95_latency_ms": metrics.get("p95_latency_ms"),
        "failure_rate": metrics.get("failure_rate"),
        "drift_score": metrics.get("drift_score"),
        "model_size_mb": metrics.get("model_size_mb"),
        "quality_gate_profile": gate.get("profile"),
        "max_p95_latency_ms": gate.get("max_p95_latency_ms"),
        "max_failure_rate": gate.get("max_failure_rate"),
        "max_drift_score": gate.get("max_drift_score"),
        "max_model_size_mb": gate.get("max_model_size_mb"),
        "passed": bool(gate.get("passed")),
        "reject_reason": gate.get("reject_reason"),
    }
    updated["quality_gates"] = _quality_gates(report)

    deployment = updated.setdefault("deployment", {})
    deployment["stage"] = deployment.get("stage", "candidate")
    passed = bool(gate.get("passed"))
    deployment["promotion_blocked"] = not passed
    deployment["rollout_strategy"] = "canary" if passed else "none"
    if not passed:
        deployment.pop("canary_percent", None)
    if gate.get("reject_reason"):
        deployment["block_reason"] = gate["reject_reason"]
    else:
        deployment.pop("block_reason", None)
    return updated


def _quality_gates(report: dict[str, Any]) -> dict[str, Any]:
    profiles = report.get("quality_gates") or report.get("quality_gate_profiles", {})
    if not profiles:
        active = report.get("quality_gate", {})
        return {
            _profile_key(active.get("profile", "cloud-sim")): {
                key: active.get(key)
                for key in (
                    "max_p95_latency_ms",
                    "max_failure_rate",
                    "max_drift_score",
                    "max_model_size_mb",
                )
                if active.get(key) is not None
            }
        }
    return {
        _profile_key(name): {
            key: gate.get(key)
            for key in (
                "max_p95_latency_ms",
                "max_failure_rate",
                "max_drift_score",
                "max_model_size_mb",
            )
            if gate.get(key) is not None
        }
        for name, gate in profiles.items()
    }


def _profile_key(profile: str) -> str:
    return normalize_profile_name(profile)


if __name__ == "__main__":
    import typer

    app = typer.Typer(help="Build or update a model manifest from an evaluation report.")

    @app.command()
    def cli(
        base: Path = typer.Option(..., "--base", help="Base model manifest YAML."),
        eval_report: Path = typer.Option(..., "--eval-report", help="Evaluation report JSON."),
        output: Path = typer.Option(..., "--output", help="Output manifest YAML path."),
    ) -> None:
        main(base, eval_report, output)

    app()
