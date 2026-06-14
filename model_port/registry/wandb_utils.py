from __future__ import annotations

import os
from typing import Any


def wandb_project(data: dict[str, Any]) -> str:
    training = data.get("training", {})
    return os.getenv("WANDB_PROJECT", training.get("wandb_project", "model-port"))


def wandb_registry_target_path(collection: str, registry_name: str | None = None) -> str:
    registry = registry_name or os.getenv("WANDB_REGISTRY_NAME", "Model")
    return f"wandb-registry-{registry}/{collection}"


def artifact_aliases(version: str, report: dict[str, Any] | None) -> list[str]:
    version_alias = f"v{version}"
    aliases = ["candidate"]
    if not report:
        return [*aliases, version_alias]

    gate = report.get("quality_gate", {})
    if gate.get("passed"):
        return [*aliases, "staging", version_alias]

    reject_reason = gate.get("reject_reason")
    if _is_latency_rejection(reject_reason):
        aliases.append("rejected-latency")
    elif reject_reason:
        aliases.append("rejected-quality")
    else:
        aliases.append("rejected")
    aliases.append(version_alias)
    return aliases


def _is_latency_rejection(reject_reason: Any) -> bool:
    return str(reject_reason or "") == "p95_latency_ms_exceeded"
