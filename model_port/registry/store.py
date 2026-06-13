from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from model_port.common.config import load_yaml


@dataclass(frozen=True)
class ModelRegistration:
    vendor: str
    model_name: str
    version: str
    manifest_path: str
    stage: str = "candidate"
    quality_gate_passed: bool = False


class JsonModelRegistry:
    def __init__(self, path: Path) -> None:
        self.path = path

    def register(self, req: ModelRegistration) -> dict[str, Any]:
        manifest_path = Path(req.manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest_path does not exist: {manifest_path}")

        manifest = load_yaml(manifest_path)
        record = record_from_manifest(manifest, manifest_path, req)
        registry = self.load()
        registry[record["id"]] = record
        self.save(registry)
        return record

    def list(self) -> list[dict[str, Any]]:
        return list(self.load().values())

    def get(self, model_id: str) -> dict[str, Any] | None:
        return self.load().get(model_id)

    def get_by_parts(self, vendor: str, model_name: str, version: str) -> dict[str, Any] | None:
        return self.get(model_id(vendor, model_name, version))

    def promote(self, model_id_value: str, target_stage: str) -> dict[str, Any] | None:
        registry = self.load()
        record = registry.get(model_id_value)
        if not record:
            return None

        if not record.get("quality_gate_passed", False):
            return {
                "status": "blocked",
                "reason": "quality_gate_failed",
                "details": promotion_block_details(record),
            }

        from_stage = str(record.get("stage", "candidate"))
        record["stage"] = target_stage
        registry[model_id_value] = record
        self.save(registry)
        return {
            "status": "promoted",
            "from_stage": from_stage,
            "to_stage": target_stage,
            "model": model_id_value,
        }

    def load(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, registry: dict[str, dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def model_id(vendor: str, model_name: str, version: str) -> str:
    return f"{vendor}.{model_name}.{version}"


def record_from_manifest(
    manifest: dict[str, Any],
    manifest_path: Path,
    req: ModelRegistration,
) -> dict[str, Any]:
    model = manifest.get("model", {})
    evaluation = manifest.get("evaluation", {})
    deployment = manifest.get("deployment", {})
    vendor = req.vendor or model.get("vendor")
    model_name = req.model_name or model.get("name")
    version = req.version or str(model.get("version"))
    quality_passed = bool(evaluation.get("passed", req.quality_gate_passed))
    stage = req.stage or deployment.get("stage", "candidate")
    return {
        "id": model_id(vendor, model_name, version),
        "vendor": vendor,
        "model_name": model_name,
        "version": version,
        "task": model.get("task"),
        "runtime": model.get("runtime"),
        "artifact_uri": model.get("artifact_uri"),
        "manifest_path": str(manifest_path),
        "stage": stage,
        "quality_gate_passed": quality_passed,
        "promotion_blocked": bool(deployment.get("promotion_blocked", not quality_passed)),
        "evaluation": evaluation,
        "deployment": deployment,
    }


def promotion_block_details(record: dict[str, Any]) -> dict[str, Any]:
    evaluation = record.get("evaluation", {})
    return {
        "p95_latency_ms": evaluation.get("p95_latency_ms"),
        "max_p95_latency_ms": evaluation.get("max_p95_latency_ms"),
        "drift_score": evaluation.get("drift_score"),
        "max_drift_score": evaluation.get("max_drift_score"),
        "failure_rate": evaluation.get("failure_rate"),
        "max_failure_rate": evaluation.get("max_failure_rate"),
        "reject_reason": evaluation.get("reject_reason"),
    }
