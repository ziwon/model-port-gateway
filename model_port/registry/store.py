from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from model_port.common.config import load_yaml

ALLOWED_PROMOTION_TARGETS = {"staging", "production"}
ALLOWED_STAGE_TRANSITIONS = {
    "candidate": {"staging"},
    "staging": {"production"},
    "production": set(),
}


@dataclass(frozen=True)
class ModelRegistration:
    vendor: str
    model_name: str
    version: str
    manifest_path: str
    stage: str = "candidate"


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

        result = validate_promotion(record, model_id_value, target_stage)
        if result is not None:
            return result

        from_stage = str(record.get("stage", "candidate"))
        record["stage"] = target_stage
        record.setdefault("deployment", {})["stage"] = target_stage
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


class SqliteModelRegistry:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._ensure_schema()

    def register(self, req: ModelRegistration) -> dict[str, Any]:
        manifest_path = Path(req.manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest_path does not exist: {manifest_path}")

        manifest = load_yaml(manifest_path)
        record = record_from_manifest(manifest, manifest_path, req)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO models (
                    id, vendor, model_name, version, task, runtime, artifact_uri,
                    manifest_path, stage, quality_gate_passed, promotion_blocked,
                    evaluation_json, deployment_json, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(id) DO UPDATE SET
                    vendor = excluded.vendor,
                    model_name = excluded.model_name,
                    version = excluded.version,
                    task = excluded.task,
                    runtime = excluded.runtime,
                    artifact_uri = excluded.artifact_uri,
                    manifest_path = excluded.manifest_path,
                    stage = excluded.stage,
                    quality_gate_passed = excluded.quality_gate_passed,
                    promotion_blocked = excluded.promotion_blocked,
                    evaluation_json = excluded.evaluation_json,
                    deployment_json = excluded.deployment_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                _record_to_row(record),
            )
        return record

    def list(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM models ORDER BY id").fetchall()
        return [_row_to_record(row) for row in rows]

    def get(self, model_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
        if row is None:
            return None
        return _row_to_record(row)

    def get_by_parts(self, vendor: str, model_name: str, version: str) -> dict[str, Any] | None:
        return self.get(model_id(vendor, model_name, version))

    def promote(self, model_id_value: str, target_stage: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id_value,)).fetchone()
            if row is None:
                return None

            record = _row_to_record(row)
            result = validate_promotion(record, model_id_value, target_stage)
            if result is not None:
                return result

            from_stage = str(record.get("stage", "candidate"))
            record["stage"] = target_stage
            record.setdefault("deployment", {})["stage"] = target_stage
            conn.execute(
                """
                UPDATE models
                SET stage = ?, deployment_json = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    target_stage,
                    json.dumps(record.get("deployment", {})),
                    model_id_value,
                ),
            )
        return {
            "status": "promoted",
            "from_stage": from_stage,
            "to_stage": target_stage,
            "model": model_id_value,
        }

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS models (
                    id TEXT PRIMARY KEY,
                    vendor TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    task TEXT,
                    runtime TEXT,
                    artifact_uri TEXT,
                    manifest_path TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    quality_gate_passed INTEGER NOT NULL,
                    promotion_blocked INTEGER NOT NULL,
                    evaluation_json TEXT NOT NULL,
                    deployment_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )


def open_model_registry(
    backend: str | None = None,
    path: Path | None = None,
) -> JsonModelRegistry | SqliteModelRegistry:
    backend_name = (backend or os.getenv("MODEL_PORT_REGISTRY_BACKEND", "json")).lower()
    registry_path = path or default_registry_path(backend_name)
    if backend_name == "json":
        return JsonModelRegistry(registry_path)
    if backend_name == "sqlite":
        return SqliteModelRegistry(registry_path)
    raise ValueError("MODEL_PORT_REGISTRY_BACKEND must be one of: json, sqlite")


def default_registry_path(backend: str | None = None) -> Path:
    backend_name = (backend or os.getenv("MODEL_PORT_REGISTRY_BACKEND", "json")).lower()
    env_path = os.getenv("MODEL_PORT_REGISTRY_PATH")
    if env_path:
        return Path(env_path)
    if backend_name == "sqlite":
        return Path("artifacts/registry/models.db")
    return Path("artifacts/registry/models.json")


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
    quality_passed = bool(evaluation.get("passed", False))
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


def validate_promotion(
    record: dict[str, Any],
    model_id_value: str,
    target_stage: str,
) -> dict[str, Any] | None:
    from_stage = str(record.get("stage", "candidate"))
    if target_stage not in ALLOWED_PROMOTION_TARGETS:
        return invalid_transition_response(
            model_id_value,
            from_stage,
            target_stage,
            "invalid_target_stage",
        )

    if target_stage not in ALLOWED_STAGE_TRANSITIONS.get(from_stage, set()):
        return invalid_transition_response(
            model_id_value,
            from_stage,
            target_stage,
            "invalid_stage_transition",
        )

    if not record.get("quality_gate_passed", False):
        return {
            "status": "blocked",
            "reason": "quality_gate_failed",
            "details": promotion_block_details(record),
        }
    return None


def invalid_transition_response(
    model_id_value: str,
    from_stage: str,
    target_stage: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "status": "blocked",
        "reason": reason,
        "details": {
            "model": model_id_value,
            "from_stage": from_stage,
            "target_stage": target_stage,
            "allowed_targets": sorted(ALLOWED_STAGE_TRANSITIONS.get(from_stage, set())),
        },
    }


def _record_to_row(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record["id"],
        record["vendor"],
        record["model_name"],
        record["version"],
        record.get("task"),
        record.get("runtime"),
        record.get("artifact_uri"),
        record["manifest_path"],
        record["stage"],
        int(bool(record["quality_gate_passed"])),
        int(bool(record["promotion_blocked"])),
        json.dumps(record.get("evaluation", {})),
        json.dumps(record.get("deployment", {})),
    )


def _row_to_record(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "vendor": row["vendor"],
        "model_name": row["model_name"],
        "version": row["version"],
        "task": row["task"],
        "runtime": row["runtime"],
        "artifact_uri": row["artifact_uri"],
        "manifest_path": row["manifest_path"],
        "stage": row["stage"],
        "quality_gate_passed": bool(row["quality_gate_passed"]),
        "promotion_blocked": bool(row["promotion_blocked"]),
        "evaluation": json.loads(row["evaluation_json"]),
        "deployment": json.loads(row["deployment_json"]),
    }
