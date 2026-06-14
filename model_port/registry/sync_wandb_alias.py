from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import typer
from rich import print

from model_port.common.config import load_yaml
from model_port.registry.store import JsonModelRegistry
from model_port.registry.wandb_utils import (
    lifecycle_aliases,
    wandb_project,
    wandb_registry_target_path,
)

app = typer.Typer(help="Sync W&B Registry aliases from the model-port registry stage.")


@app.command()
def main(
    model_id: str = typer.Option(..., "--model-id", help="Registry model id."),
    stage: str | None = typer.Option(
        None,
        "--stage",
        help="Expected current stage. If set, it must match the local registry record.",
    ),
    registry_path: Path = typer.Option(
        Path("artifacts/registry/models.json"),
        "--registry-path",
        help="Local model-port registry JSON path.",
    ),
    source_alias: str | None = typer.Option(
        None,
        "--source-alias",
        help="Existing W&B Registry alias to update. Defaults to v<model version>.",
    ),
    aliases: str | None = typer.Option(
        None,
        "--aliases",
        help="Comma-separated aliases. Defaults are derived from the local stage.",
    ),
    dry_run: bool = False,
) -> None:
    registry = JsonModelRegistry(registry_path)
    record = registry.get(model_id)
    if not record:
        raise typer.BadParameter(f"model not found in local registry: {model_id}")

    record_stage = str(record.get("stage", "candidate"))
    if stage and stage != record_stage:
        raise typer.BadParameter(
            f"stage mismatch for {model_id}: local registry has {record_stage!r}, "
            f"but --stage was {stage!r}"
        )

    manifest_path = Path(str(record["manifest_path"]))
    if not manifest_path.exists():
        raise typer.BadParameter(f"manifest_path does not exist: {manifest_path}")

    data = load_yaml(manifest_path)
    data.setdefault("deployment", {})["stage"] = record_stage
    report = _report_from_record(record)
    model = data["model"]
    version = str(record["version"])
    aliases_for_stage = _parse_aliases(aliases) or lifecycle_aliases(
        version,
        record_stage,
        report,
    )

    registry_name = os.getenv("WANDB_REGISTRY_NAME", "Model")
    collection = os.getenv("WANDB_REGISTRY_COLLECTION", str(record["model_name"]))
    target_path = wandb_registry_target_path(collection, registry_name)
    artifact_alias = source_alias or f"v{version}"
    artifact_ref = f"{target_path}:{artifact_alias}"
    entity = os.getenv("WANDB_ENTITY", "").strip()

    if dry_run:
        print("[yellow]Dry run W&B alias sync[/yellow]")
        print({
            "model_id": model_id,
            "stage": record_stage,
            "registry_name": registry_name,
            "collection": collection,
            "target_path": target_path,
            "artifact_ref": artifact_ref,
            "entity": entity or None,
            "aliases": aliases_for_stage,
        })
        return

    if not entity:
        raise typer.BadParameter(
            "WANDB_ENTITY must be set before syncing W&B Registry aliases. "
            "Use the same team/entity used by model_port.registry.wandb_register."
        )

    import wandb

    run = wandb.init(
        entity=entity,
        project=wandb_project(data),
        job_type="registry-sync",
        name=f"sync-{model_id}-{record_stage}",
        tags=["registry", "alias-sync", model["vendor"], model["task"]],
    )
    try:
        artifact = run.use_artifact(artifact_ref)
        linked = run.link_artifact(
            artifact=artifact,
            target_path=target_path,
            aliases=aliases_for_stage,
        )
        if linked is not None:
            linked.wait()
        print(
            f"[green]Synced W&B aliases for {model_id} at stage {record_stage}: "
            f"{', '.join(aliases_for_stage)}[/green]"
        )
    finally:
        run.finish()


def _report_from_record(record: dict[str, Any]) -> dict[str, Any]:
    evaluation = record.get("evaluation", {})
    metrics = {
        "p95_latency_ms": evaluation.get("p95_latency_ms"),
        "failure_rate": evaluation.get("failure_rate"),
        "drift_score": evaluation.get("drift_score"),
        "model_size_mb": evaluation.get("model_size_mb"),
        "accuracy": evaluation.get("accuracy"),
    }
    quality_gate = {
        "profile": evaluation.get("quality_gate_profile"),
        "max_p95_latency_ms": evaluation.get("max_p95_latency_ms"),
        "max_failure_rate": evaluation.get("max_failure_rate"),
        "max_drift_score": evaluation.get("max_drift_score"),
        "max_model_size_mb": evaluation.get("max_model_size_mb"),
        "min_accuracy": evaluation.get("min_accuracy"),
        "passed": bool(record.get("quality_gate_passed", False)),
        "reject_reason": evaluation.get("reject_reason"),
    }
    return {"metrics": metrics, "quality_gate": quality_gate}


def _parse_aliases(aliases: str | None) -> list[str]:
    if not aliases:
        return []
    return [alias.strip() for alias in aliases.split(",") if alias.strip()]


if __name__ == "__main__":
    app()
