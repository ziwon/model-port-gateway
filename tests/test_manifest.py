import pytest

from model_port.common.config import dump_yaml, load_yaml
from model_port.registry.build_manifest import build_manifest
from model_port.registry.store import JsonModelRegistry, ModelRegistration
from model_port.registry.wandb_utils import (
    artifact_aliases,
    wandb_project,
    wandb_registry_target_path,
)


def test_api_registration_rejects_client_quality_gate_field():
    pytest.importorskip("fastapi")
    from pydantic import ValidationError

    from model_port.api.main import ModelRegistration as ApiModelRegistration

    try:
        ApiModelRegistration(
            vendor="vendor-demo",
            model_name="smart-captioner",
            version="0.1.0",
            manifest_path="artifacts/manifests/vendor-demo-smart-captioner-0.1.0.yaml",
            stage="candidate",
            quality_gate_passed=True,
        )
    except ValidationError as exc:
        assert "quality_gate_passed" in str(exc)
    else:
        raise AssertionError("quality_gate_passed must not be accepted from clients")


def test_manifest_has_required_sections():
    data = load_yaml("configs/model_manifest.example.yaml")
    assert "model" in data
    assert "training" in data
    assert data["model"]["name"] == "smart-captioner"


def test_wandb_project_comes_from_training_section(monkeypatch):
    monkeypatch.delenv("WANDB_PROJECT", raising=False)

    assert wandb_project(load_yaml("configs/model_manifest.example.yaml")) == "model-port"


def test_wandb_project_env_overrides_manifest(monkeypatch):
    monkeypatch.setenv("WANDB_PROJECT", "override-project")

    assert wandb_project(load_yaml("configs/model_manifest.example.yaml")) == "override-project"


def test_wandb_registry_target_path_defaults_to_model_registry(monkeypatch):
    monkeypatch.delenv("WANDB_REGISTRY_NAME", raising=False)

    assert wandb_registry_target_path("smart-captioner") == "wandb-registry-Model/smart-captioner"


def test_wandb_registry_target_path_accepts_registry_override(monkeypatch):
    monkeypatch.setenv("WANDB_REGISTRY_NAME", "Edge")

    assert wandb_registry_target_path("edge-object-classifier") == (
        "wandb-registry-Edge/edge-object-classifier"
    )


def test_wandb_aliases_mark_latency_rejection():
    aliases = artifact_aliases(
        "0.1.0",
        {
            "quality_gate": {
                "passed": False,
                "reject_reason": "p95_latency_ms_exceeded",
            }
        },
    )

    assert aliases == ["candidate", "rejected-latency", "v0.1.0"]


def test_wandb_aliases_mark_quality_rejection():
    aliases = artifact_aliases(
        "0.2.0",
        {
            "quality_gate": {
                "passed": False,
                "reject_reason": "accuracy_below_threshold",
            }
        },
    )

    assert aliases == ["candidate", "rejected-quality", "v0.2.0"]


def test_wandb_aliases_mark_staging_candidate():
    aliases = artifact_aliases(
        "0.3.0",
        {
            "quality_gate": {
                "passed": True,
                "reject_reason": None,
            }
        },
    )

    assert aliases == ["candidate", "staging", "v0.3.0"]


def test_build_manifest_blocks_failed_quality_gate():
    manifest = load_yaml("configs/model_manifest.example.yaml")
    report = {
        "model_name": "smart-captioner",
        "version": "0.1.0",
        "vendor": "vendor-demo",
            "metrics": {
            "accuracy": 0.9,
            "caption_length_mean": 54.5,
            "p50_latency_ms": 2098.4701,
            "p95_latency_ms": 2117.4231,
            "failure_rate": 0.0,
            "drift_score": 0.0388,
            "model_size_mb": 12.5,
        },
        "quality_gate": {
            "profile": "edge-target",
            "min_accuracy": 0.8,
            "max_p95_latency_ms": 100.0,
            "max_failure_rate": 0.01,
            "max_drift_score": 0.2,
            "max_model_size_mb": 100.0,
            "passed": False,
            "reject_reason": "p95_latency_ms_exceeded",
        },
        "inference": {
            "max_new_tokens": 16,
            "do_sample": False,
            "num_beams": 1,
        },
        "quality_gate_profiles": {
            "cloud-sim": {
                "max_p95_latency_ms": 3000.0,
                "max_failure_rate": 0.01,
                "max_drift_score": 0.2,
                "passed": True,
            },
            "edge-target": {
                "min_accuracy": 0.8,
                "max_p95_latency_ms": 100.0,
                "max_failure_rate": 0.01,
                "max_drift_score": 0.2,
                "max_model_size_mb": 100.0,
                "passed": False,
                "reject_reason": "p95_latency_ms_exceeded",
            },
        },
    }

    updated = build_manifest(manifest, report)

    assert updated["inference"]["max_new_tokens"] == 16
    assert updated["evaluation"]["p95_latency_ms"] == 2117.4231
    assert updated["evaluation"]["accuracy"] == 0.9
    assert updated["evaluation"]["min_accuracy"] == 0.8
    assert updated["evaluation"]["model_size_mb"] == 12.5
    assert updated["evaluation"]["passed"] is False
    assert updated["evaluation"]["reject_reason"] == "p95_latency_ms_exceeded"
    assert updated["quality_gates"]["cloud_sim"]["max_p95_latency_ms"] == 3000.0
    assert updated["quality_gates"]["edge_target"]["max_p95_latency_ms"] == 100.0
    assert updated["quality_gates"]["edge_target"]["min_accuracy"] == 0.8
    assert updated["quality_gates"]["edge_target"]["max_model_size_mb"] == 100.0
    assert updated["deployment"]["stage"] == "candidate"
    assert updated["deployment"]["promotion_blocked"] is True
    assert updated["deployment"]["rollout_strategy"] == "none"
    assert "canary_percent" not in updated["deployment"]


def test_registry_blocks_failed_quality_gate_promotion(tmp_path):
    manifest = build_manifest(
        load_yaml("configs/model_manifest.example.yaml"),
        {
            "model_name": "smart-captioner",
            "version": "0.1.0",
            "vendor": "vendor-demo",
            "metrics": {
                "p95_latency_ms": 2117.4231,
                "failure_rate": 0.0,
                "drift_score": 0.0388,
            },
            "quality_gate": {
                "profile": "edge-target",
                "max_p95_latency_ms": 100.0,
                "max_failure_rate": 0.01,
                "max_drift_score": 0.2,
                "passed": False,
                "reject_reason": "p95_latency_ms_exceeded",
            },
        },
    )
    manifest_path = tmp_path / "manifest.yaml"
    dump_yaml(manifest, manifest_path)

    registry = JsonModelRegistry(tmp_path / "registry" / "models.json")
    record = registry.register(
        ModelRegistration(
            vendor="vendor-demo",
            model_name="smart-captioner",
            version="0.1.0",
            manifest_path=str(manifest_path),
            stage="candidate",
        )
    )
    assert record["id"] == "vendor-demo.smart-captioner.0.1.0"

    promote_resp = registry.promote(
        "vendor-demo.smart-captioner.0.1.0",
        "staging",
    )

    assert promote_resp == {
        "status": "blocked",
        "reason": "quality_gate_failed",
        "details": {
            "p95_latency_ms": 2117.4231,
            "max_p95_latency_ms": 100.0,
            "drift_score": 0.0388,
            "max_drift_score": 0.2,
            "failure_rate": 0.0,
            "max_failure_rate": 0.01,
            "reject_reason": "p95_latency_ms_exceeded",
        },
    }


def test_registry_promotes_passed_quality_gate(tmp_path):
    manifest = build_manifest(
        load_yaml("configs/model_manifest.example.yaml"),
        {
            "model_name": "smart-captioner",
            "version": "0.1.1",
            "vendor": "vendor-demo",
            "metrics": {
                "p95_latency_ms": 1800.0,
                "failure_rate": 0.0,
                "drift_score": 0.0388,
            },
            "quality_gate": {
                "profile": "cloud-sim",
                "max_p95_latency_ms": 3000.0,
                "max_failure_rate": 0.01,
                "max_drift_score": 0.2,
                "passed": True,
                "reject_reason": None,
            },
        },
    )
    manifest_path = tmp_path / "manifest.yaml"
    dump_yaml(manifest, manifest_path)

    registry = JsonModelRegistry(tmp_path / "registry" / "models.json")
    registry.register(
        ModelRegistration(
            vendor="vendor-demo",
            model_name="smart-captioner",
            version="0.1.1",
            manifest_path=str(manifest_path),
            stage="candidate",
        )
    )

    promote_resp = registry.promote(
        "vendor-demo.smart-captioner.0.1.1",
        "staging",
    )

    assert promote_resp == {
        "status": "promoted",
        "from_stage": "candidate",
        "to_stage": "staging",
        "model": "vendor-demo.smart-captioner.0.1.1",
    }


def test_registry_derives_quality_gate_from_manifest_only(tmp_path):
    manifest = build_manifest(
        load_yaml("configs/model_manifest.example.yaml"),
        {
            "model_name": "smart-captioner",
            "version": "0.1.2",
            "vendor": "vendor-demo",
            "metrics": {
                "p95_latency_ms": 2117.4231,
                "failure_rate": 0.0,
                "drift_score": 0.0388,
            },
            "quality_gate": {
                "profile": "edge-target",
                "max_p95_latency_ms": 100.0,
                "max_failure_rate": 0.01,
                "max_drift_score": 0.2,
                "passed": False,
                "reject_reason": "p95_latency_ms_exceeded",
            },
        },
    )
    manifest_path = tmp_path / "manifest.yaml"
    dump_yaml(manifest, manifest_path)

    registry = JsonModelRegistry(tmp_path / "registry" / "models.json")
    record = registry.register(
        ModelRegistration(
            vendor="vendor-demo",
            model_name="smart-captioner",
            version="0.1.2",
            manifest_path=str(manifest_path),
            stage="candidate",
        )
    )

    assert record["quality_gate_passed"] is False
    assert record["promotion_blocked"] is True
