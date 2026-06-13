from model_port.common.config import dump_yaml, load_yaml
from model_port.registry.build_manifest import build_manifest
from model_port.registry.store import JsonModelRegistry, ModelRegistration
from model_port.registry.wandb_utils import wandb_project


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


def test_build_manifest_blocks_failed_quality_gate():
    manifest = load_yaml("configs/model_manifest.example.yaml")
    report = {
        "model_name": "smart-captioner",
        "version": "0.1.0",
        "vendor": "vendor-demo",
        "metrics": {
            "caption_length_mean": 54.5,
            "p50_latency_ms": 2098.4701,
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
                "max_p95_latency_ms": 100.0,
                "max_failure_rate": 0.01,
                "max_drift_score": 0.2,
                "passed": False,
                "reject_reason": "p95_latency_ms_exceeded",
            },
        },
    }

    updated = build_manifest(manifest, report)

    assert updated["inference"]["max_new_tokens"] == 16
    assert updated["evaluation"]["p95_latency_ms"] == 2117.4231
    assert updated["evaluation"]["passed"] is False
    assert updated["evaluation"]["reject_reason"] == "p95_latency_ms_exceeded"
    assert updated["quality_gates"]["cloud_sim"]["max_p95_latency_ms"] == 3000.0
    assert updated["quality_gates"]["edge_target"]["max_p95_latency_ms"] == 100.0
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
            quality_gate_passed=False,
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
            quality_gate_passed=True,
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
