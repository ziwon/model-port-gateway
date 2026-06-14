import pytest

from model_port.common.dataset import normalize_caption_row, split_rows
from model_port.common.images import load_image
from model_port.common.quality import (
    evaluate_quality_gate,
    evaluate_all_quality_gates,
    quality_gate_config,
    quality_gate_profiles,
)
from model_port.common.tracking import wandb_enabled


def test_normalize_caption_row_accepts_aliases():
    row = normalize_caption_row({
        "image": "images/sample.jpg",
        "prompt": "Describe this image.",
        "response": "A sample image.",
    })

    assert row == {
        "image_path": "images/sample.jpg",
        "prompt": "Describe this image.",
        "answer": "A sample image.",
    }


def test_normalize_caption_row_requires_contract_fields():
    with pytest.raises(ValueError, match="missing required fields"):
        normalize_caption_row({"prompt": "Describe this image."})


def test_split_rows_keeps_train_and_eval_when_possible():
    rows = [{"image_path": str(index), "prompt": "p", "answer": "a"} for index in range(4)]

    train_rows, eval_rows = split_rows(rows, 0.9)

    assert len(train_rows) == 3
    assert len(eval_rows) == 1


def test_quality_gates_accept_canonical_and_legacy_names():
    validation = {
        "active_profile": "cloud_sim",
        "quality_gates": {
            "cloud_sim": {
                "max_p95_latency_ms": 3000,
                "max_failure_rate": 0.01,
                "max_drift_score": 0.2,
            },
        },
        "quality_gate_profiles": {
            "edge-target": {
                "max_p95_latency_ms": 100,
                "max_failure_rate": 0.01,
                "max_drift_score": 0.2,
            },
        },
    }
    metrics = {
        "p95_latency_ms": 2000.0,
        "failure_rate": 0.0,
        "drift_score": 0.0,
    }

    assert set(quality_gate_profiles(validation)) == {"cloud_sim", "edge_target"}
    assert quality_gate_config(validation, "edge-target")["max_p95_latency_ms"] == 100
    results = evaluate_all_quality_gates(metrics, validation)
    assert results["cloud-sim"]["passed"] is True
    assert results["edge-target"]["passed"] is False


def test_quality_gate_checks_accuracy_and_model_size():
    metrics = {
        "accuracy": 0.75,
        "p95_latency_ms": 20.0,
        "failure_rate": 0.0,
        "drift_score": 0.0,
        "model_size_mb": 12.0,
    }
    gate = {
        "profile": "edge-target",
        "min_accuracy": 0.8,
        "max_p95_latency_ms": 100,
        "max_failure_rate": 0.01,
        "max_drift_score": 0.2,
        "max_model_size_mb": 10,
    }

    result = evaluate_quality_gate(metrics, gate)

    assert result["passed"] is False
    assert result["reject_reason"] == "model_size_mb_exceeded"
    assert result["min_accuracy"] == 0.8
    assert result["max_model_size_mb"] == 10


def test_wandb_enabled_requires_offline_mode_or_api_key(monkeypatch):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    assert wandb_enabled() is False

    monkeypatch.setenv("WANDB_MODE", "offline")
    assert wandb_enabled() is True

    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_API_KEY", "x" * 40)
    assert wandb_enabled() is False

    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setenv("WANDB_API_KEY", "x" * 40)
    assert wandb_enabled() is True


def test_remote_image_urls_require_explicit_opt_in(monkeypatch, tmp_path):
    monkeypatch.delenv("MODEL_PORT_ALLOW_REMOTE_IMAGES", raising=False)

    with pytest.raises(ValueError, match="Remote image URLs are disabled"):
        load_image("https://example.com/sample.jpg", tmp_path)


def test_transformers_multimodal_auto_class_when_installed():
    transformers = pytest.importorskip("transformers")

    assert hasattr(transformers, "AutoModelForMultimodalLM")
