import pytest

from model_port.common.dataset import normalize_caption_row, split_rows
from model_port.common.quality import (
    evaluate_all_quality_gates,
    quality_gate_config,
    quality_gate_profiles,
)


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
