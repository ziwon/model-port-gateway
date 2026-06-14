from model_port.pipelines.eval_wandb import quality_gate_table_rows, wandb_summary


def test_wandb_summary_promotes_eval_metrics():
    report = _sample_report()

    assert wandb_summary(report) == {
        "p50_latency_ms": 2098.47,
        "p95_latency_ms": 2117.42,
        "runtime_sec": 67.8374,
        "samples_per_second": 0.4717,
        "steps_per_second": 0.4717,
        "failure_rate": 0.0,
        "drift_score": 0.0388,
        "quality_gate_passed": False,
        "quality_gate_profile": "edge-target",
        "quality_gate_reject_reason": "p95_latency_ms_exceeded",
        "quality_gate_p95_latency_value": 2117.42,
        "quality_gate_p95_latency_threshold": 100.0,
        "quality_gate_p95_latency_passed": False,
        "quality_gate_failure_rate_value": 0.0,
        "quality_gate_failure_rate_threshold": 0.01,
        "quality_gate_failure_rate_passed": True,
        "quality_gate_drift_score_value": 0.0388,
        "quality_gate_drift_score_threshold": 0.2,
        "quality_gate_drift_score_passed": True,
    }


def test_quality_gate_table_rows_show_latency_rejection():
    rows = quality_gate_table_rows(_sample_report())

    assert rows == [
        {
            "metric": "p95_latency_ms",
            "value": 2117.42,
            "threshold": 100.0,
            "passed": False,
            "profile": "edge-target",
        },
        {
            "metric": "failure_rate",
            "value": 0.0,
            "threshold": 0.01,
            "passed": True,
            "profile": "edge-target",
        },
        {
            "metric": "drift_score",
            "value": 0.0388,
            "threshold": 0.2,
            "passed": True,
            "profile": "edge-target",
        },
    ]


def test_quality_gate_table_rows_include_classifier_gates():
    rows = quality_gate_table_rows({
        "metrics": {
            "accuracy": 0.95,
            "p95_latency_ms": 12.0,
            "failure_rate": 0.0,
            "drift_score": 0.02,
            "model_size_mb": 9.5,
        },
        "quality_gate": {
            "profile": "edge-target",
            "min_accuracy": 0.8,
            "max_p95_latency_ms": 100.0,
            "max_failure_rate": 0.01,
            "max_drift_score": 0.2,
            "max_model_size_mb": 100.0,
            "passed": True,
            "reject_reason": None,
        },
    })

    assert rows[0] == {
        "metric": "accuracy",
        "value": 0.95,
        "threshold": 0.8,
        "passed": True,
        "profile": "edge-target",
    }
    assert rows[-1] == {
        "metric": "model_size_mb",
        "value": 9.5,
        "threshold": 100.0,
        "passed": True,
        "profile": "edge-target",
    }


def _sample_report():
    return {
        "metrics": {
            "p50_latency_ms": 2098.47,
            "p95_latency_ms": 2117.42,
            "runtime_sec": 67.8374,
            "samples_per_second": 0.4717,
            "steps_per_second": 0.4717,
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
    }
