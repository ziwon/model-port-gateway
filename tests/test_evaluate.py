from model_port.pipelines.eval_wandb import quality_gate_table_rows, wandb_summary


def test_wandb_summary_promotes_eval_metrics():
    report = _sample_report()

    assert wandb_summary(report) == {
        "p50_latency_ms": 2098.47,
        "p95_latency_ms": 2117.42,
        "failure_rate": 0.0,
        "drift_score": 0.0388,
        "quality_gate_passed": False,
        "quality_gate_profile": "edge-target",
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


def _sample_report():
    return {
        "metrics": {
            "p50_latency_ms": 2098.47,
            "p95_latency_ms": 2117.42,
            "failure_rate": 0.0,
            "drift_score": 0.0388,
        },
        "quality_gate": {
            "profile": "edge-target",
            "max_p95_latency_ms": 100.0,
            "max_failure_rate": 0.01,
            "max_drift_score": 0.2,
            "passed": False,
        },
    }
