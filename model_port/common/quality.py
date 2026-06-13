from __future__ import annotations

from typing import Any


DEFAULT_GATE = {
    "max_p95_latency_ms": 3000,
    "max_failure_rate": 0.01,
    "max_drift_score": 0.2,
}


def normalize_profile_name(profile: str) -> str:
    return profile.replace("-", "_")


def display_profile_name(profile: str) -> str:
    return normalize_profile_name(profile).replace("_", "-")


def quality_gate_profiles(validation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}

    for name, gate in validation.get("quality_gate_profiles", {}).items():
        profiles[normalize_profile_name(str(name))] = dict(gate)

    for name, gate in validation.get("quality_gates", {}).items():
        profiles[normalize_profile_name(str(name))] = dict(gate)

    if not profiles:
        active = dict(validation.get("quality_gate", {}))
        profile = normalize_profile_name(str(active.get("profile", "cloud_sim")))
        profiles[profile] = {
            key: value for key, value in active.items() if key != "profile" and value is not None
        }

    return profiles


def active_quality_profile(validation: dict[str, Any], profile: str | None = None) -> str:
    if profile:
        return normalize_profile_name(profile)
    if validation.get("active_profile"):
        return normalize_profile_name(str(validation["active_profile"]))
    active_gate = validation.get("quality_gate", {})
    return normalize_profile_name(str(active_gate.get("profile", "cloud_sim")))


def quality_gate_config(validation: dict[str, Any], profile: str | None = None) -> dict[str, Any]:
    profile_name = active_quality_profile(validation, profile)
    active_gate = dict(validation.get("quality_gate", {}))
    active_gate.pop("profile", None)
    merged = {
        "profile": display_profile_name(profile_name),
        **DEFAULT_GATE,
    }
    merged.update({
        key: validation[key]
        for key in ("max_p95_latency_ms", "max_failure_rate", "max_drift_score")
        if key in validation
    })
    merged.update(active_gate)
    merged.update(quality_gate_profiles(validation).get(profile_name, {}))
    merged["profile"] = display_profile_name(profile_name)
    return merged


def evaluate_quality_gate(metrics: dict[str, float], gate: dict[str, Any]) -> dict[str, Any]:
    max_p95 = float(gate.get("max_p95_latency_ms", DEFAULT_GATE["max_p95_latency_ms"]))
    max_failure_rate = float(gate.get("max_failure_rate", DEFAULT_GATE["max_failure_rate"]))
    max_drift_score = float(gate.get("max_drift_score", DEFAULT_GATE["max_drift_score"]))
    failures: list[str] = []
    if metrics["p95_latency_ms"] > max_p95:
        failures.append("p95_latency_ms_exceeded")
    if metrics["failure_rate"] > max_failure_rate:
        failures.append("failure_rate_exceeded")
    if metrics["drift_score"] > max_drift_score:
        failures.append("drift_score_exceeded")
    if "max_model_size_mb" in gate and metrics.get("model_size_mb", 0.0) > float(
        gate["max_model_size_mb"]
    ):
        failures.append("model_size_mb_exceeded")

    result = {
        "profile": gate.get("profile", "cloud-sim"),
        "max_p95_latency_ms": max_p95,
        "max_failure_rate": max_failure_rate,
        "max_drift_score": max_drift_score,
        "passed": not failures,
        "reject_reason": failures[0] if failures else None,
    }
    if "max_model_size_mb" in gate:
        result["max_model_size_mb"] = float(gate["max_model_size_mb"])
    return result


def evaluate_all_quality_gates(
    metrics: dict[str, float],
    validation: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        display_profile_name(profile): evaluate_quality_gate(
            metrics,
            quality_gate_config(validation, profile),
        )
        for profile in quality_gate_profiles(validation)
    }
