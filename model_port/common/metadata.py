from __future__ import annotations

from typing import Any


def model_metadata(cfg: Any) -> dict[str, str]:
    training = cfg.training
    return {
        "vendor": str(cfg.vendor),
        "model_name": str(training.get("model_name", "smart-captioner")),
        "version": str(training.get("version", "0.1.0")),
    }
