from __future__ import annotations

import os
from typing import Any


def wandb_project(data: dict[str, Any]) -> str:
    training = data.get("training", {})
    return os.getenv("WANDB_PROJECT", training.get("wandb_project", "model-port"))
