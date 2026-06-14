from __future__ import annotations

import os


def wandb_enabled() -> bool:
    mode = os.getenv("WANDB_MODE", "").lower()
    if mode == "disabled":
        return False
    if mode == "offline":
        return True
    return len(os.getenv("WANDB_API_KEY", "")) >= 40


def wandb_skip_message() -> str:
    return "Skipping W&B logging: set WANDB_API_KEY, WANDB_MODE=offline, or WANDB_MODE=disabled"
