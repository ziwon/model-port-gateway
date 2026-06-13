from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class FlexibleModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class DatasetConfig(FlexibleModel):
    name: str | None = None
    split: str = "train"
    local_jsonl: str | None = None
    max_samples: int | None = None
    train_ratio: float = 0.9


class LoraConfig(FlexibleModel):
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: str | list[str] = "auto"


class TrainingConfig(FlexibleModel):
    method: str = "lora"
    model_name: str = "smart-captioner"
    version: str = "0.1.0"
    output_dir: str = "artifacts/models/smolvlm2-lora"
    num_train_epochs: float = 1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    max_seq_length: int = 1024
    gradient_checkpointing: bool = False
    fp16: bool = False
    bf16: bool = False
    load_in_4bit: bool = False
    lora: LoraConfig = Field(default_factory=LoraConfig)


class InferenceConfig(FlexibleModel):
    max_new_tokens: int = 64
    do_sample: bool = False
    num_beams: int = 1
    temperature: float | None = None


class WandbConfig(FlexibleModel):
    project: str = "model-port"
    job_type: str = "finetune"
    tags: list[str] = Field(default_factory=list)


class ValidationConfig(FlexibleModel):
    active_profile: str | None = None
    min_exact_caption_score: float = 0.0
    quality_gate: dict[str, Any] = Field(default_factory=dict)
    quality_gates: dict[str, dict[str, Any]] = Field(default_factory=dict)
    quality_gate_profiles: dict[str, dict[str, Any]] = Field(default_factory=dict)


class TrainConfig(FlexibleModel):
    project: str = "model-port"
    vendor: str = "vendor-demo"
    base_model: str
    dataset: DatasetConfig
    training: TrainingConfig
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(data: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def load_train_config(path: str | Path) -> TrainConfig:
    return TrainConfig.model_validate(load_yaml(path))
