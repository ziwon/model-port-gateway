from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Any

import typer
from rich import print

from model_port.common.config import load_train_config
from model_port.common.dataset import read_caption_jsonl, split_rows
from model_port.common.images import load_image
from model_port.common.metadata import model_metadata

app = typer.Typer(help="Fine-tune a VLM with W&B tracking. Dry-run by default for scaffold validation.")

DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _require_training_deps() -> dict[str, Any]:
    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForMultimodalLM,
            AutoProcessor,
            BitsAndBytesConfig,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )
    except ImportError as exc:
        raise typer.BadParameter(
            "Training dependencies are missing. Install them with "
            "`pip install -e '.[train,dev,api]'` before running without --dry-run."
        ) from exc

    return {
        "torch": torch,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoModelForMultimodalLM": AutoModelForMultimodalLM,
        "AutoProcessor": AutoProcessor,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "Trainer": Trainer,
        "TrainerCallback": TrainerCallback,
        "TrainingArguments": TrainingArguments,
    }


class SmolVLMDataCollator:
    def __init__(self, processor: Any, dataset_dir: Path, max_seq_length: int) -> None:
        self.processor = processor
        self.dataset_dir = dataset_dir
        self.max_seq_length = max_seq_length

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        images = [load_image(str(example["image_path"]), self.dataset_dir) for example in examples]
        texts = [self._format_example(example) for example in examples]

        # Do not truncate: the processor expands each image into a fixed block
        # of image tokens, and truncating would desync those tokens from the
        # image features and raise a token-count mismatch.
        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        labels = batch["input_ids"].clone()

        pad_token_id = self.processor.tokenizer.pad_token_id
        labels[labels == pad_token_id] = -100

        image_token_id = getattr(self.processor, "image_token_id", None)
        if image_token_id is None:
            image_token = getattr(self.processor, "image_token", None) or "<image>"
            image_token_id = self.processor.tokenizer.convert_tokens_to_ids(image_token)
        if isinstance(image_token_id, int) and image_token_id >= 0:
            labels[labels == image_token_id] = -100

        batch["labels"] = labels
        return batch

    def _format_example(self, example: dict[str, Any]) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": str(example["prompt"])},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": str(example["answer"])}],
            },
        ]
        return self.processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=False,
        )


def _training_args_kwargs(training_args_cls: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    params = inspect.signature(training_args_cls.__init__).parameters
    if "eval_strategy" in params and "evaluation_strategy" in kwargs:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    elif "evaluation_strategy" in params and "eval_strategy" in kwargs:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    return {key: value for key, value in kwargs.items() if key in params}


def _wandb_config(cfg: Any) -> dict[str, Any]:
    training = cfg.training
    dataset_cfg = cfg.dataset
    lora_cfg = training.get("lora", {})
    metadata = model_metadata(cfg)
    return {
        "vendor": metadata["vendor"],
        "base_model": cfg.base_model,
        "dataset": dataset_cfg.get("local_jsonl", dataset_cfg.get("name")),
        "method": "lora",
        "lora_r": int(lora_cfg.get("r", 8)),
        "runtime": "transformers",
    }


def _print_contract(cfg: Any) -> None:
    dataset = cfg.dataset.get("local_jsonl", cfg.dataset.get("name"))
    print(f"\\[model-port] vendor={cfg.vendor}")
    print(f"\\[model-port] base_model={cfg.base_model}")
    print(f"\\[model-port] dataset={dataset}")
    print(f"\\[model-port] training_method={cfg.training.get('method')}")
    print(f"\\[model-port] wandb_project={cfg.wandb.get('project', 'model-port')}")
    print(f"\\[model-port] output_dir={cfg.training.get('output_dir')}")


def _start_wandb_run(cfg: Any, job_type: str) -> Any:
    import wandb

    api_key = os.getenv("WANDB_API_KEY", "")
    mode = os.getenv("WANDB_MODE", "online")
    if mode != "offline" and len(api_key) < 40:
        raise typer.BadParameter(
            "WANDB_API_KEY is missing or too short. For local W&B, open "
            "http://localhost:8081, create/sign in to the local account, copy that "
            "local API key into .env, then recreate trainer. Use a cloud W&B key only "
            "when WANDB_BASE_URL points to https://api.wandb.ai. For no server logging, "
            "set WANDB_MODE=offline."
        )

    metadata = model_metadata(cfg)
    return wandb.init(
        project=os.getenv("WANDB_PROJECT", cfg.wandb.get("project", "model-port")),
        name=f"{metadata['vendor']}-{metadata['model_name']}-{metadata['version']}",
        job_type=job_type,
        tags=cfg.wandb.get("tags", []),
        config=_wandb_config(cfg),
    )


def _finish_wandb_run(run: Any | None) -> None:
    if run is not None:
        run.finish()


def _build_wandb_metrics_callback(trainer_callback_cls: Any, torch: Any) -> Any:
    class WandbMetricsCallback(trainer_callback_cls):
        def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
            if not logs:
                return

            try:
                import wandb
            except ImportError:
                return
            if wandb.run is None:
                return

            payload: dict[str, Any] = {}
            if "loss" in logs:
                payload["train/loss"] = logs["loss"]
            if "learning_rate" in logs:
                payload["train/learning_rate"] = logs["learning_rate"]
            if "epoch" in logs:
                payload["train/epoch"] = logs["epoch"]
            if torch.cuda.is_available():
                payload["system/gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)

            if payload:
                wandb.log(payload, step=state.global_step)

    return WandbMetricsCallback()


def _run_lora_training(cfg: Any, config_path: Path) -> None:
    deps = _require_training_deps()
    torch = deps["torch"]
    AutoModelForMultimodalLM = deps["AutoModelForMultimodalLM"]
    AutoProcessor = deps["AutoProcessor"]
    BitsAndBytesConfig = deps["BitsAndBytesConfig"]
    LoraConfig = deps["LoraConfig"]
    Trainer = deps["Trainer"]
    TrainerCallback = deps["TrainerCallback"]
    TrainingArguments = deps["TrainingArguments"]
    get_peft_model = deps["get_peft_model"]
    prepare_model_for_kbit_training = deps["prepare_model_for_kbit_training"]

    training = cfg.training
    dataset_cfg = cfg.dataset
    output_dir = Path(training.get("output_dir", "artifacts/models/smolvlm2-lora"))
    local_jsonl = Path(dataset_cfg.get("local_jsonl", ""))
    if not local_jsonl:
        raise typer.BadParameter("dataset.local_jsonl is required for local SmolVLM2 training")

    try:
        rows = read_caption_jsonl(local_jsonl, dataset_cfg.get("max_samples"))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    train_rows, eval_rows = split_rows(rows, float(dataset_cfg.get("train_ratio", 0.9)))

    load_in_4bit = bool(training.get("load_in_4bit", False))
    compute_dtype = torch.bfloat16 if training.get("bf16", False) else torch.float16
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    print("[cyan]Loading processor and model[/cyan]")
    try:
        processor = AutoProcessor.from_pretrained(cfg.base_model)
    except (ImportError, ValueError) as exc:
        raise typer.BadParameter(
            "Failed to load SmolVLM processor. Make sure the trainer image was rebuilt "
            "after installing the train extra, including torchvision and num2words: "
            "`docker compose build trainer && docker compose up -d --force-recreate trainer`."
        ) from exc

    model = AutoModelForMultimodalLM.from_pretrained(
        cfg.base_model,
        device_map="auto",
        torch_dtype=compute_dtype,
        quantization_config=quantization_config,
    )

    if training.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = training.get("lora", {})
    target_modules = lora_cfg.get("target_modules", "auto")
    if target_modules == "auto":
        target_modules = DEFAULT_LORA_TARGET_MODULES
    elif isinstance(target_modules, str):
        target_modules = [module.strip() for module in target_modules.split(",") if module.strip()]

    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 8)),
        lora_alpha=int(lora_cfg.get("alpha", 16)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    collator = SmolVLMDataCollator(
        processor=processor,
        dataset_dir=local_jsonl.parent,
        max_seq_length=int(training.get("max_seq_length", 1024)),
    )
    report_to = ["wandb"] if cfg.wandb.get("project") else []
    args_kwargs = _training_args_kwargs(
        TrainingArguments,
        {
            "output_dir": str(output_dir),
            "num_train_epochs": float(training.get("num_train_epochs", 1)),
            "per_device_train_batch_size": int(training.get("per_device_train_batch_size", 1)),
            "per_device_eval_batch_size": int(training.get("per_device_eval_batch_size", 1)),
            "gradient_accumulation_steps": int(training.get("gradient_accumulation_steps", 1)),
            "learning_rate": float(training.get("learning_rate", 2e-5)),
            "logging_steps": int(training.get("logging_steps", 10)),
            "save_steps": int(training.get("save_steps", 100)),
            "save_total_limit": int(training.get("save_total_limit", 2)),
            "bf16": bool(training.get("bf16", False)),
            "fp16": bool(training.get("fp16", False)),
            "remove_unused_columns": False,
            "report_to": report_to,
            "run_name": f"{cfg.project}-{cfg.vendor}-smolvlm2-lora",
            "evaluation_strategy": "steps" if eval_rows else "no",
            "eval_steps": int(training.get("eval_steps", 100)),
            "dataloader_num_workers": int(training.get("dataloader_num_workers", 0)),
        },
    )

    print("[cyan]Starting LoRA training[/cyan]")
    wandb_run = _start_wandb_run(cfg, job_type=cfg.wandb.get("job_type", "finetune"))
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**args_kwargs),
        train_dataset=train_rows,
        eval_dataset=eval_rows or None,
        data_collator=collator,
        callbacks=[_build_wandb_metrics_callback(TrainerCallback, torch)],
    )
    try:
        trainer.train()

        output_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(output_dir))
        processor.save_pretrained(output_dir)
        (output_dir / "train_config.yaml").write_text(
            config_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        print(f"[green]Saved LoRA adapter and processor to {output_dir}[/green]")
    finally:
        _finish_wandb_run(wandb_run)


@app.command()
def main(
    config: Path = typer.Option(..., "--config", "-c", help="Training config YAML."),
    dry_run: bool = False,
    log_dry_run: bool = typer.Option(
        False,
        "--log-dry-run",
        help="Create a lightweight W&B run for the dry-run contract check.",
    ),
) -> None:
    cfg = load_train_config(config)
    _print_contract(cfg)

    if dry_run:
        if log_dry_run:
            run = _start_wandb_run(cfg, job_type="finetune-dry-run")
            try:
                import wandb

                wandb.log({"pipeline/dry_run": 1})
            finally:
                _finish_wandb_run(run)
        print("\\[model-port] dry_run=true")
        return

    method = str(cfg.training.get("method", "lora")).lower()
    if method != "lora":
        raise typer.BadParameter(f"Unsupported training.method={method!r}; only 'lora' is implemented")

    _run_lora_training(cfg, config)


if __name__ == "__main__":
    app()
