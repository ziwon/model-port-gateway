from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import typer
from rich import print

from model_port.common.config import dump_yaml, load_train_config, load_yaml
from model_port.common.images import load_image
from model_port.common.tracking import wandb_enabled

app = typer.Typer(help="Train a small edge-friendly image classifier.")


def _require_train_deps() -> dict[str, Any]:
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from torchvision import models, transforms
    except ImportError as exc:
        raise typer.BadParameter(
            "Classifier training dependencies are missing. Install/rebuild the train extra with "
            "`pip install -e '.[train,dev,api]'` or `docker compose build trainer`."
        ) from exc

    return {
        "torch": torch,
        "DataLoader": DataLoader,
        "Dataset": Dataset,
        "models": models,
        "transforms": transforms,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "image_path" not in row or "label" not in row:
                raise typer.BadParameter(
                    f"{path}:{line_number} must include image_path and label fields."
                )
            rows.append(row)
    if not rows:
        raise typer.BadParameter(f"No classifier rows found in {path}.")
    return rows


def _classes(cfg: Any, rows: list[dict[str, Any]]) -> list[str]:
    configured = cfg.dataset.get("classes")
    classes = list(configured) if configured else sorted({str(row["label"]) for row in rows})
    missing = sorted({str(row["label"]) for row in rows} - set(classes))
    if missing:
        raise typer.BadParameter(f"Dataset includes labels missing from config classes: {missing}")
    return classes


def _transforms(deps: dict[str, Any], image_size: int) -> Any:
    transforms = deps["transforms"]
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _dataset_class(deps: dict[str, Any]) -> type[Any]:
    Dataset = deps["Dataset"]

    class SceneDataset(Dataset):
        def __init__(
            self,
            rows: list[dict[str, Any]],
            dataset_dir: Path,
            label_to_id: dict[str, int],
            transform: Any,
        ) -> None:
            self.rows = rows
            self.dataset_dir = dataset_dir
            self.label_to_id = label_to_id
            self.transform = transform

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> tuple[Any, int]:
            row = self.rows[index]
            image = load_image(row["image_path"], self.dataset_dir)
            return self.transform(image), self.label_to_id[str(row["label"])]

    return SceneDataset


def _make_model(
    deps: dict[str, Any],
    num_classes: int,
    pretrained: bool = False,
    freeze_backbone: bool = False,
) -> Any:
    models = deps["models"]
    torch = deps.get("torch")
    weights = None
    if pretrained:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
    if pretrained or torch is None:
        model = models.mobilenet_v3_small(weights=weights)
    else:
        model = models.mobilenet_v3_small(
            weights=weights,
            norm_layer=lambda channels: torch.nn.GroupNorm(1, channels),
        )
    in_features = model.classifier[-1].in_features
    if torch is None:
        model.classifier[-1] = model.classifier[-1].__class__(in_features, num_classes)
    else:
        model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    if freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    return model


def _split_rows(rows: list[dict[str, Any]], train_ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(rows) < 2:
        return rows, rows
    split_at = min(max(1, int(len(rows) * train_ratio)), len(rows) - 1)
    return rows[:split_at], rows[split_at:]


@app.command()
def main(
    config: Path = typer.Option(..., "--config", help="Classifier train config YAML."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config and dataset only."),
) -> None:
    cfg = load_train_config(config)
    dataset_path = Path(cfg.dataset.local_jsonl or "")
    if not dataset_path.exists():
        raise typer.BadParameter(f"Dataset does not exist: {dataset_path}")

    rows = _read_jsonl(dataset_path)
    if cfg.dataset.max_samples:
        rows = rows[: int(cfg.dataset.max_samples)]
    classes = _classes(cfg, rows)
    train_rows, eval_rows = _split_rows(rows, float(cfg.dataset.train_ratio))
    metadata = {
        "vendor": cfg.vendor,
        "model_name": cfg.training.model_name,
        "version": cfg.training.version,
        "dataset": str(dataset_path),
        "num_rows": len(rows),
        "num_train_rows": len(train_rows),
        "num_eval_rows": len(eval_rows),
        "classes": classes,
    }
    print(f"[model-port] classifier={cfg.training.model_name}")
    print(f"[model-port] version={cfg.training.version}")
    print(f"[model-port] dataset={dataset_path}")
    print(f"[model-port] classes={','.join(classes)}")
    print(f"[model-port] output_dir={cfg.training.output_dir}")
    if dry_run:
        return

    deps = _require_train_deps()
    torch = deps["torch"]
    DataLoader = deps["DataLoader"]
    dataset_dir = Path(cfg.dataset.get("root_dir", dataset_path.parent))
    image_size = int(cfg.dataset.get("image_size", 224))
    batch_size = int(cfg.training.get("per_device_train_batch_size", 16))
    epochs = int(cfg.training.get("num_train_epochs", 3))
    label_to_id = {label: index for index, label in enumerate(classes)}
    SceneDataset = _dataset_class(deps)
    train_dataset = SceneDataset(train_rows, dataset_dir, label_to_id, _transforms(deps, image_size))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _make_model(
        deps,
        len(classes),
        pretrained=bool(cfg.training.get("pretrained", False)),
        freeze_backbone=bool(cfg.training.get("freeze_backbone", False)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(cfg.training.learning_rate),
    )
    criterion = torch.nn.CrossEntropyLoss()

    wandb_run = None
    if wandb_enabled():
        try:
            import wandb

            wandb_run = wandb.init(
                project=os.getenv("WANDB_PROJECT", cfg.wandb.project),
                name=f"{cfg.vendor}-{cfg.training.model_name}-{cfg.training.version}-train",
                job_type="train-classifier",
                config={**metadata, "base_model": cfg.base_model, "runtime": "torchvision"},
            )
        except ImportError:
            wandb_run = None

    global_step = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        seen = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            global_step += 1
            batch_size_seen = int(labels.numel())
            total_loss += float(loss.item()) * batch_size_seen
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            seen += batch_size_seen
            if wandb_run is not None:
                wandb_run.log({
                    "train/loss": float(loss.item()),
                    "train/epoch": epoch + 1,
                    "train/learning_rate": float(cfg.training.learning_rate),
                }, step=global_step)

        epoch_loss = total_loss / max(seen, 1)
        epoch_accuracy = correct / max(seen, 1)
        print(f"[model-port] epoch={epoch + 1} loss={epoch_loss:.4f} accuracy={epoch_accuracy:.4f}")
        if wandb_run is not None:
            wandb_run.log({
                "train/epoch_loss": epoch_loss,
                "train/accuracy": epoch_accuracy,
            }, step=global_step)

    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")
    (output_dir / "classes.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")
    dump_yaml(load_yaml(config), output_dir / "train_config.yaml")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if wandb_run is not None:
        wandb_run.finish()
    print(f"[model-port] saved classifier artifact to {output_dir}")


if __name__ == "__main__":
    app()
