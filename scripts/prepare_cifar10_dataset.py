from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import typer

app = typer.Typer(help="Prepare a small CIFAR-10 JSONL image-classification dataset.")


def _require_deps() -> dict[str, Any]:
    try:
        from torchvision.datasets import CIFAR10
    except ImportError as exc:
        raise typer.BadParameter(
            "CIFAR-10 preparation requires torchvision. Install the train extra with "
            "`pip install -e '.[train,dev,api]'` or use the trainer container."
        ) from exc
    return {"CIFAR10": CIFAR10}


@app.command()
def main(
    output: Path = typer.Option(
        Path("data/cifar10_subset.jsonl"),
        "--output",
        "--out",
        help="JSONL file to write.",
    ),
    num_samples: int = typer.Option(
        500,
        "--num-samples",
        "--limit",
        min=10,
        help="Number of examples to export.",
    ),
    seed: int = typer.Option(42, "--seed", help="Deterministic sampling seed."),
    download: bool = typer.Option(True, "--download/--no-download", help="Download CIFAR-10 if missing."),
) -> None:
    deps = _require_deps()
    CIFAR10 = deps["CIFAR10"]
    root = output.parent / "public"
    image_root = output.parent / "cifar10_images"
    output.parent.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)

    dataset = CIFAR10(root=str(root), train=True, download=download)
    class_names = list(dataset.classes)
    indices_by_label: dict[int, list[int]] = {index: [] for index in range(len(class_names))}
    for index, label in enumerate(dataset.targets):
        indices_by_label[int(label)].append(index)

    rng = random.Random(seed)
    for indices in indices_by_label.values():
        rng.shuffle(indices)

    per_class = max(1, num_samples // len(class_names))
    selected: list[int] = []
    for label in range(len(class_names)):
        selected.extend(indices_by_label[label][:per_class])
    remaining = num_samples - len(selected)
    if remaining > 0:
        extras = [
            index
            for label in range(len(class_names))
            for index in indices_by_label[label][per_class:]
        ]
        rng.shuffle(extras)
        selected.extend(extras[:remaining])
    rng.shuffle(selected)

    with output.open("w", encoding="utf-8") as f:
        for row_index, dataset_index in enumerate(selected[:num_samples]):
            image, label_id = dataset[dataset_index]
            label = class_names[int(label_id)]
            relative_path = Path("cifar10_images") / label / f"sample_{row_index:05d}.png"
            image_path = output.parent / relative_path
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(image_path)
            row = {
                "image_path": relative_path.as_posix(),
                "label": label,
                "source_dataset": "cifar10",
                "source_index": int(dataset_index),
            }
            f.write(json.dumps(row) + "\n")

    print(f"wrote {min(num_samples, len(selected))} CIFAR-10 rows to {output}")


if __name__ == "__main__":
    app()
