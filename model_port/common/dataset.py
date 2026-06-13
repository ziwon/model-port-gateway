from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_caption_jsonl(path: Path, max_samples: int | None) -> list[dict[str, str]]:
    if not path.exists():
        raise ValueError(f"Dataset file does not exist: {path}")

    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            try:
                rows.append(normalize_caption_row(row))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no} has invalid dataset row: {exc}") from exc
            if max_samples and len(rows) >= max_samples:
                break

    if not rows:
        raise ValueError(f"Dataset file has no usable rows: {path}")
    return rows


def normalize_caption_row(row: dict[str, Any]) -> dict[str, str]:
    image = row.get("image_path", row.get("image"))
    prompt = row.get("prompt")
    answer = row.get("answer", row.get("response"))
    missing = [
        field
        for field, value in {
            "image_path": image,
            "prompt": prompt,
            "answer": answer,
        }.items()
        if value is None
    ]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    return {
        "image_path": str(image),
        "prompt": str(prompt),
        "answer": str(answer),
    }


def split_rows(
    rows: list[dict[str, str]],
    train_ratio: float,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if len(rows) == 1:
        return rows, []

    train_size = int(len(rows) * train_ratio)
    train_size = min(max(train_size, 1), len(rows) - 1)
    return rows[:train_size], rows[train_size:]
