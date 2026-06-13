from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any
from urllib.request import urlopen


def load_image(value: str, dataset_dir: Path) -> Any:
    from PIL import Image

    if value.startswith("placeholder://"):
        return Image.new("RGB", (224, 224), color=(245, 245, 245))

    if value.startswith(("http://", "https://")):
        with urlopen(value, timeout=30) as response:
            return Image.open(io.BytesIO(response.read())).convert("RGB")

    if value.startswith("data:image"):
        _, encoded = value.split(",", 1)
        return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")

    image_path = Path(value)
    if not image_path.is_absolute():
        image_path = dataset_dir / image_path
    if not image_path.exists():
        raise ValueError(f"Image file does not exist: {image_path}")
    return Image.open(image_path).convert("RGB")
