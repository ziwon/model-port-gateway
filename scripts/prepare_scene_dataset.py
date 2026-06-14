from __future__ import annotations

import json
import random
from pathlib import Path

import typer
from PIL import Image, ImageDraw

app = typer.Typer(help="Create a synthetic scene-classification JSONL dataset.")

CLASSES = ["indoor", "outdoor", "desk", "road", "person"]


def _draw_indoor(draw: ImageDraw.ImageDraw) -> None:
    draw.rectangle((0, 0, 224, 224), fill=(230, 220, 205))
    draw.rectangle((26, 42, 96, 112), fill=(170, 205, 230), outline=(90, 110, 130), width=4)
    draw.rectangle((124, 130, 206, 184), fill=(140, 90, 55))
    draw.rectangle((134, 108, 196, 136), fill=(100, 70, 50))
    draw.ellipse((48, 150, 88, 190), fill=(95, 125, 95))


def _draw_outdoor(draw: ImageDraw.ImageDraw) -> None:
    draw.rectangle((0, 0, 224, 120), fill=(135, 196, 235))
    draw.rectangle((0, 120, 224, 224), fill=(70, 155, 80))
    draw.ellipse((26, 24, 74, 72), fill=(245, 210, 70))
    draw.rectangle((154, 78, 170, 162), fill=(95, 65, 40))
    draw.ellipse((124, 36, 200, 104), fill=(45, 130, 70))


def _draw_desk(draw: ImageDraw.ImageDraw) -> None:
    draw.rectangle((0, 0, 224, 224), fill=(218, 226, 232))
    draw.rectangle((0, 146, 224, 224), fill=(125, 82, 48))
    draw.rectangle((70, 72, 154, 128), fill=(42, 53, 70))
    draw.rectangle((80, 82, 144, 116), fill=(116, 170, 210))
    draw.rectangle((90, 130, 136, 144), fill=(55, 65, 80))
    draw.rectangle((28, 122, 62, 160), fill=(245, 245, 240), outline=(70, 60, 50), width=3)


def _draw_road(draw: ImageDraw.ImageDraw) -> None:
    draw.rectangle((0, 0, 224, 108), fill=(160, 205, 230))
    draw.rectangle((0, 108, 224, 224), fill=(70, 74, 78))
    draw.polygon([(92, 108), (132, 108), (188, 224), (36, 224)], fill=(62, 66, 70))
    for y in range(124, 214, 30):
        draw.rectangle((106, y, 118, y + 18), fill=(240, 230, 120))
    draw.rectangle((22, 82, 74, 118), fill=(70, 150, 80))
    draw.rectangle((150, 78, 204, 118), fill=(70, 150, 80))


def _draw_person(draw: ImageDraw.ImageDraw) -> None:
    draw.rectangle((0, 0, 224, 224), fill=(214, 224, 232))
    draw.ellipse((88, 36, 136, 84), fill=(232, 178, 130))
    draw.rectangle((78, 86, 146, 156), fill=(65, 105, 165))
    draw.line((84, 100, 42, 144), fill=(232, 178, 130), width=12)
    draw.line((140, 100, 182, 144), fill=(232, 178, 130), width=12)
    draw.line((94, 156, 76, 210), fill=(45, 65, 95), width=14)
    draw.line((130, 156, 148, 210), fill=(45, 65, 95), width=14)


DRAWERS = {
    "indoor": _draw_indoor,
    "outdoor": _draw_outdoor,
    "desk": _draw_desk,
    "road": _draw_road,
    "person": _draw_person,
}


def _write_image(path: Path, label: str, index: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(index)
    base = tuple(rng.randint(0, 10) for _ in range(3))
    image = Image.new("RGB", (224, 224), color=(240 - base[0], 240 - base[1], 240 - base[2]))
    draw = ImageDraw.Draw(image)
    DRAWERS[label](draw)
    image.save(path, format="JPEG", quality=90)


@app.command()
def main(
    output: Path = typer.Option(
        Path("data/scene_classification.jsonl"),
        "--output",
        "--out",
        help="JSONL file to write.",
    ),
    num_samples: int = typer.Option(
        200,
        "--num-samples",
        "--limit",
        min=5,
        help="Number of sample rows and images to create.",
    ),
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for index in range(num_samples):
            label = CLASSES[index % len(CLASSES)]
            relative_image_path = Path("scene_images") / label / f"sample_{index:04d}.jpg"
            _write_image(output.parent / relative_image_path, label, index)
            row = {
                "image_path": relative_image_path.as_posix(),
                "label": label,
            }
            f.write(json.dumps(row) + "\n")

    print(f"wrote {num_samples} scene rows to {output}")


if __name__ == "__main__":
    app()
