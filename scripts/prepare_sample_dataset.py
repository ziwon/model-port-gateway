from __future__ import annotations

import json
from pathlib import Path

import typer
from PIL import Image, ImageDraw

app = typer.Typer(help="Create a tiny local image-caption JSONL sample for dry-runs.")

SAMPLE_CAPTIONS = [
    "A person is sitting at a desk with a laptop.",
    "A dog is running on the grass.",
    "A cyclist is waiting near a crosswalk.",
    "A bowl of fruit is placed on a kitchen table.",
    "A train is stopped beside a platform.",
    "A person is holding a coffee cup near a window.",
    "A backpack is resting on a wooden chair.",
    "A street sign is visible above parked cars.",
]


def _draw_person_at_laptop(draw: ImageDraw.ImageDraw) -> None:
    draw.rectangle((0, 168, 320, 240), fill=(126, 86, 55))
    draw.rectangle((142, 118, 226, 166), fill=(45, 55, 74))
    draw.rectangle((152, 126, 216, 156), fill=(120, 170, 210))
    draw.ellipse((72, 62, 112, 102), fill=(235, 181, 135))
    draw.line((92, 102, 92, 158), fill=(35, 70, 110), width=16)
    draw.line((90, 122, 136, 154), fill=(235, 181, 135), width=8)


def _draw_dog_on_grass(draw: ImageDraw.ImageDraw) -> None:
    draw.rectangle((0, 138, 320, 240), fill=(72, 150, 75))
    draw.ellipse((92, 108, 188, 158), fill=(150, 92, 45))
    draw.ellipse((174, 84, 224, 128), fill=(150, 92, 45))
    draw.polygon([(198, 84), (220, 58), (222, 96)], fill=(95, 55, 30))
    for x in (112, 144, 176, 204):
        draw.line((x, 150, x - 10, 190), fill=(85, 50, 30), width=8)


def _draw_cyclist_crosswalk(draw: ImageDraw.ImageDraw) -> None:
    draw.rectangle((0, 142, 320, 240), fill=(74, 78, 82))
    for x in range(20, 300, 44):
        draw.rectangle((x, 176, x + 28, 190), fill=(238, 238, 220))
    draw.ellipse((84, 150, 136, 202), outline=(20, 20, 20), width=5)
    draw.ellipse((190, 150, 242, 202), outline=(20, 20, 20), width=5)
    draw.line((110, 176, 164, 116, 214, 176, 110, 176), fill=(25, 90, 140), width=5)
    draw.ellipse((150, 70, 178, 98), fill=(235, 181, 135))
    draw.line((164, 98, 164, 138), fill=(230, 70, 60), width=10)


def _draw_fruit_bowl(draw: ImageDraw.ImageDraw) -> None:
    draw.rectangle((0, 150, 320, 240), fill=(146, 93, 50))
    draw.pieslice((92, 104, 228, 210), start=0, end=180, fill=(230, 230, 235))
    colors = [(220, 40, 40), (245, 175, 50), (80, 150, 70), (130, 80, 180)]
    for offset, color in zip((0, 32, 64, 96), colors, strict=False):
        draw.ellipse((86 + offset, 92, 124 + offset, 130), fill=color)


def _draw_train_platform(draw: ImageDraw.ImageDraw) -> None:
    draw.rectangle((0, 152, 320, 240), fill=(120, 120, 120))
    draw.rectangle((30, 80, 258, 148), fill=(45, 95, 165))
    for x in range(52, 220, 48):
        draw.rectangle((x, 94, x + 30, 126), fill=(180, 215, 235))
    draw.rectangle((258, 92, 300, 148), fill=(35, 75, 125))
    draw.line((0, 184, 320, 184), fill=(230, 210, 80), width=5)
    draw.line((0, 212, 320, 212), fill=(80, 80, 80), width=6)


def _draw_coffee_window(draw: ImageDraw.ImageDraw) -> None:
    draw.rectangle((32, 28, 156, 132), fill=(170, 210, 235), outline=(80, 100, 120), width=4)
    draw.line((94, 28, 94, 132), fill=(80, 100, 120), width=3)
    draw.ellipse((212, 62, 252, 102), fill=(235, 181, 135))
    draw.rectangle((198, 110, 266, 174), fill=(80, 120, 160))
    draw.rectangle((134, 140, 176, 188), fill=(245, 245, 240), outline=(80, 60, 50), width=3)
    draw.arc((166, 150, 198, 178), start=-80, end=80, fill=(80, 60, 50), width=4)


def _draw_backpack_chair(draw: ImageDraw.ImageDraw) -> None:
    draw.rectangle((118, 90, 212, 150), fill=(130, 84, 44))
    draw.line((128, 150, 104, 220), fill=(95, 62, 34), width=7)
    draw.line((202, 150, 226, 220), fill=(95, 62, 34), width=7)
    draw.rounded_rectangle((124, 76, 200, 170), radius=18, fill=(45, 105, 90))
    draw.line((146, 82, 132, 158), fill=(25, 65, 55), width=5)
    draw.rectangle((150, 126, 190, 154), fill=(35, 80, 70))


def _draw_street_sign_cars(draw: ImageDraw.ImageDraw) -> None:
    draw.rectangle((0, 146, 320, 240), fill=(75, 80, 84))
    draw.rectangle((40, 122, 130, 158), fill=(185, 50, 55))
    draw.rectangle((180, 116, 280, 158), fill=(50, 105, 180))
    for x in (58, 112, 202, 258):
        draw.ellipse((x, 148, x + 22, 170), fill=(20, 20, 20))
    draw.line((160, 56, 160, 148), fill=(80, 80, 80), width=6)
    draw.rectangle((110, 46, 210, 78), fill=(45, 135, 80), outline=(245, 245, 245), width=3)


DRAW_SCENES = [
    _draw_person_at_laptop,
    _draw_dog_on_grass,
    _draw_cyclist_crosswalk,
    _draw_fruit_bowl,
    _draw_train_platform,
    _draw_coffee_window,
    _draw_backpack_chair,
    _draw_street_sign_cars,
]


def _write_sample_image(path: Path, index: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (320, 240), color=(235, 238, 232))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 320, 92), fill=(205, 225, 238))
    DRAW_SCENES[(index - 1) % len(DRAW_SCENES)](draw)
    image.save(path, format="JPEG", quality=90)


@app.command()
def main(
    output: Path = typer.Option(
        Path("data/sample_captions.jsonl"),
        "--output",
        "--out",
        help="JSONL file to write.",
    ),
    num_samples: int = typer.Option(
        64,
        "--num-samples",
        "--limit",
        min=1,
        help="Number of sample rows and images to create.",
    ),
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as f:
        for index in range(1, num_samples + 1):
            # Store paths relative to the JSONL location; the training collator
            # resolves them against the dataset file's directory.
            relative_image_path = Path("images") / f"sample_{index:03d}.jpg"
            _write_sample_image(output.parent / relative_image_path, index)
            row = {
                "image_path": relative_image_path.as_posix(),
                "prompt": "Describe this image.",
                "answer": SAMPLE_CAPTIONS[(index - 1) % len(SAMPLE_CAPTIONS)],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {num_samples} rows to {output}")


if __name__ == "__main__":
    app()
