from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

IMAGE = Path("yolo_dataset/images/train/44000077_test.jpg")
LABEL = Path("yolo_dataset/labels/train/44000077_test.txt")
OUTPUT = Path("training/44000077_visual_check.jpg")

CLASS_NAMES = [
    "open",
    "short",
    "mousebite",
    "spur",
    "copper",
    "pin-hole",
]

image = Image.open(IMAGE).convert("RGB")
draw = ImageDraw.Draw(image)

W, H = image.size

print(f"Image size: {W} x {H}")

with open(LABEL, "r") as f:
    for line in f:
        cls, xc, yc, w, h = line.split()

        cls = int(cls)
        xc = float(xc)
        yc = float(yc)
        w = float(w)
        h = float(h)

        # YOLO normalized coordinates -> pixel coordinates
        x_center = xc * W
        y_center = yc * H
        box_width = w * W
        box_height = h * H

        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=3
        )

        label = CLASS_NAMES[cls]

        draw.text(
            (x1, max(0, y1 - 18)),
            label,
            fill="red"
        )

        print(
            f"{label:10s}: "
            f"({x1}, {y1}) -> ({x2}, {y2})"
        )

image.save(OUTPUT, quality=95)

print()
print(f"Saved: {OUTPUT}")
