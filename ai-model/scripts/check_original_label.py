from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("dataset/DeepPCB-master/PCBData")

sample = "44000077"

test_image = ROOT / "group44000/44000" / f"{sample}_test.jpg"
temp_image = ROOT / "group44000/44000" / f"{sample}_temp.jpg"
annotation = ROOT / "group44000/44000_not" / f"{sample}.txt"

CLASS_NAMES = {
    1: "open",
    2: "short",
    3: "mousebite",
    4: "spur",
    5: "copper",
    6: "pin-hole",
}

print("TEST :", test_image)
print("TEMP :", temp_image)
print("LABEL:", annotation)
print()

def draw_labels(image_path, output_path):

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    print(f"{image_path.name}: {image.size}")

    with open(annotation, "r") as f:
        for line in f:

            x1, y1, x2, y2, cls = map(int, line.split())

            name = CLASS_NAMES[cls]

            print(
                f"class={cls} ({name:9s}) "
                f"box=({x1},{y1}) -> ({x2},{y2})"
            )

            draw.rectangle(
                [x1, y1, x2, y2],
                outline="red",
                width=3
            )

            draw.text(
                (x1, max(0, y1 - 15)),
                f"{cls}: {name}",
                fill="red"
            )

    image.save(output_path)

    print("Saved:", output_path)
    print()


draw_labels(
    test_image,
    "training/44000077_TEST_labeled.jpg"
)

draw_labels(
    temp_image,
    "training/44000077_TEMP_labeled.jpg"
)

