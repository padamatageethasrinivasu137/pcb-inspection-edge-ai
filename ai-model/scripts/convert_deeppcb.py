from pathlib import Path
import random
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path("dataset/DeepPCB-master/PCBData")

TRAINVAL_FILE = ROOT / "trainval.txt"
TEST_FILE = ROOT / "test.txt"

OUTPUT_ROOT = Path("yolo_dataset")

VAL_RATIO = 0.10
RANDOM_SEED = 42


# DeepPCB class -> YOLO class
CLASS_NAMES = {
    1: "open",
    2: "short",
    3: "mousebite",
    4: "spur",
    5: "copper",
    6: "pin-hole",
}

CLASS_MAP = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
}


# ============================================================
# READ OFFICIAL LIST
# ============================================================

def read_list(list_file):

    entries = []

    with open(list_file, "r") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 2:
                raise ValueError(
                    f"Unexpected line in {list_file}:\n{line}"
                )

            image_rel = parts[0]
            annotation_rel = parts[1]

            entries.append(
                (image_rel, annotation_rel)
            )

    return entries


# ============================================================
# CONVERT ONE ANNOTATION
# ============================================================

def convert_annotation(annotation_path, image_width, image_height):

    yolo_lines = []

    with open(annotation_path, "r") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                raise ValueError(
                    f"Invalid annotation line in {annotation_path}:\n"
                    f"{line}"
                )

            x1, y1, x2, y2, deep_class = map(
                int,
                parts
            )

            if deep_class not in CLASS_MAP:
                raise ValueError(
                    f"Unknown DeepPCB class {deep_class} "
                    f"in {annotation_path}"
                )

            # DeepPCB class -> YOLO class
            yolo_class = CLASS_MAP[deep_class]

            # Convert:
            #
            # x1 y1 x2 y2
            #
            # to:
            #
            # x_center y_center width height

            x_center = (x1 + x2) / 2.0
            y_center = (y1 + y2) / 2.0

            width = x2 - x1
            height = y2 - y1

            # Normalize to 0-1
            x_center /= image_width
            y_center /= image_height
            width /= image_width
            height /= image_height

            yolo_lines.append(
                f"{yolo_class} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{width:.6f} "
                f"{height:.6f}"
            )

    return yolo_lines


# ============================================================
# PROCESS ONE DATASET SPLIT
# ============================================================

def process(entries, split):

    image_output_dir = OUTPUT_ROOT / "images" / split
    label_output_dir = OUTPUT_ROOT / "labels" / split

    image_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    label_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    successful = 0
    failed = 0

    print()
    print("----------------------------------------")
    print(f"Processing {split}")
    print("----------------------------------------")

    for index, (image_rel, annotation_rel) in enumerate(entries, 1):

        try:

            # ------------------------------------------------
            # Original image path from trainval.txt/test.txt
            #
            # Example:
            #
            # group44000/44000/44000077.jpg
            #
            # Actual DeepPCB image:
            #
            # group44000/44000/44000077_test.jpg
            # ------------------------------------------------

            image_base = ROOT / image_rel

            image_path = image_base.with_name(
                image_base.stem + "_test.jpg"
            )

            # ------------------------------------------------
            # Annotation path is already provided by DeepPCB
            # ------------------------------------------------

            annotation_path = ROOT / annotation_rel

            # Check files exist
            if not image_path.exists():

                raise FileNotFoundError(
                    f"Image not found:\n{image_path}"
                )

            if not annotation_path.exists():

                raise FileNotFoundError(
                    f"Annotation not found:\n{annotation_path}"
                )

            # ------------------------------------------------
            # Read image dimensions
            # ------------------------------------------------

            with Image.open(image_path) as image:

                image_width, image_height = image.size

            # ------------------------------------------------
            # Convert annotation
            # ------------------------------------------------

            yolo_lines = convert_annotation(
                annotation_path,
                image_width,
                image_height
            )

            # ------------------------------------------------
            # Output filenames
            # ------------------------------------------------

            output_image_name = (
                image_base.stem + "_test.jpg"
            )

            output_label_name = (
                image_base.stem + "_test.txt"
            )

            output_image = (
                image_output_dir /
                output_image_name
            )

            output_label = (
                label_output_dir /
                output_label_name
            )

            # ------------------------------------------------
            # Copy image
            # ------------------------------------------------

            image_data = image_path.read_bytes()

            output_image.write_bytes(image_data)

            # ------------------------------------------------
            # Write YOLO label
            # ------------------------------------------------

            with open(output_label, "w") as f:

                f.write(
                    "\n".join(yolo_lines)
                )

                if yolo_lines:
                    f.write("\n")

            successful += 1

            # Show progress every 100 images
            if index % 100 == 0:

                print(
                    f"{index}/{len(entries)} processed"
                )

        except Exception as e:

            failed += 1

            print(
                f"ERROR processing {image_rel}: {e}"
            )

    print()
    print(
        f"{split}: successfully processed "
        f"{successful}/{len(entries)}"
    )

    if failed:

        print(
            f"{split}: FAILED = {failed}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("========================================")
    print(" DeepPCB -> YOLO conversion")
    print("========================================")
    print()

    print(f"DeepPCB root: {ROOT}")
    print()

    # --------------------------------------------------------
    # Read official DeepPCB lists
    # --------------------------------------------------------

    trainval = read_list(TRAINVAL_FILE)
    test = read_list(TEST_FILE)

    print(
        f"Official trainval entries: {len(trainval)}"
    )

    print(
        f"Official test entries:     {len(test)}"
    )

    print()

    # --------------------------------------------------------
    # Split trainval into train + validation
    # --------------------------------------------------------

    random.seed(RANDOM_SEED)

    random.shuffle(trainval)

    val_count = int(
        len(trainval) * VAL_RATIO
    )

    val_entries = trainval[:val_count]
    train_entries = trainval[val_count:]

    print(
        f"Training entries:          {len(train_entries)}"
    )

    print(
        f"Validation entries:        {len(val_entries)}"
    )

    print(
        f"Test entries:              {len(test)}"
    )

    print()

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    process(
        train_entries,
        "train"
    )

    process(
        val_entries,
        "val"
    )

    process(
        test,
        "test"
    )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print()
    print("========================================")
    print(" Conversion complete")
    print("========================================")


if __name__ == "__main__":
    main()
