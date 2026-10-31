import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]

# Support the Pi deployment, repository deployment and local model directory.
MODEL_CANDIDATES = [
    BASE_DIR / "best.onnx",
    BASE_DIR.parent / "app" / "best.onnx",
    REPO_ROOT / "ai-model" / "weights" / "best.onnx",
]

MODEL_PATH = next(
    (path for path in MODEL_CANDIDATES if path.is_file()),
    MODEL_CANDIDATES[0],
)

CLASS_NAMES = [
    "open",
    "short",
    "mousebite",
    "spur",
    "copper",
    "pin-hole",
]

IMG_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run standalone YOLO11n ONNX inference."
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the PCB test image.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output image path. Defaults to <image>_result.jpg.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    image_path = Path(args.image).expanduser().resolve()

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = image_path.with_name(
            f"{image_path.stem}_result.jpg"
        )

    # --------------------------------------------------
    # Validate files
    # --------------------------------------------------

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"ONNX model not found: {MODEL_PATH}"
        )

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Input image not found: {image_path}"
        )

    # --------------------------------------------------
    # Load ONNX model
    # --------------------------------------------------

    print("Loading model...")
    print("Model:", MODEL_PATH)

    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )

    input_info = session.get_inputs()[0]
    input_name = input_info.name

    print("Model loaded")
    print("Input name:", input_name)
    print("Input shape:", input_info.shape)
    print("Execution provider:", session.get_providers()[0])

    # --------------------------------------------------
    # Read image
    # --------------------------------------------------

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Could not decode image: {image_path}"
        )

    original_h, original_w = image.shape[:2]

    print(f"Image size: {original_w} x {original_h}")

    # --------------------------------------------------
    # Preprocess
    # --------------------------------------------------
    # Diagnostic pipeline: direct resize, not letterbox.

    resized = cv2.resize(
        image,
        (IMG_SIZE, IMG_SIZE),
    )

    rgb = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB,
    )

    input_image = rgb.transpose(2, 0, 1)
    input_image = input_image.astype(np.float32) / 255.0
    input_image = np.expand_dims(input_image, axis=0)

    # --------------------------------------------------
    # Run inference
    # --------------------------------------------------

    print("Running inference...")

    outputs = session.run(
        None,
        {input_name: input_image},
    )

    output = outputs[0]

    print("Output shape:", output.shape)

    # --------------------------------------------------
    # Decode YOLO11 output
    # --------------------------------------------------

    predictions = output[0].T

    boxes = []
    scores = []
    class_ids = []

    for prediction in predictions:
        x_center, y_center, width, height = prediction[:4]

        class_scores = prediction[4:]
        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])

        if not np.isfinite(confidence):
            continue

        if confidence < CONF_THRESHOLD:
            continue

        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2

        # Scale coordinates back to original image dimensions.
        x1 *= original_w / IMG_SIZE
        x2 *= original_w / IMG_SIZE
        y1 *= original_h / IMG_SIZE
        y2 *= original_h / IMG_SIZE

        boxes.append([
            int(x1),
            int(y1),
            int(x2 - x1),
            int(y2 - y1),
        ])

        scores.append(confidence)
        class_ids.append(class_id)

    # --------------------------------------------------
    # Non-Maximum Suppression
    # --------------------------------------------------

    if boxes:
        indices = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            CONF_THRESHOLD,
            IOU_THRESHOLD,
        )

        indices = np.array(indices).flatten()
    else:
        indices = []

    print()
    print("DETECTIONS")
    print("----------")

    if len(indices) == 0:
        print("No defects detected.")

    else:
        for i in indices:
            x, y, w, h = boxes[i]

            class_name = CLASS_NAMES[class_ids[i]]
            confidence = scores[i]

            print(
                f"{class_name:10s} "
                f"confidence={confidence:.2f} "
                f"box=({x}, {y}, {w}, {h})"
            )

            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2,
            )

            label = f"{class_name} {confidence:.2f}"

            cv2.putText(
                image,
                label,
                (x, max(y - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    # --------------------------------------------------
    # Save result
    # --------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(
            f"Could not save output image: {output_path}"
        )

    print()
    print("Saved result to:", output_path)


if __name__ == "__main__":
    main()
