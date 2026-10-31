import argparse
import time
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

IMG_SIZE = 640
WARMUP_RUNS = 5
BENCHMARK_RUNS = 50


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Benchmark YOLO11n ONNX CPU inference."
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the PCB test image.",
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=BENCHMARK_RUNS,
        help="Number of timed inference runs.",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=WARMUP_RUNS,
        help="Number of warm-up inference runs.",
    )

    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be at least 1")

    if args.warmup < 0:
        parser.error("--warmup cannot be negative")

    return args


def main():
    args = parse_arguments()

    image_path = Path(args.image).expanduser().resolve()

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

    print("Loading model:", MODEL_PATH)

    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    print("Execution provider:", session.get_providers()[0])

    # --------------------------------------------------
    # Read and preprocess image
    # --------------------------------------------------

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Could not decode image: {image_path}"
        )

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
    # Warm-up
    # --------------------------------------------------

    print()
    print(f"Warm-up runs: {args.warmup}")

    for _ in range(args.warmup):
        session.run(
            None,
            {input_name: input_image},
        )

    # --------------------------------------------------
    # Benchmark
    # --------------------------------------------------

    print(f"Benchmark runs: {args.runs}")

    start = time.perf_counter()

    for _ in range(args.runs):
        session.run(
            None,
            {input_name: input_image},
        )

    end = time.perf_counter()

    total = end - start
    average = total / args.runs
    fps = 1.0 / average

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print()
    print("===== Raspberry Pi 5 CPU ONNX Benchmark =====")
    print(f"Image:             {image_path.name}")
    print(f"Runs:              {args.runs}")
    print(f"Total time:        {total:.3f} s")
    print(f"Average inference: {average * 1000:.2f} ms")
    print(f"Throughput:        {fps:.2f} images/s")
    print("===============================================")


if __name__ == "__main__":
    main()
