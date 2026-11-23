# Deployment Guide

## 1. Requirements

The deployment requires:

- Raspberry Pi 4 running a compatible 64-bit Linux environment.
- Python environment compatible with the pinned dependencies.
- The exported YOLO11n ONNX model.
- Network access from the browser device to the Raspberry Pi for the web interface.

The documented software environment is Python 3.13.5, Flask 3.1.3, NumPy 2.5.3, ONNX Runtime 1.30.0 and OpenCV Headless 5.0.0.93.

Check package availability for the specific Raspberry Pi operating system and Python version before installation.

## 2. Clone the Repository

```bash
git lfs install

git clone https://github.com/padamatageethasrinivasu137/pcb-inspection-edge-ai.git
cd pcb-inspection-edge-ai

git lfs pull
```

## 3. Create a Virtual Environment

```bash
cd raspberry-pi
python3 -m venv .venv
source .venv/bin/activate
```

## 4. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Install the ONNX Model

The exported YOLO11n ONNX model is tracked using Git LFS at:

`ai-model/weights/best.onnx`

After cloning, run `git lfs pull` from the repository root to retrieve the model.

The Flask application searches for the model in this order:

1. `raspberry-pi/app/best.onnx` (optional app-local copy).
2. `ai-model/weights/best.onnx` (canonical repository copy).

No separate release download is required.

## 6. Run the Flask Application

```bash
cd app
python image_demo.py
```

The application is configured to listen on port 5001 and bind to `0.0.0.0`.

From a device on the same network, open:

`http://<RASPBERRY_PI_IP>:5001`

Replace the placeholder with the Raspberry Pi's IP address.

## 7. Run Standalone Inference

From the `raspberry-pi` directory, provide an input image:

```bash
python tests/test_inference.py --image /path/to/pcb_image.jpg
```

An optional output path can be supplied:

```bash
python tests/test_inference.py \
  --image /path/to/pcb_image.jpg \
  --output /path/to/result.jpg
```

The standalone test uses direct resizing to 640 × 640. It does not use the Flask application's letterbox preprocessing.

## 8. Run the Benchmark

```bash
python tools/benchmark.py --image /path/to/pcb_image.jpg
```

The default benchmark uses five warm-up runs and fifty timed inference runs. These values can be changed using `--warmup` and `--runs`.

The current benchmark measures ONNX session inference time after preprocessing. It does not measure the complete Flask request or image-processing pipeline.

## 9. Security Notes

- Do not expose the Flask development server directly to untrusted networks.
- The ONNX model is tracked using Git LFS. Keep PyTorch checkpoints (`best.pt`, `last.pt`), private keys, credentials and uploaded user images out of Git.
- Use an appropriate production server and access controls for wider deployment.
