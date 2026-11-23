# PCB Inspection Edge AI

An edge-AI system for detecting and localizing printed circuit board (PCB) defects using **YOLO11n**, **ONNX Runtime**, and a **Raspberry Pi 4**. The project covers DeepPCB dataset preparation, model training and evaluation, ONNX export, and browser-based image inspection with inference performed locally on the Raspberry Pi CPU.

## Overview

The workflow separates model development from edge deployment:

1. Prepare and convert DeepPCB annotations into YOLO format.
2. Train and evaluate a six-class YOLO11n object-detection model.
3. Export the selected model to ONNX.
4. Run ONNX inference on Raspberry Pi 4 using ONNX Runtime's CPU execution provider.
5. Upload a PCB image through a Flask web interface and view the annotated detections.

The current application is an **image-upload inspection workflow**. An integrated live-camera pipeline and FPGA acceleration are not implemented.

## Detected Defect Classes

| Class ID | Defect |
|---:|---|
| 0 | Open |
| 1 | Short |
| 2 | Mousebite |
| 3 | Spur |
| 4 | Copper |
| 5 | Pin-hole |

## System Architecture

```text
DeepPCB dataset
      |
      v
Annotation inspection and YOLO conversion
      |
      v
YOLO11n training and evaluation (Apple M3 / MPS)
      |
      v
Best PyTorch checkpoint
      |
      v
ONNX export (640 x 640)
      |
      v
Raspberry Pi 4
      |
      v
Flask image-upload interface
      |
      v
OpenCV preprocessing -> ONNX Runtime CPU inference
      |
      v
Prediction decoding -> class-aware NMS
      |
      v
Annotated PCB image and detection summary
```

The Flask application preserves image aspect ratio during preprocessing using letterboxing, converts the image to RGB, normalizes it, and runs the ONNX model locally. Detection boxes are mapped back to the uploaded image and displayed with class labels and confidence scores.

## Repository Structure

```text
pcb-inspection-edge-ai/
├── ai-model/
│   ├── configs/
│   │   └── data.yaml
│   ├── scripts/
│   │   ├── check_original_label.py
│   │   ├── convert_deeppcb.py
│   │   ├── prepare_data_config.py
│   │   └── visual_check.py
│   ├── visuals/
│   └── weights/
│       ├── best.onnx
│       └── README.md
├── docs/
│   ├── benchmark_report.md
│   ├── deployment_guide.md
│   ├── model_methodology.md
│   └── system_architecture.md
├── raspberry-pi/
│   ├── app/
│   │   └── image_demo.py
│   ├── tests/
│   │   └── test_inference.py
│   ├── tools/
│   │   └── benchmark.py
│   ├── requirements.txt
│   └── README.md
├── results/
│   ├── model/
│   │   └── train-2/
│   │       ├── results.csv
│   │       ├── confusion_matrix.png
│   │       └── test/
│   │           └── metrics.md
│   └── raspberry-pi/
│       └── README.md
├── .gitattributes
├── .gitignore
└── README.md
```

The original dataset, converted dataset, PyTorch checkpoints, generated training runs, and local machine-specific dataset configuration are not included in Git.

## Dataset

The project uses the **DeepPCB** dataset with the following prepared splits:

| Split | Images | Label files |
|---|---:|---:|
| Training | 900 | 900 |
| Validation | 100 | 100 |
| Test | 500 | 500 |
| **Total** | **1,500** | **1,500** |

The original dataset and converted YOLO dataset must be obtained and prepared locally. They are excluded from this repository.

Dataset preparation scripts are in `ai-model/scripts/`:

- `check_original_label.py` — inspect original DeepPCB annotations.
- `convert_deeppcb.py` — convert annotations to YOLO format.
- `visual_check.py` — inspect converted bounding boxes visually.
- `prepare_data_config.py` — generate a machine-specific dataset configuration.

See [AI model development](ai-model/README.md) for dataset locations and preparation instructions.

## Model Development

The selected training run is `train-2`.

| Parameter | Configuration |
|---|---|
| Model | YOLO11n object detection |
| Input resolution | 640 × 640 |
| Classes | 6 |
| Epochs | 50 |
| Batch size | 8 |
| Training images | 900 |
| Validation images | 100 |
| Training device | Apple M3, PyTorch MPS |
| Deployment format | ONNX |

### Held-out test-set results

The selected `best.pt` checkpoint was evaluated on the 500-image DeepPCB test split, containing 3,140 annotated object instances.

| Metric | Test result |
|---|---:|
| Precision | 0.956 |
| Recall | 0.937 |
| mAP@50 | 0.970 |
| mAP@50–95 | 0.734 |

These are **PyTorch test-set evaluation metrics**. They are not measurements of the exported ONNX model or Raspberry Pi inference accuracy. Per-class results and evaluation plots are available in [test metrics](results/model/train-2/test/metrics.md).

## Raspberry Pi Application

The deployment application is in `raspberry-pi/app/image_demo.py`.

Current capabilities:

- Browser-based JPG, JPEG, and PNG image upload (10 MB upload limit).
- Local YOLO11n inference through ONNX Runtime `CPUExecutionProvider`.
- Image preprocessing and prediction decoding.
- Class-aware non-maximum suppression.
- Bounding-box and confidence visualization.
- Detection summary and model diagnostics.

The application binds to `0.0.0.0` on port `5001`. A browser on the same network can access it at `http://<RASPBERRY_PI_IP>:5001`.

### Recorded Raspberry Pi inference benchmark

Two preliminary Raspberry Pi 4 benchmark runs were recorded using one PCB image, five warm-up runs, and 50 timed runs.

| Measurement | Run 1 | Run 2 |
|---|---:|---:|
| Average inference latency | 457.32 ms | 451.25 ms |
| Throughput | 2.19 images/s | 2.22 images/s |

These values measure only the ONNX Runtime `session.run()` call. Image decoding, preprocessing, postprocessing, Flask request handling, and result rendering are excluded. The benchmark script uses direct resizing, whereas the Flask application uses letterbox preprocessing. Therefore, these values are **not end-to-end application response times**.

See the [benchmark report](docs/benchmark_report.md) for methodology and limitations.

## Setup and Run

### 1. Clone the repository and retrieve the model

Git LFS is required because the exported ONNX model is tracked using Git LFS.

```bash
git lfs install
git clone https://github.com/padamatageethasrinivasu137/pcb-inspection-edge-ai.git
cd pcb-inspection-edge-ai
git lfs pull
```

Verify that the ONNX file was retrieved:

```bash
ls -lh ai-model/weights/best.onnx
file ai-model/weights/best.onnx
```

The application checks for an optional app-local model at `raspberry-pi/app/best.onnx` first, then uses the canonical repository model at `ai-model/weights/best.onnx`.

### 2. Install Raspberry Pi dependencies

Use a compatible 64-bit Raspberry Pi OS and Python environment. The deployment environment documented for this project used Python 3.13.5; package availability can vary by OS and architecture.

```bash
cd raspberry-pi
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Start the web application

```bash
cd app
python image_demo.py
```

From another device on the same network, open:

```text
http://<RASPBERRY_PI_IP>:5001
```

Replace the placeholder with the Raspberry Pi's IP address.

### 4. Run standalone inference

From the `raspberry-pi` directory, with the virtual environment activated:

```bash
python tests/test_inference.py --image /path/to/pcb_image.jpg
```

Optionally specify where the annotated output should be saved:

```bash
python tests/test_inference.py \
  --image /path/to/pcb_image.jpg \
  --output /path/to/result.jpg
```

This standalone diagnostic script uses direct resizing to 640 × 640 and does not use the Flask application's letterbox preprocessing.

### 5. Run the inference benchmark

```bash
python tools/benchmark.py --image /path/to/pcb_image.jpg
```

Optional run counts:

```bash
python tools/benchmark.py \
  --image /path/to/pcb_image.jpg \
  --warmup 10 \
  --runs 100
```

The benchmark measures ONNX model execution only. See [Raspberry Pi deployment](raspberry-pi/README.md) and the [deployment guide](docs/deployment_guide.md) for additional details.

## Model and Data Distribution

- `ai-model/weights/best.onnx` — exported YOLO11n model, tracked with Git LFS.
- `best.pt` and `last.pt` — PyTorch checkpoints, maintained locally and not tracked.
- DeepPCB source data and converted YOLO dataset — maintained locally and not tracked.
- Generated training runs — only selected records and evaluation artifacts are included.

Install Git LFS before cloning and run `git lfs pull` to retrieve the ONNX model. See [model weights documentation](ai-model/weights/README.md).

## Limitations

- The current interface processes uploaded images; live-camera integration is not implemented.
- Edge inference currently uses the Raspberry Pi CPU; FPGA acceleration is not implemented.
- The recorded Raspberry Pi benchmark uses one image and measures model execution rather than complete application latency.
- Raspberry Pi benchmark software and hardware details are only partially recorded; results should be treated as a preliminary baseline.
- The standalone inference and benchmark utilities use direct resizing, unlike the Flask application's letterbox preprocessing.

## Documentation

- [System architecture](docs/system_architecture.md)
- [Model methodology and evaluation](docs/model_methodology.md)
- [Deployment guide](docs/deployment_guide.md)
- [Raspberry Pi benchmark report](docs/benchmark_report.md)
- [AI model development](ai-model/README.md)
- [Raspberry Pi module](raspberry-pi/README.md)
- [Training and test artifacts](results/model/train-2/README.md)

## Hardware and Software

- **Model development:** Apple MacBook Air M3 with PyTorch MPS.
- **Edge target:** Raspberry Pi 4.
- **Inference runtime:** ONNX Runtime with CPUExecutionProvider.
- **Application:** Python, Flask, OpenCV, and NumPy.
