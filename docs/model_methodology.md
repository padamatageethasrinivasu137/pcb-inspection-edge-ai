# Model Methodology

## 1. Overview

This project uses a YOLO11n object-detection model to identify defects in printed circuit board (PCB) images. The model is trained using the DeepPCB dataset and exported to ONNX for CPU-based edge inference.

## 2. Dataset

The project uses DeepPCB with the following prepared splits:

| Split | Images | Label files |
|---|---:|---:|
| Training | 900 | 900 |
| Validation | 100 | 100 |
| Test | 500 | 500 |
| **Total** | **1500** | **1500** |

The original dataset and converted YOLO dataset are not distributed in this repository.

## 3. Defect Categories

The model detects six classes:

| Class ID | Defect |
|---:|---|
| 0 | Open |
| 1 | Short |
| 2 | Mousebite |
| 3 | Spur |
| 4 | Copper |
| 5 | Pin-hole |

## 4. Dataset Preparation

The preparation pipeline consists of:

1. Inspecting original DeepPCB annotations.
2. Converting annotations into YOLO bounding-box format.
3. Checking converted labels and bounding boxes visually.
4. Generating a machine-specific dataset configuration.

The scripts are available under `ai-model/scripts/`.

The portable dataset configuration is `ai-model/configs/data.yaml`. The machine-specific `data.local.yaml` is generated locally and excluded from Git.

## 5. Model Training

The selected training run is `train-2`.

| Parameter | Value |
|---|---|
| Architecture | YOLO11n |
| Task | Object detection |
| Epochs | 50 |
| Input resolution | 640 × 640 |
| Batch size | 8 |
| Optimizer | Auto |
| Device | Apple MPS |
| Pretrained weights | YOLO11n |
| Training images | 900 |
| Validation images | 100 |

The recorded training configuration is available in `results/model/train-2/args.yaml`.

## 6. Model Export and Edge Deployment

The trained PyTorch checkpoint was exported to ONNX format for deployment using ONNX Runtime.

The Raspberry Pi application uses:

- ONNX Runtime CPUExecutionProvider
- OpenCV image processing
- NumPy tensor operations
- Flask for browser-based image inspection

Model binaries are excluded from ordinary Git commits and are intended to be distributed separately through versioned releases.

## 7. Test Evaluation

The selected `train-2` PyTorch checkpoint was evaluated on the held-out test split containing 500 images and 3,140 annotated instances.

| Metric | Result |
|---|---:|
| Precision | 0.956 |
| Recall | 0.937 |
| mAP@0.50 | 0.970 |
| mAP@0.50:0.95 | 0.734 |

These metrics are from PyTorch test evaluation, not ONNX or Raspberry Pi evaluation.

Detailed per-class metrics and evaluation plots are available in `results/model/train-2/test/metrics.md`.

## 8. Reproducibility

The repository contains the preparation scripts, configuration template, training records and evaluation plots.

The original dataset, generated dataset, Python environment and model binaries are maintained separately.
