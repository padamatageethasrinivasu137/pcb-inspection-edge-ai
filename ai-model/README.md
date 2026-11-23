# AI Model Development

This module contains the dataset preparation, annotation conversion,
YOLO11n training and model export workflow for PCB defect inspection.

## 1. Defect Classes

| YOLO ID | Defect |
|---:|---|
| 0 | Open |
| 1 | Short |
| 2 | Mousebite |
| 3 | Spur |
| 4 | Copper |
| 5 | Pin-hole |

## 2. Dataset

The project uses the DeepPCB dataset.

| Split | Images | Labels |
|---|---:|---:|
| Train | 900 | 900 |
| Validation | 100 | 100 |
| Test | 500 | 500 |

The original DeepPCB dataset and converted YOLO dataset are maintained
locally and are not included in Git.

Expected local dataset locations, relative to the repository root:

- dataset/DeepPCB-master/PCBData/
- yolo_dataset/images/
- yolo_dataset/labels/

## 3. Dataset Preparation

The scripts in `scripts/` provide:

- `check_original_label.py`: inspect original DeepPCB annotations.
- `convert_deeppcb.py`: convert DeepPCB annotations into YOLO format.
- `visual_check.py`: visually inspect converted bounding boxes.
- `prepare_data_config.py`: generate a local dataset configuration.

Run dataset preparation from the repository root. The converter expects
the original dataset at `dataset/DeepPCB-master/PCBData/`.

## 4. Dataset Configuration

The tracked `configs/data.yaml` is a portable template.

Generate a local configuration using:

`python3 ai-model/scripts/prepare_data_config.py`

This creates `configs/data.local.yaml` with the absolute dataset path
for the current machine. That local file is excluded from Git.

## 5. Training Setup

Recorded training configuration:

- Model: YOLO11n
- Image size: 640 x 640
- Epochs: 50
- Batch size: 8
- Device: Apple MPS
- Training split: 900 images
- Validation split: 100 images

The recorded training command was:

`yolo detect train model=yolo11n.pt data=ai-model/configs/data.local.yaml epochs=50 imgsz=640 batch=8 device=mps`

The original pretrained checkpoint and training environment are maintained
in the local Mac workspace.

## 6. Model Export

The selected training run is `train-2`.

The exported ONNX model was generated with:

`yolo export model=ai-model/weights/best.pt format=onnx imgsz=640 batch=1 simplify=True`

## 7. Model Artifacts

The `weights/` directory contains:

- `best.pt`: best PyTorch checkpoint, maintained locally and not tracked.
- `last.pt`: final PyTorch checkpoint, maintained locally and not tracked.
- `best.onnx`: exported ONNX model, tracked using Git LFS.

The ONNX model is available through Git LFS. The PyTorch checkpoints, datasets and generated training runs are excluded from Git.

See `weights/README.md` for instructions to retrieve the ONNX model.

## 8. Evaluation Records

The selected run's records are stored under:

`results/model/train-2/`

They include the training history CSV and confusion matrix.

The CSV records training and validation metrics. Separate official test-set
evaluation results should be documented independently.

## 9. Reproducibility

Keep the original dataset, converted dataset, Python virtual environment
and generated training runs outside version control.

See the repository root README for the overall system and deployment
documentation.
