# YOLO11n Training and Evaluation: train-2

This directory contains the selected final 50-epoch YOLO11n training run and its test-set evaluation.

## Directory structure

- `results.csv`: Epoch-wise training and validation metrics.
- `confusion_matrix.png`: Confusion matrix generated during training/validation.
- `test/`: Independent test-split evaluation artifacts and metrics.

## Test-set evaluation

The selected `best.pt` checkpoint was evaluated against the DeepPCB test split containing 500 images and 3,140 ground-truth object instances.

See [`test/metrics.md`](test/metrics.md) for overall metrics, class-wise performance, evaluation configuration, and generated plots.

The test-set results are kept separate from the training and validation records.

## Model weights

The `.pt` and `.onnx` model files are not stored in this directory. See the repository's model weights documentation for distribution information.
