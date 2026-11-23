# YOLO11n Test-Set Evaluation

## Evaluation configuration

| Parameter | Value |
|---|---|
| Model | YOLO11n |
| Checkpoint | `best.pt` from `train-2` |
| Dataset | DeepPCB |
| Dataset split | Test |
| Test images | 500 |
| Ground-truth object instances | 3,140 |
| Input image size | 640 × 640 |
| Batch size | 8 |
| Device | Apple M3 MPS |
| Ultralytics version | 8.4.174 |
| PyTorch version | 2.14.1 |

## Overall performance

| Metric | Score |
|---|---:|
| Precision | 0.956 |
| Recall | 0.937 |
| mAP@50 | 0.970 |
| mAP@50–95 | 0.734 |

## Class-wise performance

| Class | Images | Instances | Precision | Recall | mAP@50 | mAP@50–95 |
|---|---:|---:|---:|---:|---:|---:|
| Open | 482 | 659 | 0.966 | 0.942 | 0.974 | 0.653 |
| Short | 368 | 478 | 0.904 | 0.881 | 0.916 | 0.615 |
| Mousebite | 413 | 586 | 0.982 | 0.940 | 0.982 | 0.724 |
| Spur | 373 | 483 | 0.970 | 0.926 | 0.971 | 0.719 |
| Copper | 403 | 464 | 0.976 | 0.968 | 0.987 | 0.873 |
| Pin-hole | 438 | 470 | 0.936 | 0.968 | 0.987 | 0.817 |

## Inference speed

| Stage | Time per image |
|---|---:|
| Preprocessing | 0.2 ms |
| Inference | 3.8 ms |
| Postprocessing | 9.9 ms |

These timings were measured on an Apple M3 using MPS and are hardware- and configuration-dependent.

## Evaluation artifacts

- `confusion_matrix.png`: Test-set confusion matrix.
- `confusion_matrix_normalized.png`: Normalized test-set confusion matrix.
- `BoxPR_curve.png`: Precision-recall curve.
- `BoxP_curve.png`: Precision versus confidence.
- `BoxR_curve.png`: Recall versus confidence.
- `BoxF1_curve.png`: F1 score versus confidence.

## Notes

- These results were obtained by evaluating `best.pt` on the test split using Ultralytics validation with `split=test`.
- The test split contains 500 images and 500 corresponding label files.
- Training and validation metrics are documented separately.
- The evaluation was performed on the PyTorch checkpoint, not the exported ONNX model.
- The reported inference speed is not a Raspberry Pi benchmark.
