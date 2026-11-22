# PCB Inspection Edge AI

An edge-based PCB defect inspection system using YOLO11n, ONNX Runtime and Raspberry Pi.

The project aims to detect and localize defects in printed circuit boards (PCBs), with inference performed locally on embedded hardware.

## 1. Defect Categories

The system identifies six defect classes:

| Class ID | Defect |
|---:|---|
| 0 | Open |
| 1 | Short |
| 2 | Mousebite |
| 3 | Spur |
| 4 | Copper |
| 5 | Pin-hole |

## 2. System Architecture

The development and deployment workflow consists of:

1. DeepPCB dataset preparation and annotation conversion.
2. YOLO11n model training and validation.
3. Export of the trained model to ONNX.
4. Deployment on Raspberry Pi using ONNX Runtime.
5. Image inspection through a Flask web application.

## 3. Repository Structure

```text
pcb-inspection-edge-ai/
├── ai-model/
│   ├── configs/
│   ├── scripts/
│   ├── visuals/
│   └── weights/
├── raspberry-pi/
│   ├── app/
│   ├── tests/
│   ├── tools/
│   └── requirements.txt
├── results/
│   └── model/
│       └── train-2/
├── docs/
├── demo/
├── .gitignore
└── README.md
```

## 4. Dataset

The project uses the DeepPCB dataset.

| Dataset split | Images |
|---|---:|
| Training | 900 |
| Validation | 100 |
| Testing | 500 |
| **Total** | **1,500** |

The original dataset and converted YOLO dataset are maintained locally and excluded from Git.

See [AI Model Development](ai-model/README.md) for dataset preparation and conversion.

## 5. Model Development

| Parameter | Configuration |
|---|---|
| Architecture | YOLO11n |
| Input resolution | 640 × 640 |
| Number of classes | 6 |
| Epochs | 50 |
| Batch size | 8 |
| Training device | Apple M3 / PyTorch MPS |
| Deployment format | ONNX |

The selected training run is `train-2`.

Training records and the confusion matrix are available in `results/model/train-2/`.

## 6. Edge Deployment

The Raspberry Pi application uses:

- Python
- Flask
- OpenCV
- NumPy
- ONNX Runtime
- ONNX Runtime CPUExecutionProvider

The current implementation supports image upload, PCB defect detection and annotated result visualization through a web interface.

See [Raspberry Pi Deployment](raspberry-pi/README.md) for setup and execution instructions.

## 7. Development Status

| Component | Status |
|---|---|
| Dataset preparation and conversion | Completed |
| YOLO11n training | Completed |
| ONNX export | Completed |
| Raspberry Pi CPU inference | Verified |
| Flask image inspection interface | Implemented and route-tested |
| Live camera integration | Not yet integrated |

## 8. Hardware

- Apple MacBook Air M3 — model development and training.
- Raspberry Pi 4 — edge inference and application hosting.

## 9. Model Distribution

Trained model binaries are excluded from ordinary Git commits.

The intended distribution method is versioned GitHub Releases. See `ai-model/weights/README.md` for artifact details.

## 10. License

License to be determined.
