# Raspberry Pi Deployment

This module contains the Raspberry Pi implementation of the PCB defect inspection system. It runs the trained YOLO11n model in ONNX format using ONNX Runtime on the CPU and provides a Flask-based image inspection interface.

## 1. Implementation Overview

The current application supports:

- PCB image uploads through a web browser.
- Image preprocessing and YOLO11n inference.
- Detection of six PCB defect categories.
- Bounding-box visualization and annotated results.
- Detection summaries.
- Local inference using ONNX Runtime CPUExecutionProvider.

## 2. Defect Classes

| Class ID | Defect |
|---:|---|
| 0 | Open |
| 1 | Short |
| 2 | Mousebite |
| 3 | Spur |
| 4 | Copper |
| 5 | Pin-hole |

## 3. Repository Structure

```text
raspberry-pi/
├── app/
│   └── image_demo.py
│   └── best.onnx          # downloaded separately
├── tests/
│   └── test_inference.py
├── tools/
│   └── benchmark.py
└── requirements.txt
```

## 4. Verified Software Environment

The application was tested with:

| Component | Version |
|---|---|
| Python | 3.13.5 |
| Flask | 3.1.3 |
| NumPy | 2.5.3 |
| ONNX Runtime | 1.30.0 |
| OpenCV Headless | 5.0.0.93 |
| Inference provider | CPUExecutionProvider |

The pinned dependencies are listed in `requirements.txt`.

## 5. Model Setup

The application uses the exported `best.onnx` model.

The current application expects the model at:

`raspberry-pi/app/best.onnx`

Model binaries are excluded from ordinary Git commits. A versioned
GitHub Release will be used to distribute the matching model artifact.
Until a release is published, the model must be supplied separately
and placed at the expected location.

## 6. Installation

On the Raspberry Pi, clone the repository and enter the deployment directory:

```bash
git clone https://github.com/padamatageethasrinivasu137/pcb-inspection-edge-ai.git
cd pcb-inspection-edge-ai/raspberry-pi
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Place `best.onnx` in the `app/` directory as described above.

## 7. Running the Application

Start the Flask application:

```bash
cd app
python image_demo.py
```

The application listens on port `5001` and binds to `0.0.0.0`.

Open the following address from a device on the same network:

`http://<RASPBERRY_PI_IP>:5001`

Replace `<RASPBERRY_PI_IP>` with the Raspberry Pi's actual IP address.

## 8. Testing and Benchmarking

The repository includes:

- `tests/test_inference.py` — standalone ONNX inference test.
- `tools/benchmark.py` — inference benchmarking utility.

Run the inference test from the Raspberry Pi deployment directory:

```bash
python tests/test_inference.py
```

Run the benchmark:

```bash
python tools/benchmark.py
```

The main Flask application uses letterbox preprocessing. The standalone test and benchmark scripts currently use direct image resizing, so their results should not be treated as identical application-level measurements.

## 9. Verification and Performance

The following checks have been completed:

- ONNX model loading.
- CPUExecutionProvider availability.
- Single-image inference.
- Flask home route returning HTTP 200.
- Missing-image upload returning HTTP 400.
- Python syntax checks.
- Dependency consistency check.

A single-image inference test previously recorded approximately 496 ms. Actual performance depends on image content, system load and hardware conditions.

## 10. Current Limitations

- The current application uses image uploads rather than an integrated live-camera pipeline.
- FPGA acceleration is not yet integrated.
- Inference currently uses the Raspberry Pi CPU.
- Older camera-streaming experiments are not part of the primary application workflow.

## 11. Security and Deployment Notes

- Do not commit model binaries, virtual environments, private keys, certificates or uploaded images.
- Do not expose the Flask development server directly to an untrusted network.
- Use an appropriate production deployment and access-control configuration for wider use.
