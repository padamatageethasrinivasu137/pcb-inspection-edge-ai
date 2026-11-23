# Edge Inference Benchmark Report

## 1. Purpose

This report records CPU inference performance of the exported YOLO11n ONNX model on the target Raspberry Pi hardware.

## 2. Benchmark Implementation

The benchmark utility is:

`raspberry-pi/tools/benchmark.py`

Default settings:

| Parameter | Value |
|---|---:|
| Model input resolution | 640 × 640 |
| Warm-up runs | 5 |
| Timed runs | 50 |
| Execution provider | ONNX Runtime CPUExecutionProvider |
| Preprocessing | Direct resize, RGB conversion and normalization |

The script reports total timed duration, average inference latency and throughput.

## 3. Hardware and Software

| Item | Measured configuration |
|---|---|
| Board | Raspberry Pi 4 |
| Model | YOLO11n |
| Model format | ONNX |
| Execution provider | ONNX Runtime CPUExecutionProvider |
| Input resolution | 640 × 640 |
| Python version | 3.13.5 |
| RAM | Not recorded |
| Operating system | Not recorded |
| ONNX Runtime version | Not recorded |
| OpenCV version | Not recorded |
| Model checksum | Not recorded |

## 4. Results

Two preliminary benchmark runs were recorded using one PCB test image.

| Measurement | Run 1 | Run 2 |
|---|---:|---:|
| Warm-up runs | 5 | 5 |
| Timed runs | 50 | 50 |
| Total timed duration | 22.866 s | 22.562 s |
| Average inference latency | 457.32 ms | 451.25 ms |
| Throughput | 2.19 images/s | 2.22 images/s |

The preliminary Raspberry Pi 4 CPU inference baseline is approximately 451–457 ms per image.

The following metrics are not measured by the current benchmark script:

- Median latency
- Maximum latency
- CPU utilization
- Memory usage

## 5. Measurement Scope

The benchmark measures only the ONNX Runtime inference call (`session.run()`) using Python's `time.perf_counter()`.

The following operations occur outside the timed loop:

- Model loading and ONNX Runtime session initialization.
- Image decoding.
- Resizing to 640 × 640.
- BGR-to-RGB conversion.
- Tensor conversion, normalization and batch-dimension expansion.

The benchmark uses direct resizing, whereas the Flask application uses letterbox preprocessing. It excludes Flask request handling, image uploads, application preprocessing, postprocessing and result rendering.

Therefore, these measurements represent model inference latency, not complete application response time.
## 6. Reproducibility

Run from the `raspberry-pi` directory:

```bash
python tools/benchmark.py --image /path/to/pcb_image.jpg
```

For example, to use 10 warm-up runs and 100 timed runs:

```bash
python tools/benchmark.py \
  --image /path/to/pcb_image.jpg \
  --warmup 10 \
  --runs 100
```

Record the board model, RAM, operating system, software versions, model checksum and command used alongside every measurement.

## 7. Mac and Raspberry Pi Comparison

MacBook M3 PyTorch test-evaluation timing and Raspberry Pi ONNX CPU timing represent different hardware and software configurations.

They must be reported separately and must not be presented as a direct like-for-like comparison.
