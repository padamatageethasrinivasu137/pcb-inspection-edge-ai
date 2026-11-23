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
| RAM | To be recorded |
| Operating system | To be recorded |
| Python version | To be recorded |
| ONNX Runtime version | To be recorded |
| OpenCV version | To be recorded |
| Model file/checksum | To be recorded |

## 4. Results

No verified Raspberry Pi benchmark measurements are included in this report yet.

| Measurement | Result |
|---|---|
| Warm-up runs | 5 by default |
| Timed runs | 50 by default |
| Average inference latency | Pending |
| Throughput | Pending |
| Median latency | Not measured by current script |
| Maximum latency | Not measured by current script |
| CPU utilization | Not measured by current script |
| Memory usage | Not measured by current script |

## 5. Measurement Scope

The current benchmark measures the ONNX Runtime session call only. Image decoding and preprocessing are performed before the timed loop.

The benchmark uses direct resizing, whereas the Flask application uses letterbox preprocessing. Therefore, benchmark results should not be interpreted as complete application response time.

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
