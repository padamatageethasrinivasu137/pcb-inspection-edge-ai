# Raspberry Pi 4 — ONNX Performance Benchmark

## 1. Hardware and Software

| Parameter | Configuration |
|---|---|
| Hardware | Raspberry Pi 4 |
| Model | YOLO11n |
| Model format | ONNX |
| Execution provider | ONNX Runtime CPUExecutionProvider |
| Input resolution | 640 × 640 |
| Python | 3.13.5 |

## 2. Benchmark Methodology

The benchmark uses a single PCB test image and measures ONNX model execution time.

- Warm-up runs: 5
- Timed inference runs: 50
- Preprocessing: Direct resize to 640 × 640
- Timing: Python `time.perf_counter()`
- Measurement scope: Model inference only

The benchmark does not include Flask request handling, image upload, preprocessing, postprocessing or result rendering.

## 3. Recorded Results

| Metric | Run 1 | Run 2 |
|---|---:|---:|
| Warm-up runs | 5 | 5 |
| Timed runs | 50 | 50 |
| Total time | 22.866 s | 22.562 s |
| Average inference | 457.32 ms | 451.25 ms |
| Throughput | 2.19 images/s | 2.22 images/s |

## 4. Summary

The two benchmark runs produced similar inference times, establishing a preliminary Raspberry Pi 4 CPU inference baseline of approximately 451–457 ms per image.

These measurements are specific to the tested model, input image, software environment and hardware conditions. They should not be interpreted as complete application response times.

## 5. Limitations

- Only one test image was used.
- Measurements cover model execution, not the complete inspection pipeline.
- The diagnostic benchmark uses direct resizing, whereas the main Flask application uses letterbox preprocessing.
- Results may vary with system load and thermal conditions.

Future benchmarking should include multiple images, repeated sessions and complete application latency measurements.
