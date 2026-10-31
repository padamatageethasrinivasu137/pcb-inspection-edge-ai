# PCB Inspection Edge AI

An edge-based PCB defect inspection system using YOLO11n and Raspberry Pi.

## Objective

Detect and localize six PCB defect categories:

1. Open
2. Short
3. Mousebite
4. Spur
5. Copper
6. Pin-hole

## Project Components

- `ai-model/` - Dataset preparation, model training, evaluation and export.
- `raspberry-pi/` - ONNX inference and Flask inspection application.
- `docs/` - Technical documentation and project report.
- `results/` - Model evaluation and hardware benchmarks.
- `demo/` - Demonstration materials.

## Current Status

- YOLO11n training: Completed.
- ONNX export: Completed.
- Raspberry Pi CPU inference: Completed.
- Flask image inspection interface: Implemented.
- FPGA acceleration: Planned.

## Hardware

- MacBook Air M3 for AI model development.
- Raspberry Pi for edge inference and application hosting.
- Efinix Ti60 FPGA planned for future acceleration.

## License

To be determined.
