# Model Weights

This directory stores the model artifacts used by the PCB inspection project.

- best.onnx: Exported YOLO11n model used for Raspberry Pi CPU inference. Tracked using Git LFS.
- best.pt: Best PyTorch checkpoint. Kept locally and not tracked.
- last.pt: Final PyTorch checkpoint. Kept locally and not tracked.

## Retrieving the ONNX Model

Install Git LFS before cloning:

    git lfs install
    git clone https://github.com/padamatageethasrinivasu137/pcb-inspection-edge-ai.git
    cd pcb-inspection-edge-ai
    git lfs pull

Verify the model file:

    ls -lh ai-model/weights/best.onnx
    file ai-model/weights/best.onnx

The Raspberry Pi application uses this canonical model path when no app-local copy exists. See raspberry-pi/README.md.
