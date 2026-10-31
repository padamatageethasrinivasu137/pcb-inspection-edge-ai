# Local Model Weights

This directory stores the trained model artifacts locally.

- best.pt: best PyTorch checkpoint from the selected training run.
- last.pt: final PyTorch checkpoint.
- best.onnx: exported ONNX model used for Raspberry Pi CPU inference.

Model binaries are excluded from normal Git commits.
Use a versioned GitHub Release to distribute model artifacts.
