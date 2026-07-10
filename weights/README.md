# Weights

| File | Purpose |
|---|---|
| `nautilus-aquavision-v0.1.0-alpha.pt` | PyTorch checkpoint for Ultralytics inference, validation, or fine-tuning |
| `nautilus-aquavision-v0.1.0-alpha.onnx` | ONNX opset 20 deployment model, fixed input 960×960 |

The ONNX file passed `onnx.checker.check_model` and a zero-input CPU smoke test with ONNX Runtime 1.23.2.

These weights are YOLO26-derived and distributed under AGPL-3.0. See the root `LICENSE`, `NOTICE`, and `DATASETS.md`.
