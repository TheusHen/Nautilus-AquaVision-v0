#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Re-export the PyTorch checkpoint to ONNX."""
from ultralytics import YOLO

model = YOLO("weights/nautilus-aquavision-v0.1.0-alpha.pt")
model.export(format="onnx", imgsz=960, simplify=True, dynamic=False, nms=False)
