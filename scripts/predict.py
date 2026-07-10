#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Run Nautilus AquaVision inference with Ultralytics on PT or ONNX weights."""
from __future__ import annotations

import argparse
from pathlib import Path
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="weights/nautilus-aquavision-v0.1.0-alpha.onnx")
    parser.add_argument("--source", required=True, help="Image, directory, video, webcam index, or URL")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.395, help="Balanced threshold from the validation F1 curve")
    parser.add_argument("--device", default=None, help="Examples: cpu, 0, 0,1")
    parser.add_argument("--project", default="runs/predict")
    parser.add_argument("--name", default="aquavision")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(str(model_path))
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=True,
        project=args.project,
        name=args.name,
    )
    if results:
        print(f"Saved results to: {results[0].save_dir}")


if __name__ == "__main__":
    main()
