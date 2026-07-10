#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Validate Nautilus AquaVision against a YOLO data.yaml file."""
from __future__ import annotations

import argparse
from pathlib import Path
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="weights/nautilus-aquavision-v0.1.0-alpha.pt")
    parser.add_argument("--data", required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    for value in (args.model, args.data):
        if not Path(value).exists():
            raise FileNotFoundError(value)

    metrics = YOLO(args.model).val(data=args.data, imgsz=args.imgsz, device=args.device, plots=True)
    print(metrics.results_dict)


if __name__ == "__main__":
    main()
