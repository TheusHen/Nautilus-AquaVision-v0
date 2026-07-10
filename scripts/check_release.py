#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Perform lightweight integrity checks on the release files."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import onnx

ROOT = Path(__file__).resolve().parents[1]
ONNX_PATH = ROOT / "weights" / "nautilus-aquavision-v0.1.0-alpha.onnx"
SUMMARY = ROOT / "metrics" / "summary.json"
CHECKSUMS = ROOT / "SHA256SUMS"

model = onnx.load(ONNX_PATH)
onnx.checker.check_model(model)
metadata = {item.key: item.value for item in model.metadata_props}
assert metadata.get("task") == "detect"
assert "floating_object" in metadata.get("names", "")
assert json.loads(SUMMARY.read_text())["onnx"]["opset"] == 20

expected = {}
for line in CHECKSUMS.read_text().splitlines():
    digest, rel = line.split("  ", 1)
    expected[rel] = digest
for rel, digest in expected.items():
    p = ROOT / rel
    got = hashlib.sha256(p.read_bytes()).hexdigest()
    assert got == digest, f"Checksum mismatch: {rel}"
print("Release checks passed.")
