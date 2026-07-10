# Model Card — Nautilus AquaVision v0

## Model details

- **Model name:** Nautilus AquaVision v0
- **Release:** `0.1.0-alpha`
- **Developer:** Matheus Henrique (`@TheusHen`)
- **Task:** single-class object detection
- **Class:** `floating_object`
- **Base architecture:** Ultralytics YOLO26n
- **Input:** RGB, 960 × 960 during training/export
- **Parameters:** 2,504,190 in the training graph; 2,375,031 fused
- **Compute:** approximately 5.2 GFLOPs fused
- **Formats:** PyTorch `.pt` and ONNX opset 20
- **License:** GNU AGPL-3.0; dataset terms remain separately applicable

## Intended use

The model is intended for research, prototyping, dataset triage, and perception experiments involving floating objects or litter on river, lake, coastal, and similar water surfaces.

Potential applications include:

- candidate detection for a human review pipeline;
- perception experiments for the Nautilus surface-cleaning project;
- environmental monitoring prototypes;
- pre-annotation of newly collected images;
- edge-device performance research.

## Out-of-scope use

Do not use this model as the only input for collision avoidance, autonomous navigation, environmental enforcement, emergency response, or decisions that could harm people, animals, property, or ecosystems. It is not a classifier of material, ownership, toxicity, recyclability, or biological species.

## Training setup

| Property | Value |
|---|---|
| Framework | Ultralytics 8.4.91 |
| PyTorch | 2.10.0+cu128 |
| Hardware | NVIDIA Tesla T4, 14.56 GB VRAM |
| Epochs | 100 |
| Best epoch | 88 |
| Image size | 960 |
| Batch size | 8 |
| Optimizer | auto-selected by Ultralytics |
| AMP | enabled |
| Cosine LR | enabled |
| Training time | 4.733 hours |
| Seed | 0 |

Full arguments are in `configs/train_args.yaml`.

## Dataset

| Split | Images |
|---|---:|
| Train | 5,282 |
| Validation | 667 |
| Test | 654 |
| Total | 6,603 |

The Ultralytics label-distribution plot reports **32,085 instances in the training split**. Validation contains **4,210 instances**.

Verified source attribution is documented in `DATASETS.md`. Original images and annotations are not included.

## Evaluation

Best-checkpoint evaluation on the 667-image validation split:

| Metric | Value |
|---|---:|
| Precision | 0.8971 |
| Recall | 0.8501 |
| mAP@0.50 | 0.9290 |
| mAP@0.50:0.95 | 0.6538 |
| Maximum validation F1 | approximately 0.87 at confidence 0.395 |

These values are in-domain validation metrics, not proof of performance on arbitrary waterways or deployment cameras.

## ONNX details

- **File:** `weights/nautilus-aquavision-v0.1.0-alpha.onnx`
- **Size:** approximately 9.6 MB
- **Input:** `images`, float32, `[1, 3, 960, 960]`
- **Output:** `output0`, float32, `[1, 300, 6]`
- **End-to-end:** yes
- **Export NMS:** disabled
- **Dynamic shapes:** no
- **ONNX checker:** passed
- **CPU Runtime smoke test:** passed with ONNX Runtime

## Limitations and failure modes

- The model uses a single broad class; detections are not necessarily human-made litter.
- Reflections, foam, leaves, rocks, shoreline structures, algae, packaging, and other objects may be confused.
- Small-object performance is sensitive to camera resolution and distance.
- Confidence calibration may change across cameras and environments.
- Validation images may share visual properties with training sources, so domain-shift performance is unknown.
- No dedicated fairness, privacy, adversarial robustness, weather, night, or safety validation has been performed.
- The release lacks the original `manifest.csv`; exact accepted-image counts per upstream source cannot be independently reconstructed from the files provided for packaging.

## Recommended deployment checks

1. Collect a local holdout set from the final camera and environment.
2. Measure false positives per minute and missed-object rate, not only mAP.
3. Tune the confidence threshold using operational costs.
4. Add temporal tracking and multi-frame confirmation when used on video.
5. Keep a human in the loop for consequential decisions.
6. Log model version, threshold, camera settings, and inference backend.

## Reproducibility

- The sanitized notebook is in `notebooks/train_aquavision.ipynb`.
- Full checkpoint training history is in `metrics/training_history.csv`.
- Model and artifact hashes are in `SHA256SUMS`.
- Dataset creation logic is in `scripts/aquavision_builder.py`.

## Environmental note

The released model is small enough for many edge experiments, but actual energy use depends on resolution, frame rate, backend, and hardware. Benchmark the lowest resolution and cadence that still meets the application requirement.
