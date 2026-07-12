# ONNX Requirements Report

## Model
- File: `/workspaces/Nautilus-AquaVision-v0/weights/nautilus-aquavision-v0.1.0-alpha.onnx`
- Size on disk: **9.55 MiB**
- Estimated weights: **9.42 MiB**
- Parameters: **2,469,576**
- Nodes: **383**
- Opsets: `{"ai.onnx": 20}`

## Tested Inputs
- `images`: `[1, 3, 960, 960]`

## Local Benchmarks

| Provider | Threads | Median | P95 | Inferences/s | Peak model RAM | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| CPUExecutionProvider | 1 | 198.15 ms | 210.88 ms | 5.05 | 214.65 MiB | não medido |
| CPUExecutionProvider | 2 | 105.19 ms | 115.16 ms | 9.51 | 200.46 MiB | não medido |
| CPUExecutionProvider | 4 | 101.79 ms | 145.13 ms | 9.82 | 201.69 MiB | não medido |

## Performance Tiers

- **usable** — CPUExecutionProvider, 1 threads: 198.15 ms mediana, 210.88 ms P95, 5.05 inferences/s.
- **best measured** — CPUExecutionProvider, 2 threads: 105.19 ms mediana, 115.16 ms P95, 9.51 inferences/s.
- **best measured** — CPUExecutionProvider, 4 threads: 101.79 ms mediana, 145.13 ms P95, 9.82 inferences/s.

## Estimated Requirements

- **Minimum system RAM:** 4 GiB
- **Recommended system RAM:** 8 GiB
- **Minimum process budget:** 257.58 MiB
- **Recommended process budget:** 321.98 MiB
- **VRAM:** not measured; the benchmark did not use a monitorable NVIDIA GPU.

- **Minimum free space:** 11.46 MiB
- **Recommended free space:** 23.88 MiB

## Limitations and Recommendations

- The model contains FP32 weights. FP16 can roughly halve the weight footprint; INT8 may approach 1/4, but accuracy, operator support, and real gains must be validated.
- For GPU, compare CUDA/TensorRT with CPU: unsupported operators may fall back to CPU, causing transfers and performance loss.

> CPU/GPU requirements for other machines cannot be derived precisely from the ONNX file alone. This report measures the current machine. To recommend specific hardware targets, maintain a database of real CPU/GPU benchmarks and compare the same input scenario.
