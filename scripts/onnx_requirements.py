from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import psutil
from onnx import TensorProto


DTYPE_BYTES = {
    TensorProto.FLOAT: 4,
    TensorProto.UINT8: 1,
    TensorProto.INT8: 1,
    TensorProto.UINT16: 2,
    TensorProto.INT16: 2,
    TensorProto.INT32: 4,
    TensorProto.INT64: 8,
    TensorProto.STRING: 0,
    TensorProto.BOOL: 1,
    TensorProto.FLOAT16: 2,
    TensorProto.DOUBLE: 8,
    TensorProto.UINT32: 4,
    TensorProto.UINT64: 8,
    TensorProto.COMPLEX64: 8,
    TensorProto.COMPLEX128: 16,
    TensorProto.BFLOAT16: 2,
}

ORT_TO_NUMPY = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(int16)": np.int16,
    "tensor(int8)": np.int8,
    "tensor(uint64)": np.uint64,
    "tensor(uint32)": np.uint32,
    "tensor(uint16)": np.uint16,
    "tensor(uint8)": np.uint8,
    "tensor(bool)": np.bool_,
}


@dataclass
class BenchmarkResult:
    provider: str
    threads: int | None
    session_load_ms: float
    median_ms: float
    p95_ms: float
    mean_ms: float
    throughput_per_s: float
    baseline_rss_bytes: int
    loaded_rss_bytes: int
    peak_rss_bytes: int
    model_rss_delta_bytes: int
    peak_gpu_delta_bytes: int | None
    profile_file: str | None
    error: str | None = None


def human_bytes(value: int | float | None) -> str:
    if value is None:
        return "não medido"
    value = float(value)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    for unit in units:
        if abs(value) < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TiB"


def ceil_gib(value_bytes: float, minimum: int = 1) -> int:
    return max(minimum, math.ceil(value_bytes / (1024**3)))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, q))


def parse_shape_args(items: list[str]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Shape inválido: {item!r}. Use nome=1,3,224,224")
        name, dims_text = item.split("=", 1)
        dims = [int(x.strip()) for x in dims_text.split(",") if x.strip()]
        if not name.strip() or not dims or any(d <= 0 for d in dims):
            raise ValueError(f"Shape inválido: {item!r}")
        result[name.strip()] = dims
    return result


def tensor_dims(value_info: Any) -> list[int | str | None]:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return []
    dims: list[int | str | None] = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value") and dim.dim_value > 0:
            dims.append(int(dim.dim_value))
        elif dim.HasField("dim_param") and dim.dim_param:
            dims.append(dim.dim_param)
        else:
            dims.append(None)
    return dims


def model_disk_size(model_path: Path, model: onnx.ModelProto) -> int:
    total = model_path.stat().st_size
    seen: set[Path] = {model_path.resolve()}
    for initializer in model.graph.initializer:
        if initializer.data_location != TensorProto.EXTERNAL:
            continue
        entries = {entry.key: entry.value for entry in initializer.external_data}
        location = entries.get("location")
        if not location:
            continue
        external = (model_path.parent / location).resolve()
        if external not in seen and external.exists():
            total += external.stat().st_size
            seen.add(external)
    return total


def inspect_model(model_path: Path) -> tuple[onnx.ModelProto, dict[str, Any]]:
    # For models larger than 2 GB, path-based checking avoids loading the full model in memory.
    onnx.checker.check_model(str(model_path), full_check=False)
    model = onnx.load(str(model_path), load_external_data=False)

    inferred = model
    shape_inference_warning = None
    try:
        inferred = onnx.shape_inference.infer_shapes(model, check_type=False, strict_mode=False)
    except Exception as exc:  # shape inference is best-effort
        shape_inference_warning = str(exc)

    initializer_names = {x.name for x in model.graph.initializer}
    inputs = []
    for value in inferred.graph.input:
        if value.name in initializer_names:
            continue
        elem_type = value.type.tensor_type.elem_type
        inputs.append(
            {
                "name": value.name,
                "shape": tensor_dims(value),
                "dtype": TensorProto.DataType.Name(elem_type)
                if elem_type
                else "UNKNOWN",
            }
        )

    outputs = []
    for value in inferred.graph.output:
        elem_type = value.type.tensor_type.elem_type
        outputs.append(
            {
                "name": value.name,
                "shape": tensor_dims(value),
                "dtype": TensorProto.DataType.Name(elem_type)
                if elem_type
                else "UNKNOWN",
            }
        )

    parameter_count = 0
    weight_bytes = 0
    dtype_hist = Counter()
    for tensor in model.graph.initializer:
        elements = math.prod(tensor.dims) if tensor.dims else 1
        parameter_count += elements
        bytes_per_value = DTYPE_BYTES.get(tensor.data_type, 0)
        weight_bytes += elements * bytes_per_value
        dtype_hist[TensorProto.DataType.Name(tensor.data_type)] += elements

    op_hist = Counter(node.op_type for node in model.graph.node)
    opsets = {
        (item.domain or "ai.onnx"): item.version for item in model.opset_import
    }

    info = {
        "model_path": str(model_path.resolve()),
        "onnx_ir_version": model.ir_version,
        "producer_name": model.producer_name,
        "producer_version": model.producer_version,
        "opsets": opsets,
        "graph_name": model.graph.name,
        "nodes": len(model.graph.node),
        "operator_histogram": dict(op_hist.most_common()),
        "parameters": parameter_count,
        "weight_bytes_estimate": weight_bytes,
        "disk_bytes": model_disk_size(model_path, model),
        "weight_dtype_histogram": dict(dtype_hist),
        "inputs": inputs,
        "outputs": outputs,
        "dynamic_inputs": [
            x["name"]
            for x in inputs
            if any(not isinstance(d, int) or d <= 0 for d in x["shape"])
        ],
        "shape_inference_warning": shape_inference_warning,
    }
    return model, info


def build_inputs(
    session: ort.InferenceSession,
    shape_overrides: dict[str, list[int]],
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, list[int]]]:
    rng = np.random.default_rng(seed)
    feeds: dict[str, np.ndarray] = {}
    resolved_shapes: dict[str, list[int]] = {}

    for inp in session.get_inputs():
        shape: list[int] = []
        override = shape_overrides.get(inp.name)
        for index, dim in enumerate(inp.shape):
            if override is not None:
                if len(override) != len(inp.shape):
                    raise ValueError(
                        f"{inp.name}: the model expects {len(inp.shape)} dimensions, "
                        f"but --shape received {len(override)}."
                    )
                shape = list(override)
                break
            if isinstance(dim, int) and dim > 0:
                shape.append(dim)
            else:
                raise ValueError(
                    f"Dynamic input {inp.name!r}, shape={inp.shape}. "
                    f"Please provide --shape {inp.name}=..."
                )

        dtype = ORT_TO_NUMPY.get(inp.type)
        if dtype is None:
            raise TypeError(
                f"Input type is not yet supported by the generator: "
                f"{inp.name} ({inp.type})."
            )

        # Safe defaults for most benchmarks. For models that depend on content,
        # real feeds can be accepted in a future iteration.
        if np.issubdtype(dtype, np.floating):
            data = rng.standard_normal(shape).astype(dtype)
        elif dtype == np.bool_:
            data = np.ones(shape, dtype=dtype)
        elif np.issubdtype(dtype, np.integer):
            lowered = inp.name.lower()
            if "mask" in lowered:
                data = np.ones(shape, dtype=dtype)
            elif "token" in lowered or "input_ids" in lowered:
                data = rng.integers(0, 1000, size=shape, dtype=dtype)
            else:
                data = np.zeros(shape, dtype=dtype)
        else:
            raise TypeError(f"Could not generate data for {inp.name}: {dtype}")

        feeds[inp.name] = data
        resolved_shapes[inp.name] = shape

    unknown = set(shape_overrides) - {x.name for x in session.get_inputs()}
    if unknown:
        raise ValueError(f"--shape contém entradas inexistentes: {sorted(unknown)}")

    return feeds, resolved_shapes


class ResourceSampler:
    def __init__(self, interval_s: float = 0.005) -> None:
        self.interval_s = interval_s
        self.process = psutil.Process(os.getpid())
        self.peak_rss = self.process.memory_info().rss
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self.gpu_available = False
        self.gpu_handle = None
        self.gpu_initial_used = 0
        self.gpu_peak_used = 0
        self._nvml = None

        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._nvml = pynvml
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.gpu_initial_used = int(
                pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle).used
            )
            self.gpu_peak_used = self.gpu_initial_used
            self.gpu_available = True
        except Exception:
            self.gpu_available = False

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                self.peak_rss = max(
                    self.peak_rss, self.process.memory_info().rss
                )
            except psutil.Error:
                pass

            if self.gpu_available and self._nvml and self.gpu_handle:
                try:
                    used = int(
                        self._nvml.nvmlDeviceGetMemoryInfo(self.gpu_handle).used
                    )
                    self.gpu_peak_used = max(self.gpu_peak_used, used)
                except Exception:
                    pass
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self) -> tuple[int, int | None]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        gpu_delta = (
            max(0, self.gpu_peak_used - self.gpu_initial_used)
            if self.gpu_available
            else None
        )
        if self._nvml:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
        return self.peak_rss, gpu_delta


def provider_chain(requested: str) -> list[str]:
    available = ort.get_available_providers()
    if requested == "cpu":
        return ["CPUExecutionProvider"]
    if requested == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDAExecutionProvider is not available. "
                "Install onnxruntime-gpu and compatible CUDA/cuDNN dependencies."
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if requested == "tensorrt":
        if "TensorrtExecutionProvider" not in available:
            raise RuntimeError("TensorrtExecutionProvider is not available.")
        return [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
    if "TensorrtExecutionProvider" in available:
        return [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def run_benchmark(
    model_path: Path,
    requested_provider: str,
    threads: int | None,
    shape_overrides: dict[str, list[int]],
    warmup: int,
    runs: int,
    seed: int,
    enable_profile: bool,
) -> tuple[BenchmarkResult, dict[str, list[int]]]:
    process = psutil.Process(os.getpid())
    baseline_rss = process.memory_info().rss

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads is not None:
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
    if enable_profile:
        options.enable_profiling = True
        options.profile_file_prefix = "onnx_requirements_profile"

    providers = provider_chain(requested_provider)
    start_load = time.perf_counter()
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=providers,
    )
    load_ms = (time.perf_counter() - start_load) * 1000
    loaded_rss = process.memory_info().rss

    feeds, resolved_shapes = build_inputs(session, shape_overrides, seed)

    for _ in range(warmup):
        session.run(None, feeds)

    sampler = ResourceSampler()
    sampler.start()
    timings_ms: list[float] = []
    try:
        for _ in range(runs):
            started = time.perf_counter()
            session.run(None, feeds)
            timings_ms.append((time.perf_counter() - started) * 1000)
    finally:
        peak_rss, gpu_delta = sampler.stop()

    profile_file = session.end_profiling() if enable_profile else None
    median_ms = statistics.median(timings_ms)
    result = BenchmarkResult(
        provider=session.get_providers()[0],
        threads=threads,
        session_load_ms=load_ms,
        median_ms=median_ms,
        p95_ms=percentile(timings_ms, 95),
        mean_ms=statistics.fmean(timings_ms),
        throughput_per_s=1000.0 / median_ms if median_ms > 0 else float("inf"),
        baseline_rss_bytes=baseline_rss,
        loaded_rss_bytes=loaded_rss,
        peak_rss_bytes=peak_rss,
        model_rss_delta_bytes=max(0, peak_rss - baseline_rss),
        peak_gpu_delta_bytes=gpu_delta,
        profile_file=profile_file,
    )
    return result, resolved_shapes


def choose_thread_candidates() -> list[int]:
    physical = psutil.cpu_count(logical=False) or 1
    logical = psutil.cpu_count(logical=True) or physical
    candidates = {1, max(1, physical // 2), physical, logical}
    return sorted(candidates)


def classify_benchmarks(
    benchmarks: list[BenchmarkResult],
    target_latency_ms: float | None,
) -> list[dict[str, Any]]:
    valid = [b for b in benchmarks if b.error is None and math.isfinite(b.median_ms)]
    if not valid:
        return []

    best = min(b.median_ms for b in valid)
    rows = []
    for b in sorted(valid, key=lambda x: x.median_ms, reverse=True):
        if target_latency_ms is not None:
            ratio = b.p95_ms / target_latency_ms
            if ratio <= 0.6:
                label = "best / ample headroom"
            elif ratio <= 1.0:
                label = "recommended"
            elif ratio <= 2.0:
                label = "usable"
            else:
                label = "minimum functional"
        else:
            ratio = b.median_ms / best
            if ratio <= 1.05:
                label = "best measured"
            elif ratio <= 1.25:
                label = "recommended"
            elif ratio <= 2.0:
                label = "usable"
            else:
                label = "minimum functional"

        rows.append(
            {
                "provider": b.provider,
                "threads": b.threads,
                "classification": label,
                "median_ms": b.median_ms,
                "p95_ms": b.p95_ms,
                "throughput_per_s": b.throughput_per_s,
            }
        )
    return rows


def estimate_requirements(
    static: dict[str, Any],
    benchmarks: list[BenchmarkResult],
) -> dict[str, Any]:
    valid = [b for b in benchmarks if b.error is None]
    if not valid:
        return {}

    fastest = min(valid, key=lambda b: b.median_ms)
    memory_reference = max(valid, key=lambda b: b.model_rss_delta_bytes)

    app_peak = max(
        memory_reference.model_rss_delta_bytes,
        int(static["weight_bytes_estimate"] * 1.10),
    )
    system = platform.system().lower()
    # Approximate reserve for the OS and services. This is intentionally heuristic.
    os_reserve = 4 * 1024**3 if system == "windows" else 2 * 1024**3

    min_app = int(app_peak * 1.20)
    recommended_app = int(app_peak * 1.50)
    min_system = min_app + os_reserve
    recommended_system = recommended_app + os_reserve + 2 * 1024**3

    gpu_values = [
        b.peak_gpu_delta_bytes
        for b in valid
        if b.peak_gpu_delta_bytes is not None and b.peak_gpu_delta_bytes > 0
    ]
    gpu_peak = max(gpu_values) if gpu_values else None

    disk = static["disk_bytes"]
    return {
        "measured_best": {
            "provider": fastest.provider,
            "threads": fastest.threads,
            "median_ms": fastest.median_ms,
            "p95_ms": fastest.p95_ms,
            "throughput_per_s": fastest.throughput_per_s,
        },
        "ram": {
            "measured_model_process_peak_bytes": app_peak,
            "minimum_app_budget_bytes": min_app,
            "recommended_app_budget_bytes": recommended_app,
            "minimum_system_ram_gib": ceil_gib(min_system, 4),
            "recommended_system_ram_gib": ceil_gib(recommended_system, 8),
            "note": (
                "Estimate includes a heuristic reserve for the operating system. "
                "Other programs, multiple sessions, and concurrency require more RAM."
            ),
        },
        "vram": {
            "measured_peak_delta_bytes": gpu_peak,
            "minimum_vram_gib": ceil_gib(gpu_peak * 1.20, 1)
            if gpu_peak
            else None,
            "recommended_vram_gib": ceil_gib(gpu_peak * 1.50, 1)
            if gpu_peak
            else None,
            "note": (
                "The NVIDIA measurement is the increase in total GPU memory and may include "
                "other processes; on WDDM it may not be available."
            ),
        },
        "storage": {
            "model_and_external_data_bytes": disk,
            "minimum_free_bytes": int(disk * 1.20),
            "recommended_free_bytes": int(disk * 2.50),
            "note": (
                "The recommended headroom allows optimized models, caches, and temporary files; "
                "TensorRT may create additional engine cache."
            ),
        },
    }


def optimization_notes(static: dict[str, Any]) -> list[str]:
    notes = []
    dtypes = static["weight_dtype_histogram"]
    if dtypes.get("FLOAT", 0):
        notes.append(
            "The model contains FP32 weights. FP16 can roughly halve the weight footprint; "
            "INT8 may approach 1/4, but accuracy, operator support, and real gains must "
            "be validated."
        )
    if static["dynamic_inputs"]:
        notes.append(
            "The model has dynamic dimensions. Run the report for each important real-world "
            "scenario such as batch size, resolution, and sequence length."
        )
    if any(op in static["operator_histogram"] for op in ("Loop", "Scan", "If")):
        notes.append(
            "The graph contains control flow. Random inputs may not exercise the worst-case "
            "path; use real data for a reliable report."
        )
    notes.append(
        "For GPU, compare CUDA/TensorRT with CPU: unsupported operators may fall back to "
        "CPU, causing transfers and performance loss."
    )
    return notes


def markdown_report(report: dict[str, Any]) -> str:
    s = report["static"]
    req = report.get("requirements", {})
    lines = [
        "# ONNX Requirements Report",
        "",
        "## Model",
        f"- File: `{s['model_path']}`",
        f"- Size on disk: **{human_bytes(s['disk_bytes'])}**",
        f"- Estimated weights: **{human_bytes(s['weight_bytes_estimate'])}**",
        f"- Parameters: **{s['parameters']:,}**",
        f"- Nodes: **{s['nodes']:,}**",
        f"- Opsets: `{json.dumps(s['opsets'], ensure_ascii=False)}`",
        "",
        "## Tested Inputs",
    ]

    for name, shape in report.get("resolved_shapes", {}).items():
        lines.append(f"- `{name}`: `{shape}`")

    lines += ["", "## Local Benchmarks", ""]
    lines.append(
        "| Provider | Threads | Median | P95 | Inferences/s | Peak model RAM | Peak VRAM |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for b in report["benchmarks"]:
        if b.get("error"):
            lines.append(
                f"| {b['provider']} | {b['threads']} | erro | erro | erro | erro | erro |"
            )
            continue
        lines.append(
            "| {provider} | {threads} | {median:.2f} ms | {p95:.2f} ms | "
            "{ips:.2f} | {ram} | {vram} |".format(
                provider=b["provider"],
                threads=b["threads"] if b["threads"] is not None else "auto",
                median=b["median_ms"],
                p95=b["p95_ms"],
                ips=b["throughput_per_s"],
                ram=human_bytes(b["model_rss_delta_bytes"]),
                vram=human_bytes(b["peak_gpu_delta_bytes"]),
            )
        )

    lines += ["", "## Performance Tiers", ""]
    for item in report["performance_tiers"]:
        lines.append(
            f"- **{item['classification']}** — {item['provider']}, "
            f"{item['threads'] or 'auto'} threads: "
            f"{item['median_ms']:.2f} ms mediana, "
            f"{item['p95_ms']:.2f} ms P95, "
            f"{item['throughput_per_s']:.2f} inferences/s."
        )

    if req:
        ram = req["ram"]
        vram = req["vram"]
        storage = req["storage"]
        lines += [
            "",
            "## Estimated Requirements",
            "",
            f"- **Minimum system RAM:** {ram['minimum_system_ram_gib']} GiB",
            f"- **Recommended system RAM:** {ram['recommended_system_ram_gib']} GiB",
            f"- **Minimum process budget:** {human_bytes(ram['minimum_app_budget_bytes'])}",
            f"- **Recommended process budget:** {human_bytes(ram['recommended_app_budget_bytes'])}",
            (
                f"- **Minimum VRAM:** {vram['minimum_vram_gib']} GiB"
                if vram["minimum_vram_gib"] is not None
                else "- **VRAM:** not measured; the benchmark did not use a monitorable NVIDIA GPU."
            ),
            (
                f"- **Recommended VRAM:** {vram['recommended_vram_gib']} GiB"
                if vram["recommended_vram_gib"] is not None
                else ""
            ),
            f"- **Minimum free space:** {human_bytes(storage['minimum_free_bytes'])}",
            f"- **Recommended free space:** {human_bytes(storage['recommended_free_bytes'])}",
        ]

    lines += [
        "",
        "## Limitations and Recommendations",
        "",
    ]
    for note in report["notes"]:
        lines.append(f"- {note}")

    lines += [
        "",
        "> CPU/GPU requirements for other machines cannot be derived precisely from the ONNX file alone. "
        "This report measures the current machine. To recommend specific hardware targets, maintain a "
        "database of real CPU/GPU benchmarks and compare the same input scenario.",
        "",
    ]
    return "\n".join(line for line in lines if line != "" or True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyzes and benchmarks the requirements of an ONNX model."
    )
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--shape",
        action="append",
        default=[],
        help="Dynamic input shape: name=1,3,224,224. Can be repeated.",
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "cpu", "cuda", "tensorrt"],
        default="auto",
    )
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--threads",
        default="sweep",
        help="'sweep', 'auto', or an integer. Sweep tests several CPU thread counts.",
    )
    parser.add_argument("--target-latency-ms", type=float)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--json", type=Path, default=Path("onnx_requirements.json"))
    parser.add_argument(
        "--markdown", type=Path, default=Path("onnx_requirements.md")
    )
    args = parser.parse_args()

    if not args.model.exists():
        parser.error(f"Model not found: {args.model}")
    if args.runs < 3 or args.warmup < 0:
        parser.error("--runs must be >= 3 and --warmup >= 0")

    shape_overrides = parse_shape_args(args.shape)
    _, static = inspect_model(args.model)

    requested_provider = args.provider
    if requested_provider != "cpu":
        thread_candidates: list[int | None] = [None]
    elif args.threads == "sweep":
        thread_candidates = choose_thread_candidates()
    elif args.threads == "auto":
        thread_candidates = [None]
    else:
        value = int(args.threads)
        if value <= 0:
            parser.error("--threads must be positive")
        thread_candidates = [value]

    # In auto mode without GPU, sweep CPU to show the performance curve.
    if requested_provider == "auto":
        available = ort.get_available_providers()
        has_gpu = any(
            x in available
            for x in ("TensorrtExecutionProvider", "CUDAExecutionProvider")
        )
        if not has_gpu and args.threads == "sweep":
            requested_provider = "cpu"
            thread_candidates = choose_thread_candidates()

    benchmarks: list[BenchmarkResult] = []
    resolved_shapes: dict[str, list[int]] = {}
    for threads in thread_candidates:
        try:
            result, shapes = run_benchmark(
                model_path=args.model,
                requested_provider=requested_provider,
                threads=threads,
                shape_overrides=shape_overrides,
                warmup=args.warmup,
                runs=args.runs,
                seed=args.seed,
                enable_profile=args.profile,
            )
            resolved_shapes = shapes
        except Exception as exc:
            result = BenchmarkResult(
                provider=requested_provider,
                threads=threads,
                session_load_ms=float("nan"),
                median_ms=float("nan"),
                p95_ms=float("nan"),
                mean_ms=float("nan"),
                throughput_per_s=float("nan"),
                baseline_rss_bytes=0,
                loaded_rss_bytes=0,
                peak_rss_bytes=0,
                model_rss_delta_bytes=0,
                peak_gpu_delta_bytes=None,
                profile_file=None,
                error=f"{type(exc).__name__}: {exc}",
            )
        benchmarks.append(result)

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": {
            "os": platform.platform(),
            "python": platform.python_version(),
            "cpu": platform.processor() or platform.machine(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "system_ram_bytes": psutil.virtual_memory().total,
            "onnx_version": onnx.__version__,
            "onnxruntime_version": ort.__version__,
            "available_providers": ort.get_available_providers(),
        },
        "static": static,
        "resolved_shapes": resolved_shapes,
        "benchmarks": [asdict(x) for x in benchmarks],
        "performance_tiers": classify_benchmarks(
            benchmarks, args.target_latency_ms
        ),
        "requirements": estimate_requirements(static, benchmarks),
        "notes": optimization_notes(static),
        "methodology": {
            "minimum_memory_multiplier": 1.20,
            "recommended_memory_multiplier": 1.50,
            "benchmark_inputs": "synthetic",
            "warning": (
                "Synthetic measurements do not replace benchmarks with real inputs "
                "and a full pipeline."
            ),
        },
    }

    args.json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    args.markdown.write_text(markdown_report(report), encoding="utf-8")

    print(markdown_report(report))
    print(f"\nJSON: {args.json.resolve()}")
    print(f"Markdown: {args.markdown.resolve()}")

    errors = [b.error for b in benchmarks if b.error]
    if errors and len(errors) == len(benchmarks):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
