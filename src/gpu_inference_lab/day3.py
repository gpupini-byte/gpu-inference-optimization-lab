"""Day 3 fair comparison of CUDA PyTorch and ONNX Runtime I/O Binding."""
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image

from .data import make_inputs
from .day2 import _export_onnx, compare_outputs, measure_data_transfer
from .environment import collect_environment
from .models import load_resnet18


@dataclass(frozen=True)
class BenchmarkConfig:
    batch_sizes: tuple[int, ...] = (1, 8, 32)
    runs: int = 3
    warmup_runs: int = 10
    repeats: int = 30
    seed: int = 2025

    def validate(self) -> None:
        if self.batch_sizes != (1, 8, 32):
            raise ValueError("Day 3 comparison uses the fixed batch set (1, 8, 32)")
        if self.runs < 1:
            raise ValueError("runs must be at least 1")
        if self.warmup_runs < 0:
            raise ValueError("warmup_runs cannot be negative")
        if self.repeats < 2:
            raise ValueError("repeats must be at least 2")


def _sync() -> None:
    torch.cuda.synchronize()


def _summarize(samples_ms: list[float], batch_size: int) -> dict[str, Any]:
    avg = mean(samples_ms)
    return {
        "mean_ms": avg,
        "median_ms": median(samples_ms),
        "p95_ms": float(np.percentile(samples_ms, 95)),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "samples_ms": samples_ms,
        "throughput_images_per_second": batch_size / (avg / 1_000),
    }


def _measure_series(
    call: Callable[[], Any],
    batch_size: int,
    config: BenchmarkConfig,
    measure_torch_memory: bool,
) -> tuple[dict[str, Any], Any]:
    """Run first-call timing, then independent warmup/sample rounds."""
    runs: list[dict[str, Any]] = []
    last_output: Any = None
    for run_index in range(config.runs):
        _sync()
        baseline = torch.cuda.memory_allocated() if measure_torch_memory else None
        if measure_torch_memory:
            torch.cuda.reset_peak_memory_stats()
        started = perf_counter()
        last_output = call()
        _sync()
        first_call_ms = (perf_counter() - started) * 1_000

        for _ in range(config.warmup_runs):
            last_output = call()
        _sync()

        samples: list[float] = []
        for _ in range(config.repeats):
            _sync()
            started = perf_counter()
            last_output = call()
            _sync()
            samples.append((perf_counter() - started) * 1_000)
        row = {"run": run_index + 1, "first_call_ms": first_call_ms, **_summarize(samples, batch_size)}
        if measure_torch_memory:
            row["baseline_allocated_bytes"] = baseline
            row["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            row["peak_increment_bytes"] = max(0, row["peak_allocated_bytes"] - int(baseline or 0))
        else:
            row["baseline_allocated_bytes"] = None
            row["peak_allocated_bytes"] = None
            row["peak_increment_bytes"] = None
        runs.append(row)

    samples = [value for row in runs for value in row["samples_ms"]]
    aggregate = {
        **_summarize(samples, batch_size),
        "runs": runs,
        "independent_run_count": len(runs),
        "per_run_mean_ms": [row["mean_ms"] for row in runs],
        "first_call_each_run_ms": [row["first_call_ms"] for row in runs],
        "peak_allocated_bytes_by_run": [row["peak_allocated_bytes"] for row in runs],
        "peak_increment_bytes_by_run": [row["peak_increment_bytes"] for row in runs],
        "peak_allocated_bytes": max((row["peak_allocated_bytes"] or 0 for row in runs), default=0) if measure_torch_memory else None,
    }
    return aggregate, last_output


def _torch_call(model: torch.nn.Module, inputs: torch.Tensor, fp16: bool) -> torch.Tensor:
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=fp16):
        return model(inputs)


def _to_cpu_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output.detach().float().cpu()
    if hasattr(output, "numpy"):
        return torch.from_numpy(output.numpy()).float()
    return torch.from_numpy(np.asarray(output)).float()


def _bind_cuda_io(session: Any, input_name: str, output_name: str, inputs_cuda: torch.Tensor) -> Any:
    binding = session.io_binding()
    binding.bind_input(
        name=input_name,
        device_type="cuda",
        device_id=0,
        element_type=np.float32,
        shape=tuple(inputs_cuda.shape),
        buffer_ptr=inputs_cuda.data_ptr(),
    )
    # ORT allocates a CUDA output OrtValue with the dynamically inferred shape.
    binding.bind_output(name=output_name, device_type="cuda", device_id=0)
    binding.synchronize_inputs()
    return binding


def _validate_real_images(model: torch.nn.Module, session: Any, input_name: str, output_name: str, manifest_path: Path) -> dict[str, Any]:
    from torchvision.models import ResNet18_Weights

    weights = ResNet18_Weights.DEFAULT
    transform = weights.transforms()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    images: list[torch.Tensor] = []
    records: list[dict[str, Any]] = []
    for item in manifest["images"]:
        image_path = manifest_path.parent / item["file"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Validation image missing: {image_path}. Run scripts/fetch_validation_images.py")
        images.append(transform(Image.open(image_path).convert("RGB")))
        records.append(item)
    batch = torch.stack(images).to("cuda")
    with torch.inference_mode():
        fp32_logits = _torch_call(model, batch, fp16=False)
        fp16_logits = _torch_call(model, batch, fp16=True)

    binding = _bind_cuda_io(session, input_name, output_name, batch)
    session.run_with_iobinding(binding)
    binding.synchronize_outputs()
    _sync()
    onnx_value = binding.get_outputs()[0]
    if onnx_value.device_name().lower() != "cuda":
        raise RuntimeError(f"Real-image ONNX output is not device-resident: {onnx_value.device_name()}")
    onnx_logits = _to_cpu_tensor(onnx_value)
    fp32_cpu = fp32_logits.detach().float().cpu()
    fp16_cpu = fp16_logits.detach().float().cpu()
    categories = weights.meta["categories"]

    per_image = []
    for index, item in enumerate(records):
        expected = int(item["expected_imagenet_index"])
        ref_pred = int(fp32_cpu[index].argmax().item())
        fp16_pred = int(fp16_cpu[index].argmax().item())
        onnx_pred = int(onnx_logits[index].argmax().item())
        per_image.append({
            "file": item["file"],
            "expected_label": item["expected_imagenet_label"],
            "expected_index": expected,
            "pytorch_fp32_prediction": categories[ref_pred],
            "pytorch_fp32_index": ref_pred,
            "pytorch_fp32_correct": ref_pred == expected,
            "pytorch_fp16_prediction": categories[fp16_pred],
            "pytorch_fp16_index": fp16_pred,
            "pytorch_fp16_correct": fp16_pred == expected,
            "pytorch_fp16_agrees_with_fp32": fp16_pred == ref_pred,
            "onnx_cuda_prediction": categories[onnx_pred],
            "onnx_cuda_index": onnx_pred,
            "onnx_cuda_correct": onnx_pred == expected,
            "onnx_cuda_agrees_with_fp32": onnx_pred == ref_pred,
        })
    total = len(per_image)
    if total == 0:
        raise ValueError("Real-image manifest is empty")
    return {
        "dataset_name": manifest["dataset_name"],
        "count": total,
        "purpose": manifest["purpose"],
        "prediction_agreement_vs_fp32": {
            "pytorch_fp16": sum(x["pytorch_fp16_agrees_with_fp32"] for x in per_image) / total,
            "onnx_cuda": sum(x["onnx_cuda_agrees_with_fp32"] for x in per_image) / total,
        },
        "top1_accuracy_on_this_subset": {
            "pytorch_fp32": sum(x["pytorch_fp32_correct"] for x in per_image) / total,
            "pytorch_fp16": sum(x["pytorch_fp16_correct"] for x in per_image) / total,
            "onnx_cuda": sum(x["onnx_cuda_correct"] for x in per_image) / total,
        },
        "per_image": per_image,
    }


def _create_session(onnx_path: Path, profile_path: Path) -> tuple[Any, dict[str, Any]]:
    import onnxruntime as ort

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(f"ONNX Runtime CUDA EP unavailable: {ort.get_available_providers()}")
    preload_started = perf_counter()
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls(directory="")
    preload_seconds = perf_counter() - preload_started
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = str(profile_path)
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    started = perf_counter()
    session = ort.InferenceSession(str(onnx_path), options, providers=[("CUDAExecutionProvider", {"device_id": 0})])
    init_seconds = perf_counter() - started
    providers = session.get_providers()
    if not providers or providers[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"CUDA EP is not the first active provider: {providers}")
    return session, {
        "version": ort.__version__,
        "providers_available": ort.get_available_providers(),
        "providers_active": providers,
        "provider_order_verified": True,
        "cpu_fallback_disabled": True,
        "dll_preload_seconds": preload_seconds,
        "session_initialization_seconds": init_seconds,
    }


def run_benchmark(output_dir: Path, config: BenchmarkConfig | None = None, include_real_images: bool = True) -> dict[str, Any]:
    config = config or BenchmarkConfig()
    config.validate()
    if not torch.cuda.is_available():
        raise RuntimeError("Milestone 3 requires CUDA for all three execution modes")

    environment = collect_environment()
    try:
        git_root = subprocess.run(["git", "rev-parse", "--show-toplevel"], check=True, capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        git_root = None
    output_dir.mkdir(parents=True, exist_ok=True)
    model, model_init_seconds = load_resnet18(torch.device("cuda"), pretrained=True)
    onnx_path = output_dir / "resnet18_milestone3.onnx"
    export_seconds = _export_onnx(model, make_inputs(1, seed=config.seed), onnx_path)
    session, ort_info = _create_session(onnx_path, output_dir / "onnxruntime_profile")
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    result: dict[str, Any] = {
        "schema_version": 3,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_context": {
            "working_directory": environment["working_directory"],
            "git_root": git_root,
            "canonical_project_root": git_root or environment["working_directory"],
        },
        "environment": environment,
        "model": {
            "architecture": "torchvision ResNet-18",
            "weights": "ResNet18_Weights.DEFAULT (ImageNet1K_V1)",
            "weights_dtype": "float32",
            "input_dtype": "float32",
            "input_shape_per_image": [3, 224, 224],
            "output_semantics": "1000 ImageNet logits, float32; no softmax",
            "weight_initialization_seconds": model_init_seconds,
            "onnx_export_seconds": export_seconds,
        },
        "configuration": asdict(config),
        "timing_scopes": {
            "device_resident": "Input tensor is already on CUDA; time forward/inference only; output remains on CUDA. CUDA synchronization brackets every sample.",
            "end_to_end": "Input begins as the same CPU tensor; include host-to-device transfer, inference, and output transfer to CPU. CUDA synchronization brackets every sample.",
            "separate_transfer": "Blocking CPU-to-CUDA copy plus allocation and synchronization, no inference; measured separately for each batch.",
            "first_call": "First call in each independent run and configuration; separate from warmup and steady samples.",
        },
        "methods": [],
        "onnx_runtime": {**ort_info, "model_file": str(onnx_path), "profile_node_provider_counts": None, "profile_verified_cuda_only": False},
        "real_image_validation": {"status": "not_requested"},
        "memory_measurement_note": "PyTorch peak allocated bytes report only the PyTorch caching allocator. ONNX Runtime allocates CUDA memory outside this counter; ORT peak GPU allocation is not claimed or compared.",
    }

    correctness_outputs: dict[tuple[str, str, int], torch.Tensor] = {}
    transfer_by_batch: dict[int, dict[str, Any]] = {}
    labels = {
        "device_resident": {
            "pytorch_fp32": "PyTorch FP32 (device-resident)",
            "pytorch_fp16": "PyTorch autocast FP16 (device-resident)",
            "onnx_fp32": "ONNX Runtime FP32 I/O Binding (device-resident)",
        },
        "end_to_end": {
            "pytorch_fp32": "PyTorch FP32 (CPU input to CPU output)",
            "pytorch_fp16": "PyTorch autocast FP16 (CPU input to CPU output)",
            "onnx_fp32": "ONNX Runtime FP32 session.run (CPU input to CPU output)",
        },
    }

    for batch_size in config.batch_sizes:
        cpu_inputs = make_inputs(batch_size, seed=config.seed, device="cpu").contiguous()
        transfer_by_batch[batch_size] = measure_data_transfer(cpu_inputs)
        cuda_inputs = cpu_inputs.to("cuda", non_blocking=False)
        _sync()

        device_specs = [
            ("pytorch_fp32", lambda: _torch_call(model, cuda_inputs, fp16=False), True),
            ("pytorch_fp16", lambda: _torch_call(model, cuda_inputs, fp16=True), True),
        ]
        for method, call, tracks_memory in device_specs:
            timing, last_output = _measure_series(call, batch_size, config, tracks_memory)
            cpu_output = _to_cpu_tensor(last_output)
            correctness_outputs[("device_resident", method, batch_size)] = cpu_output
            result["methods"].append({
                "scope": "device_resident",
                "method": method,
                "label": labels["device_resident"][method],
                "batch_size": batch_size,
                "input_shape": list(cuda_inputs.shape),
                "output_shape": list(cpu_output.shape),
                "timing_scope": result["timing_scopes"]["device_resident"],
                "standalone_transfer": transfer_by_batch[batch_size],
                "aggregate": timing,
                "gpu_peak_allocated_bytes": timing["peak_allocated_bytes"],
                "numerical_comparison": None,
            })

        binding_started = perf_counter()
        binding = _bind_cuda_io(session, input_name, output_name, cuda_inputs)
        binding_setup_seconds = perf_counter() - binding_started

        def ort_device_call() -> Any:
            session.run_with_iobinding(binding)
            binding.synchronize_outputs()
            _sync()
            outputs = binding.get_outputs()
            if not outputs or outputs[0].device_name().lower() != "cuda":
                raise RuntimeError("ONNX I/O Binding produced an output outside CUDA")
            return outputs[0]

        ort_timing, ort_output = _measure_series(ort_device_call, batch_size, config, measure_torch_memory=False)
        ort_cpu_output = _to_cpu_tensor(ort_output)
        correctness_outputs[("device_resident", "onnx_fp32", batch_size)] = ort_cpu_output
        result["methods"].append({
            "scope": "device_resident",
            "method": "onnx_fp32",
            "label": labels["device_resident"]["onnx_fp32"],
            "batch_size": batch_size,
            "input_shape": list(cuda_inputs.shape),
            "output_shape": list(ort_cpu_output.shape),
            "output_device_verified": "cuda",
            "io_binding_setup_seconds": binding_setup_seconds,
            "timing_scope": result["timing_scopes"]["device_resident"],
            "standalone_transfer": transfer_by_batch[batch_size],
            "aggregate": ort_timing,
            "gpu_peak_allocated_bytes": None,
            "numerical_comparison": None,
        })

        # Retain host copies for comparison, then free resident input/output
        # references before measuring the end-to-end paths' peak allocations.
        del binding, ort_output, ort_cpu_output, cuda_inputs
        _sync()

        def pytorch_e2e(fp16: bool) -> torch.Tensor:
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=fp16):
                inputs_cuda = cpu_inputs.to("cuda", non_blocking=False)
                logits = model(inputs_cuda)
                output_cpu = logits.to("cpu", copy=True)
            del inputs_cuda, logits
            return output_cpu

        def onnx_e2e() -> np.ndarray:
            return session.run([output_name], {input_name: cpu_inputs.numpy()})[0]

        for method, call, tracks_memory in (
            ("pytorch_fp32", lambda: pytorch_e2e(False), True),
            ("pytorch_fp16", lambda: pytorch_e2e(True), True),
            ("onnx_fp32", onnx_e2e, False),
        ):
            timing, last_output = _measure_series(call, batch_size, config, tracks_memory)
            cpu_output = _to_cpu_tensor(last_output)
            correctness_outputs[("end_to_end", method, batch_size)] = cpu_output
            result["methods"].append({
                "scope": "end_to_end",
                "method": method,
                "label": labels["end_to_end"][method],
                "batch_size": batch_size,
                "input_shape": [batch_size, 3, 224, 224],
                "output_shape": list(cpu_output.shape),
                "timing_scope": result["timing_scopes"]["end_to_end"],
                "standalone_transfer": transfer_by_batch[batch_size],
                "aggregate": timing,
                "gpu_peak_allocated_bytes": timing["peak_allocated_bytes"],
                "numerical_comparison": None,
            })

        for scope in ("device_resident", "end_to_end"):
            reference = correctness_outputs[(scope, "pytorch_fp32", batch_size)]
            for method in ("pytorch_fp16", "onnx_fp32"):
                candidate = correctness_outputs[(scope, method, batch_size)]
                comparison = compare_outputs(reference, candidate)
                row = next(item for item in result["methods"] if item["scope"] == scope and item["method"] == method and item["batch_size"] == batch_size)
                row["numerical_comparison"] = comparison
        del cpu_inputs

    if include_real_images:
        manifest_path = Path(__file__).resolve().parents[2] / "data" / "real_validation" / "manifest.json"
        try:
            real_validation = _validate_real_images(model, session, input_name, output_name, manifest_path)
            result["real_image_validation"] = {"status": "ok", **real_validation}
        except FileNotFoundError as exc:
            result["real_image_validation"] = {"status": "images_missing", "reason": str(exc)}

    profile_path = Path(session.end_profiling())
    profile_events = json.loads(profile_path.read_text(encoding="utf-8"))
    node_events = [event for event in profile_events if event.get("cat") == "Node" and "provider" in event.get("args", {})]
    provider_counts: dict[str, int] = {}
    for event in node_events:
        provider = event["args"]["provider"]
        provider_counts[provider] = provider_counts.get(provider, 0) + 1
    result["onnx_runtime"].update({
        "profile_node_provider_counts": provider_counts,
        "profile_path": str(profile_path),
        "profile_verified_cuda_only": bool(provider_counts.get("CUDAExecutionProvider")) and not provider_counts.get("CPUExecutionProvider"),
    })
    if not result["onnx_runtime"]["profile_verified_cuda_only"]:
        raise RuntimeError(f"ONNX profile did not show CUDA-only node placement: {provider_counts}")

    result["artifacts"] = {
        "json": "day3_benchmark.json",
        "report": "day3_report.md",
        "comparison_chart": "day3_comparison.png",
        "onnx_model": onnx_path.name,
        "onnx_profile": Path(result["onnx_runtime"]["profile_path"]).name,
        "real_image_manifest": "../../data/real_validation/manifest.json",
    }
    _write_chart(result, output_dir / "day3_comparison.png")
    (output_dir / "day3_benchmark.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    _write_report(result, output_dir / "day3_report.md")
    return result


def _write_chart(result: dict[str, Any], output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {
        "pytorch_fp32": ("PyTorch FP32", "tab:blue", "o"),
        "pytorch_fp16": ("PyTorch autocast FP16", "tab:orange", "s"),
        "onnx_fp32": ("ONNX Runtime FP32", "tab:green", "^"),
    }
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for scope, col, title in (("device_resident", 0, "Device-resident inference"), ("end_to_end", 1, "End-to-end CPU input/output")):
        rows = [row for row in result["methods"] if row["scope"] == scope]
        for method, (label, color, marker) in styles.items():
            subset = sorted((row for row in rows if row["method"] == method), key=lambda row: row["batch_size"])
            if not subset:
                continue
            batches = [row["batch_size"] for row in subset]
            means = [row["aggregate"]["mean_ms"] for row in subset]
            lower = [max(0, row["aggregate"]["mean_ms"] - row["aggregate"]["min_ms"]) for row in subset]
            upper = [max(0, row["aggregate"]["p95_ms"] - row["aggregate"]["mean_ms"]) for row in subset]
            axes[0, col].errorbar(batches, means, yerr=[lower, upper], label=label, color=color, marker=marker, capsize=3)
            throughputs = [row["aggregate"]["throughput_images_per_second"] for row in subset]
            axes[1, col].plot(batches, throughputs, label=label, color=color, marker=marker)
        axes[0, col].set_title(title)
        axes[0, col].set_xlabel("Batch size")
        axes[0, col].set_ylabel("Latency (ms), mean with min-to-p95 whiskers")
        axes[1, col].set_xlabel("Batch size")
        axes[1, col].set_ylabel("Throughput (images/s)")
        for ax in (axes[0, col], axes[1, col]):
            ax.set_xticks([1, 8, 32])
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
    figure.suptitle("ResNet-18 inference: matched weights and inputs, separated timing scopes")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _write_report(result: dict[str, Any], path: Path) -> None:
    env = result["environment"]
    lines = [
        "# GPU Inference Optimization Lab — Milestone 3", "",
        f"Run timestamp (UTC): `{result['created_at_utc']}`", "",
        "## Environment and protocol", "",
        f"- Python: `{env['python'].split()[0]}`; executable `{Path(env['python_executable']).name}`",
        f"- PyTorch: `{env['pytorch']}`; CUDA build `{env['torch_cuda_build']}`; cuDNN `{env['cudnn_version']}`",
        f"- GPU: `{env['devices'][0]['name']}` ({env['devices'][0]['total_memory_bytes'] // (1024 ** 2)} MiB, CC {env['devices'][0]['compute_capability']})",
        f"- CPU: `{env.get('cpu_name', 'unavailable')}`; driver `{(env.get('nvidia_smi') or {}).get('driver_version', 'unavailable')}`",
        f"- Git root: `{Path(result['repository_context']['git_root']).name if result['repository_context']['git_root'] else 'not initialized'}`",
        f"- Runs: {result['configuration']['runs']}; warmups per run/configuration: {result['configuration']['warmup_runs']}; timed calls per run/configuration: {result['configuration']['repeats']}.",
        "- Model initialization, ONNX export, ONNX session initialization, first calls, steady-state samples, and standalone H2D copies are separate fields in the JSON.", "",
        "All inference methods use one pretrained ResNet-18 checkpoint, the same fixed-seed float32 NCHW input values, batch sizes 1/8/32, and 1000-class float32 logits with no softmax. CUDA execution is synchronized before and after each timed sample.", "",
        "## Matched-scope comparison", "",
        "Device-resident timings exclude input and output transfers. PyTorch receives an existing CUDA tensor; ONNX Runtime uses I/O Binding with that same CUDA tensor pointer and an ORT-allocated CUDA output. End-to-end timings start with the same CPU tensor and finish with CPU logits; PyTorch explicitly copies input and output, while ONNX uses `session.run` and its CPU I/O path.", "",
        "### Device-resident inference", "",
        "| Method | Batch | Mean ms | Median ms | p95 ms | Images/s | PyTorch peak allocated MiB |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["methods"]:
        if row["scope"] != "device_resident":
            continue
        agg = row["aggregate"]
        peak = f"{agg['peak_allocated_bytes'] / (1024 ** 2):.2f}" if agg["peak_allocated_bytes"] is not None else "not visible to torch allocator"
        lines.append(f"| {row['label']} | {row['batch_size']} | {agg['mean_ms']:.4f} | {agg['median_ms']:.4f} | {agg['p95_ms']:.4f} | {agg['throughput_images_per_second']:.2f} | {peak} |")
    lines += ["", "### End-to-end CPU input to CPU output", "", "| Method | Batch | Mean ms | Median ms | p95 ms | Images/s | PyTorch peak allocated MiB |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in result["methods"]:
        if row["scope"] != "end_to_end":
            continue
        agg = row["aggregate"]
        peak = f"{agg['peak_allocated_bytes'] / (1024 ** 2):.2f}" if agg["peak_allocated_bytes"] is not None else "not visible to torch allocator"
        lines.append(f"| {row['label']} | {row['batch_size']} | {agg['mean_ms']:.4f} | {agg['median_ms']:.4f} | {agg['p95_ms']:.4f} | {agg['throughput_images_per_second']:.2f} | {peak} |")
    lines += ["", "![Matched timing scopes](day3_comparison.png)", "", "## Output comparison", "", "Prediction agreement compares argmax classes; it is distinct from numerical closeness. `allclose` uses rtol=0.01 and atol=0.01.", "", "| Scope | Batch | Method vs PyTorch FP32 | Allclose | Max absolute error | Mean absolute error | RMSE | Top-1 agreement |", "|---|---:|---|:---:|---:|---:|---:|---:|"]
    for row in result["methods"]:
        check = row.get("numerical_comparison")
        if not check:
            continue
        lines.append(f"| {row['scope']} | {row['batch_size']} | {row['method']} | {check['allclose']} | {check['max_absolute_error']:.6g} | {check['mean_absolute_error']:.6g} | {check['rmse']:.6g} | {check['top1_agreement']:.1%} |")

    onnx = result["onnx_runtime"]
    lines += ["", "## ONNX CUDA and memory accounting", "", f"ORT `{onnx['version']}` providers: `{onnx['providers_active']}`. CPU fallback disabled: `{onnx['cpu_fallback_disabled']}`. Profile assignment counts: `{onnx['profile_node_provider_counts']}`. CUDA-only profile verification: `{onnx['profile_verified_cuda_only']}`.", "", result["memory_measurement_note"], ""]

    validation = result["real_image_validation"]
    lines += ["## Real-image smoke validation", ""]
    if validation["status"] == "ok":
        lines.append(f"Set: `{validation['dataset_name']}` ({validation['count']} images). Prediction agreement and top-1 accuracy are different measurements:")
        lines.append(f"- FP16 agreement with PyTorch FP32: {validation['prediction_agreement_vs_fp32']['pytorch_fp16']:.1%}; exact-label accuracy on this tiny subset: {validation['top1_accuracy_on_this_subset']['pytorch_fp16']:.1%}.")
        lines.append(f"- ONNX CUDA agreement with PyTorch FP32: {validation['prediction_agreement_vs_fp32']['onnx_cuda']:.1%}; exact-label accuracy on this tiny subset: {validation['top1_accuracy_on_this_subset']['onnx_cuda']:.1%}.")
        lines.append("")
        lines.append("| Image | Expected label | PyTorch FP32 prediction | FP16 prediction | ONNX prediction |")
        lines.append("|---|---|---|---|---|")
        for item in validation["per_image"]:
            lines.append(f"| {item['file']} | {item['expected_label']} | {item['pytorch_fp32_prediction']} | {item['pytorch_fp16_prediction']} | {item['onnx_cuda_prediction']} |")
    elif validation["status"] == "images_missing":
        lines.append(f"Not run: {validation['reason']}")
    else:
        lines.append("Not requested.")
    lines += ["", "This three-image public-domain/CC0 smoke set checks preprocessing and semantic output flow only. It is not a meaningful estimate of ImageNet accuracy. Image sources and license evidence are listed in `data/real_validation/README.md`.", "", "## Interpretation and limitations", "", "Use device-resident comparisons to study framework/model execution; use end-to-end comparisons to study a CPU-fed, CPU-consumed request. The I/O Binding output allocation is owned by ONNX Runtime. PyTorch peak allocated memory omits ORT's CUDA allocator, CUDA context, and other non-PyTorch allocations, so there is no directly comparable ORT peak number here. The small batch-one path can be dispatch-bound. Results are specific to this laptop, driver, thermal/power state, software versions, and synthetic benchmark inputs. No performance improvement is assumed; inspect the recorded distributions.", "", "## Reproduce", "", "```powershell", r".\.venv\Scripts\python.exe -m pytest", r".\.venv\Scripts\python.exe scripts/fetch_validation_images.py", r".\.venv\Scripts\python.exe -m gpu_inference_lab.day3 --output-dir results/milestone3 --runs 3 --warmup-runs 10 --repeats 30", "```", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Matched PyTorch/ONNX CUDA validation using I/O Binding")
    parser.add_argument("--output-dir", type=Path, default=Path("results/milestone3"))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--skip-real-images", action="store_true")
    args = parser.parse_args()
    config = BenchmarkConfig(runs=args.runs, warmup_runs=args.warmup_runs, repeats=args.repeats, seed=args.seed)
    result = run_benchmark(args.output_dir, config, include_real_images=not args.skip_real_images)
    print(json.dumps({"json": str(args.output_dir / "day3_benchmark.json"), "report": str(args.output_dir / "day3_report.md"), "cuda_only_profile": result["onnx_runtime"]["profile_verified_cuda_only"], "real_image_status": result["real_image_validation"]["status"]}, indent=2))


if __name__ == "__main__":
    main()
