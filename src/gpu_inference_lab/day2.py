"""Day 2 precision and ONNX CUDA benchmark, correctness checks, and artifacts."""
from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any, Callable

import numpy as np
import torch

from .data import make_inputs
from .environment import collect_environment
from .models import load_resnet18


def _sync() -> None:
    torch.cuda.synchronize()


def _summary(samples_ms: list[float], batch_size: int) -> dict[str, Any]:
    ordered = sorted(samples_ms)
    avg = mean(samples_ms)
    return {
        "mean_ms": avg,
        "median_ms": median(samples_ms),
        "p95_ms": float(np.percentile(ordered, 95)),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "samples_ms": samples_ms,
        "throughput_images_per_second": batch_size / (avg / 1_000),
    }


def measure_data_transfer(inputs_cpu: torch.Tensor, warmup_runs: int = 3, repeats: int = 20) -> dict[str, Any]:
    """Measure blocking CPU-to-CUDA tensor copies, including allocation and sync."""
    samples: list[float] = []
    for _ in range(warmup_runs):
        copied = inputs_cpu.to("cuda", non_blocking=False)
        _sync()
        del copied
    for _ in range(repeats):
        _sync()
        started = perf_counter()
        copied = inputs_cpu.to("cuda", non_blocking=False)
        _sync()
        samples.append((perf_counter() - started) * 1_000)
        del copied
    return {
        **_summary(samples, int(inputs_cpu.shape[0])),
        "warmup_runs": warmup_runs,
        "scope": "blocking host-to-device copy plus destination allocation; synchronized; no model inference",
    }


def _torch_forward(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    use_fp16: bool,
) -> torch.Tensor:
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
            return model(inputs)


def benchmark_torch_run(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    batch_size: int,
    use_fp16: bool,
    warmup_runs: int,
    repeats: int,
) -> tuple[dict[str, Any], torch.Tensor]:
    """Measure one independent warmup+timing run and return one result tensor."""
    torch.cuda.reset_peak_memory_stats()
    _sync()
    baseline_bytes = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    result = _torch_forward(model, inputs, use_fp16)
    _sync()
    first_forward_ms = (perf_counter() - started) * 1_000
    first_forward_peak = torch.cuda.max_memory_allocated()

    steady_baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup_runs):
        result = _torch_forward(model, inputs, use_fp16)
    _sync()

    samples: list[float] = []
    for _ in range(repeats):
        _sync()
        started = perf_counter()
        result = _torch_forward(model, inputs, use_fp16)
        _sync()
        samples.append((perf_counter() - started) * 1_000)

    summary = _summary(samples, batch_size)
    summary.update({
        "warmup_runs": warmup_runs,
        "first_forward_ms": first_forward_ms,
        "first_forward_peak_allocated_bytes": first_forward_peak,
        "steady_baseline_allocated_bytes": steady_baseline,
        "steady_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "steady_peak_increment_bytes": max(0, torch.cuda.max_memory_allocated() - steady_baseline),
        "scope": "model forward only, input preallocated on CUDA, output remains on CUDA; synchronization outside every timed call; autocast enabled only for FP16 configuration",
    })
    # baseline_bytes is retained for an explicit audit check in output.
    summary["pre_first_forward_allocated_bytes"] = baseline_bytes
    return summary, result


def compare_outputs(reference: torch.Tensor, candidate: torch.Tensor, rtol: float = 1e-2, atol: float = 1e-2) -> dict[str, Any]:
    ref = reference.detach().float().cpu()
    cand = candidate.detach().float().cpu()
    difference = (cand - ref).abs()
    relative = difference / ref.abs().clamp_min(1e-8)
    rmse = float(torch.sqrt(torch.mean((cand - ref) ** 2)).item())
    return {
        "rtol": rtol,
        "atol": atol,
        "allclose": bool(torch.allclose(cand, ref, rtol=rtol, atol=atol)),
        "max_absolute_error": float(difference.max().item()),
        "mean_absolute_error": float(difference.mean().item()),
        "rmse": rmse,
        "max_relative_error": float(relative.max().item()),
        "top1_agreement": float((cand.argmax(1) == ref.argmax(1)).float().mean().item()),
        "reference_top1": ref.argmax(1).tolist(),
        "candidate_top1": cand.argmax(1).tolist(),
    }


def _export_onnx(model: torch.nn.Module, example: torch.Tensor, path: Path) -> float:
    """Export a dynamic-batch ONNX graph through PyTorch's Dynamo exporter."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_model = copy.deepcopy(model).to("cpu").eval()
    cpu_example = example[:1].detach().cpu()
    started = perf_counter()
    # The exporter prints a Unicode checkmark; suppress its console log so
    # Windows code pages do not make otherwise successful export fail.
    with contextlib.redirect_stdout(io.StringIO()):
        torch.onnx.export(
            cpu_model,
            (cpu_example,),
            str(path),
            input_names=["images"],
            output_names=["logits"],
            dynamo=True,
            dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=32)},),
            report=False,
        )
    return perf_counter() - started


def benchmark_onnx_cuda(
    path: Path,
    inputs_cpu: torch.Tensor,
    warmup_runs: int,
    repeats: int,
    runs: int,
) -> dict[str, Any]:
    """Run ONNX CUDA only with CPU fallback disabled and profiled assignments."""
    import onnxruntime as ort

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(f"CUDAExecutionProvider unavailable; available={ort.get_available_providers()}")
    # Importing torch first preloads its CUDA/cuDNN DLLs for compatible ORT wheels.
    preload_started = perf_counter()
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls(directory="")
    dll_preload_seconds = perf_counter() - preload_started
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = str(path.parent / "onnxruntime_profile")
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session_started = perf_counter()
    session = ort.InferenceSession(
        str(path), options, providers=[("CUDAExecutionProvider", {"device_id": 0})]
    )
    session_initialization_seconds = perf_counter() - session_started
    active_providers = session.get_providers()
    if not active_providers or active_providers[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"CUDAExecutionProvider is not active first provider: {active_providers}")
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result: dict[str, Any] = {
        "providers_available": ort.get_available_providers(),
        "providers_active": active_providers,
        "provider_verification": "CUDAExecutionProvider active; CPU EP fallback disabled by session configuration; profile checked after runs",
        "version": ort.__version__,
        "onnx_version": __import__("onnx").__version__,
        "onnxscript_version": __import__("onnxscript").__version__,
        "dll_preload_seconds": dll_preload_seconds,
        "session_initialization_seconds": session_initialization_seconds,
        "runs_by_batch": [],
        "gpu_peak_allocated_bytes": None,
        "memory_note": "ONNX Runtime uses its own CUDA allocator; PyTorch peak allocation counters do not measure ORT allocations.",
        "timing_scope": "session.run request; includes CPU input conversion, host-to-device transfer, graph execution, and output copy to CPU",
    }
    for batch_size in (1, 8, 32):
        cpu_np = inputs_cpu[:batch_size].contiguous().numpy()
        input_tensor = torch.from_numpy(cpu_np)
        first_started = perf_counter()
        first_out = session.run([output_name], {input_name: cpu_np})[0]
        first_ms = (perf_counter() - first_started) * 1_000
        batch_runs = []
        for run_index in range(runs):
            for _ in range(warmup_runs):
                session.run([output_name], {input_name: cpu_np})
            samples = []
            for _ in range(repeats):
                started = perf_counter()
                output = session.run([output_name], {input_name: cpu_np})[0]
                samples.append((perf_counter() - started) * 1_000)
            batch_runs.append({"run": run_index + 1, **_summary(samples, batch_size), "warmup_runs": warmup_runs})
        aggregate_samples = [x for row in batch_runs for x in row["samples_ms"]]
        result["runs_by_batch"].append({
            "batch_size": batch_size,
            "first_session_run_ms": first_ms,
            "input_shape": list(input_tensor.shape),
            "output_shape": list(first_out.shape),
            "runs": batch_runs,
            "aggregate": _summary(aggregate_samples, batch_size),
            "numerical_comparison_pending": True,
        })

    profile_path = Path(session.end_profiling())
    events = json.loads(profile_path.read_text(encoding="utf-8"))
    node_events = [e for e in events if e.get("cat") == "Node" and "provider" in e.get("args", {})]
    provider_counts: dict[str, int] = {}
    for event in node_events:
        provider = event["args"]["provider"]
        provider_counts[provider] = provider_counts.get(provider, 0) + 1
    if not provider_counts.get("CUDAExecutionProvider"):
        raise RuntimeError(f"Profile shows no CUDA node execution: {provider_counts}")
    if provider_counts.get("CPUExecutionProvider"):
        raise RuntimeError(f"CPU fallback nodes detected in CUDA benchmark: {provider_counts}")
    result["profile_node_provider_counts"] = provider_counts
    result["profile_path"] = str(profile_path)
    result["profile_verified_cuda_only"] = True
    return result


def _aggregate_runs(run_rows: list[dict[str, Any]], batch_size: int) -> dict[str, Any]:
    flat = [sample for row in run_rows for sample in row["samples_ms"]]
    return {
        **_summary(flat, batch_size),
        "independent_run_count": len(run_rows),
        "per_run_mean_ms": [row["mean_ms"] for row in run_rows],
        "per_run_median_ms": [row["median_ms"] for row in run_rows],
        "per_run_p95_ms": [row["p95_ms"] for row in run_rows],
        "per_run_first_forward_ms": [row["first_forward_ms"] for row in run_rows],
        "per_run_peak_allocated_bytes": [row["steady_peak_allocated_bytes"] for row in run_rows],
    }


def run_day2(output_dir: Path, runs: int = 3, warmup_runs: int = 10, repeats: int = 30, onnx_enabled: bool = True) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Day 2 CUDA precision benchmarks require a CUDA-enabled PyTorch runtime")
    env = collect_environment()
    try:
        git_root = subprocess.run(["git", "rev-parse", "--show-toplevel"], check=True, capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        git_root = None
    model, model_init_seconds = load_resnet18(torch.device("cuda"), pretrained=True)
    data: dict[str, Any] = {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": env,
        "repository_context": {"working_directory": env["working_directory"], "git_root": git_root, "git_root_status": "found" if git_root else "no Git root found from the working directory"},
        "model": {"architecture": "ResNet-18", "weights": "ResNet18_Weights.DEFAULT (ImageNet1K_V1)", "dtype_weights": "float32", "image_shape": [3, 224, 224]},
        "configuration": {"batch_sizes": [1, 8, 32], "seed": 2025, "runs": runs, "warmup_runs_per_run": warmup_runs, "repeats_per_run": repeats, "precision_modes": ["CUDA FP32", "CUDA autocast FP16"]},
        "model_initialization_seconds": model_init_seconds,
        "benchmarks": [],
        "onnx": {"status": "not_requested" if not onnx_enabled else "pending"},
        "methodology": {
            "weights_and_inputs": "one pretrained FP32 ResNet-18; deterministic CPU tensors generated once per batch and copied unchanged for both CUDA modes",
            "fp32_timing": "model forward only; CUDA input already resident; output remains CUDA; inference_mode; synchronization before and after every sample",
            "fp16_timing": "same FP32-weight model under torch.autocast(cuda, float16); input already resident; output remains CUDA; inference_mode; synchronized samples",
            "data_transfer": "host-to-device copy measured separately with blocking copies and CUDA synchronization",
            "first_forward": "first call for that precision/batch configuration; reported apart from warmed steady-state samples",
            "gpu_memory": "torch.cuda.max_memory_allocated for PyTorch allocator; steady peak includes model and input baseline and is reported with incremental peak",
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    cpu_inputs: dict[int, torch.Tensor] = {}
    fp32_refs: dict[int, torch.Tensor] = {}
    fp16_outputs: dict[int, torch.Tensor] = {}
    for batch_size in (1, 8, 32):
        cpu_batch = make_inputs(batch_size, seed=2025, device="cpu")
        cpu_inputs[batch_size] = cpu_batch
        transfer = measure_data_transfer(cpu_batch)
        gpu_batch = cpu_batch.to("cuda")
        _sync()
        for mode, use_fp16 in (("cuda_fp32", False), ("cuda_autocast_fp16", True)):
            per_run = []
            last_output = None
            try:
                for run_index in range(runs):
                    row, last_output = benchmark_torch_run(model, gpu_batch, batch_size, use_fp16, warmup_runs, repeats)
                    row["run"] = run_index + 1
                    per_run.append(row)
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                data["benchmarks"].append({"backend": "pytorch_cuda", "precision": mode, "batch_size": batch_size, "status": "out_of_memory", "reason": str(exc).splitlines()[0], "data_transfer": transfer, "runs": per_run, "aggregate": None, "numerical_check": None})
                continue
            if use_fp16:
                fp16_outputs[batch_size] = last_output.detach().float().cpu()
            else:
                fp32_refs[batch_size] = last_output.detach().float().cpu()
            data["benchmarks"].append({
                "backend": "pytorch_cuda",
                "precision": mode,
                "batch_size": batch_size,
                "input_shape": list(gpu_batch.shape),
                "data_transfer": transfer,
                "runs": per_run,
                "status": "ok",
                "aggregate": _aggregate_runs(per_run, batch_size),
                "numerical_check": None,
            })
        del gpu_batch

    for row in data["benchmarks"]:
        if row["precision"] == "cuda_autocast_fp16" and row.get("status") == "ok" and row["batch_size"] in fp32_refs and row["batch_size"] in fp16_outputs:
            row["numerical_check"] = compare_outputs(fp32_refs[row["batch_size"]], fp16_outputs[row["batch_size"]])

    if onnx_enabled:
        try:
            onnx_path = output_dir / "resnet18_dynamic.onnx"
            export_seconds = _export_onnx(model, cpu_inputs[1], onnx_path)
            onnx_result = benchmark_onnx_cuda(onnx_path, cpu_inputs[32], warmup_runs, repeats, runs)
            onnx_result["export_seconds"] = export_seconds
            # Numerical comparison uses one deterministic batch per shape.
            import onnxruntime as ort
            # The runtime session was closed after benchmark; re-open a short-lived CUDA session for correctness only.
            verify_options = ort.SessionOptions()
            verify_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
            session = ort.InferenceSession(str(onnx_path), verify_options, providers=[("CUDAExecutionProvider", {"device_id": 0})])
            if not session.get_providers() or session.get_providers()[0] != "CUDAExecutionProvider":
                raise RuntimeError(f"CUDAExecutionProvider inactive for numerical verification: {session.get_providers()}")
            for row in onnx_result["runs_by_batch"]:
                n = row["batch_size"]
                logits = session.run(["logits"], {"images": cpu_inputs[n].numpy()})[0]
                row["numerical_comparison"] = compare_outputs(fp32_refs[n], torch.from_numpy(logits))
                row.pop("numerical_comparison_pending", None)
            data["onnx"] = {"status": "ok", **onnx_result, "model_file": str(onnx_path)}
        except Exception as exc:  # ONNX is optional; retain PyTorch FP16/FP32 results on exporter or provider failure.
            data["onnx"] = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}

    _write_charts(data, output_dir)
    data["artifacts"] = {"json": "day2_benchmark.json", "report": "day2_report.md", "latency_chart": "day2_latency.png", "throughput_chart": "day2_throughput.png"}
    json_path = output_dir / "day2_benchmark.json"
    json_path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    _write_markdown_report(data, output_dir / "day2_report.md")
    return data


def _write_charts(data: dict[str, Any], output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = [("cuda_fp32", "PyTorch CUDA FP32"), ("cuda_autocast_fp16", "PyTorch CUDA autocast FP16")]
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, label in modes:
        rows = [x for x in data["benchmarks"] if x["precision"] == key and x.get("aggregate")]
        ax.plot([r["batch_size"] for r in rows], [r["aggregate"]["mean_ms"] for r in rows], marker="o", label=label)
    ax.set(xlabel="Batch size", ylabel="Mean steady-state latency (ms)", title="ResNet-18 CUDA inference latency")
    ax.set_xticks([1, 8, 32]); ax.grid(True, alpha=0.3); ax.legend(); fig.tight_layout()
    fig.savefig(output_dir / "day2_latency.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for key, label in modes:
        rows = [x for x in data["benchmarks"] if x["precision"] == key and x.get("aggregate")]
        ax.plot([r["batch_size"] for r in rows], [r["aggregate"]["throughput_images_per_second"] for r in rows], marker="o", label=label)
    ax.set(xlabel="Batch size", ylabel="Images per second", title="ResNet-18 CUDA inference throughput")
    ax.set_xticks([1, 8, 32]); ax.grid(True, alpha=0.3); ax.legend(); fig.tight_layout()
    fig.savefig(output_dir / "day2_throughput.png", dpi=160); plt.close(fig)


def _write_markdown_report(data: dict[str, Any], path: Path) -> None:
    lines = [
        "# Day 2 Inference Benchmark Report", "",
        f"Generated: `{data['created_at_utc']}`", "",
        "## Environment", "",
        f"- Python: `{data['environment']['python'].split()[0]}` (`{data['environment']['python_executable']}`)",
        f"- Working directory: `{data['repository_context']['working_directory']}`; Git root: `{data['repository_context']['git_root'] or 'not present'}`",
        f"- PyTorch: `{data['environment']['pytorch']}`; CUDA build `{data['environment']['torch_cuda_build']}`",
        f"- GPU: `{data['environment']['devices'][0]['name']}`; compute capability `{data['environment']['devices'][0]['compute_capability']}`",
        f"- CPU: `{data['environment'].get('cpu_name', 'unavailable')}`; PyTorch threads `{data['environment']['cpu_threads']}`",
        f"- Driver: `{(data['environment'].get('nvidia_smi') or {}).get('driver_version', 'unavailable')}`",
        f"- Protocol: {data['configuration']['runs']} runs, {data['configuration']['warmup_runs_per_run']} warmups and {data['configuration']['repeats_per_run']} timed calls per run/configuration.", "",
        "## PyTorch CUDA Results", "",
        "FP32 and autocast FP16 use the same FP32 model weights and deterministic inputs. These figures time only a model call with the input already on CUDA and output left on CUDA; host-to-device transfer is measured separately.", "",
        "| Precision | Batch | Mean ms | Median ms | p95 ms | Throughput images/s | Peak allocated MiB | FP16 vs FP32 allclose | Top-1 agreement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in data["benchmarks"]:
        if not row.get("aggregate"):
            lines.append(f"| {row['precision']} | {row['batch_size']} | OOM | — | — | — | — | — | — |")
            continue
        mem = max(row["aggregate"]["per_run_peak_allocated_bytes"]) / (1024 ** 2)
        check = row["numerical_check"]
        top1 = f"{check['top1_agreement']:.3f}" if check else "—"
        lines.append(
            f"| {row['precision']} | {row['batch_size']} | {row['aggregate']['mean_ms']:.4f} | {row['aggregate']['median_ms']:.4f} | {row['aggregate']['p95_ms']:.4f} | {row['aggregate']['throughput_images_per_second']:.2f} | {mem:.2f} | {check['allclose'] if check else '—'} | {top1} |"
        )
    lines += ["", "## Observed FP16 Tradeoff", "", "Percentages below compare this run's FP16 autocast result with CUDA FP32. Lower latency is better; measurements reflect this GPU and this synthetic input workload.", "", "| Batch | Mean latency change | Throughput change | Peak allocated memory change |", "|---:|---:|---:|---:|"]
    for batch_size in (1, 8, 32):
        fp32 = next((r for r in data["benchmarks"] if r["batch_size"] == batch_size and r["precision"] == "cuda_fp32" and r.get("aggregate")), None)
        fp16 = next((r for r in data["benchmarks"] if r["batch_size"] == batch_size and r["precision"] == "cuda_autocast_fp16" and r.get("aggregate")), None)
        if fp32 and fp16:
            a, b = fp32["aggregate"], fp16["aggregate"]
            latency_change = (b["mean_ms"] / a["mean_ms"] - 1) * 100
            throughput_change = (b["throughput_images_per_second"] / a["throughput_images_per_second"] - 1) * 100
            mem32 = max(a["per_run_peak_allocated_bytes"])
            mem16 = max(b["per_run_peak_allocated_bytes"])
            memory_change = (mem16 / mem32 - 1) * 100
            latency_label = f"{abs(latency_change):.1f}% slower" if latency_change > 0 else f"{abs(latency_change):.1f}% lower"
            lines.append(f"| {batch_size} | {latency_label} | {throughput_change:+.1f}% | {memory_change:+.1f}% |")
    lines += ["", "![Latency](day2_latency.png)", "", "![Throughput](day2_throughput.png)", "", "## Transfer, startup, and numerical details", ""]
    for row in data["benchmarks"]:
        if not row.get("aggregate"):
            lines.append(f"- **{row['precision']} batch {row['batch_size']}**: `{row.get('status', 'skipped')}` — {row.get('reason', 'no timing data')}")
            continue
        lines.append(f"- **{row['precision']} batch {row['batch_size']}**: separate H2D transfer mean {row['data_transfer']['mean_ms']:.4f} ms, median {row['data_transfer']['median_ms']:.4f} ms, p95 {row['data_transfer']['p95_ms']:.4f} ms.")
        first_forward = ", ".join(f"{x:.4f}" for x in row["aggregate"]["per_run_first_forward_ms"])
        lines.append(f"  First-forward per run: {first_forward} ms. FP32 model initialization: {data['model_initialization_seconds']:.4f} s (shared across precision configurations).")
        if row["numerical_check"]:
            e = row["numerical_check"]
            lines.append(f"  FP16 numerical check: allclose={e['allclose']} with rtol={e['rtol']}, atol={e['atol']}; max abs error={e['max_absolute_error']:.6g}; mean abs error={e['mean_absolute_error']:.6g}; RMSE={e['rmse']:.6g}; max relative error={e['max_relative_error']:.6g}.")
    lines += ["", "## ONNX Runtime CUDA", ""]
    if data["onnx"]["status"] == "ok":
        onnx = data["onnx"]
        lines.append(f"ONNX Runtime `{onnx['version']}` active providers: `{onnx['providers_active']}`; profile provider assignments: `{onnx['profile_node_provider_counts']}`. CPU fallback was disabled and the profile contained CUDA nodes only.")
        lines.append("Timing is ONNX `session.run`, including CPU input handling, H2D transfer, graph execution and D2H output copy. It is an end-to-end request measurement and is not directly comparable to the PyTorch device-resident forward-only figures above.")
        lines.append(f"Versions: ONNX `{onnx['onnx_version']}`, ONNX Script `{onnx['onnxscript_version']}`. DLL preload: {onnx['dll_preload_seconds']:.4f} s; ONNX session initialization: {onnx['session_initialization_seconds']:.4f} s; model export: {onnx['export_seconds']:.4f} s (all excluded from steady-state timing).")
        lines.append("")
        lines.append("| Batch | First session run ms | Mean ms | Median ms | p95 ms | Throughput images/s | Allclose | Top-1 agreement vs PyTorch FP32 |")
        lines.append("|---:|---:|---:|---:|---:|---:|:---:|---:|")
        for row in onnx["runs_by_batch"]:
            agg = row["aggregate"]; check = row["numerical_comparison"]
            lines.append(f"| {row['batch_size']} | {row['first_session_run_ms']:.4f} | {agg['mean_ms']:.4f} | {agg['median_ms']:.4f} | {agg['p95_ms']:.4f} | {agg['throughput_images_per_second']:.2f} | {check['allclose']} | {check['top1_agreement']:.3f} |")
        lines.append("")
        lines.append(f"Dynamo ONNX export time: {onnx['export_seconds']:.3f} s (excluded from inference timing); profile file: `{onnx['profile_path']}`.")
    else:
        lines.append(f"ONNX status: `{data['onnx']['status']}`. {data['onnx'].get('reason', 'Not requested.')}")
    lines += ["", "## Interpretation and limitations", "", "These are observed measurements, not guaranteed improvements. FP16 autocast retains FP32 model parameters and selects reduced precision for eligible CUDA operations. A speedup is shape-, hardware-, clock-, and workload-dependent; small batch latency can be launch/dispatch bound. Relative error can look large for logits close to zero; the allclose check also applies the stated absolute tolerance. Peak allocation is measured through PyTorch's allocator and does not include all driver or context memory. Synthetic random inputs represent tensor shape and dtype, not image preprocessing or production traffic. Independent runs share one process/model and are not cold process launches.", "", "## Reproduce", "", "```powershell", ".\\.venv\\Scripts\\python.exe -m pytest", ".\\.venv\\Scripts\\python.exe -m gpu_inference_lab.day2 --output-dir results --runs 3 --warmup-runs 10 --repeats 30", "```", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 2 CUDA FP32/FP16 and ONNX Runtime benchmark")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--skip-onnx", action="store_true")
    args = parser.parse_args()
    result = run_day2(args.output_dir, args.runs, args.warmup_runs, args.repeats, onnx_enabled=not args.skip_onnx)
    print(json.dumps({"json": str(args.output_dir / "day2_benchmark.json"), "report": str(args.output_dir / "day2_report.md"), "onnx_status": result["onnx"]["status"]}, indent=2))


if __name__ == "__main__":
    main()
