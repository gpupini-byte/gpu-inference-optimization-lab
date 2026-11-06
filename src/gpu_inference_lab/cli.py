"""Command-line entry point for the Day 1 benchmark matrix."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .benchmark import benchmark_inference
from .data import make_inputs
from .environment import collect_environment
from .models import load_resnet18


def run_benchmarks(output: Path, warmup_runs: int = 10, repeats: int = 50) -> dict[str, Any]:
    env = collect_environment()
    results: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": {"architecture": "ResNet-18", "weights": "ResNet18_Weights.DEFAULT (ImageNet1K_V1)", "input_shape_per_image": [3, 224, 224], "input_dtype": "float32"},
        "configuration": {"batch_sizes": [1, 8, 32], "warmup_runs": warmup_runs, "repeats": repeats, "seed": 2025},
        "environment": env,
        "benchmarks": [],
        "limitations": ["Synthetic fixed-seed normal tensors are used; preprocessing and data loading are excluded.", "GPU clocks, power mode, temperature, and background processes can affect results.", "CPU and GPU use the same inputs and model weights; inference outputs are not copied back during timing."],
    }

    # Reuse the same loaded weights across batch sizes for a backend. Init time
    # remains separately recorded and is never folded into steady-state timing.
    for backend in ("cpu", "cuda"):
        if backend == "cuda" and not env["cuda_available"]:
            results["benchmarks"].append({"backend": backend, "status": "skipped", "reason": "CUDA is unavailable in this PyTorch runtime"})
            continue
        device = torch.device(backend)
        try:
            model, init_seconds = load_resnet18(device, pretrained=True)
            backend_result: dict[str, Any] = {
                "backend": backend,
                "status": "ok",
                "dtype": "float32",
                "model_initialization_seconds": init_seconds,
                "batches": [],
            }
            for batch_size in (1, 8, 32):
                try:
                    inputs = make_inputs(batch_size, seed=2025, device=device)
                    timing = benchmark_inference(model, inputs, device, warmup_runs, repeats)
                    # Assert the output contract once outside the measured region.
                    with torch.inference_mode():
                        sample_output = model(inputs)
                    batch_data: dict[str, Any] = {
                        "batch_size": batch_size,
                        "input_shape": list(inputs.shape),
                        "output_shape": list(sample_output.shape),
                        **timing,
                    }
                except torch.cuda.OutOfMemoryError as exc:
                    if device.type != "cuda":
                        raise
                    torch.cuda.empty_cache()
                    batch_data = {"batch_size": batch_size, "status": "out_of_memory", "reason": str(exc).splitlines()[0]}
                finally:
                    if "inputs" in locals():
                        del inputs
                backend_result["batches"].append(batch_data)
            results["benchmarks"].append(backend_result)
            del model
        except torch.cuda.OutOfMemoryError as exc:
            if device.type != "cuda":
                raise
            torch.cuda.empty_cache()
            results["benchmarks"].append({"backend": backend, "status": "out_of_memory", "reason": str(exc).splitlines()[0]})

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ResNet-18 CPU and CUDA FP32 inference")
    parser.add_argument("--output", type=Path, default=Path("results/day1_benchmark.json"))
    parser.add_argument("--warmup-runs", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    results = run_benchmarks(args.output, args.warmup_runs, args.repeats)
    print(json.dumps({"output": str(args.output), "environment": results["environment"], "benchmarks": [
        {"backend": b["backend"], "status": b["status"], "batches": [
            {"batch_size": x["batch_size"], "mean_ms": x.get("steady_state", {}).get("latency_ms", {}).get("mean"), "throughput_images_per_second": x.get("steady_state", {}).get("throughput_images_per_second"), "status": x.get("status", "ok")}
            for x in b.get("batches", [])]
        } for b in results["benchmarks"]]}, indent=2))


if __name__ == "__main__":
    main()
