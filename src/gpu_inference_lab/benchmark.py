"""Synchronized inference timing with warmup and distribution summaries."""
from __future__ import annotations

from statistics import mean, median, quantiles
from time import perf_counter
from typing import Any

import torch


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_inference(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    device: torch.device,
    warmup_runs: int = 10,
    repeats: int = 50,
) -> dict[str, Any]:
    """Measure batch-call latency; transfer and model creation are excluded."""
    if repeats < 2:
        raise ValueError("repeats must be at least 2 for distribution statistics")
    model.eval()
    # Deliberately separate the first call, synchronization, and warmup from the
    # steady-state samples. Inputs already reside on the target device.
    synchronize(device)
    cold_started = perf_counter()
    with torch.inference_mode():
        model(inputs)
    synchronize(device)
    cold_start_ms = (perf_counter() - cold_started) * 1_000

    with torch.inference_mode():
        for _ in range(warmup_runs):
            model(inputs)
    synchronize(device)

    samples_ms: list[float] = []
    with torch.inference_mode():
        for _ in range(repeats):
            synchronize(device)
            started = perf_counter()
            model(inputs)
            synchronize(device)
            samples_ms.append((perf_counter() - started) * 1_000)

    ordered = sorted(samples_ms)
    p95 = quantiles(ordered, n=20, method="inclusive")[18]
    batch_size = int(inputs.shape[0])
    avg_ms = mean(samples_ms)
    return {
        "cold_start_first_forward_ms": cold_start_ms,
        "steady_state": {
            "repeats": repeats,
            "warmup_runs": warmup_runs,
            "latency_ms": {
                "mean": avg_ms,
                "median": median(samples_ms),
                "p95": p95,
                "min": min(samples_ms),
                "max": max(samples_ms),
                "samples": samples_ms,
            },
            "throughput_images_per_second": batch_size / (avg_ms / 1_000),
        },
        "timing_scope": "model forward only; inputs preallocated on target device; per-call host wall clock with CUDA synchronization",
    }
