import torch
import pytest

from gpu_inference_lab.benchmark import benchmark_inference


class TinyModel(torch.nn.Module):
    def forward(self, x):
        return x + 1


def test_cpu_benchmark_reports_distribution_and_throughput():
    x = torch.zeros(2, 3)
    result = benchmark_inference(TinyModel(), x, torch.device("cpu"), warmup_runs=1, repeats=3)
    steady = result["steady_state"]
    assert len(steady["latency_ms"]["samples"]) == 3
    assert steady["latency_ms"]["min"] <= steady["latency_ms"]["median"] <= steady["latency_ms"]["max"]
    assert steady["throughput_images_per_second"] > 0
    assert result["cold_start_first_forward_ms"] >= 0


def test_benchmark_requires_multiple_samples():
    with pytest.raises(ValueError):
        benchmark_inference(TinyModel(), torch.zeros(1), torch.device("cpu"), repeats=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_synchronization_path():
    x = torch.zeros(2, 3, device="cuda")
    result = benchmark_inference(TinyModel().cuda(), x, torch.device("cuda"), warmup_runs=1, repeats=3)
    assert result["steady_state"]["throughput_images_per_second"] > 0
