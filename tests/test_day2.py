import json
from pathlib import Path

import pytest
import torch

from gpu_inference_lab.day2 import benchmark_torch_run, compare_outputs


def test_compare_outputs_reports_metrics_and_top1():
    ref = torch.tensor([[1.0, 0.25], [0.1, 2.0]])
    cand = torch.tensor([[1.001, 0.249], [0.101, 1.999]])
    metrics = compare_outputs(ref, cand, rtol=0.01, atol=0.01)
    assert metrics["allclose"]
    assert metrics["top1_agreement"] == 1.0
    assert metrics["max_absolute_error"] == pytest.approx(0.001, abs=1e-6)
    assert metrics["rmse"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_autocast_inference_run_returns_output_and_memory_metrics():
    model = torch.nn.Conv2d(3, 4, kernel_size=1).cuda().eval()
    inputs = torch.randn(2, 3, 8, 8, device="cuda")
    fp32, ref = benchmark_torch_run(model, inputs, 2, False, warmup_runs=1, repeats=3)
    fp16, pred = benchmark_torch_run(model, inputs, 2, True, warmup_runs=1, repeats=3)
    assert len(fp16["samples_ms"]) == 3
    assert fp16["steady_peak_allocated_bytes"] >= fp16["steady_baseline_allocated_bytes"]
    assert compare_outputs(ref, pred)["allclose"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_onnx_cuda_execution_provider_smoke(tmp_path):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("Relu", ["input"], ["output"])],
        "cuda_provider_smoke",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    model_path = tmp_path / "smoke.onnx"
    onnx.save(model, model_path)
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = str(tmp_path / "ort_cuda_smoke_profile")
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(str(model_path), options, providers=[("CUDAExecutionProvider", {"device_id": 0})])
    assert session.get_providers()[0] == "CUDAExecutionProvider"
    output = session.run(["output"], {"input": [[-1.0, 2.0]]})[0]
    assert output.tolist() == [[0.0, 2.0]]
    profile = json.loads(Path(session.end_profiling()).read_text(encoding="utf-8"))
    providers = {event.get("args", {}).get("provider") for event in profile if event.get("cat") == "Node"}
    assert "CUDAExecutionProvider" in providers
    assert "CPUExecutionProvider" not in providers
