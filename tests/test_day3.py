import json
from pathlib import Path

import pytest
import torch

from gpu_inference_lab.day3 import BenchmarkConfig, _bind_cuda_io


def test_benchmark_config_defaults_and_validation():
    config = BenchmarkConfig()
    config.validate()
    assert config.batch_sizes == (1, 8, 32)
    with pytest.raises(ValueError, match="at least 2"):
        BenchmarkConfig(repeats=1).validate()
    with pytest.raises(ValueError, match="fixed batch set"):
        BenchmarkConfig(batch_sizes=(1, 4)).validate()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_onnx_iobinding_device_input_and_output(tmp_path):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("Relu", ["images"], ["logits"])],
        "iobinding_correctness",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    model_path = tmp_path / "iobinding.onnx"
    onnx.save(model, model_path)

    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = str(tmp_path / "iobinding_profile")
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(str(model_path), options, providers=[("CUDAExecutionProvider", {"device_id": 0})])
    assert session.get_providers()[0] == "CUDAExecutionProvider"

    inputs = torch.tensor([[-1.0, 2.0]], device="cuda", dtype=torch.float32)
    binding = _bind_cuda_io(session, "images", "logits", inputs)
    session.run_with_iobinding(binding)
    binding.synchronize_outputs()
    torch.cuda.synchronize()
    output = binding.get_outputs()[0]
    assert output.device_name().lower() == "cuda"
    assert output.numpy().tolist() == [[0.0, 2.0]]

    profile = json.loads(Path(session.end_profiling()).read_text(encoding="utf-8"))
    providers = {event.get("args", {}).get("provider") for event in profile if event.get("cat") == "Node"}
    assert providers == {"CUDAExecutionProvider"}
