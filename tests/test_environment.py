from gpu_inference_lab.environment import collect_environment


def test_environment_records_runtime_fields():
    result = collect_environment()
    assert result["python"]
    assert result["pytorch"]
    assert isinstance(result["cuda_available"], bool)
    assert isinstance(result["devices"], list)
