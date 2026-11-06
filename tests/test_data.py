import torch
import pytest

from gpu_inference_lab.data import make_inputs


def test_inputs_are_reproducible_and_have_expected_shape():
    a = make_inputs(2, seed=7)
    b = make_inputs(2, seed=7)
    assert a.shape == (2, 3, 224, 224)
    assert a.dtype == torch.float32
    assert torch.equal(a, b)


def test_inputs_reject_nonpositive_batch():
    with pytest.raises(ValueError):
        make_inputs(0)
