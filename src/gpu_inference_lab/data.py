"""Deterministic synthetic inputs shared by every execution backend."""
import torch


def make_inputs(batch_size: int, seed: int = 2025, device: torch.device | str = "cpu") -> torch.Tensor:
    """Generate fixed-seed ImageNet-shaped float32 tensors in NCHW layout."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    inputs = torch.randn(batch_size, 3, 224, 224, generator=generator, dtype=torch.float32)
    return inputs.to(device)
