"""Model construction kept separate from all timed inference work."""
from time import perf_counter

import torch
from torchvision.models import ResNet18_Weights, resnet18


def load_resnet18(device: torch.device, pretrained: bool = True) -> tuple[torch.nn.Module, float]:
    """Build ResNet-18, optionally load ImageNet weights, and return init seconds.

    Initialization includes weight download/loading if needed and device transfer;
    it is deliberately not part of inference latency.
    """
    started = perf_counter()
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights).eval().to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return model, perf_counter() - started
