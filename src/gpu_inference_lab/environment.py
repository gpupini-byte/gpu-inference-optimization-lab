"""Capture observed runtime and hardware details for reproducible results."""
import platform
import os
import subprocess
import sys
from typing import Any

import torch
import torchvision
import numpy


def _nvidia_smi() -> dict[str, str] | None:
    """Return driver and currently reported GPU memory where nvidia-smi exists."""
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=5,
        )
        rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        if not rows:
            return None
        name, driver, total_mib, free_mib = [part.strip() for part in rows[0].split(",", 3)]
        return {"name": name, "driver_version": driver, "memory_total_mib": total_mib, "memory_free_at_capture_mib": free_mib}
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _cpu_name() -> str:
    """Return the Windows processor model when available, with a portable fallback."""
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except (ImportError, OSError):
        return platform.processor() or platform.machine()


def collect_environment() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    devices = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append({
                "index": index,
                "name": props.name,
                "total_memory_bytes": props.total_memory,
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count,
            })
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cpu_name": _cpu_name(),
        "working_directory": os.getcwd(),
        "pytorch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": numpy.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": cuda_available,
        "devices": devices,
        "cpu_threads": torch.get_num_threads(),
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "nvidia_smi": _nvidia_smi(),
    }
