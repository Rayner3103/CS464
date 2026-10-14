"""Which CUDA kernels does a launch closure actually run? (e.g. the cuBLAS kernel behind torch.matmul)"""
from __future__ import annotations

import torch
from torch.profiler import ProfilerActivity, profile


def cuda_kernel_names(fn) -> list[str]:
    """Names of the device kernels launched by one call of `fn`, in launch order (excluding our own fills)."""
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = []
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA and e.name not in names:
            names.append(e.name)
    return names
