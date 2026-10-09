"""B1: torch.matmul on the default cuBLAS path, i.e. what a typical user gets.

Torch's global matmul flags are left at their defaults and recorded in env.json.
"""
from __future__ import annotations

import torch


class TorchMatmul:
    family = "torch_matmul"

    def make_runner(self, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor | None = None):
        c = out if out is not None else torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)

        def run():
            return torch.matmul(a, b, out=c)
        return run
