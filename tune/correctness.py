"""Correctness gate (plan 3.4).

Reference: fp32 cuBLAS GEMM of the (fp16-rounded) inputs with TF32 disabled.
Inputs are N(0,1), so each reference element has std ~ sqrt(K).

Pass criterion, elementwise:  |C - ref| <= rtol * |ref| + atol_frac * sqrt(K)
  - rtol covers rounding the fp32 result to the output dtype (2^-11 for fp16, 2^-8 for bf16) plus
    accumulation-order differences.
  - atol_frac * sqrt(K) is a fraction of one output std, small enough that any wrong/garbage tile fails.
The output buffer is pre-filled with NaN, so a kernel that leaves elements unwritten also fails.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass

import torch

TOLERANCES = {
    torch.float16: {"rtol": 1e-2, "atol_frac": 1e-2},
    torch.bfloat16: {"rtol": 3e-2, "atol_frac": 3e-2},
}


@contextmanager
def _strict_fp32():
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def make_inputs(M: int, N: int, K: int, dtype=torch.float16, seed: int = 0, device: str = "cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    a = torch.randn((M, K), generator=g, device=device, dtype=torch.float32).to(dtype)
    b = torch.randn((K, N), generator=g, device=device, dtype=torch.float32).to(dtype)
    return a, b


def placed_inputs(M: int, N: int, K: int, dtype=torch.float16, seed: int = 0, placement: int = 0,
                  device: str = "cuda"):
    """Same values as make_inputs, but at a different physical placement for each `placement` index.

    A seeded 1-64 MiB pad is allocated first so the caching allocator hands out different blocks.
    Returns (a, b, pad); keep `pad` alive while a and b are in use.
    """
    pad = None
    if placement:
        import random
        mib = random.Random(seed * 7919 + placement).randint(1, 64)
        pad = torch.empty(mib * 2**20, dtype=torch.uint8, device=device)
    a, b = make_inputs(M, N, K, dtype, seed, device)
    return a, b, pad


def reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    with _strict_fp32():
        return a.float() @ b.float()


@dataclass
class CheckResult:
    valid: bool
    failure_reason: str | None
    max_abs_err: float
    max_rel_err_vs_scale: float  # max |err| / sqrt(K): error in units of output std

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def check_output(c: torch.Tensor, ref: torch.Tensor, K: int, dtype) -> CheckResult:
    tol = TOLERANCES[dtype]
    c32 = c.float()
    if not torch.isfinite(c32).all():
        n_bad = int((~torch.isfinite(c32)).sum())
        return CheckResult(False, f"non_finite_output({n_bad})", math.inf, math.inf)
    err = (c32 - ref).abs()
    scale = math.sqrt(max(K, 1))
    bound = tol["rtol"] * ref.abs() + tol["atol_frac"] * scale
    max_abs = float(err.max()) if err.numel() else 0.0
    ok = bool((err <= bound).all())
    return CheckResult(ok, None if ok else "tolerance_exceeded", max_abs, max_abs / scale)


def validate(backend, M: int, N: int, K: int, dtype=torch.float16, seed: int = 0,
             ref: torch.Tensor | None = None, inputs=None) -> CheckResult:
    """Run `backend.make_runner(a, b, out)` once into a NaN-filled buffer and compare to the fp32 reference.

    Compile errors and CUDA faults are caught and reported as failures, never raised.
    """
    try:
        a, b = inputs if inputs is not None else make_inputs(M, N, K, dtype, seed)
        if ref is None:
            ref = reference(a, b)
        out = torch.full((M, N), float("nan"), device=a.device, dtype=dtype)
        c = backend.make_runner(a, b, out=out)()
        torch.cuda.synchronize()
        if c.data_ptr() != out.data_ptr():
            out = c
        return check_output(out, ref, K, dtype)
    except Exception as e:  # compile failures (e.g. out of shared memory), launch errors
        msg = f"{type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}"
        return CheckResult(False, msg, math.nan, math.nan)
