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


PLACEMENT_MAX_TRIES = 8


def placed_inputs(M: int, N: int, K: int, dtype=torch.float16, seed: int = 0, placement: int = 0,
                  device: str = "cuda", avoid: set[int] | frozenset[int] = frozenset()):
    """Same values as make_inputs, but at a different device address for each `placement` index.

    Placement 0 is make_inputs as it is. Placement p > 0 copies each operand into a fresh buffer at an offset of a whole
    number of MiB (1-64, drawn per operand from (seed, placement, shape)), so A and B land on other pages and at another
    A-B distance. Offsets are multiples of 1 MiB, so the base pointers keep the allocator's alignment (the kernels'
    16-byte specializations do not change). A view's address is still drawn again (up to PLACEMENT_MAX_TRIES times) when
    it is in `avoid`: callers pass the addresses of the placements they already timed, because the buffer can reuse a
    block a previous placement freed.

    Why not a pad allocated before make_inputs (the protocol up to 2026-11-19, deep check HAR-7): operands <= 1 MiB come
    from PyTorch's small-block pool, which a large pad never touches, so they got back the blocks the previous placement
    had just freed (1 of 3 placements distinct), and a large pad could hand placement 2 placement 0's block (2 of 3).
    Returns (a, b, None); the third value is kept for the callers' unpacking (the views hold their buffers).
    """
    a, b = make_inputs(M, N, K, dtype, seed, device)
    if not placement:
        return a, b, None
    import random
    rng = random.Random(f"placement:{seed}:{placement}:{M}x{N}x{K}")  # str seed: stable across processes
    out = []
    for t in (a, b):
        for _ in range(PLACEMENT_MAX_TRIES):
            shift = rng.randint(1, 64) * 2**20 // t.element_size()
            v = torch.empty(shift + t.numel(), dtype=t.dtype, device=device)[shift:].view(t.shape)
            if v.data_ptr() not in avoid:
                break
            del v
        v.copy_(t)
        out.append(v)
    return out[0], out[1], None


CONTEXT_POISON_MARKERS = ("illegal", "misaligned", "unspecified launch failure", "device-side assert",
                          "an illegal instruction")


def context_poisoned(reason: str | None) -> bool:
    """True when a failure reason is a CUDA error that leaves the context unusable (sticky): every later CUDA call in
    the process fails with it, so the process must stop instead of recording the next candidates as invalid."""
    r = (reason or "").lower()
    return any(x in r for x in CONTEXT_POISON_MARKERS)


class ContextPoisoned(RuntimeError):
    """A candidate faulted the CUDA context (see `context_poisoned`); `culprit` names it. Nothing else can run on the
    GPU in this process."""

    def __init__(self, culprit: str, reason: str):
        super().__init__(f"{culprit} poisoned the CUDA context: {reason}")
        self.culprit, self.reason = culprit, reason


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

    Compile errors and CUDA faults are caught and reported as failures, never raised. A fault that poisons the context
    (`context_poisoned(result.failure_reason)`) must stop the caller: nothing after it in the process can run.
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
