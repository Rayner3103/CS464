"""Parameterized Triton GEMM family (plan 6.1): C[M,N] = A[M,K] @ B[K,N], row-major NN, fp16/bf16 in, fp32 acc.

Tunables: BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, GROUP_M (L2 swizzle), SPLIT_K.
EVEN_K is a constexpr specialization chosen per shape: when K % (BLOCK_K * SPLIT_K) == 0 the K-mask is dropped.

SPLIT_K > 1 is deterministic: each split writes an fp32 partial tile to a [SPLIT_K, M, N] workspace and a
second kernel reduces + casts. No atomics, so results are bitwise reproducible.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import triton
import triton.language as tl

_TL_DTYPE = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16}


@dataclass(frozen=True)
class MatmulConfig:
    BLOCK_M: int
    BLOCK_N: int
    BLOCK_K: int
    num_warps: int = 4
    num_stages: int = 3
    GROUP_M: int = 8
    SPLIT_K: int = 1

    def key(self) -> str:
        return (f"bm{self.BLOCK_M}_bn{self.BLOCK_N}_bk{self.BLOCK_K}_w{self.num_warps}_s{self.num_stages}"
                f"_g{self.GROUP_M}_sk{self.SPLIT_K}")

    def to_dict(self) -> dict:
        return asdict(self)


@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cs, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, SPLIT_K: tl.constexpr, EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    # Grouped ordering: GROUP_M consecutive tile-rows share B tiles in L2.
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # Wrap out-of-range rows/cols onto valid memory instead of masking loads; they are masked at the store.
    offs_am = offs_m % M
    offs_bn = offs_n % N
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            k_remaining = K - k * BLOCK_K * SPLIT_K
            a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_K * SPLIT_K * stride_bk

    c_ptrs = c_ptr + pid_k * stride_cs + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _splitk_reduce_kernel(ws_ptr, c_ptr, MN, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < MN
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in range(SPLIT_K):
        acc += tl.load(ws_ptr + s * MN + offs, mask=mask, other=0.0)
    tl.store(c_ptr + offs, acc.to(c_ptr.dtype.element_ty), mask=mask)


def even_k(K: int, cfg: MatmulConfig) -> bool:
    return K % (cfg.BLOCK_K * cfg.SPLIT_K) == 0


def grid(M: int, N: int, cfg: MatmulConfig) -> tuple[int, int]:
    return (triton.cdiv(M, cfg.BLOCK_M) * triton.cdiv(N, cfg.BLOCK_N), cfg.SPLIT_K)


class TritonMatmul:
    """Pre-allocates output/workspace so the launch closure does no allocation (timing-safe)."""

    family = "triton_nn_v1"

    def __init__(self, cfg: MatmulConfig):
        self.cfg = cfg

    def make_runner(self, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor | None = None):
        cfg = self.cfg
        assert a.ndim == 2 and b.ndim == 2 and a.shape[1] == b.shape[0], (a.shape, b.shape)
        assert a.dtype == b.dtype and a.dtype in _TL_DTYPE, a.dtype
        M, K = a.shape
        N = b.shape[1]
        c = out if out is not None else torch.empty((M, N), device=a.device, dtype=a.dtype)
        g = grid(M, N, cfg)
        ek = even_k(K, cfg)
        if cfg.SPLIT_K == 1:
            def run():
                _matmul_kernel[g](
                    a, b, c, M, N, K,
                    a.stride(0), a.stride(1), b.stride(0), b.stride(1), 0, c.stride(0), c.stride(1),
                    BLOCK_M=cfg.BLOCK_M, BLOCK_N=cfg.BLOCK_N, BLOCK_K=cfg.BLOCK_K,
                    GROUP_M=cfg.GROUP_M, SPLIT_K=1, EVEN_K=ek,
                    num_warps=cfg.num_warps, num_stages=cfg.num_stages,
                )
                return c
            return run

        assert c.is_contiguous(), "split-K reduce writes C as a flat buffer"
        ws = torch.empty((cfg.SPLIT_K, M, N), device=a.device, dtype=torch.float32)
        RBLOCK = 1024
        rgrid = (triton.cdiv(M * N, RBLOCK),)

        def run():
            _matmul_kernel[g](
                a, b, ws, M, N, K,
                a.stride(0), a.stride(1), b.stride(0), b.stride(1), ws.stride(0), ws.stride(1), ws.stride(2),
                BLOCK_M=cfg.BLOCK_M, BLOCK_N=cfg.BLOCK_N, BLOCK_K=cfg.BLOCK_K,
                GROUP_M=cfg.GROUP_M, SPLIT_K=cfg.SPLIT_K, EVEN_K=ek,
                num_warps=cfg.num_warps, num_stages=cfg.num_stages,
            )
            _splitk_reduce_kernel[rgrid](ws, c, M * N, SPLIT_K=cfg.SPLIT_K, BLOCK=RBLOCK, num_warps=4)
            return c
        return run


def matmul(a: torch.Tensor, b: torch.Tensor, cfg: MatmulConfig) -> torch.Tensor:
    return TritonMatmul(cfg).make_runner(a, b)()
