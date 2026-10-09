"""Config space and pruning rules for the Triton family (plan 6.1, 6.2).

Pruning is split in two so it ports across GPUs:
  * static rules  - depend only on the config and the hardware limits (run once per GPU)
  * shape rules   - depend on (M, N, K) as well (run per shape when building the work list)
Each rule has a name; `prune` returns the reasons a config was dropped so the pruning is auditable.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

from kernels.triton_matmul import MatmulConfig

SPACE = {
    "BLOCK_M": [16, 32, 64, 128, 256],
    "BLOCK_N": [32, 64, 128, 256],
    "BLOCK_K": [16, 32, 64, 128],
    "num_warps": [2, 4, 8],
    "num_stages": [2, 3, 4, 5],
    "GROUP_M": [1, 4, 8, 16],
    "SPLIT_K": [1, 2, 4, 8, 16],
}


@dataclass(frozen=True)
class HwLimits:
    smem_per_block_bytes: int   # opt-in max dynamic shared memory per block
    regs_per_sm: int
    sm_count: int

    @classmethod
    def from_device(cls, device: int = 0) -> "HwLimits":
        import torch
        p = torch.cuda.get_device_properties(device)
        return cls(p.shared_memory_per_block_optin, p.regs_per_multiprocessor, p.multi_processor_count)


def full_space() -> list[MatmulConfig]:
    keys = list(SPACE)
    return [MatmulConfig(**dict(zip(keys, vals))) for vals in itertools.product(*SPACE.values())]


def smem_bytes(cfg: MatmulConfig, elem_bytes: int = 2) -> int:
    # Triton's pipeliner keeps (num_stages - 1) A+B tiles in flight; budget num_stages to be conservative.
    return cfg.num_stages * (cfg.BLOCK_M * cfg.BLOCK_K + cfg.BLOCK_K * cfg.BLOCK_N) * elem_bytes


def acc_regs_per_thread(cfg: MatmulConfig) -> float:
    return cfg.BLOCK_M * cfg.BLOCK_N / (cfg.num_warps * 32)


def static_violations(cfg: MatmulConfig, hw: HwLimits, elem_bytes: int = 2) -> list[str]:
    reasons = []
    if smem_bytes(cfg, elem_bytes) > hw.smem_per_block_bytes:
        reasons.append("smem_exceeds_limit")
    # fp32 accumulator alone above 128 regs/thread leaves too little of the 255 cap for operands -> spills.
    if acc_regs_per_thread(cfg) > 128:
        reasons.append("acc_regs_gt_128")
    # Tiny tiles on many warps: under 8 accumulator elements per thread, warps idle on the MMA.
    if acc_regs_per_thread(cfg) < 8:
        reasons.append("acc_elems_per_thread_lt_8")
    # mma.sync m16n8k16: each warp needs at least a 16x8 slab of the output; also BLOCK_K >= 16.
    if cfg.BLOCK_M * cfg.BLOCK_N < cfg.num_warps * 16 * 8:
        reasons.append("tile_too_small_for_warps")
    return reasons


def shape_violations(cfg: MatmulConfig, M: int, N: int, K: int, *, max_workspace_bytes: int = 512 * 2**20) -> list[str]:
    reasons = []
    # Tile far larger than the problem: >2x the smallest power of two covering the dim (min tile 16).
    def cover(x: int) -> int:
        return max(16, 1 << (x - 1).bit_length())
    if cfg.BLOCK_M > 2 * cover(M):
        reasons.append("block_m_oversized")
    if cfg.BLOCK_N > 2 * cover(N):
        reasons.append("block_n_oversized")
    if cfg.BLOCK_K > 2 * cover(K):
        reasons.append("block_k_oversized")
    if cfg.SPLIT_K > 1:
        # Each split must own at least one full BLOCK_K step, else extra splits do nothing but add a reduction.
        if K < cfg.SPLIT_K * cfg.BLOCK_K:
            reasons.append("split_k_starved")
        ws_elems = cfg.SPLIT_K * M * N
        if ws_elems * 4 > max_workspace_bytes or ws_elems >= 2**31:
            reasons.append("split_k_workspace_too_large")
    # GROUP_M only matters if there are more tile-rows than the group.
    if cfg.GROUP_M > 1 and -(-M // cfg.BLOCK_M) <= 1:
        reasons.append("group_m_noop")
    return reasons


def prune(configs: list[MatmulConfig], hw: HwLimits, shape: tuple[int, int, int] | None = None,
          elem_bytes: int = 2) -> tuple[list[MatmulConfig], dict[str, int]]:
    kept, dropped = [], {}
    for c in configs:
        reasons = static_violations(c, hw, elem_bytes)
        if shape is not None:
            reasons += shape_violations(c, *shape)
        if reasons:
            for r in reasons:
                dropped[r] = dropped.get(r, 0) + 1
        else:
            kept.append(c)
    return kept, dropped


# Hand-picked starting configs for the week-1 check (plan 13): one per regime we expect to matter.
HANDPICKED = {
    "large_square": MatmulConfig(128, 256, 32, num_warps=8, num_stages=3, GROUP_M=8),
    "medium": MatmulConfig(128, 128, 32, num_warps=4, num_stages=4, GROUP_M=8),
    "small_tile": MatmulConfig(64, 64, 32, num_warps=4, num_stages=4, GROUP_M=8),
    "skinny_m": MatmulConfig(16, 64, 128, num_warps=4, num_stages=4, GROUP_M=1),
    "large_k_splitk": MatmulConfig(64, 64, 64, num_warps=4, num_stages=3, GROUP_M=1, SPLIT_K=8),
}
