"""B0: roofline. Spec peaks come from tune.env.spec_peaks; measured peaks come from microbenchmarks here.

measured compute peak = best achieved TFLOP/s of torch.matmul over a few large, tile-friendly squares
measured bandwidth    = device-to-device copy of a large buffer, counting read + write bytes
"""
from __future__ import annotations

import torch

from tune.timing import TimingConfig, benchmark


def measure_bandwidth_gbps(nbytes: int = 1 << 30, device: int = 0) -> dict:
    src = torch.empty(nbytes // 4, dtype=torch.float32, device=f"cuda:{device}").uniform_()
    dst = torch.empty_like(src)
    r = benchmark(lambda: dst.copy_(src), TimingConfig(cache_mode="hot", max_time_s=3.0), device)
    return {"copy_bytes": nbytes, "copy_median_us": r.median_us,
            "dram_gbps_measured": 2 * nbytes / (r.median_us * 1e-6) / 1e9,
            "dram_gbps_measured_best": 2 * nbytes / (r.min_us * 1e-6) / 1e9}


def measure_peak_tflops(sizes=(4096, 6144, 8192), dtype=torch.float16, device: int = 0) -> dict:
    best, per = 0.0, {}
    for n in sizes:
        a = torch.randn(n, n, device=f"cuda:{device}", dtype=dtype)
        b = torch.randn(n, n, device=f"cuda:{device}", dtype=dtype)
        c = torch.empty_like(a)
        r = benchmark(lambda: torch.matmul(a, b, out=c), TimingConfig(cache_mode="hot", max_time_s=3.0), device)
        tf = 2 * n**3 / (r.median_us * 1e-6) / 1e12
        per[n] = {"median_us": r.median_us, "tflops": tf, "clock_mhz": r.telemetry_after.get("clock_mhz")}
        best = max(best, tf)
        del a, b, c
    torch.cuda.empty_cache()
    return {"fp16_f32acc_tflops_measured": best, "per_size": per}


def gemm_bytes(M: int, N: int, K: int, elem_bytes: int = 2) -> int:
    """Compulsory DRAM traffic: read A and B once, write C once."""
    return elem_bytes * (M * K + K * N + M * N)


def attainable_tflops(M: int, N: int, K: int, peak_tflops: float, bw_gbps: float, elem_bytes: int = 2) -> float:
    flops = 2 * M * N * K
    t_min = max(flops / (peak_tflops * 1e12), gemm_bytes(M, N, K, elem_bytes) / (bw_gbps * 1e9))
    return flops / t_min / 1e12
