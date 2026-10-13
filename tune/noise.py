"""Noise-floor study (plan 3.3) and thermal-drift run.

noise: for each shape in the noise set, time fixed kernels with exactly `repeats` samples, for `rounds`
rounds per session. Shapes are visited in a shuffled order each round, so slow drift spreads over shapes
instead of aliasing onto one. Rows go to noise.jsonl, tagged with session_id and round. Re-running on
other days with new session ids gives the across-session component.

drift: run one fixed large GEMM continuously for N minutes. Every ~1 s, log the per-launch time and
GPU clock/temp/power to drift.csv.
"""
from __future__ import annotations

import csv
import random
import time
import uuid
from pathlib import Path

import torch

from tune.correctness import make_inputs, placed_inputs
from tune.env import gpu_telemetry, provenance, write_env
from tune.records import RecordWriter
from tune.runner import DTYPES, LAYOUT, candidates_for
from tune.timing import TimingConfig, benchmark, pool


def run_noise(shapes: list[dict], out_dir: Path, *, backends=("torch", "triton:bm128_bn128_bk32_w4_s4_g8_sk1"),
              repeats: int = 100, rounds: int = 3, cache_modes=("cold", "hot"), dtype: str = "fp16",
              session_id: str | None = None, seed: int = 0, n_allocs: int = 1, log=print) -> Path:
    session_id = session_id or "noise-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    env = write_env(out_dir, session_id=session_id, kind="noise",
                    extra={"noise_params": {"repeats": repeats, "rounds": rounds, "n_allocs": n_allocs,
                                            "backends": list(backends), "cache_modes": list(cache_modes)}})
    prov = provenance(env)
    rng = random.Random(seed)
    path = out_dir / "noise.jsonl"
    tdtype = DTYPES[dtype]
    t0 = time.perf_counter()
    with RecordWriter(path) as w:
        for rnd in range(rounds):
            order = list(shapes)
            rng.shuffle(order)
            for i, s in enumerate(order):
                M, N, K = s["M"], s["N"], s["K"]
                cands = [c for spec in backends for c in candidates_for(spec, M, N, K)]
                timed = {(c.config_key, cm): [] for c in cands for cm in cache_modes}
                for alloc in range(n_allocs):
                    # Fresh placement each round as well, so rounds sample different physical layouts.
                    a, b, pad = placed_inputs(M, N, K, tdtype, placement=rnd * n_allocs + alloc)
                    for c in cands:
                        fn = c.backend.make_runner(a, b)
                        for cm in cache_modes:
                            cfg = TimingConfig(cache_mode=cm, min_reps=repeats, max_reps=repeats, target_rse=0.0,
                                               max_time_s=60.0)
                            timed[(c.config_key, cm)].append(benchmark(fn, cfg))
                    del a, b, pad
                for c in cands:
                    for cm in cache_modes:
                        parts = timed[(c.config_key, cm)]
                        r = pool(parts)
                        tel = r.telemetry_after
                        w.write({"hw_id": env["hw_id"], "driver": env["driver"], "torch": env["versions"]["torch"],
                                 "triton": env["versions"]["triton"], "cuda": env["versions"]["torch_cuda"],
                                 "M": M, "N": N, "K": K, "dtype": dtype, "layout": LAYOUT,
                                 "kernel_family": c.family, "config_key": c.config_key, "config": c.config,
                                 "valid": True, "cache_mode": cm, "batch": 1, "use_graph": False,
                                 "time_median_us": r.median_us, "time_p10_us": r.p10_us, "time_p90_us": r.p90_us,
                                 "time_min_us": r.min_us, "time_max_us": r.max_us, "time_iqr_us": r.iqr_us,
                                 "rse": r.rse, "n_samples": r.n_samples, "converged": r.converged,
                                 "n_allocs": n_allocs, "alloc_medians_us": [x.median_us for x in parts],
                                 "tflops": 2 * M * N * K / (r.median_us * 1e-6) / 1e12,
                                 "clock_mhz": tel.get("clock_mhz"), "temp_c": tel.get("temp_c"),
                                 "power_w": tel.get("power_w"), "throttle_reasons": tel.get("throttle_reasons"),
                                 "timestamp": time.time(), "session_id": session_id, "round": rnd,
                                 "suite": "noise", "suite_origin": s.get("suite_origin")},
                                samples_us=r.samples_us)
                torch.cuda.empty_cache()
                log(f"[noise r{rnd} {i + 1}/{len(order)}] {M}x{N}x{K} | {time.perf_counter() - t0:.0f}s")
    return path


def run_drift(out_dir: Path, minutes: float = 10.0, n: int = 8192, interval_s: float = 1.0, log=print) -> Path:
    stamp = time.strftime('%Y%m%d-%H%M%S')
    write_env(out_dir, session_id=f"drift-{stamp}", kind="drift")
    a, b = make_inputs(n, n, n)
    c = torch.empty((n, n), device="cuda", dtype=a.dtype)
    path = out_dir / f"drift-{stamp}.csv"
    t_end = time.perf_counter() + minutes * 60
    t0 = time.perf_counter()
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["t_s", "iters", "median_us", "min_us", "max_us", "clock_mhz", "mem_clock_mhz", "temp_c",
                     "power_w", "throttle_reasons"])
        while time.perf_counter() < t_end:
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            times = []
            t_int = time.perf_counter()
            while time.perf_counter() - t_int < interval_s:
                s.record()
                torch.matmul(a, b, out=c)
                e.record()
                e.synchronize()
                times.append(s.elapsed_time(e) * 1e3)
            times.sort()
            tel = gpu_telemetry()
            wr.writerow([round(time.perf_counter() - t0, 2), len(times), times[len(times) // 2], times[0], times[-1],
                         tel["clock_mhz"], tel["mem_clock_mhz"], tel["temp_c"], tel["power_w"], tel["throttle_reasons"]])
            f.flush()
            if int(time.perf_counter() - t0) % 60 < interval_s:
                log(f"[drift {time.perf_counter() - t0:.0f}s] {times[len(times) // 2]:.0f}us {tel}")
    return path
