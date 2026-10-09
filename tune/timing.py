"""GPU timing protocol (plan 3.1).

- CUDA events around the launch, never Python wall-clock.
- Time-based warmup, then adaptive repeats until relative std-error < target or a cap is hit.
- Raw samples kept; median is the primary statistic.
- cache_mode="cold": L2 flushed (outside the timed region) before every sample.
  cache_mode="hot": no flush; operands stay L2-resident across back-to-back calls.
- A GPU-side sleep is queued ahead of each chunk so the stream is backlogged: kernels then start
  back-to-back on the device and the events measure device time, not Python launch latency.
- batch > 1 times `batch` back-to-back launches per sample (amortized per-launch = sample / batch);
  use_graph=True captures those launches into a CUDA graph to strip launch overhead entirely.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np
import torch

from tune.env import gpu_telemetry

CACHE_MODES = ("cold", "hot")


@dataclass
class TimingConfig:
    cache_mode: str = "cold"
    warmup_ms: float = 25.0         # time-based warmup after the first (compile) call
    min_warmup_iters: int = 3
    min_reps: int = 10
    max_reps: int = 1000
    target_rse: float = 0.01        # stop once std/sqrt(n)/mean drops below this
    max_time_s: float = 2.0         # wall-clock cap per measurement (excluding warmup)
    batch: int = 1                  # launches per timed sample
    use_graph: bool = False
    flush_bytes: int | None = None  # default: max(8 * L2, 64 MiB)
    keep_raw: bool = True

    def __post_init__(self):
        if self.cache_mode not in CACHE_MODES:
            raise ValueError(f"cache_mode must be one of {CACHE_MODES}, got {self.cache_mode!r}")
        if self.cache_mode == "cold" and self.batch != 1:
            raise ValueError("cold mode flushes L2 before each sample; batch>1 would make launches 2..n hot")
        if self.batch < 1 or self.min_reps < 1 or self.max_reps < self.min_reps:
            raise ValueError("need batch >= 1 and 1 <= min_reps <= max_reps")


@dataclass
class TimingResult:
    """All times in microseconds, per launch (sample time / batch)."""
    median_us: float
    mean_us: float
    std_us: float
    min_us: float
    max_us: float
    p10_us: float
    p25_us: float
    p75_us: float
    p90_us: float
    p99_us: float
    n_samples: int
    rse: float
    converged: bool
    cache_mode: str
    batch: int
    use_graph: bool
    wall_s: float
    telemetry_before: dict = field(default_factory=dict)
    telemetry_after: dict = field(default_factory=dict)
    samples_us: list[float] | None = None

    @property
    def iqr_us(self) -> float:
        return self.p75_us - self.p25_us

    def summary(self) -> dict:
        d = asdict(self)
        d.pop("samples_us")
        d["iqr_us"] = self.iqr_us
        return d


def pool(results: list["TimingResult"], rse_target: float = 0.01) -> "TimingResult":
    """Merge measurements of the same kernel taken on different input allocations into one result.

    Physical placement of the operands shifts memory-bound kernels by several percent
    (docs/methodology.md, "allocation noise"); pooling the raw samples over fresh allocations averages it out.
    """
    if len(results) == 1:
        return results[0]
    samples = [s for r in results for s in (r.samples_us or [])]
    stats = summarize(samples, rse_target=rse_target)
    stats["converged"] = all(r.converged for r in results)
    first, last = results[0], results[-1]
    return TimingResult(**stats, cache_mode=first.cache_mode, batch=first.batch, use_graph=first.use_graph,
                        wall_s=sum(r.wall_s for r in results), telemetry_before=first.telemetry_before,
                        telemetry_after=last.telemetry_after, samples_us=samples)


def summarize(samples_us, *, rse_target: float = 0.01) -> dict:
    s = np.asarray(samples_us, dtype=np.float64)
    n = len(s)
    mean = float(s.mean())
    std = float(s.std(ddof=1)) if n > 1 else 0.0
    rse = std / math.sqrt(n) / mean if n > 1 and mean > 0 else math.inf
    q = np.percentile(s, [10, 25, 50, 75, 90, 99])
    return {"median_us": float(q[2]), "mean_us": mean, "std_us": std, "min_us": float(s.min()),
            "max_us": float(s.max()), "p10_us": float(q[0]), "p25_us": float(q[1]), "p75_us": float(q[3]),
            "p90_us": float(q[4]), "p99_us": float(q[5]), "n_samples": n, "rse": rse, "converged": rse <= rse_target}


class _L2Flusher:
    """Overwrites a buffer much larger than L2 so no operand lines survive into the timed region."""
    _cache: dict[tuple[int, int], torch.Tensor] = {}

    def __init__(self, nbytes: int | None, device: int):
        if nbytes is None:
            l2 = torch.cuda.get_device_properties(device).L2_cache_size
            nbytes = max(8 * l2, 64 * 2**20)
        key = (device, nbytes)
        if key not in self._cache:
            self._cache[key] = torch.empty(nbytes // 4, dtype=torch.int32, device=f"cuda:{device}")
        self.buf = self._cache[key]

    def __call__(self):
        self.buf.zero_()


class _ClockRate:
    """Cycles/sec of the GPU-side sleep primitive, estimated once per process."""
    hz: float | None = None

    @classmethod
    def get(cls) -> float:
        if cls.hz is None:
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            cycles = 10_000_000
            torch.cuda._sleep(1000)
            s.record()
            torch.cuda._sleep(cycles)
            e.record()
            e.synchronize()
            cls.hz = cycles / (s.elapsed_time(e) / 1e3)
        return cls.hz


def _backlog(seconds: float):
    """Queue a device-side spin so the CPU can enqueue a whole chunk before the GPU reaches it."""
    if seconds > 0:
        torch.cuda._sleep(int(min(seconds, 0.2) * _ClockRate.get()))


def benchmark(fn: Callable[[], object], cfg: TimingConfig | None = None, device: int = 0) -> TimingResult:
    """Time `fn` (which must only enqueue GPU work on the current stream)."""
    cfg = cfg or TimingConfig()
    torch.cuda.set_device(device)
    flush = _L2Flusher(cfg.flush_bytes, device) if cfg.cache_mode == "cold" else None

    if cfg.batch > 1 and not cfg.use_graph:
        def run():
            for _ in range(cfg.batch):
                fn()
    else:
        run = fn

    # First call triggers JIT compilation / cuBLAS heuristics; never timed.
    fn()
    torch.cuda.synchronize()

    if cfg.use_graph:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):  # graph capture needs a warmed-up side stream
            fn()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(cfg.batch):
                fn()
        run = graph.replay

    # Warmup: estimate per-sample cost, then keep running until warmup_ms of device time has elapsed.
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(cfg.min_warmup_iters):
        if flush:
            flush()
        run()
    e.record()
    e.synchronize()
    est_ms = max(s.elapsed_time(e) / cfg.min_warmup_iters, 1e-3)
    extra = int(cfg.warmup_ms / est_ms)
    for _ in range(min(extra, 10_000)):
        if flush:
            flush()
        run()
    torch.cuda.synchronize()

    # Per-sample CPU enqueue cost, used to size the backlog sleep.
    t0 = time.perf_counter()
    if flush:
        flush()
    run()
    enqueue_s = time.perf_counter() - t0
    torch.cuda.synchronize()

    tel_before = gpu_telemetry(device)
    samples: list[float] = []
    t_start = time.perf_counter()
    chunk = int(min(max(50.0 / est_ms, 5), 100))  # ~50 ms of work per chunk
    if not cfg.use_graph:
        # The driver's pending-launch queue is ~1k entries; past that the CPU blocks, the backlog drains,
        # and we would measure Python launch rate instead of device time.
        chunk = min(chunk, max(1, 256 // cfg.batch))
    converged = False
    while True:
        n = min(chunk, cfg.max_reps - len(samples))
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        _backlog(2.0 * enqueue_s * n)
        for i in range(n):
            if flush:
                flush()
            starts[i].record()
            run()
            ends[i].record()
        ends[-1].synchronize()
        samples.extend(st.elapsed_time(en) * 1e3 / cfg.batch for st, en in zip(starts, ends))

        stats = summarize(samples, rse_target=cfg.target_rse)
        converged = len(samples) >= cfg.min_reps and stats["converged"]
        if converged or len(samples) >= cfg.max_reps or time.perf_counter() - t_start > cfg.max_time_s:
            break
    wall = time.perf_counter() - t_start
    tel_after = gpu_telemetry(device)

    stats = summarize(samples, rse_target=cfg.target_rse)
    stats["converged"] = converged
    return TimingResult(**stats, cache_mode=cfg.cache_mode, batch=cfg.batch, use_graph=cfg.use_graph,
                        wall_s=wall, telemetry_before=tel_before, telemetry_after=tel_after,
                        samples_us=samples if cfg.keep_raw else None)
