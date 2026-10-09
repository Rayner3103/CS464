import math

import pytest
import torch

from tests.conftest import requires_cuda
from tune.timing import TimingConfig, benchmark, summarize


def test_summarize_basic_stats():
    s = summarize([1.0, 2.0, 3.0, 4.0, 100.0])
    assert s["median_us"] == 3.0
    assert s["min_us"] == 1.0 and s["max_us"] == 100.0
    assert s["n_samples"] == 5
    assert s["p25_us"] == 2.0 and s["p75_us"] == 4.0


def test_summarize_rse_and_convergence():
    s = summarize([10.0] * 50)
    assert s["rse"] == 0.0 and s["converged"]
    s = summarize([1.0, 100.0])
    assert s["rse"] > 0.01 and not s["converged"]
    assert math.isinf(summarize([5.0])["rse"])


def test_config_validation():
    with pytest.raises(ValueError):
        TimingConfig(cache_mode="warm")
    with pytest.raises(ValueError):
        TimingConfig(cache_mode="cold", batch=10)  # batching would make launches 2..n hot
    with pytest.raises(ValueError):
        TimingConfig(min_reps=10, max_reps=5)


@requires_cuda
def test_benchmark_matches_known_sleep():
    # A device-side spin of a known cycle count: event timing must recover its duration.
    from tune.timing import _ClockRate
    cycles = int(_ClockRate.get() * 200e-6)  # ~200 us
    r = benchmark(lambda: torch.cuda._sleep(cycles), TimingConfig(cache_mode="hot", max_time_s=0.5))
    assert 150 < r.median_us < 260, r.median_us
    assert r.n_samples >= 10
    assert r.samples_us is not None and len(r.samples_us) == r.n_samples


@requires_cuda
def test_respects_rep_caps():
    a = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    r = benchmark(lambda: a @ a, TimingConfig(cache_mode="hot", min_reps=37, max_reps=37, target_rse=0.0))
    assert r.n_samples == 37 and not r.converged


@requires_cuda
def test_cold_is_not_faster_than_hot_for_l2_resident_kernel():
    # 1024^2 fp16 operands (2 MiB each) fit in L2 on recent GPUs: hot should be <= cold (plus noise).
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    c = torch.empty_like(a)
    hot = benchmark(lambda: torch.matmul(a, a, out=c), TimingConfig(cache_mode="hot"))
    cold = benchmark(lambda: torch.matmul(a, a, out=c), TimingConfig(cache_mode="cold"))
    assert hot.median_us <= cold.median_us * 1.05


@requires_cuda
def test_graph_amortized_per_launch():
    a = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    c = torch.empty_like(a)
    g = benchmark(lambda: torch.matmul(a, a, out=c), TimingConfig(cache_mode="hot", batch=50, use_graph=True))
    single = benchmark(lambda: torch.matmul(a, a, out=c), TimingConfig(cache_mode="hot"))
    assert g.batch == 50
    # Per-launch graph time excludes per-sample event overhead, so it can't be meaningfully larger.
    assert g.median_us <= single.median_us * 1.1
