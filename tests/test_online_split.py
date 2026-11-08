"""`tune.online.search_shape` = `propose_lt` + `propose_triton` (ranking, needs the models) + `measure` (GPU half, needs none),
and the key round trip that lets another process / interpreter measure a proposal. CPU only: validation, timing, inputs
and the cuBLASLt candidate sources are replaced by fakes."""
import itertools

import pytest
import torch

import tune.online as online

KEYS = ["bm64_bn64_bk32_w4_s3_g1_sk1", "bm16_bn32_bk32_w2_s2_g1_sk1", "bm128_bn64_bk32_w4_s3_g1_sk2"]


class _R:
    def __init__(self, **k):
        self.__dict__.update(k)


def _key(b):
    return b.cfg.key() if hasattr(b.cfg, "key") else b.key


class _LT:
    def __init__(self, key, family="cublaslt", cfg=None):
        self.key, self.family, self.cfg, self.problem = key, family, cfg or {"algo_id": key}, "P"

    def make_runner(self, a, b):
        return _key(self)


@pytest.fixture
def fakes(monkeypatch):
    from kernels.triton_matmul import TritonMatmul
    from tune.sweep import parse_key
    log = []
    clock = itertools.count()
    monkeypatch.setattr(online.time, "perf_counter", lambda: float(next(clock)))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda *a: None)
    monkeypatch.setattr(online, "placed_inputs", lambda *a, **k: ("a", "b", None))
    monkeypatch.setattr(online, "reference", lambda a, b: "ref")
    monkeypatch.setattr(online, "validate", lambda b, *a, **k: log.append(("validate", _key(b))) or
                        _R(valid=not _key(b).startswith("bm16"), failure_reason=None))
    monkeypatch.setattr(TritonMatmul, "make_runner", lambda self, a, b: _key(self))
    monkeypatch.setattr("tune.timing.benchmark", lambda runner, cfg: log.append(("bench", runner)) or
                        _R(median_us=100.0 + len(runner)))
    monkeypatch.setattr("baselines.cublaslt.candidates", lambda M, N, K, dtype, n:
                        [_LT(f"h{i}_a{i}_t1_s1_sk1_r0_sw0_c0") for i in range(min(n, 3))])
    monkeypatch.setattr("baselines.cublaslt_rank.candidates", lambda M, N, K, dtype, m, problem=None, skip=None:
                        [_LT(f"r{i}_a{9 + i}_t3_s1_sk1_r0_sw0_c0", "cublaslt_rank") for i in range(m)])
    monkeypatch.setattr("baselines.cublaslt_knn.from_key", lambda key, M, N, K, dtype:
                        None if "a99" in key else _LT(key, "cublaslt_rank"))

    class Ranker:
        pool_keys = KEYS
        cfgs = [parse_key(k) for k in KEYS]

        def rank(self, M, N, K, k, order="plain"):
            log.append(("rank", k))
            return KEYS[:k]
    return log, Ranker()


def test_search_shape_is_propose_then_measure(fakes):
    log, ranker = fakes
    full = online.search_shape(ranker, 64, 64, 64, 3, torch.float16, "diverse", lt_n=4, lt_rank=2)
    lt, lt_setup = online.propose_lt(64, 64, 64, torch.float16, 4, 0, 2)
    tri, rank_s = online.propose_triton(ranker, 64, 64, 64, 3, "diverse")
    part = online.measure(64, 64, 64, torch.float16, lt, tri, lt_n=4)
    strip = lambda cs: [{k: v for k, v in c.items() if not k.endswith("_s")} for c in cs]  # noqa: E731
    assert strip(part["candidates"]) == strip(full["candidates"])
    assert [c["config_key"] for c in full["candidates"]][:3] == KEYS and full["lt_rank"] == 2 and "lt_setup_s" in full
    assert not full["candidates"][1]["valid"] and "cold_us" not in full["candidates"][1]
    # timing order of the original search_shape: without a router the Triton half is timed first, then cuBLASLt
    log.clear()
    online.search_shape(ranker, 64, 64, 64, 3, torch.float16, "diverse", lt_n=4, lt_rank=2)
    benched = [x[1] for x in log if x[0] == "bench"]
    assert benched[:2] == [KEYS[0], KEYS[2]] and benched[2].startswith("h0_") and benched[-1].startswith("r1_")
    assert log[0] == ("rank", 3)                     # ranking happens after the cuBLASLt proposals, before any timing


def test_a_router_skip_never_ranks_the_triton_half(fakes):
    log, ranker = fakes

    class Router:
        threshold = 0.02

        def skip_triton(self, M, N, K, t):
            return True, 0.001
    r = online.search_shape(ranker, 64, 64, 64, 3, torch.float16, "diverse", lt_n=4, router=Router(), confirm=2)
    assert r["router"]["skipped"] and r["rank_s"] == 0.0 and not [x for x in log if x[0] == "rank"]
    assert all(c["family"] == "cublaslt" for c in r["candidates"])
    assert sum(1 for c in r["candidates"] if c["confirm_us"] != c["cold_us"]) <= 2


def test_proposal_keys_round_trip_through_another_process(fakes):
    log, ranker = fakes
    lt, _ = online.propose_lt(64, 64, 64, torch.float16, 4, 0, 2)
    tri, _ = online.propose_triton(ranker, 64, 64, 64, 3)
    keys = online.proposal_keys(tri + lt) + [{"family": "cublaslt_rank", "key": "r9_a99_t1_s1_sk1_r0_sw0_c0"},
                                              {"family": "cublaslt", "key": "h7_gone"}]
    back = online.backends_from_keys(keys, 64, 64, 64, torch.float16, lt_n=4)
    assert [k for _, k, _, _ in back] == [k["key"] for k in keys[:-2]]          # unknown keys are dropped, order kept
    assert [s for *_, s in back] == [k["setup_s"] for k in keys[:-2]]
    assert back[0][2].cfg.key() == KEYS[0]


def test_inputs_that_do_not_fit_are_a_skip(fakes, monkeypatch):
    log, ranker = fakes

    def oom(*a, **k):
        raise torch.OutOfMemoryError("CUDA out of memory. Tried")
    monkeypatch.setattr(online, "placed_inputs", oom)
    r = online.search_shape(ranker, 1 << 20, 1 << 20, 64, 3, lt_n=4)
    assert r["skipped"].startswith("oom_inputs") and r["candidates"] == [] and not log
    assert online.measure(8, 8, 8, torch.float16, [], [], lt_n=0)["skipped"].startswith("oom_inputs")


def test_no_ranker_is_a_cuBLASLt_only_search_stored_with_k0(fakes, tmp_path):
    from tune.winner_cache import WinnerCache
    log, ranker = fakes
    assert online.propose_triton(None, 64, 64, 64, 3) == ([], 0.0) and online.propose_triton(ranker, 64, 64, 64, 0) == ([], 0.0)
    wc = WinnerCache(tmp_path / "w.jsonl", env={"hw_id": "gpu"})
    r = online.select_kernel(online.LT_ONLY, 64, 64, 64, calls=10**9, lt_n=4, lt_rank=0, lt_knn=0, cache=wc, use_bundle=False,
                             router=True)
    assert r["k"] == 0 and r["reason"] == "lt_only" and r["choice"]["family"] == "cublaslt"
    assert all(c["family"] == "cublaslt" for c in r["candidates"]) and not [x for x in log if x[0] == "rank"]
    e = wc.get(64, 64, 64, "fp16", k=0)
    assert e["k"] == 0 and e["ranker"] is None and wc.get(64, 64, 64, "fp16", k=1) is None   # a ranker search is wider
    again = online.select_kernel(online.LT_ONLY, 64, 64, 64, lt_n=4, lt_rank=0, lt_knn=0, cache=wc, use_bundle=False)
    assert again["cached"] and again["choice"]["key"] == r["choice"]["key"]
    # defaults (lt_rank 5, lt_knn auto) are overridden: the heuristic alone, no proposals
    online.select_kernel(online.LT_ONLY, 96, 64, 64, lt_n=4, cache=wc, use_bundle=False)
    e = wc.get(96, 64, 64, "fp16", k=0)
    assert e["k"] == 0 and e["lt_rank"] == 0 and e["lt_knn"] == 0
    none = online.select_kernel(online.LT_ONLY, 64, 64, 64, lt_n=0, cache=wc, use_bundle=False)
    assert none["choice"] is None and none["reason"] == "no_candidates"
