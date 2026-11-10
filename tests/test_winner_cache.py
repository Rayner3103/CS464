"""Winner cache (`tune/winner_cache.py`) and its use by `tune.online.select_kernel`. CPU only: the environment is injected
and `search_shape` is replaced by a fake, so no GPU is needed."""
import json

import pytest

import tune.online as online
from tune.winner_cache import WinnerCache, backend_for

ENV = {"hw_id": "gpu-a", "triton": "3.7.1", "cublaslt": 130000, "driver": "595"}
KEY = "bm64_bn64_bk32_w4_s3_g1_sk1"


def _cands(k=3, lt=0, winner=KEY):
    c = [{"config_key": winner if i == 0 else f"bm32_bn32_bk32_w4_s{2 + i}_g1_sk1", "valid": True, "failure_reason": None,
          "compile_s": 0.1, "cold_us": 10.0 + i, "time_s": 0.01} for i in range(k)]
    return c + [{"config_key": f"h{i}_a1_t2_s3_sk1_r0_sw0_c0", "family": "cublaslt", "valid": True, "failure_reason": None,
                 "compile_s": 0.0, "cold_us": 20.0 + i, "time_s": 0.01} for i in range(lt)]


@pytest.fixture
def fake_search(monkeypatch):
    class Calls(list):
        knn: list
    calls = Calls()
    calls_knn = calls.knn = []
    calls.rank = []
    calls.router = []

    def fake(ranker, M, N, K, k, dtype, order, lt_n, lt_knn=0, lt_rank=0, router=None, confirm=0):
        calls.append((M, N, K, k, lt_n))
        calls.router.append(router)
        calls_knn.append(lt_knn)
        calls.rank.append(lt_rank)
        return {"rank_s": 0.1, "candidates": _cands(k, lt_n)}
    monkeypatch.setattr(online, "search_shape", fake)
    return calls


def test_put_get_roundtrip_and_reload(tmp_path):
    wc = WinnerCache(tmp_path / "w.jsonl", env=ENV)
    ch = {"key": KEY, "family": "triton", "mode": "cold", "rank": 0, "compile_s": 0.3, "time_s": 0.03}
    assert wc.get(64, 64, 64, "fp16", k=3, lt_n=0) is None
    row = wc.put(64, 64, 64, "fp16", k=3, lt_n=0, choice=ch, candidates=_cands())
    assert row["winner_us"] == 10.0 and row["choice"]["key"] == KEY
    again = WinnerCache(tmp_path / "w.jsonl", env=ENV)         # a new process reads what was appended
    assert len(again) == 1 and again.get(64, 64, 64, "fp16", k=3, lt_n=0)["choice"]["key"] == KEY
    assert again.get(64, 64, 64, "bf16", k=3, lt_n=0) is None   # dtype is part of the key


def test_reuse_needs_an_equally_wide_search(tmp_path):
    wc = WinnerCache(tmp_path / "w.jsonl", env=ENV)
    wc.put(8, 8, 8, "fp16", k=3, lt_n=4, choice={"key": KEY, "family": "triton"})
    assert wc.get(8, 8, 8, "fp16", k=3, lt_n=4) and wc.get(8, 8, 8, "fp16", k=1, lt_n=0)
    assert wc.get(8, 8, 8, "fp16", k=5, lt_n=4) is None         # frequent shape: search again
    assert wc.get(8, 8, 8, "fp16", k=3, lt_n=16) is None        # more cuBLASLt candidates asked for
    wc.put(8, 8, 8, "fp16", k=5, lt_n=16, choice={"key": "bm16_bn32_bk32_w2_s2_g1_sk1", "family": "triton"})
    assert wc.get(8, 8, 8, "fp16", k=5, lt_n=16)["choice"]["key"] == "bm16_bn32_bk32_w2_s2_g1_sk1"  # last line wins


def test_other_environment_and_damaged_lines_are_ignored(tmp_path):
    p = tmp_path / "w.jsonl"
    WinnerCache(p, env=ENV).put(8, 8, 8, "fp16", k=3, lt_n=0, choice={"key": KEY, "family": "triton"})
    with open(p, "a") as f:
        f.write('{"schema": 1, "env": {"hw_id": "gpu-a"')       # a torn last line after a crash
    assert WinnerCache(p, env=ENV).get(8, 8, 8, "fp16") is not None
    for field, v in (("cublaslt", 130100), ("triton", "3.8.0"), ("hw_id", "gpu-b"), ("driver", "600")):
        assert WinnerCache(p, env={**ENV, field: v}).get(8, 8, 8, "fp16") is None, field
    assert len(p.read_text().splitlines()) == 2                 # nothing is ever deleted from the file


def test_no_valid_winner_is_not_stored(tmp_path):
    wc = WinnerCache(tmp_path / "w.jsonl", env=ENV)
    assert wc.put(8, 8, 8, "fp16", k=3, lt_n=0, choice={"key": None, "mode": None}) is None
    assert not (tmp_path / "w.jsonl").exists()


def test_select_kernel_searches_once_then_serves_from_cache(tmp_path, fake_search):
    wc = WinnerCache(tmp_path / "w.jsonl", env=ENV)
    kw = dict(cache=wc, use_bundle=False, lt_n=4)
    a = online.select_kernel(None, 64, 64, 64, **kw)
    b = online.select_kernel(None, 64, 64, 64, **kw)
    assert len(fake_search) == 1 and not a["cached"] and b["cached"]
    assert b["choice"]["key"] == a["choice"]["key"] == KEY and b["reason"] == "cached" and b["candidates"] == []
    assert b["rank_s"] == 0.0 and b["k"] == 3
    # a caller that later marks the shape frequent (k = 5) pays one wider search, which then also serves k = 3
    c = online.select_kernel(None, 64, 64, 64, calls=2_000_000, **kw)
    d = online.select_kernel(None, 64, 64, 64, **kw)
    assert not c["cached"] and d["cached"] and d["k"] == 5 and [x[3] for x in fake_search] == [3, 5]
    # another shape is a miss
    online.select_kernel(None, 128, 64, 64, **kw)
    assert len(fake_search) == 3 and len(wc) == 2


def test_select_kernel_cache_off_always_searches(tmp_path, fake_search, monkeypatch):
    for cache in (None, False):
        online.select_kernel(None, 64, 64, 64, cache=cache, use_bundle=False)
        online.select_kernel(None, 64, 64, 64, cache=cache, use_bundle=False)
    assert len(fake_search) == 4
    monkeypatch.setenv("TUNE_WINNER_CACHE", "off")
    assert WinnerCache.default() is None


def test_a_broken_cache_never_breaks_selection(tmp_path, fake_search):
    class Broken:
        def get(self, *a, **k):
            raise OSError("disk gone")

        def put(self, *a, **k):
            raise OSError("disk gone")
    with pytest.warns(UserWarning):
        sel = online.select_kernel(None, 64, 64, 64, cache=Broken(), use_bundle=False)
    assert sel["choice"]["key"] == KEY and not sel["cached"]


def test_stale_cublaslt_winner_is_a_miss(tmp_path, fake_search, monkeypatch):
    wc = WinnerCache(tmp_path / "w.jsonl", env=ENV)
    wc.put(64, 64, 64, "fp16", k=3, lt_n=4, lt_knn=online.KNN_DEFAULT,
           choice={"key": "h0_a9_t9_s9_sk1_r0_sw0_c0", "family": "cublaslt"})
    monkeypatch.setattr("tune.winner_cache.backend_for", lambda *a, **k: None)  # the library no longer returns that key
    sel = online.select_kernel(None, 64, 64, 64, cache=wc, use_bundle=False, lt_n=4)
    assert not sel["cached"] and len(fake_search) == 1


def test_backend_for_triton_choice():
    b = backend_for({"key": KEY, "family": "triton"}, 64, 64, 64)
    assert b.cfg.key() == KEY and hasattr(b, "make_runner")
    assert backend_for({"key": None}, 64, 64, 64) is None


def test_entry_is_plain_json(tmp_path):
    wc = WinnerCache(tmp_path / "w.jsonl", env=ENV)
    wc.put(8, 8, 8, "fp16", k=3, lt_n=0, choice={"key": KEY, "family": "triton", "mode": "graph"},
           candidates=[{"config_key": KEY, "graph_us": 4.2, "cold_us": 9.0}], order="diverse", ranker="lgbm_rank/v2")
    row = json.loads((tmp_path / "w.jsonl").read_text())
    assert row["winner_us"] == 4.2 and row["ranker"] == "lgbm_rank/v2" and row["env"] == ENV


@pytest.fixture
def no_lt_ranker(monkeypatch):
    monkeypatch.setattr("baselines.cublaslt_rank.ranker", lambda stem=None: None)


@pytest.fixture
def with_lt_ranker(monkeypatch):
    monkeypatch.setattr("baselines.cublaslt_rank.ranker", lambda stem=None: object())


def test_without_a_ranker_model_the_default_search_uses_knn_proposals(tmp_path, fake_search, no_lt_ranker):
    wc = WinnerCache(tmp_path / "w.jsonl", env=ENV)
    assert online.KNN_DEFAULT == 8 and online.RANK_DEFAULT == 5
    online.select_kernel(None, 64, 64, 64, cache=wc, use_bundle=False, lt_n=4)
    assert fake_search.knn == [8] and fake_search.rank == [0]
    assert wc.get(64, 64, 64, "fp16", k=3, lt_n=4, lt_knn=8) and wc.get(64, 64, 64, "fp16", k=3, lt_n=4, lt_knn=5)


def test_with_a_ranker_model_the_default_search_uses_it_instead_of_knn(tmp_path, fake_search, with_lt_ranker):
    wc = WinnerCache(tmp_path / "w.jsonl", env=ENV)
    online.select_kernel(None, 64, 64, 64, cache=wc, use_bundle=False, lt_n=4)
    assert fake_search.knn == [0] and fake_search.rank == [5]
    assert wc.get(64, 64, 64, "fp16", k=3, lt_n=4, lt_rank=5) and wc.get(64, 64, 64, "fp16", k=3, lt_n=4, lt_rank=3)
    assert wc.get(64, 64, 64, "fp16", k=3, lt_n=4, lt_rank=8) is None            # a wider ranker search was asked for
    online.select_kernel(None, 64, 64, 64, cache=wc, use_bundle=False, lt_n=4)    # served from the cache
    assert len(fake_search) == 1
    online.select_kernel(None, 64, 64, 64, cache=wc, use_bundle=False, lt_n=4, lt_rank=0, lt_knn=8)  # opt into the proposals
    assert fake_search.knn == [0, 8] and fake_search.rank == [5, 0]


def test_knn_and_ranker_are_off_for_a_triton_only_search(tmp_path, fake_search, with_lt_ranker):
    wc = WinnerCache(tmp_path / "w.jsonl", env=ENV)
    online.select_kernel(None, 128, 64, 64, cache=wc, use_bundle=False, lt_n=0)
    assert fake_search.knn == [0] and fake_search.rank == [0]
    online.select_kernel(None, 128, 64, 64, cache=wc, use_bundle=False, lt_n=0)
    assert len(fake_search) == 1                                                  # served from the cache


def test_entry_without_proposals_is_a_miss_when_they_are_requested(tmp_path):
    wc = WinnerCache(tmp_path / "w.jsonl", env=ENV)
    wc.put(32, 32, 32, "fp16", k=3, lt_n=4, choice={"key": KEY, "family": "triton"})   # written before proposals existed
    assert wc.get(32, 32, 32, "fp16", k=3, lt_n=4)["choice"]["key"] == KEY
    assert wc.get(32, 32, 32, "fp16", k=3, lt_n=4, lt_knn=8) is None and wc.get(32, 32, 32, "fp16", k=3, lt_n=4, lt_rank=5) is None


class _FakeBooster:
    def __init__(self, p):
        self.p = p

    def predict(self, X):
        return [self.p] * len(X)


def _router(p, thr=0.02):
    from tune.router import Router
    return Router(_FakeBooster(p), {"threshold": thr, "hw_id": None, "n_tune_shapes": 1, "oof_auc": 0.8})


def test_router_skips_triton_only_below_its_threshold():
    assert _router(0.01).skip_triton(64, 64, 64, 10.0) == (True, 0.01)
    skip, p = _router(0.30).skip_triton(64, 64, 64, 10.0)
    assert skip is False and p == 0.30
    from tune.router import FEATURES, features
    assert len(features(64, 4096, 11008, 120.0)) == len(FEATURES)


def test_router_is_on_only_when_the_bundle_does_not_serve_the_kernels(tmp_path, fake_search, monkeypatch):
    import kernels.bundle as bundle
    rt = _router(0.3)
    monkeypatch.setattr("tune.router.router", lambda stem=None: rt)
    kw = dict(use_bundle=True, lt_n=4)
    monkeypatch.setattr(bundle, "install", lambda *a, **k: {"installed": True, "matches": True})
    online.select_kernel(None, 64, 64, 64, cache=None, **kw)                       # bundle serves Triton: router off in auto
    monkeypatch.setattr(bundle, "install", lambda *a, **k: {"installed": True, "matches": False, "mismatch": ["triton"]})
    online.select_kernel(None, 64, 64, 65, cache=None, **kw)                       # bundle built for another Triton: compiles are real
    monkeypatch.setattr(bundle, "install", lambda *a, **k: {"installed": False, "reason": "no bundle"})
    online.select_kernel(None, 64, 64, 66, cache=None, **kw)
    online.select_kernel(None, 64, 64, 67, cache=None, use_bundle=False, lt_n=0)   # Triton-only search: nothing to route
    online.select_kernel(None, 64, 64, 68, cache=None, use_bundle=False, lt_n=4, router=False)
    assert fake_search.router == [None, rt, rt, None, None]
    monkeypatch.setattr(bundle, "install", lambda *a, **k: {"installed": True, "matches": True})
    online.select_kernel(None, 64, 64, 69, cache=None, router=True, **kw)          # forced on even with a serving bundle
    assert fake_search.router[-1] is rt
