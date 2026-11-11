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


def test_a_cache_that_cannot_be_built_never_breaks_selection(tmp_path, fake_search, monkeypatch):
    """`cache=True` builds the default cache inside select_kernel: reading the environment (NVML, the kernel source, the
    library versions) can fail, and that is a cache problem like any other."""
    monkeypatch.setenv("TUNE_WINNER_CACHE", str(tmp_path / "w.jsonl"))

    def boom(**k):
        raise FileNotFoundError("kernels/triton_matmul.py")
    monkeypatch.setattr("tune.winner_cache.environment", boom)
    monkeypatch.setattr(online, "model_ids", lambda ranker: None)
    with pytest.warns(UserWarning, match="winner cache unavailable"):
        sel = online.select_kernel(None, 64, 64, 64, cache=True, use_bundle=False, lt_n=0)
    assert sel["choice"]["key"] == KEY and not sel["cached"] and not (tmp_path / "w.jsonl").exists()


def test_damaged_bytes_and_wrong_types_cost_their_line_not_the_cache(tmp_path, fake_search, monkeypatch):
    p = tmp_path / "w.jsonl"
    WinnerCache(p, env=ENV).put(64, 64, 64, "fp16", k=3, lt_n=0, choice={"key": KEY, "family": "triton"})
    good = p.read_bytes()
    bad_type = json.dumps({"schema": 2, "env": ENV, "M": "x", "N": 8, "K": 8, "dtype": "fp16", "k": 3, "lt_n": 0}).encode()
    p.write_bytes(b"\xff\xfe\x00 not utf-8\n" + bad_type + b"\n" + good)
    wc = WinnerCache(p, env=ENV)                                                    # used to raise UnicodeDecodeError
    assert len(wc) == 1 and wc.get(64, 64, 64, "fp16", k=3)["choice"]["key"] == KEY
    monkeypatch.setenv("TUNE_WINNER_CACHE", str(p))
    monkeypatch.setattr("tune.winner_cache.environment", lambda **k: dict(ENV))
    monkeypatch.setattr(online, "model_ids", lambda ranker: None)
    assert online.select_kernel(None, 64, 64, 64, cache=True, use_bundle=False, lt_n=0)["cached"]
    assert wc.compact() == {"before": 3, "after": 1} and p.read_bytes() == good    # compaction drops the unreadable lines


def _model(stem, text):
    for ext in (".txt", ".json"):
        (stem.parent / (stem.name + ext)).write_text(text + ext)
    return stem


def test_model_id_names_the_files_content(tmp_path):
    from tune.winner_cache import model_id
    a = model_id(_model(tmp_path / "lgbm_rank_v2", "trees"))
    assert a.startswith("lgbm_rank_v2:") and len(a.split(":")[1]) == 12 and a == model_id(tmp_path / "lgbm_rank_v2")
    assert model_id(_model(tmp_path / "lgbm_rank_v2", "other trees")) != a          # retrained under the same name
    assert model_id(None) is None and model_id(tmp_path / "missing") is None


def test_default_cache_declares_the_models_so_a_retrained_ranker_is_searched_again(tmp_path, fake_search, monkeypatch):
    """`select_kernel(cache=True)` after a retrain: the old winner came from another model's proposals. It is soft-stale
    (a miss by default), not served as current."""
    import types
    ranker = types.SimpleNamespace(stem=_model(tmp_path / "lgbm_rank_v2", "trees"), features="v2")
    monkeypatch.setenv("TUNE_WINNER_CACHE", str(tmp_path / "w.jsonl"))
    lt_stem = _model(tmp_path / "lt_rank", "lt")
    monkeypatch.setattr("baselines.cublaslt_rank.default_stem", lambda: lt_stem)
    monkeypatch.setattr("tune.router.default_stem", lambda: tmp_path / "no_router")
    monkeypatch.setattr("tune.winner_cache.environment",
                        lambda models=None, **k: {**ENV, **({"models": models} if models is not None else {})})
    kw = dict(cache=True, use_bundle=False, lt_n=0)
    a, b = online.select_kernel(ranker, 64, 64, 64, **kw), online.select_kernel(ranker, 64, 64, 64, **kw)
    assert not a["cached"] and b["cached"] and len(fake_search) == 1
    ids = json.loads((tmp_path / "w.jsonl").read_text())["env"]["models"]
    assert ids == online.model_ids(ranker) and ids["triton"].startswith("lgbm_rank_v2:") and ids["router"] is None
    assert ids["lt_rank"].startswith("lt_rank:")
    _model(tmp_path / "lgbm_rank_v2", "retrained trees")
    c = online.select_kernel(ranker, 64, 64, 64, **kw)
    assert not c["cached"] and len(fake_search) == 2                                # searched again with the new model
    old = WinnerCache(tmp_path / "w.jsonl", env={**ENV, "models": online.model_ids(ranker)})
    assert old.get(64, 64, 64, "fp16", k=3)["stale"] == []                          # ... and that result is current
    _model(tmp_path / "lt_rank", "retrained lt")                                    # any model of the set counts
    stale = WinnerCache(tmp_path / "w.jsonl", env={**ENV, "models": online.model_ids(ranker)})
    assert stale.get(64, 64, 64, "fp16", k=3) is None
    assert stale.get(64, 64, 64, "fp16", k=3, allow_stale=True)["stale"] == ["models"]
    # a cache the caller built keeps the caller's environment: nothing is added to it
    mine = WinnerCache(tmp_path / "mine.jsonl", env=ENV)
    online.select_kernel(ranker, 64, 64, 64, cache=mine, use_bundle=False, lt_n=0)
    assert "models" not in json.loads((tmp_path / "mine.jsonl").read_text())["env"]


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


# ---------------------------------------------------------------------------------------------- schema 2: staleness tiers

ENV2 = {**ENV, "torch": "2.13.0", "cuda_runtime": "13.0", "vbios": "94.02", "kernel_src": "aaaa", "clock_lock": "1755"}
SCOPED = {**ENV2, "env_id": "py-a"}     # one interpreter's entries: the scope in which `compact` prunes superseded lines
LT_CHOICE = {"key": "h0_a1_t2_s3_sk1_r0_sw0_c0", "family": "cublaslt"}


def test_soft_stale_entry_is_served_only_when_asked(tmp_path):
    p = tmp_path / "w.jsonl"
    WinnerCache(p, env=ENV2).put(8, 8, 8, "fp16", k=3, lt_n=0, choice={"key": KEY, "family": "triton"})
    for field, v in (("driver", "600"), ("torch", "2.14.0"), ("triton", "3.8.0"), ("cuda_runtime", "13.1"),
                     ("cublaslt", 130100), ("vbios", "94.03"), ("clock_lock", "unlocked"), ("models", {"v2": "x"})):
        wc = WinnerCache(p, env={**ENV2, field: v})
        assert wc.get(8, 8, 8, "fp16") is None and len(wc) == 0, field          # default: a stale entry is a miss
        hit = wc.get(8, 8, 8, "fp16", allow_stale=True)
        assert hit["choice"]["key"] == KEY and hit["stale"] == [field] and hit["verified"] is False, field
        assert [e["choice"]["key"] for e in wc.stale_entries()] == [KEY]
    cur = WinnerCache(p, env=ENV2).get(8, 8, 8, "fp16")
    assert cur["stale"] == [] and cur["verified"] is True


def test_hard_stale_entries_are_never_served(tmp_path):
    p = tmp_path / "w.jsonl"
    WinnerCache(p, env=ENV2).put(8, 8, 8, "fp16", k=3, lt_n=0, choice={"key": KEY, "family": "triton"})
    WinnerCache(p, env=ENV2).put(9, 9, 9, "fp16", k=3, lt_n=4, choice=LT_CHOICE)
    other_gpu = WinnerCache(p, env={**ENV2, "hw_id": "gpu-b"})
    assert other_gpu.get(8, 8, 8, "fp16", allow_stale=True) is None and other_gpu.get(9, 9, 9, "fp16", allow_stale=True) is None
    new_src = WinnerCache(p, env={**ENV2, "kernel_src": "bbbb"})
    assert new_src.get(8, 8, 8, "fp16", allow_stale=True) is None                 # a Triton key now names other code
    assert new_src.get(9, 9, 9, "fp16", k=3, lt_n=4, allow_stale=True)["stale"] == ["kernel_src"]  # cuBLASLt: unaffected


def test_undeclared_fields_are_not_compared_and_schema1_lines_are_soft(tmp_path):
    p = tmp_path / "w.jsonl"
    p.write_text(json.dumps({"schema": 1, "env": ENV, "M": 8, "N": 8, "K": 8, "dtype": "fp16", "k": 3, "lt_n": 0,
                             "choice": {"key": KEY, "family": "triton"}}) + "\n")
    wc = WinnerCache(p, env=ENV2)
    hit = wc.get(8, 8, 8, "fp16", allow_stale=True)
    assert wc.get(8, 8, 8, "fp16") is None and set(hit["stale"]) == {"torch", "cuda_runtime", "vbios", "kernel_src", "clock_lock"}
    assert hit["id"]                                                               # derived from the line
    assert WinnerCache(p, env={**ENV2, "torch": None, "cuda_runtime": None, "vbios": None, "kernel_src": None,
                                  "clock_lock": None}).get(8, 8, 8, "fp16")["stale"] == []


def test_current_entry_wins_over_a_wider_stale_one_only_when_wide_enough(tmp_path):
    p = tmp_path / "w.jsonl"
    WinnerCache(p, env={**ENV2, "driver": "590"}).put(8, 8, 8, "fp16", k=5, lt_n=16, choice={"key": "old", "family": "triton"})
    wc = WinnerCache(p, env=ENV2)
    wc.put(8, 8, 8, "fp16", k=3, lt_n=16, choice={"key": KEY, "family": "triton"})
    assert wc.get(8, 8, 8, "fp16", k=3, lt_n=16, allow_stale=True)["choice"]["key"] == KEY
    assert wc.get(8, 8, 8, "fp16", k=5, lt_n=16, allow_stale=True)["choice"]["key"] == "old"
    assert wc.get(8, 8, 8, "fp16", k=5, lt_n=16) is None and wc.stale_entries() == []


def test_mark_verified_persists_for_the_same_environment_only(tmp_path):
    p = tmp_path / "w.jsonl"
    WinnerCache(p, env={**ENV2, "driver": "590"}).put(8, 8, 8, "fp16", k=3, lt_n=0, choice={"key": KEY, "family": "triton"})
    wc = WinnerCache(p, env=ENV2)
    wc.mark_verified(wc.get(8, 8, 8, "fp16", allow_stale=True))
    assert wc.get(8, 8, 8, "fp16", allow_stale=True)["verified"]
    assert WinnerCache(p, env=ENV2).get(8, 8, 8, "fp16", allow_stale=True)["verified"]          # read back by a new process
    assert not WinnerCache(p, env={**ENV2, "torch": "x"}).get(8, 8, 8, "fp16", allow_stale=True)["verified"]


def test_compact_drops_re_searched_lines_and_keeps_unsearched_stale_ones(tmp_path):
    p = tmp_path / "w.jsonl"
    old = {**SCOPED, "driver": "590"}
    WinnerCache(p, env=old).put(8, 8, 8, "fp16", k=3, lt_n=0, choice={"key": "old8", "family": "triton"})
    WinnerCache(p, env=old).put(9, 9, 9, "fp16", k=3, lt_n=0, choice={"key": "old9", "family": "triton"})
    WinnerCache(p, env={**SCOPED, "hw_id": "gpu-b"}).put(8, 8, 8, "fp16", k=3, lt_n=0, choice={"key": "b8", "family": "triton"})
    wc = WinnerCache(p, env=SCOPED)
    wc.mark_verified(wc.get(9, 9, 9, "fp16", allow_stale=True))
    wc.put(8, 8, 8, "fp16", k=3, lt_n=0, choice={"key": "new8", "family": "triton"})      # 8^3 re-searched here
    with open(p, "a") as f:
        f.write('{"schema": 2, "env"')                                                    # torn line
    assert wc.compact() == {"before": 6, "after": 4}
    keys = [json.loads(x).get("choice", {}).get("key") for x in p.read_text().splitlines()]
    assert keys == ["old9", "b8", "new8", None]                                       # None: the verification line
    again = WinnerCache(p, env=SCOPED)
    assert again.get(8, 8, 8, "fp16")["choice"]["key"] == "new8"
    assert again.get(9, 9, 9, "fp16", allow_stale=True)["verified"]
    wc.put(10, 10, 10, "fp16", k=3, lt_n=0, choice={"key": KEY, "family": "triton"})        # writers still append after the rename
    assert WinnerCache(p, env=SCOPED).get(10, 10, 10, "fp16") is not None


def test_select_kernel_serves_a_stale_entry_after_one_correctness_check(tmp_path, fake_search, monkeypatch):
    p = tmp_path / "w.jsonl"
    WinnerCache(p, env={**ENV, "driver": "590"}).put(64, 64, 64, "fp16", k=3, lt_n=4, lt_rank=5,
                                                     choice={"key": KEY, "family": "triton"})
    checks = []

    class Ok:
        def __init__(self, v):
            self.valid = v
    monkeypatch.setattr(online, "validate", lambda b, *a, **k: checks.append(b) or Ok(True))
    monkeypatch.setattr("baselines.cublaslt_rank.ranker", lambda stem=None: object())
    kw = dict(use_bundle=False, lt_n=4)
    assert not online.select_kernel(None, 64, 64, 64, cache=WinnerCache(p, env=ENV), **kw)["cached"]  # default: a miss
    p.write_text(p.read_text().splitlines()[0] + "\n")                                     # forget that re-search
    a = online.select_kernel(None, 64, 64, 64, cache=WinnerCache(p, env=ENV), serve_stale=True, **kw)
    b = online.select_kernel(None, 64, 64, 64, cache=WinnerCache(p, env=ENV), serve_stale=True, **kw)
    assert a["cached"] and a["reason"] == "cached_stale" and a["stale"] == ["driver"] and b["reason"] == "cached_stale"
    assert len(checks) == 1 and len(fake_search) == 1                                      # verified once, never searched
    monkeypatch.setattr(online, "validate", lambda b, *a, **k: Ok(False))
    WinnerCache(p, env={**ENV, "driver": "591"}).put(65, 64, 64, "fp16", k=3, lt_n=4, lt_rank=5,
                                                    choice={"key": KEY, "family": "triton"})
    c = online.select_kernel(None, 65, 64, 64, cache=WinnerCache(p, env=ENV), serve_stale=True, **kw)
    assert not c["cached"] and len(fake_search) == 2                                       # failed the gate: searched


def test_compact_keeps_an_older_wider_stale_search(tmp_path):
    p = tmp_path / "w.jsonl"
    WinnerCache(p, env={**SCOPED, "driver": "590"}).put(8, 8, 8, "fp16", k=5, lt_n=16, choice={"key": "wide", "family": "triton"})
    WinnerCache(p, env={**SCOPED, "driver": "591"}).put(8, 8, 8, "fp16", k=3, lt_n=16, choice={"key": "narrow-old", "family": "triton"})
    wc = WinnerCache(p, env=SCOPED)
    wc.put(8, 8, 8, "fp16", k=3, lt_n=16, choice={"key": KEY, "family": "triton"})
    assert wc.compact() == {"before": 3, "after": 2}
    assert wc.get(8, 8, 8, "fp16", k=5, lt_n=16, allow_stale=True)["choice"]["key"] == "wide"
    assert wc.get(8, 8, 8, "fp16", k=3, lt_n=16)["choice"]["key"] == KEY


def _keys(p):
    return [json.loads(x).get("choice", {}).get("key") for x in p.read_text().splitlines()]


def _put(p, env, key, M=8, **kw):
    WinnerCache(p, env=env).put(M, 8, 8, "fp16", **{"k": 3, "lt_n": 16, **kw}, choice={"key": key, "family": "triton"})


def test_compact_without_env_id_never_drops_another_environments_line(tmp_path):
    """Two interpreters on one GPU sharing the file (torch 2.13 / 2.14), neither names itself: a line that is stale for
    the compactor is the other one's current entry. Whichever searched last, both keep their winner."""
    a, b = {**ENV2, "torch": "2.13.0"}, {**ENV2, "torch": "2.14.0"}
    for first, second in ((a, b), (b, a)):
        p = tmp_path / f"w{first['torch']}.jsonl"
        _put(p, first, "first")
        _put(p, second, "second")
        assert WinnerCache(p, env=a).compact() == {"before": 2, "after": 2}
        assert WinnerCache(p, env=first).get(8, 8, 8, "fp16", k=3, lt_n=16)["choice"]["key"] == "first"
        assert WinnerCache(p, env=second).get(8, 8, 8, "fp16", k=3, lt_n=16)["choice"]["key"] == "second"


def test_compact_from_a_process_that_declares_less_keeps_the_declared_entries(tmp_path):
    """No declared clock lock means the clock is not compared, so a 1755 MHz line reads as current here; it is still
    another environment's line and survives this process's compaction."""
    p = tmp_path / "w.jsonl"
    locked, undeclared = {**ENV2, "clock_lock": "1755"}, {**ENV2, "clock_lock": None}
    for env in (locked, undeclared, {**locked, "env_id": "py-a"}, {**undeclared, "env_id": "py-a"}):
        _put(p, env, f"lock-{env['clock_lock']}-{env.get('env_id')}")
    for env in (undeclared, {**undeclared, "env_id": "py-a"}):
        assert WinnerCache(p, env=env).compact() == {"before": 4, "after": 4}
    assert _keys(p) == ["lock-1755-None", "lock-None-None", "lock-1755-py-a", "lock-None-py-a"]
    assert WinnerCache(p, env={**locked, "env_id": "py-a"}).get(8, 8, 8, "fp16", k=3, lt_n=16)["choice"]["key"] == "lock-1755-py-a"
    p2 = tmp_path / "w2.jsonl"                                # the two-line case of the review, no env_id anywhere
    _put(p2, locked, "at-1755")
    _put(p2, undeclared, "undeclared")
    assert WinnerCache(p2, env=undeclared).compact() == {"before": 2, "after": 2}
    assert WinnerCache(p2, env=locked).get(8, 8, 8, "fp16", k=3, lt_n=16)["choice"]["key"] == "at-1755"


def test_compact_drops_exact_duplicates_everywhere_and_superseded_lines_only_in_its_own_scope(tmp_path):
    p = tmp_path / "w.jsonl"
    old_a, old_b = {**SCOPED, "driver": "590"}, {**SCOPED, "driver": "590", "env_id": "py-b"}
    _put(p, {**ENV2, "driver": "590"}, "plain-old")          # no env_id: stale here, kept (nobody says whose it is)
    _put(p, ENV2, "plain-1")
    _put(p, ENV2, "plain-2")                                 # same environment as the line before: replaces it
    _put(p, old_b, "b-old")                                  # another interpreter's scope: untouched
    _put(p, {**SCOPED, "env_id": "py-b"}, "b-new")
    _put(p, old_a, "a-old")                                  # this scope, stale, re-searched below: dropped
    _put(p, SCOPED, "a-new")
    wc = WinnerCache(p, env=SCOPED)
    assert wc.compact() == {"before": 7, "after": 5}
    assert _keys(p) == ["plain-old", "plain-2", "b-old", "b-new", "a-new"]
    assert WinnerCache(p, env=ENV2).compact() == {"before": 5, "after": 5}      # a compactor without env_id prunes no scope


def test_compact_keeps_a_stale_line_written_after_the_current_one(tmp_path):
    """The interpreter was upgraded and searched again while this compactor still holds the old environment (a daemon's
    cached copy): the newer line is the live one and must survive."""
    p = tmp_path / "w.jsonl"
    _put(p, SCOPED, "before-upgrade")
    _put(p, {**SCOPED, "torch": "2.14.0"}, "after-upgrade")
    assert WinnerCache(p, env=SCOPED).compact() == {"before": 2, "after": 2}
    assert WinnerCache(p, env={**SCOPED, "torch": "2.14.0"}).get(8, 8, 8, "fp16", k=3, lt_n=16)["choice"]["key"] == "after-upgrade"
    assert WinnerCache(p, env={**SCOPED, "torch": "2.14.0"}).compact() == {"before": 2, "after": 1}   # the upgraded side prunes
    assert _keys(p) == ["after-upgrade"]


def test_environment_env_id_is_only_stored_when_given(monkeypatch):
    import tune.winner_cache as wc
    for name in ("_nvml_handle", "hw_id"):
        monkeypatch.setattr(f"tune.env.{name}", lambda *a: None)
    monkeypatch.setattr("baselines.cublaslt.library_info", lambda: {"version": 1})
    assert "env_id" not in wc.environment() and wc.environment(env_id="py-a")["env_id"] == "py-a"


def test_a_put_after_a_torn_line_is_not_glued_onto_it(tmp_path):
    p = tmp_path / "w.jsonl"
    p.write_text('{"schema": 2, "env"')
    WinnerCache(p, env=ENV2).put(8, 8, 8, "fp16", k=3, lt_n=0, choice={"key": KEY, "family": "triton"})
    assert WinnerCache(p, env=ENV2).get(8, 8, 8, "fp16")["choice"]["key"] == KEY


def test_parseable_lines_missing_key_fields_are_skipped(tmp_path):
    p = tmp_path / "w.jsonl"
    p.write_text(json.dumps({"schema": 2, "env": ENV2, "M": 8}) + "\n")
    wc = WinnerCache(p, env=ENV2)
    assert len(wc) == 0 and wc.compact() == {"before": 1, "after": 0}


def test_a_triton_winner_stored_without_a_family_is_still_hard_on_kernel_source(tmp_path):
    p = tmp_path / "w.jsonl"
    WinnerCache(p, env=ENV2).put(8, 8, 8, "fp16", k=3, lt_n=0, choice={"key": KEY})          # family None
    assert WinnerCache(p, env={**ENV2, "kernel_src": "bbbb"}).get(8, 8, 8, "fp16", allow_stale=True) is None


def test_mark_verified_writes_one_line_per_entry_and_environment(tmp_path):
    p = tmp_path / "w.jsonl"
    WinnerCache(p, env={**ENV2, "driver": "590"}).put(8, 8, 8, "fp16", k=3, lt_n=0, choice={"key": KEY, "family": "triton"})
    wc = WinnerCache(p, env=ENV2)
    hit = wc.get(8, 8, 8, "fp16", allow_stale=True)
    wc.mark_verified(hit)
    wc.mark_verified(hit)
    wc.mark_verified(wc.get(8, 8, 8, "fp16", allow_stale=True))
    assert len(p.read_text().splitlines()) == 2


def _writer(path, start, n):
    wc = WinnerCache(path, env=ENV2)
    for i in range(start, start + n):
        wc.put(i, 8, 8, "fp16", k=3, lt_n=0, choice={"key": KEY, "family": "triton"})


def _compactor(path, n):
    wc = WinnerCache(path, env=ENV2)
    for _ in range(n):
        wc.compact()


def test_concurrent_puts_and_compactions_lose_nothing(tmp_path):
    import multiprocessing as mp
    p = tmp_path / "w.jsonl"
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_writer, args=(p, s, 60)) for s in (1000, 2000)]
    procs += [ctx.Process(target=_compactor, args=(p, 40)) for _ in range(2)]
    for x in procs:
        x.start()
    for x in procs:
        x.join(60)
        assert x.exitcode == 0
    wc = WinnerCache(p, env=ENV2)
    assert len(wc) == 120 and not list(tmp_path.glob("*.tmp-*"))
