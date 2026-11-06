"""The learned cuBLASLt ranker and the router on another GPU: SM count from the model / device, the foreign-model guard, the
zero-shot environment, and the reduced-screen shape selection. CPU only (reads the committed 3080 snapshot models)."""
from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest

SNAP = Path(__file__).resolve().parent.parent / "reports" / "snapshot" / "lock1755"


def _copy(tmp_path, name: str, **meta) -> Path:
    for ext in (".txt", ".json"):
        shutil.copy(SNAP / (name + ext), tmp_path / (name + ext))
    j = json.loads((tmp_path / (name + ".json")).read_text())
    j.update(meta)
    (tmp_path / (name + ".json")).write_text(json.dumps(j))
    return tmp_path / name


def test_features_sms_only_changes_wave_columns():
    from baselines.cublaslt_rank import FEATURES, features
    cfgs = np.array([[5, 20, 0, 1, 0, 0, 0], [5, 17, 7, 3, 1, 1, 0]], dtype=np.int64)
    a, b = features((4096, 4096, 4096), cfgs, 68), features((4096, 4096, 4096), cfgs, 82)
    same = [i for i, f in enumerate(FEATURES) if f not in ("waves_a", "waves_b")]
    assert np.allclose(a[:, same], b[:, same], equal_nan=True)
    w = FEATURES.index("waves_a")
    assert np.allclose(a[:, w] * 68, b[:, w] * 82, rtol=1e-5)
    assert np.allclose(features((4096, 4096, 4096), cfgs), a)   # default = the 3080's 68


def test_lt_ranker_sms_from_meta_native_and_integer(tmp_path):
    from baselines.cublaslt_rank import LtRanker, device_sms
    stem = _copy(tmp_path, "lt_rank")
    assert LtRanker.load(stem).sms == 68                       # the committed model predates the `sms` field
    assert LtRanker.load(stem, "native").sms == device_sms()
    assert LtRanker.load(stem, 82).sms == 82
    (tmp_path / "x").mkdir()
    stem2 = _copy(tmp_path / "x", "lt_rank", sms=30)
    assert LtRanker.load(stem2).sms == 30                       # a retrained model carries its own GPU's count
    assert LtRanker.load(stem2, 82).sms == 82                  # an explicit value wins


def test_lt_ranker_refuses_foreign_unless_lifted(tmp_path, monkeypatch):
    import baselines.cublaslt_rank as r
    stem = _copy(tmp_path, "lt_rank", hw_id="rtx_other_sm75_30sms_6gb")
    monkeypatch.setattr(r, "_RANKERS", {})
    for k in ("TUNE_LT_RANK", "TUNE_LT_RANK_FOREIGN", "TUNE_LT_RANK_SMS"):
        monkeypatch.delenv(k, raising=False)
    with pytest.warns(UserWarning, match="disabled"):
        assert r.ranker(stem) is None
    with pytest.warns(UserWarning, match="zero-shot"):
        got = r.ranker(stem, foreign=True, sms="native")
    assert got is not None and got.sms == r.device_sms()
    # the environment does the same thing for processes that cannot pass arguments (the `cublaslt_rank:M` backend)
    monkeypatch.setattr(r, "_RANKERS", {})
    monkeypatch.setenv("TUNE_LT_RANK", str(stem)); monkeypatch.setenv("TUNE_LT_RANK_FOREIGN", "1"); monkeypatch.setenv("TUNE_LT_RANK_SMS", "82")
    assert r.default_stem() == stem
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert r.ranker().sms == 82


def test_router_sms_and_guard(tmp_path, monkeypatch):
    import tune.router as t
    stem = _copy(tmp_path, "router", hw_id="rtx_other_sm75_30sms_6gb")
    monkeypatch.setattr(t, "_ROUTERS", {})
    for k in ("TUNE_ROUTER", "TUNE_ROUTER_FOREIGN", "TUNE_ROUTER_SMS"):
        monkeypatch.delenv(k, raising=False)
    with pytest.warns(UserWarning, match="disabled"):
        assert t.router(stem) is None
    with pytest.warns(UserWarning, match="zero-shot"):
        r = t.router(stem, foreign=True, sms=30)
    assert r.sms == 30
    f68, f30 = t.features(2048, 2048, 2048, 100.0), t.features(2048, 2048, 2048, 100.0, 30)
    i = t.FEATURES.index("l_ctas_per_sm")
    assert f30[i] > f68[i] and all(np.isclose(a, b) for j, (a, b) in enumerate(zip(f68, f30)) if j != i)
    assert 0.0 <= r.p_win(2048, 2048, 2048, 100.0) <= 1.0


def test_zero_shot_lt_rank_env():
    from analysis.zero_shot import lt_rank_env
    # own model: TUNE_LT_RANK (set by the pipeline to the run's fit) must survive
    assert lt_rank_env("as_is", None) == {"TUNE_LT_RANK_FOREIGN": None, "TUNE_LT_RANK_SMS": None}
    a, n = lt_rank_env("as_is", Path("/m/lt_rank")), lt_rank_env("native", Path("/m/lt_rank"))
    assert a["TUNE_LT_RANK_FOREIGN"] == n["TUNE_LT_RANK_FOREIGN"] == "1" and a["TUNE_LT_RANK"] == "/m/lt_rank"
    assert a["TUNE_LT_RANK_SMS"] is None and n["TUNE_LT_RANK_SMS"] == "native"


def test_limit_shapes_spread():
    from baselines.cublaslt_screen import limit_shapes
    xs = list(range(100))
    assert limit_shapes(xs, 0, True) == xs and limit_shapes(xs, 200, True) == xs
    assert limit_shapes(xs, 5, False) == [0, 1, 2, 3, 4]
    sp = limit_shapes(xs, 5, True)
    assert sp[0] == 0 and sp[-1] == 99 and len(set(sp)) == 5 and sp == sorted(sp)
    assert limit_shapes(xs, 1, True) == [0]
