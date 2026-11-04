import json

import numpy as np
import pytest

from analysis import ranker_compare as rc


def _row(shape, family, key, us, mode="cold"):
    M, N, K = shape
    return {"M": M, "N": N, "K": K, "kernel_family": family, "config_key": key, "valid": True, "time_median_us": us,
            "cache_mode": mode}


def _online(shape, picks):
    """picks: {k: (key, family)} -> a joint online row; choice for lt_n' in {0, 1} reuse the same pick for simplicity."""
    M, N, K = shape
    ch = lambda k: {"key": picks[k][0], "family": picks[k][1], "compile_s": 0.1 * k, "time_s": 0.0}
    return {"M": M, "N": N, "K": K, "candidates": [{"x": 1}], "rank_s": 0.1, "lt_n": 16,
            "choice": {str(k): ch(k) for k in picks},
            "choice_lt": {f"{k}_lt{n}": ch(k) for k in picks for n in (0, 1)}}


def test_references_scores_and_paired(tmp_path):
    big, tiny = (4096, 4096, 4096), (64, 64, 64)
    cold = [_row(big, "triton_nn_v1", "t0", 1000.0), _row(big, "triton_nn_v1", "t1", 900.0),
            _row(big, "cublaslt", "h0_a", 1000.0), _row(big, "cublaslt", "h1_a", 950.0), _row(big, "cublaslt_deep", "d0", 880.0),
            _row(tiny, "triton_nn_v1", "t0", 10.0), _row(tiny, "triton_nn_v1", "t1", 12.0),
            _row(tiny, "cublaslt", "h0_a", 11.0), _row(tiny, "cublaslt_deep", "d0", 11.0)]
    hot = [_row(tiny, "triton_nn_v1", "t0", 8.0, "hot"), _row(tiny, "triton_nn_v1", "t1", 9.0, "hot"),
           _row(tiny, "cublaslt", "h0_a", 9.0, "hot"), _row(tiny, "cublaslt_deep", "d0", 9.0, "hot")]
    for name, rows in (("c.jsonl", cold), ("h.jsonl", hot)):
        (tmp_path / name).write_text("\n".join(map(json.dumps, rows)))
    good = {1: ("t1", "triton"), 3: ("t1", "triton"), 5: ("t1", "triton")}
    bad = {1: ("t0", "triton"), 3: ("t0", "triton"), 5: ("t0", "triton")}
    (tmp_path / "v1.jsonl").write_text("\n".join(json.dumps(_online(s, p)) for s, p in ((big, bad), (tiny, {k: ("h0_a", "cublaslt") for k in (1, 3, 5)}))))
    (tmp_path / "v2.jsonl").write_text("\n".join(json.dumps(_online(s, p)) for s, p in ((big, good), (tiny, {k: ("t0", "triton") for k in (1, 3, 5)}))))
    cold_d, hot_d = rc.load_stage2(tmp_path, ["c.jsonl"], "cold"), rc.load_stage2(tmp_path, ["h.jsonl"], "hot")
    ref = rc.references(cold_d, hot_d)
    assert ref[big]["small"] is False and ref[tiny]["small"] is True       # tiny's fastest candidate is < 50 us and has graph times
    assert ref[big]["b2"] == 1000.0 and ref[big]["b3"] == 950.0 and ref[big]["b3p"] == 880.0 and ref[big]["orc"] == 880.0
    assert ref[tiny]["b2"] == 9.0 and ref[tiny]["tri_orc"] == 8.0           # graph times for the small shape
    out, summary = rc.build(tmp_path, tmp_path, {"v1": "v1.jsonl", "v2": "v2.jsonl"}, "c.jsonl", "h.jsonl")
    v1, v2 = summary["scores"]["v1"]["top-3 + LT16"]["all"], summary["scores"]["v2"]["top-3 + LT16"]["all"]
    assert v1["n"] == 2 and v1["cost_s"] == pytest.approx(0.4)             # 0.1*3 compile + 0.1 rank
    assert v1["b2"]["median"] == pytest.approx(np.median([1000 / 1000, 9 / 9])) and v1["pct_lt"] == 50.0
    assert v2["b2"]["faster"] == 2 and v2["b2"]["slower"] == 0              # 900 vs B2 1000 (big); 8 us vs 9 us graph (tiny)
    p = summary["paired"]["top-3 + LT16"]
    assert p["n"] == 2 and p["b_faster"] == 2 and p["geomean_ratio"] < 1     # v2 faster than v1 on both shapes
    assert "Paired: v2 vs v1" in out.read_text()


def test_extended_oracle_sees_proposals():
    """`orc_x` also covers k-NN / ranker proposals timed in the session, so a pool that picks one cannot get negative regret; `orc` is unchanged."""
    s = (4096, 4096, 4096)
    d = {s: {"tri": {"t0": 1000.0}, "lth": {"h0_a": 1000.0, "h1_a": 950.0}, "ltd": {"d0": 900.0},
             "ltk": {"n0_a": 920.0}, "ltr": {"r0_a": 880.0, "r1_a": 940.0}}}
    ref = rc.references(d, {})
    assert ref[s]["b3p"] == 900.0 and ref[s]["orc"] == 900.0 and ref[s]["orc_x"] == 880.0
    d[s]["ltr"] = {}
    d[s]["ltk"] = {}
    assert rc.references(d, {})[s]["orc_x"] == 900.0                       # no proposals timed: identical to orc
