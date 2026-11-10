import json

import numpy as np
import pytest

from analysis import zero_shot as zs


def _screen(tmp_path, shapes):
    d = tmp_path / "sweep_screen_heldout"
    d.mkdir()
    for (M, N, K), times in shapes.items():
        rows = [{"key": k, "us": t, "status": "ok"} for k, t in times.items()]
        (d / f"{M}x{N}x{K}_fp16.json").write_text(json.dumps({"M": M, "N": N, "K": K, "complete": True, "rows": rows}))
    (d / "9x9x9_fp16.json").write_text(json.dumps({"M": 9, "N": 9, "K": 9, "complete": False, "rows": []}))
    return d


def _row(M, N, K, cands, choice, k=3):
    return {"M": M, "N": N, "K": K, "k": k, "candidates": [{"config_key": c, "valid": v} for c, v in cands],
            "choice": {str(j): {"key": choice[j - 1]} for j in range(1, k + 1)}}


def test_regret_conventions():
    t = {"a": 10.0, "b": 11.0}
    assert zs.regret(t, "a") == 0.0
    assert abs(zs.regret(t, "b") - 0.1) < 1e-12
    assert zs.regret(t, None) == zs.CAP
    assert zs.regret(t, "missing") is None


def test_load_screen_skips_incomplete(tmp_path):
    d = _screen(tmp_path, {(1, 2, 3): {"a": 5.0, "b": 6.0}})
    assert list(zs.load_screen(d)) == [(1, 2, 3)]


def _screen_with_slow(tmp_path):
    d = tmp_path / "sweep_screen_heldout"
    d.mkdir()
    rows = [{"key": "a", "us": 10.0, "status": "ok"}, {"key": "b", "us": 11.0, "status": "ok"},
            {"key": "s", "us": 30.0, "status": "slow"}, {"key": "x", "us": None, "status": "failed"}]
    (d / "1x2x3_fp16.json").write_text(json.dumps({"M": 1, "N": 2, "K": 3, "complete": True, "rows": rows}))
    (d / "4x5x6_fp16.json").write_text(json.dumps({"M": 4, "N": 5, "K": 6, "complete": True,
                                                   "rows": [{"key": "s", "us": 9.0, "status": "slow"}]}))   # no ok row: skipped
    return d


def test_load_screen_keeps_slow_rows_so_bad_picks_are_scored(tmp_path):
    d = _screen_with_slow(tmp_path)
    screen = zs.load_screen(d)
    assert list(screen) == [(1, 2, 3)] and set(screen[(1, 2, 3)]) == {"a", "b", "s"}     # failed row (no time) and the all-slow shape are out
    assert zs.regret(screen[(1, 2, 3)], "s") == pytest.approx(2.0)                        # the old behaviour made this None (unscored)
    assert set(zs.load_screen(d, include_slow=False)[(1, 2, 3)]) == {"a", "b"}


def test_summarize_counts_picks_over_50_percent():
    s = zs.summarize([0.0, 0.2, 0.6, 2.0, zs.CAP, None])
    assert s["over_50"] == 3 and s["n"] == 5 and s["unscored"] == 1


def test_summarize_and_fmt_report_p99_between_p95_and_worst():
    vals = [i / 100 for i in range(100)]                       # 0.00 .. 0.99
    s = zs.summarize(vals)
    assert s["p95"] == pytest.approx(0.9405) and s["p99"] == pytest.approx(0.9801) and s["p95"] < s["p99"] <= s["worst"]
    assert zs.fmt(s) == "49.5% / 94.0% / 98.0% (49.5%)"
    old = {k: v for k, v in s.items() if k != "p99"}           # a summary written before p99 was stored still formats
    assert zs.fmt(old) == "49.5% / 94.0% (49.5%)"
    assert zs.fmt({"n": 0}) == "n/a"


def test_build_report_has_the_over_50_column_and_counts_slow_picks(tmp_path):
    data = tmp_path / "hw_x"
    data.mkdir()
    _screen_with_slow(data)
    (data / zs.out_name("native")).write_text(json.dumps(_row(1, 2, 3, [("s", True), ("a", True), ("b", True)], ["s", "s", "a"])) + "\n")
    out, summary = zs.build(data, tmp_path / "reports", fixed=None)
    sc = summary["variants"]["native"]["scores"]
    assert sc["online-1"]["all"]["over_50"] == 1 and sc["online-1"]["all"]["unscored"] == 0 and sc["online-3"]["all"]["over_50"] == 0
    assert "regret > 50%" in out.read_text()


def test_score_rows_top1_best_of_k_and_online(tmp_path):
    screen = zs.load_screen(_screen(tmp_path, {(1, 2, 3): {"a": 10.0, "b": 12.0, "c": 20.0},
                                               (4, 5, 6): {"a": 100.0, "b": 150.0}}))
    rows = [_row(1, 2, 3, [("c", True), ("b", True), ("a", True)], ["c", "b", "a"]),
            _row(4, 5, 6, [("c", False), ("b", True), ("a", True)], [None, "b", "a"])]  # top pick failed to compile
    sc = zs.score_rows(screen, rows, 3, fixed="b")
    assert sc["top-1"][0] == 1.0 and sc["top-1"][1] == zs.CAP
    assert sc["best-of-3"] == [0.0, 0.0]
    assert sc["best-of-2"][0] == pytest.approx(0.2) and sc["online-1"] == [1.0, zs.CAP]
    assert sc["invalid_picks"] == [0, 1]
    assert sc["fixed"] == pytest.approx([0.2, 0.5])
    assert sc["_oracle_us"] == [10.0, 100.0]


def test_summarize_counts_unscored_and_no_pick():
    s = zs.summarize([0.0, 0.01, 0.1, zs.CAP, None])
    assert s["n"] == 4 and s["unscored"] == 1 and s["no_pick"] == 1
    assert s["within_eps"] == 0.5 and s["worst"] == zs.CAP
    assert np.isclose(s["median"], 0.055)


def test_build_writes_report(tmp_path):
    data = tmp_path / "hw_x"
    data.mkdir()
    d = _screen(data, {(1, 2, 3): {"a": 10.0, "b": 12.0}})
    assert d.name == "sweep_screen_heldout"
    (data / zs.out_name("native")).write_text(json.dumps(_row(1, 2, 3, [("a", True), ("b", True), ("a", True)],
                                                              ["a", "a", "a"])) + "\n")
    out, summary = zs.build(data, tmp_path / "reports", fixed="b", variants=("native", "as_is"))
    assert list(summary["variants"]) == ["native"]            # as_is has no rows: skipped, not an error
    assert summary["variants"]["native"]["scores"]["online-3"]["all"]["median"] == 0.0
    assert "Zero-shot selector transfer: `hw_x`" in out.read_text()


def test_skipped_oom_rows_are_not_scored_as_bad_picks(tmp_path):
    screen = zs.load_screen(_screen(tmp_path, {(1, 2, 3): {"a": 10.0, "b": 12.0}, (4, 5, 6): {"a": 1.0, "b": 2.0}}))
    skipped = {"M": 4, "N": 5, "K": 6, "k": 3, "candidates": [], "skipped": "oom_inputs: x",
               "choice": {str(j): {"key": None} for j in (1, 2, 3)}}
    ok = _row(1, 2, 3, [("a", True), ("b", True), ("a", True)], ["a", "a", "a"])
    sc = zs.score_rows(screen, [ok, skipped], 3, fixed="b")
    assert sc["_skipped"] == 1 and sc["_oracle_us"] == [10.0]
    assert sc["online-3"] == [0.0] and sc["fixed"] == pytest.approx([0.2])


def test_parse_rankers_names_and_out_names(tmp_path):
    from pathlib import Path
    assert zs.parse_rankers(["/m/lgbm_rank"]) == [("", Path("/m/lgbm_rank"))]
    assert zs.parse_rankers(["v1=/m/lgbm_rank", "v2=/m/lgbm_rank_v2"]) == [("v1", Path("/m/lgbm_rank")), ("v2", Path("/m/lgbm_rank_v2"))]
    with pytest.raises(ValueError):
        zs.parse_rankers(["v1=/a", "v1=/b"])
    with pytest.raises(ValueError):
        zs.parse_rankers(["bad name=/a"])
    assert zs.out_name("native") == "zero_shot_native.jsonl" and zs.out_name("native", "v2") == "zero_shot_v2_native.jsonl"
    for n in ("", "v1", "v2"):                      # legacy names, then named models, are all found
        (tmp_path / zs.out_name("native", n)).write_text("")
    assert zs.find_models(tmp_path) == ["", "v1", "v2"]


def test_build_scores_each_model_and_rows_with_larger_k(tmp_path):
    data = tmp_path / "hw_x"
    data.mkdir()
    _screen(data, {(1, 2, 3): {"a": 10.0, "b": 12.0, "c": 20.0}})
    # a k = 5 run (e.g. a KPolicy run on a frequent shape): stores choices up to 5, scores at k = 3 as well
    r5 = {"M": 1, "N": 2, "K": 3, "k": 5, "lt_n": 16,
          "candidates": [{"config_key": c, "valid": True, "family": "triton"} for c in ("c", "b", "a", "c", "b")]
                        + [{"config_key": "h0_x", "valid": True, "family": "cublaslt"}],
          "choice": {str(j): {"key": "h0_x"} for j in range(1, 6)},                        # joint choice: a cuBLASLt config
          "choice_lt": {f"{j}_lt0": {"key": "a" if j >= 3 else "c"} for j in range(1, 6)}}  # Triton-only choice
    (data / zs.out_name("native", "v2")).write_text(json.dumps(r5) + "\n")
    (data / zs.out_name("native")).write_text(json.dumps(_row(1, 2, 3, [("c", True), ("b", True), ("a", True)], ["c", "b", "a"])) + "\n")
    out, summary = zs.build(data, tmp_path / "reports", fixed=None)
    assert set(summary["variants"]) == {"native", "v2/native"}
    v2 = summary["variants"]["v2/native"]
    assert v2["joint_rows"] is True and v2["model"] == "v2"
    assert v2["scores"]["online-3"]["all"]["median"] == 0.0      # the Triton-only choice 'a' (the oracle), not the cuBLASLt key
    assert v2["scores"]["online-1"]["all"]["median"] == 1.0 and v2["scores"]["online-3"]["all"]["unscored"] == 0
    assert summary["variants"]["native"]["scores"]["online-3"]["all"]["median"] == 0.0
    text = out.read_text()
    assert "Model `v2`, variant `native`" in text and "| rank v1 | rank v2 |" in text
    out5, s5 = zs.build(data, tmp_path / "reports5", fixed=None, k=5)       # only the k = 5 rows reach k = 5
    assert set(s5["variants"]) == {"v2/native"} and "online-5" in s5["variants"]["v2/native"]["scores"]


def test_write_stage2_picks_unions_triton_candidates_and_flags_small_shapes(tmp_path):
    data = tmp_path / "hw_x"
    data.mkdir()
    d = _screen(data, {(1, 2, 3): {"a": 10.0, "b": 12.0}, (4, 5, 6): {"a": 100.0, "b": 150.0}})
    (d / "subset.json").write_text(json.dumps({"suite": "all", "shapes": [{"M": 1, "N": 2, "K": 3}, {"M": 4, "N": 5, "K": 6}]}))
    r1 = _row(1, 2, 3, [("a", True), ("b", True), ("a", True)], ["a", "a", "a"])
    r2 = dict(_row(1, 2, 3, [("b", True), ("c", True), ("a", True)], ["b", "b", "b"]),
              candidates=[{"config_key": "b", "valid": True, "family": "triton"}, {"config_key": "c", "valid": True, "family": "triton"},
                          {"config_key": "h0_x", "valid": True, "family": "cublaslt"}])
    (data / zs.out_name("native")).write_text(json.dumps(r1) + "\n")
    (data / zs.out_name("as_is", "v2")).write_text(json.dumps(r2) + "\n")
    picks, lt, n = zs.write_stage2_picks(data)
    doc = json.loads(picks.read_text())
    assert n == 1 and doc["picks"]["1x2x3"] == ["a", "b", "c"] and "by_selector" in doc      # union; no cuBLASLt key
    assert [s["M"] for s in json.loads(lt.read_text())["shapes"]] == [1]                    # only the 10 us shape is < 50 us


def test_run_makes_k_per_shape_names_files_and_resumes(tmp_path, monkeypatch):
    import tune.env, tune.online
    seen = []
    def fake_search(ranker, M, N, K, k, dtype, order, lt_n):
        seen.append((M, k, lt_n))
        c = [{"config_key": f"c{i}", "valid": True, "compile_s": 0.1, "time_s": 0.01, "cold_us": 100.0 - i, "graph_us": None}
             for i in range(k)]
        return {"rank_s": 0.0, "candidates": c}
    class FakeRanker:
        def rule(self, order):
            return {"group": "tile_split_bk"}
    monkeypatch.setattr(tune.online, "search_shape", fake_search)
    monkeypatch.setattr(tune.env, "write_env", lambda out_dir, **kw: {"hw_id": "hw", "driver": "d", "gpu": {}})
    monkeypatch.setattr(tune.env, "provenance", lambda env: {})
    monkeypatch.setattr(zs, "variant_ranker", lambda stem, variant, gpu: FakeRanker())
    monkeypatch.setattr(zs, "source_gpu_name", lambda stem: "src")
    shapes = [{"M": 8, "N": 8, "K": 8}, {"M": 16, "N": 16, "K": 16, "calls": 3_000_000}, {"M": 32, "N": 32, "K": 32, "frequent": True}]
    pol = tune.online.KPolicy(3, 5, 1_000_000)
    paths = zs.run(shapes, tmp_path / "m", tmp_path, variants=("native",), k=3, name="v2", lt_n=16, policy=pol, log=lambda m: None)
    assert [p.name for p in paths] == ["zero_shot_v2_native.jsonl"]
    rows = [json.loads(x) for x in paths[0].read_text().splitlines()]
    assert [r["k"] for r in rows] == [3, 5, 5] and all(r["lt_n"] == 16 and r["model"] == "v2" for r in rows)
    assert "choice_lt" in rows[1] and "5_lt0" in rows[1]["choice_lt"] and rows[1]["calls"] == 3_000_000
    assert [x[2] for x in seen] == [16, 16, 16]
    n = len(seen)
    zs.run(shapes, tmp_path / "m", tmp_path, variants=("native",), k=3, name="v2", lt_n=16, policy=pol, log=lambda m: None)
    assert len(seen) == n                                                     # resumed per (shape, k, lt_n)
    zs.run(shapes[:1], tmp_path / "m", tmp_path, variants=("native",), k=3, name="", log=lambda m: None)
    assert (tmp_path / "zero_shot_native.jsonl").exists() and seen[-1] == (8, 3, 0)   # the original, unnamed behaviour


def test_ranker_compare_scores_a_k3_run_without_deep_search_rows(tmp_path):
    from analysis import ranker_compare as rc
    shape = (4096, 4096, 4096)
    rows = [{"M": 4096, "N": 4096, "K": 4096, "kernel_family": f, "config_key": k, "valid": True, "time_median_us": t, "cache_mode": "cold"}
            for f, k, t in (("triton_nn_v1", "t0", 900.0), ("cublaslt", "h0_a", 1000.0), ("cublaslt", "h1_a", 950.0))]
    (tmp_path / "c.jsonl").write_text("\n".join(map(json.dumps, rows)))
    (tmp_path / "h.jsonl").write_text("")
    ch = lambda k, key, fam: {"key": key, "family": fam, "compile_s": 0.1 * k, "time_s": 0.0}
    row = {"M": 4096, "N": 4096, "K": 4096, "candidates": [{"x": 1}], "rank_s": 0.1, "lt_n": 16,
           "choice": {str(k): ch(k, "t0", "triton") for k in (1, 2, 3)},
           "choice_lt": {f"{k}_lt{n}": ch(k, "t0", "triton") for k in (1, 2, 3) for n in (0, 1, 4, 16)}}
    (tmp_path / "o.jsonl").write_text(json.dumps(row))
    out, summary = rc.build(tmp_path, tmp_path, {"zs": "o.jsonl"}, "c.jsonl", "h.jsonl")
    assert set(summary["scores"]["zs"]) == {"top-3 Triton only", "top-3 + LT16", "top-1 + LT16"}   # no k = 5 selectors
    e = summary["scores"]["zs"]["top-3 + LT16"]["all"]
    assert e["b2"]["median"] == pytest.approx(0.9) and e["b3p"]["median"] == pytest.approx(900 / 950)   # B3+ = B3 without deep rows
    with pytest.raises(ValueError):
        (tmp_path / "tri.jsonl").write_text(json.dumps({**row, "choice_lt": {}, "lt_n": 0}))   # a Triton-only run
        rc.build(tmp_path, tmp_path, {"zs": "tri.jsonl"}, "c.jsonl", "h.jsonl")


def test_picks_prints_a_ranker_compare_command_that_does_not_overwrite_the_3080_report(tmp_path, capsys):
    data = tmp_path / "hw_x"
    data.mkdir()
    _screen(data, {(1, 2, 3): {"a": 10.0, "b": 12.0}})
    (data / "sweep_screen_heldout" / "subset.json").write_text(json.dumps({"shapes": [{"M": 1, "N": 2, "K": 3}]}))
    (data / zs.out_name("native", "v1")).write_text(json.dumps(_row(1, 2, 3, [("a", True), ("b", True), ("a", True)], ["a", "a", "a"])) + "\n")
    assert zs._main(["picks", "--data", str(data)]) == 0
    out = capsys.readouterr().out
    assert "analysis.ranker_compare" in out and "--report reports/zero_shot_hw_x" in out and "v1_native=zero_shot_v1_native.jsonl" in out
