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
