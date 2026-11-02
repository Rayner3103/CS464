import json

import numpy as np

from analysis import cross_gpu as cg


def test_chosen_prefers_the_triton_only_choice_of_a_joint_row():
    row = {"choice": {"3": {"key": "h0_lt"}}, "choice_lt": {"3_lt0": {"key": "bm16_bn16_bk16_w2_s2_g1_sk1"}}}
    assert cg.chosen(row, 3)["key"].startswith("bm16")
    assert cg.chosen({"choice": {"3": {"key": "t"}}}, 3)["key"] == "t"


def test_load_rows_filters_shapes_k_and_rows_without_candidates(tmp_path):
    p = tmp_path / "o.jsonl"
    rows = [{"M": 1, "N": 2, "K": 3, "k": 5, "candidates": [1]}, {"M": 4, "N": 5, "K": 6, "k": 5, "candidates": [1]},
            {"M": 1, "N": 2, "K": 3, "k": 3, "candidates": [1]}, {"M": 1, "N": 2, "K": 3, "k": 5}]
    p.write_text("\n".join(map(json.dumps, rows)))
    assert len(cg.load_rows(p)) == 3
    assert len(cg.load_rows(p, {(1, 2, 3)}, min_k=5)) == 1


def test_regret_by_shape_drops_picks_missing_from_the_screen():
    screen = {(1, 2, 3): {"a": 10.0, "b": 11.0}}
    rows = [{"M": 1, "N": 2, "K": 3, "choice": {"3": {"key": "b"}}}, {"M": 1, "N": 2, "K": 3, "choice": {"3": {"key": "zz"}}}]
    assert abs(cg.regret_by_shape(screen, rows[:1], 3)[(1, 2, 3)] - 0.1) < 1e-12
    assert cg.regret_by_shape(screen, rows[1:], 3) == {}


def test_paired_bootstrap_interval_contains_zero_for_identical_data_and_detects_a_shift():
    x = np.array([0.01, 0.02, 0.05, 0.10, 0.00, 0.03])
    same = cg.paired_bootstrap(x, x)
    assert same["diff"] == 0 and same["lo"] == 0 and same["hi"] == 0
    shifted = cg.paired_bootstrap(x, x + 0.5)
    assert abs(shifted["diff"] - 0.5) < 1e-12 and shifted["lo"] > 0.4


def _ref(b3p, tri, small):
    return {"b3p": b3p, "tri_orc": tri, "small": small, "src": {"tri": {}, "lth": {}, "ltd": {}}}


def test_speed_vs_b3p_uses_the_tie_margin_of_each_regime():
    ref = {(1, 1, 1): _ref(100.0, 90.0, False), (2, 2, 2): _ref(10.0, 9.0, True), (3, 3, 3): _ref(100.0, 400.0, False)}
    times = {(1, 1, 1): 101.0, (2, 2, 2): 9.8, (3, 3, 3): 400.0}   # speed 0.99 (tie), 1.02 (tie below 50 us: margin 5%), 0.25 (behind)
    s = cg.speed_vs_b3p(ref, times, list(ref))
    assert s["all"]["n"] == 3 and s["all"]["ahead"] == 0 and s["all"]["behind"] == 1
    assert s[">= 50 us"]["n"] == 2 and s["< 50 us"]["n"] == 1


def test_family_ceiling_counts_shapes_within_90_percent():
    ref = {(1, 1, 1): _ref(100.0, 100.0, False), (2, 2, 2): _ref(100.0, 500.0, False), (3, 3, 3): _ref(10.0, 5.0, True)}
    c = cg.family_ceiling(ref, list(ref))
    assert c["n"] == 3 and c["ge_90pct"] == 2 and c["ahead"] == 1 and c["behind"] == 1


def test_ptx_mma_share_counts_tensor_core_and_fma_only_kernels(tmp_path):
    for name, body in (("a", ".target sm_86\n mma.sync.aligned"), ("b", ".target sm_75\n fma.rn.f32 %r1;")):
        d = tmp_path / "online_triton_cache" / "s" / name
        d.mkdir(parents=True)
        (d / "_matmul_kernel.ptx").write_text(body)
    assert cg.ptx_mma_share(tmp_path) == {"n": 2, "mma": 1, "fma_only": 1, "sm_86": 1, "sm_75": 1}
    assert cg.ptx_mma_share(tmp_path / "nowhere") is None


def test_joint_summary_counts_cublaslt_wins_and_ignores_untimed_picks():
    ref = {(1, 1, 1): {**_ref(100.0, 90.0, False), "src": {"tri": {}, "lth": {"h2_x": 105.0}, "ltd": {}}},
           (2, 2, 2): {**_ref(100.0, 90.0, False), "src": {"tri": {}, "lth": {}, "ltd": {}}}}
    cold = {s: {"tri": {}, "lth": {}, "ltd": {}} for s in ref}
    rows = [{"M": 1, "N": 1, "K": 1, "lt_n": 16, "choice": {"3": {"key": "h2_x", "family": "cublaslt"}}},
            {"M": 2, "N": 2, "K": 2, "lt_n": 16, "choice": {"3": {"key": "bm16_untimed", "family": "triton"}}}]
    j = cg.joint_summary(rows, ref, cold, 3, set(ref))
    assert j["n"] == 2 and j["cublaslt"] == 1 and j["covered"] == 1 and abs(j["median"] - 100 / 105) < 1e-9
    assert cg.joint_summary([{"M": 1, "N": 1, "K": 1, "choice": {"3": {"key": "t"}}}], ref, cold, 3, set(ref)) is None
