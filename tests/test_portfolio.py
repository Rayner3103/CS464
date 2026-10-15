import numpy as np

from analysis.portfolio import curve, greedy_portfolio, regret_matrix, set_cover


def test_greedy_and_cover_on_synthetic_regrets():
    # 4 shapes x 3 configs: c0 is best on shapes 0-1, c1 on shapes 2-3, c2 is mediocre everywhere.
    T = np.array([[1.0, 2.0, 1.2],
                  [1.0, 2.0, 1.2],
                  [3.0, 1.0, 1.2],
                  [np.inf, 1.0, 1.2]])
    R = regret_matrix(T)
    assert R.min(axis=1).tolist() == [1.0] * 4
    order = greedy_portfolio(R)
    assert order[0] == 2  # best single config by mean regret (B6-like)
    assert set(order[:3]) == {0, 1, 2}
    cv = curve(R, order)
    assert abs(cv[0]["mean"] - 0.2) < 1e-9 and cv[-1]["max"] == 0.0
    assert sorted(set_cover(R, 0.05)) == [0, 1]
    assert set_cover(R, 0.25) == [2]


def test_align_maps_columns_and_marks_missing_configs_inf():
    from analysis.portfolio import align
    T_to = np.array([[2.0, 3.0], [5.0, 1.0]])
    out = align(["b", "z", "a"], ["a", "b"], T_to)
    assert out[:, 0].tolist() == [3.0, 1.0] and out[:, 2].tolist() == [2.0, 5.0]
    assert np.isinf(out[:, 1]).all()


def test_coarse_subset_respects_split():
    from tune.sweep import HELDOUT_SPLITS, coarse_subset
    shapes = [{"M": 64, "N": 64, "K": 64 * (i + 1), "suite": "S7", "split": sp}
              for i, sp in enumerate(["tune", "heldout_region", "heldout_random", "tune"])]
    assert {s["split"] for s in coarse_subset(shapes, 10, 1e9)} == {"tune"}
    assert {s["split"] for s in coarse_subset(shapes, 10, 1e9, HELDOUT_SPLITS)} == set(HELDOUT_SPLITS)


def test_stage2_regret_uses_portfolio_prefix_against_all_rows():
    from analysis.portfolio import stage2_regret
    rows = [{"M": 1, "N": 2, "K": 3, "kernel_family": "triton_nn_v1", "valid": True, "config_key": k, "time_median_us": t}
            for k, t in (("p0", 12.0), ("p1", 10.5), ("s0", 10.0))]
    rows.append({"M": 1, "N": 2, "K": 3, "kernel_family": "triton_nn_v1", "valid": False, "config_key": "x", "time_median_us": 1.0})
    meta, reg = stage2_regret(rows, ["p0", "p1"], (1, 2))
    assert meta[0]["oracle_us"] == 10.0
    assert abs(reg[1][0] - 1.2) < 1e-9 and abs(reg[2][0] - 1.05) < 1e-9
