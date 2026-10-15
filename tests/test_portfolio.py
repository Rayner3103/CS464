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
