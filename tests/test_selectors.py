import numpy as np

from kernels.config_space import HwLimits

HW = HwLimits(smem_per_block_bytes=101376, regs_per_sm=65536, sm_count=68)  # RTX 3080


def test_knn_select_takes_neighbour_best_runnable_and_averages_for_k_gt_1():
    from analysis.selectors import knn_select
    X_train = np.array([[0.0, 0, 0], [10, 10, 10], [11, 10, 10]])
    R = np.array([[1.0, 1.1, 3.0],
                  [2.0, 1.0, 1.05],
                  [2.0, 1.3, 1.0]])
    X = np.array([[0.5, 0, 0], [10.2, 10, 10]])
    ok = np.ones((2, 3), bool)
    assert knn_select(X_train, R, X, ok, 1).tolist() == [0, 1]
    ok[0, 0] = False  # the neighbour's best is pruned on the target: fall back to its next best
    assert knn_select(X_train, R, X, ok, 1).tolist() == [1, 1]
    # k = 2 on the second target: mean log regret over shapes 1 and 2 -> config 2 (1.05 * 1.0 < 1.0 * 1.3)
    assert knn_select(X_train, R, X, np.ones((2, 3), bool), 2)[1] == 2
    # Leave-one-out never uses the shape itself.
    assert knn_select(X_train, R, X_train, np.ones((3, 3), bool), 1, np.eye(3, dtype=bool)).tolist() == [1, 2, 1]


def test_rule_is_always_runnable_and_follows_its_regimes():
    from analysis.selectors import rule_config, runnable
    for M in (1, 7, 16, 33, 128, 1000, 4096, 16384):
        for N in (1, 30, 300, 4096, 14336):
            for K in (1, 17, 300, 4096, 16384):
                assert runnable(rule_config(M, N, K, HW), M, N, K, HW)
    decode = rule_config(1, 4096, 11008, HW)
    assert (decode.BLOCK_M, decode.BLOCK_N) == (16, 128) and decode.SPLIT_K > 1  # 32 CTAs < 68 SMs: split K
    big = rule_config(8192, 8192, 8192, HW)
    assert (big.BLOCK_M, big.BLOCK_N, big.SPLIT_K, big.GROUP_M) == (64, 128, 1, 8)


def test_stage2_oracle_bar_and_pick_times():
    from analysis.selectors import stage2
    row = lambda fam, key, t, valid=True: {"M": 1, "N": 2, "K": 3, "kernel_family": fam, "config_key": key,
                                           "valid": valid, "time_median_us": t}
    rows = [row("triton_nn_v1", "a", 10.0), row("triton_nn_v1", "b", 12.0), row("triton_nn_v1", "c", 1.0, False),
            row("cublaslt", "h0", 11.0), row("cublaslt_deep", "d0", 9.5), row("torch_matmul", "default", 13.0)]
    st = stage2(rows, [{"M": 1, "N": 2, "K": 3}, {"M": 4, "N": 5, "K": 6}], {"x": ["b", "a"], "y": ["c", "a"]})
    assert st["oracle"][0] == 10.0 and st["b3plus"][0] == 9.5 and st["b1"][0] == 13.0
    assert st["x"][0] == 12.0 and np.isnan(st["y"][0])  # invalid rows never count
    assert np.isnan(st["oracle"][1]) and np.isnan(st["x"][1])


def test_shape_features_embed_and_grouped_exclusion():
    from analysis.selectors import embed, shape_features, share_two_dims
    shapes = [{"M": 2, "N": 8901, "K": 9643}, {"M": 2, "N": 8704, "K": 9643}, {"M": 64, "N": 64, "K": 64}]
    F = shape_features(shapes, 68)
    assert F["dims"].shape == (3, 3) and F["fill"].shape == (3, 13) and F["align"].shape == (3, 6)
    # 16x128 tiles: 8901 -> 70 CTAs (2 waves, last one nearly empty); 8704 -> 68 CTAs (exactly one wave).
    assert abs(F["fill"][0, 3] - 70 / 136) < 1e-9 and F["fill"][1, 3] == 1.0
    assert F["align"][2].tolist() == [1.0] * 6
    assert embed(F, F, {"fill": 0, "align": 0}).shape == (3, 3)  # zero-weight groups drop out
    X = embed(F, F, {"fill": 2.0, "align": 0})
    assert X.shape == (3, 16) and abs(X[:, :3].mean()) < 1e-9
    ex = share_two_dims(shapes, shapes)
    assert ex[0, 1] and ex[1, 0] and not ex[0, 2] and ex.diagonal().all()
