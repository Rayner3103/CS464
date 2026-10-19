import numpy as np


def test_canonical_labels_take_first_member_within_eps():
    from analysis.learned import canonical_labels
    R = np.array([[1.00, 1.01, 1.5],    # member 0 within 2% of the best (1.00)
                  [1.30, 1.00, 1.01],   # member 1 is best; member 2 ties but comes later
                  [1.30, 1.10, 1.00]])  # only member 2
    assert canonical_labels(R, [0, 1, 2]).tolist() == [0, 1, 2]
    assert canonical_labels(R, [2, 1, 0]).tolist() == [1, 0, 0]  # greedy order decides the tie


def test_grouped_folds_partition_and_exclude_shapes_sharing_two_dims():
    from analysis.learned import grouped_folds
    from analysis.selectors import share_two_dims
    rng = np.random.default_rng(0)
    shapes = [{"M": int(m), "N": int(n), "K": int(k)} for m, n, k in rng.choice([1, 16, 64, 512, 4096], (60, 3))]
    folds = grouped_folds(shapes, n_folds=5)
    tests = np.concatenate([te for _, te in folds])
    assert sorted(tests.tolist()) == list(range(60))
    ex = share_two_dims(shapes, shapes)
    for tr, te in folds:
        assert not set(tr) & set(te) and not ex[np.ix_(te, tr)].any()


def test_crossed_appends_all_shape_x_candidate_products():
    from analysis.learned import Crossed
    X = np.random.default_rng(1).normal(size=(200, 5)) * [1, 2, 3, 4, 5] + 7
    tf = Crossed(n_shape=2).fit(X)
    Z = tf.transform(X)
    assert Z.shape == (200, 5 + 2 * 3)
    assert np.allclose(Z.mean(axis=0), 0, atol=1e-4) and np.allclose(Z.std(axis=0), 1, atol=1e-3)


def _toy(S=40, C=6, seed=2):
    rng = np.random.default_rng(seed)
    Xs = rng.normal(size=(S, 4))
    # Config c is best where Xs[:, 0] falls in its band, so the label is learnable from feature 0.
    band = np.clip(((Xs[:, 0] + 2) / 4 * C).astype(int), 0, C - 1)
    R = 1.5 + rng.random((S, C))
    R[np.arange(S), band] = 1.0
    Pc = np.arange(C, dtype=float)[:, None].repeat(7, axis=1)
    Fc = np.broadcast_to(Xs[:, None, :1], (S, C, 9)).astype(np.float32).copy()
    return Xs, R, Pc, Fc


def test_fit_predict_cls_picks_runnable_members_and_falls_back():
    from analysis.learned import fit_predict_cls
    Xs, R, _, _ = _toy()
    train, ok = np.arange(30), np.ones((10, R.shape[1]), bool)
    ok[:, 0] = False
    pick, members, _ = fit_predict_cls("dt", {"P": 6, "max_depth": 4}, Xs, R, train, Xs[30:], ok, fallback=5, full=True)
    assert set(pick) <= set(members) and not (pick == 0).any()
    none = fit_predict_cls("logreg", {"P": 6, "C": 1}, Xs, R, train, Xs[30:], np.zeros_like(ok), fallback=5)
    assert (none == 5).all()


def test_fit_predict_rank_picks_from_runnable_pool():
    from analysis.learned import fit_predict_rank
    Xs, R, Pc, Fc = _toy()
    train, ok = np.arange(30), np.ones((10, R.shape[1]), bool)
    ok[:, 1] = False
    for model, hp in (("rf", {"max_features": 0.5, "min_samples_leaf": 2}), ("logreg", {"C": 1.0}),
                      ("lgbm", {"objective": "lambdarank", "num_leaves": 4, "n_estimators": 20})):
        pick, pool, _ = fit_predict_rank(model, hp, Xs, Pc, Fc, R, train, Xs[30:], Fc[30:], ok, fallback=0, full=True)
        assert set(pick) <= set(pool) and not (pick == 1).any()


def test_online_choose_uses_graph_times_below_50us_and_sums_cost():
    from tune.online import choose
    c = [{"config_key": "a", "valid": True, "compile_s": 1.0, "time_s": 0.1, "cold_us": 20.0, "graph_us": 9.0},
         {"config_key": "b", "valid": False, "compile_s": 0.5},
         {"config_key": "c", "valid": True, "compile_s": 2.0, "time_s": 0.2, "cold_us": 19.0, "graph_us": 10.0}]
    one = choose(c, 1)
    assert one["key"] == "a" and one["mode"] == "graph" and one["compile_s"] == 1.0
    three = choose(c, 3)
    assert three["key"] == "a" and three["rank"] == 0  # graph time decides: 9 < 10 although c is faster cold
    assert abs(three["compile_s"] - 3.5) < 1e-12 and abs(three["time_s"] - 0.3) < 1e-12
    big = [dict(x, cold_us=x["cold_us"] * 10) for x in c if x["valid"]]
    assert choose(big, 2)["key"] == "c" and choose(big, 2)["mode"] == "cold"
    assert choose(c[1:2], 1)["key"] is None


def test_ranker_save_load_round_trip(tmp_path):
    import lightgbm as lgb
    from analysis.learned import Ranker
    gpu = {"smem_per_block_optin_bytes": 101376, "regs_per_sm": 65536, "sm_count": 68, "smem_per_sm_bytes": 102400,
           "max_threads_per_sm": 1536}
    pool = ["bm16_bn64_bk64_w4_s3_g1_sk1", "bm64_bn64_bk32_w4_s3_g8_sk1", "bm128_bn128_bk32_w8_s3_g8_sk1",
            "bm32_bn64_bk64_w4_s4_g1_sk4"]
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 35)).astype(np.float32)
    est = lgb.LGBMRegressor(n_estimators=20, num_leaves=4, verbose=-1).fit(X, X[:, 22] - X[:, 0])
    rk = Ranker(est.booster_, pool, {"objective": "regression"}, gpu,
                {"note": "toy", "diverse_rule": {"group": "tile_split", "delta": 10.0, "plain": 1}})
    rk.save(tmp_path / "m")
    back = Ranker.load(tmp_path / "m")
    for M, N, K in ((1, 4096, 4096), (4096, 4096, 4096), (300, 77, 9000)):
        top = rk.rank(M, N, K, 3)
        assert top == back.rank(M, N, K, 3) and set(top) <= set(pool) and len(top) <= 3
        assert np.allclose(rk.scores(M, N, K), back.scores(M, N, K))
    assert back.meta["note"] == "toy"
    d = back.rank(4096, 4096, 4096, 3, order="diverse")
    assert d[0] == back.rank(4096, 4096, 4096, 1)[0] and len(d) == len(set(d))


def test_diverse_order_spreads_over_groups_within_delta_then_fills():
    from analysis.learned import PLAIN_RULE, diverse_order
    score = np.array([0.00, 0.01, 0.02, 0.04, 0.30, np.inf])
    groups = np.array([0, 0, 0, 1, 2, 3])
    assert diverse_order(score, groups, 3, PLAIN_RULE) == [0, 1, 2]
    rule = {"group": "x", "delta": 0.05, "plain": 1}
    assert diverse_order(score, groups, 3, rule) == [0, 3, 1]  # group 2 is 0.30 behind: filled from score order
    assert diverse_order(score, groups, 3, dict(rule, delta=1.0)) == [0, 3, 4]
    assert diverse_order(score, groups, 3, dict(rule, plain=2)) == [0, 1, 3]
    assert diverse_order(score, groups, 9, dict(rule, delta=1.0)) == [0, 3, 4, 1, 2]  # never the unrunnable one
    assert diverse_order(np.full(3, np.inf), groups[:3], 2, rule) == []


def test_config_groups_by_tile_split():
    from analysis.learned import config_groups
    from tune.sweep import parse_key
    cfgs = [parse_key(k) for k in ("bm64_bn64_bk32_w4_s3_g8_sk1", "bm64_bn64_bk64_w4_s4_g1_sk1",
                                   "bm64_bn64_bk32_w4_s3_g8_sk2", "bm32_bn64_bk32_w4_s3_g8_sk1")]
    g = config_groups(cfgs, "tile_split")
    assert g[0] == g[1] and len({g[0], g[2], g[3]}) == 3
    assert len(set(config_groups(cfgs, "tile_split_bk"))) == 4 and len(set(config_groups(cfgs, None))) == 4


def test_diverse_order_inner_pass_varies_warps_within_the_top_group():
    from analysis.learned import diverse_order
    score = np.array([0.00, 0.01, 0.015, 0.03, 0.04])
    outer = np.array([0, 0, 0, 1, 2])      # configs 0-2 share the top tile x BLOCK_K x split
    inner = np.array([4, 4, 8, 4, 4])      # num_warps: config 2 is the top tile with 8 warps
    base = {"group": "x", "delta": 0.05, "plain": 1}
    assert diverse_order(score, outer, 3, base) == [0, 3, 4]
    before = dict(base, inner=1, inner_group="warps", inner_delta=0.02, inner_pos="before")
    assert diverse_order(score, outer, 3, before, inner_groups=inner) == [0, 2, 3]
    after = dict(before, inner_pos="after")
    assert diverse_order(score, outer, 3, after, inner_groups=inner) == [0, 3, 4]  # outer fills k first
    assert diverse_order(score, outer, 4, after, inner_groups=inner) == [0, 3, 4, 2]
    tight = dict(before, inner_delta=0.01)
    assert diverse_order(score, outer, 3, tight, inner_groups=inner) == [0, 3, 4]  # 8-warp variant too far behind
