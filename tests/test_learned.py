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
