"""A learned ranker over cuBLASLt configurations vs the k-NN proposer, scored exactly in grouped cross-validation. CPU only.

Data: the full per-config screen of the TUNE shapes (`baselines/cublaslt_screen.py`: every enumerated config's screen time, hot L2,
3 launches). Model: a LightGBM regressor of log(t / t_best_of_shape) per (shape, config) from shape features, config parameters
(algo / tile / stages as categoricals, split-K, reduction, swizzle, custom) and fill features from the tile dimensions (CTA count and
padding waste of the tile on the problem in both orientations, waves on 68 SMs, K per split). Targets are clipped at log 8 so the
terrible configs do not dominate. Folds are GroupKFold over shapes grouped by "shares two of M, N, K" (the grouping the selectors use),
so a near-duplicate shape never trains and tests together.

Metric (exact: every config's time is known, unlike `analysis.lt_knn_scan`, which could only see a shape's stored top 8): for each
shape, the pool a method proposes is added to the best HEURISTIC-SEEDED config of that shape's own search (`seed` rows: the heuristic
top-16 and a per-algorithm heuristic search = what the shipped joint search already times); `captured` = log(best_seed / pool best) /
log(best_seed / screen best) over shapes whose screen improves on the seeds by > 5%; also "closes fully" (within 2% of the screen
best) and the mean regret over all shapes. Methods: the k-NN proposer at several (k, per-neighbour) (neighbours from the screen's own
winners, same grouped exclusion), the ranker's top-m, and their union.

    python -m analysis.lt_rank [--folds 5] [--no-cv-cache]
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np

from baselines.cublaslt_rank import CATS, FEATURES, SMS, TILE_MN, Y_CLIP, device_sms, features  # noqa: E402  (shared with the runtime ranker)
from baselines.cublaslt_search import CFG_KEYS  # noqa: E402


def load_screens(d: Path | None = None) -> dict:
    """{(M, N, K): {"cfgs": (n, 7) int array (-1 for None), "us": (n,), "seed": (n,) bool, "complete": bool}}"""
    from baselines.cublaslt_screen import load, out_dir
    d = d or out_dir()
    out = {}
    for f in sorted(d.glob("*_fp16.json.gz")):
        z = load(f)
        rows = z["rows"]
        if len(rows) < 8:
            continue
        cfgs = np.array([[-1 if v is None else v for v in r[0]] for r in rows], dtype=np.int64)
        out[(z["M"], z["N"], z["K"])] = {"cfgs": cfgs, "us": np.array([r[1] for r in rows]), "seed": np.array([r[2] for r in rows]),
                                         "complete": bool(z["screened_all"])}
    return out


def groups_of(shapes: list) -> np.ndarray:
    """Connected components of 'shares two of M, N, K' (union-find)."""
    parent = list(range(len(shapes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(shapes)):
        for j in range(i + 1, len(shapes)):
            if sum(a == b for a, b in zip(shapes[i], shapes[j])) >= 2:
                parent[find(i)] = find(j)
    return np.array([find(i) for i in range(len(shapes))])


def shape_feat(s: tuple, sms: int = SMS) -> np.ndarray:
    M, N, K = s
    return np.array([math.log2(M), math.log2(N), math.log2(K), math.log2(math.ceil(M / 128) * math.ceil(N / 128) / sms + 1e-9),
                     float(M % 8 == 0 and N % 8 == 0 and K % 8 == 0)])


def pool_score(sc: dict, pool: list[tuple]) -> float:
    """Time of the best of (heuristic seeds + pool), the pool looked up in this shape's screen; an unknown config = no gain."""
    cfg_index = sc["index"]
    ts = [sc["us"][cfg_index[p]] for p in pool if p in cfg_index]
    return min([sc["best_seed"]] + ts)


def prepare(screens: dict) -> list:
    shapes = sorted(screens)
    for s in shapes:
        d = screens[s]
        d["index"] = {tuple(int(v) for v in row): i for i, row in enumerate(d["cfgs"])}
        d["best"] = float(d["us"].min())
        d["best_seed"] = float(d["us"][d["seed"]].min()) if d["seed"].any() else float(d["us"].min())
        d["need"] = d["best_seed"] / d["best"] > 1.05
    return shapes


def summarize(rows: list[tuple[float, float, float]], need: np.ndarray) -> dict:
    """rows = (best_seed, pool_best, screen_best) per shape."""
    r = np.array(rows)
    seed, got, best = r[:, 0], r[:, 1], r[:, 2]
    cap = np.log(seed / got)[need] / np.log(seed / best)[need]
    return {"captured": float(np.mean(cap)), "full": float(np.mean((got <= 1.02 * best)[need])),
            "regret": float(np.mean(got / best - 1)), "p95": float(np.percentile(got / best - 1, 95)), "n_need": int(need.sum())}


def knn_pool(i: int, order_i: list, tops: list, k: int, per: int) -> list[tuple]:
    seen, pool = set(), []
    for j in order_i[:k]:
        for c in tops[j][:per]:
            if c not in seen:
                seen.add(c)
                pool.append(c)
    return pool


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--screen-dir", type=Path, default=None)
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--save", type=Path, default=None, help="fit on all tune shapes and save the model here (text) + .json")
    ap.add_argument("--sms", type=int, default=None,
                    help="SM count of the GPU the screen was measured on (default: this device's; the CTA / wave features use it and "
                         "it is saved in the model's json). Needed only to fit on a machine other than the screened GPU")
    a = ap.parse_args(argv)
    sms = a.sms or device_sms()
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold

    screens = load_screens(a.screen_dir)
    shapes = prepare(screens)
    n = len(shapes)
    complete = sum(screens[s]["complete"] for s in shapes)
    need = np.array([screens[s]["need"] for s in shapes])
    print(f"{n} tune shapes screened ({complete} completely), {sum(len(screens[s]['us']) for s in shapes)} (shape, config) rows; "
          f"{need.sum()} shapes where the screen beats the best heuristic seed by > 5%")

    X = [features(s, screens[s]["cfgs"], sms) for s in shapes]
    Y = [np.minimum(np.log(screens[s]["us"] / screens[s]["best"]), Y_CLIP) for s in shapes]
    grp = groups_of(shapes)
    print(f"{len(set(grp))} shape groups (shares two of M, N, K)")

    params = dict(objective="regression", learning_rate=0.06, num_leaves=63, min_data_in_leaf=60, feature_fraction=0.8,
                  bagging_fraction=0.7, bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=8, max_cat_to_onehot=8)

    def fit(idx):
        Xt = np.vstack([X[i] for i in idx]); yt = np.concatenate([Y[i] for i in idx])
        w = np.where(yt < 0.1, 3.0, 1.0)           # near-top rows count more: they decide the pool
        ds = lgb.Dataset(Xt, yt, weight=w, feature_name=FEATURES, categorical_feature=CATS, free_raw_data=False)
        return lgb.train(params, ds, num_boost_round=a.rounds)

    t0 = time.perf_counter()
    pred = [None] * n
    for f, (tr, te) in enumerate(GroupKFold(a.folds).split(np.arange(n), groups=grp)):
        m = fit(tr)
        for i in te:
            pred[i] = m.predict(X[i])
        print(f"  fold {f + 1}/{a.folds}: {len(tr)} train / {len(te)} test shapes ({time.perf_counter() - t0:.0f}s)", flush=True)

    # --- k-NN proposer on the screen's own winners (grouped exclusion), exact times
    F = np.array([shape_feat(s, sms) for s in shapes]); F = F / (F.std(0) + 1e-9)
    D = np.sqrt(((F[:, None, :] - F[None, :, :]) ** 2).sum(-1))
    order = [[j for j in np.argsort(D[i]) if j != i and grp[j] != grp[i]] for i in range(n)]
    tops = [[tuple(int(v) for v in screens[s]["cfgs"][j]) for j in np.argsort(screens[s]["us"])[:8]] for s in shapes]

    def score(poolfn):
        rows = []
        for i, s in enumerate(shapes):
            d = screens[s]
            rows.append((d["best_seed"], pool_score(d, poolfn(i)), d["best"]))
        return summarize(rows, need)

    def rank_top(i, m):
        d = screens[shapes[i]]
        ix = np.argsort(pred[i])[:m]
        return [tuple(int(v) for v in d["cfgs"][j]) for j in ix]

    print("\n== grouped CV, exact screen times; pool = best heuristic seed + the method's proposals ==")
    print(f"{'method':38s} {'pool':>5s} {'captured':>9s} {'closes fully':>12s} {'mean regret':>11s} {'p95':>6s}")
    base = score(lambda i: [])
    print(f"{'heuristic seeds only':38s} {0:5d} {base['captured']*100:8.1f}% {base['full']*100:11.0f}% {base['regret']*100:10.1f}% {base['p95']*100:5.1f}%")
    out = {"n": n, "need": int(need.sum()), "methods": {}}

    def row(label, poolfn, size):
        r = score(poolfn)
        out["methods"][label] = {**r, "pool": size}
        print(f"{label:38s} {size:5.1f} {r['captured']*100:8.1f}% {r['full']*100:11.0f}% {r['regret']*100:10.1f}% {r['p95']*100:5.1f}%")

    for k, per in ((5, 2), (8, 2), (10, 2), (15, 1)):
        size = np.mean([len(knn_pool(i, order[i], tops, k, per)) for i in range(n)])
        row(f"kNN k={k} x {per}", lambda i, k=k, per=per: knn_pool(i, order[i], tops, k, per), size)
    for m in (3, 5, 8, 10, 16, 24):
        row(f"ranker top-{m}", lambda i, m=m: rank_top(i, m), m)
    for k, per, m in ((8, 2, 5), (8, 2, 10), (5, 2, 5)):
        size = np.mean([len(set(knn_pool(i, order[i], tops, k, per)) | set(rank_top(i, m))) for i in range(n)])
        row(f"kNN {k}x{per} + ranker top-{m}", lambda i, k=k, per=per, m=m: list(set(knn_pool(i, order[i], tops, k, per)) | set(rank_top(i, m))), size)

    # rank quality overall: where does the screen's best config sit in the predicted order?
    pos = [int(np.argsort(np.argsort(pred[i]))[int(np.argmin(screens[shapes[i]]["us"]))]) + 1 for i in range(n)]
    pos = np.array(pos)
    print(f"\nrank of the screen's best config in the model's order (of ~{int(np.mean([len(screens[s]['us']) for s in shapes]))} configs): "
          f"median {np.median(pos):.0f}, p75 {np.percentile(pos, 75):.0f}, p90 {np.percentile(pos, 90):.0f}; in top 10 on {np.mean(pos <= 10)*100:.0f}% of shapes")
    imp = None
    if a.save:
        m = fit(np.arange(n))
        m.save_model(str(a.save) + ".txt")
        imp = dict(sorted(zip(FEATURES, map(float, m.feature_importance("gain"))), key=lambda kv: -kv[1]))
        from baselines.cublaslt import library_info
        from tune.env import default_data_dir
        Path(str(a.save) + ".json").write_text(json.dumps({"hw_id": default_data_dir().name, "cublaslt_version": library_info()["version"],
                                                          "features": FEATURES, "cats": CATS, "params": params, "rounds": a.rounds,
                                                          "n_tune_shapes": n, "sms": sms, "tile_mn": TILE_MN, "y_clip": Y_CLIP,
                                                          "importance_gain": imp}, indent=1))
        print("saved", a.save, "| top features by gain:", list(imp)[:8])
    return out


if __name__ == "__main__":
    main()
