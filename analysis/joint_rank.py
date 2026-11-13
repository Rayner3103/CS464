"""One score for Triton and cuBLASLt candidates: can a joint top-m pool replace the fixed pool? CPU only.

The two family rankers predict a within-family log regret (Triton: `Ranker.scores` against the Triton screen oracle; cuBLASLt:
`baselines/cublaslt_rank` against the cuBLASLt screen best), which are not comparable across families. A joint score adds each family's
expected best time on one absolute scale:

    s(c) = A_f(shape) + r_c,     A_f(shape) = log(t_best_f / roofline)  (f = Triton screen-oracle, cuBLASLt B3+)

A_f is a small LightGBM regressor on shape features fit on the 477 TUNE shapes' full-protocol stage 2 (`results_lock1755_tune477[_graph]`,
one session); r_c is the family ranker's prediction for the candidate. The pool is the m lowest scores of every candidate the search
could time (Triton diverse top-5, the heuristic top-16, the learned ranker's picks), and the choice inside it is the usual short-timing
`tune.online.choose`. A pool of m candidates costs m validations + timings (and a compile only for the Triton ones among them), so m is the
budget; when no Triton candidate is in the top m the Triton half is skipped, which makes the cost-only router a special case.

`fit` (grouped out-of-fold accuracy of A_f on tune shapes; no held-out shape read) then `score` (held-out simulation from the online run +
its stage-2 session, same files as `analysis.lt_rank_ab`). The pool sizes m are printed as a curve, none selected on held-out.

    python -m analysis.joint_rank [--online online_joint_diverse_v2_knn8_rank16.jsonl --results ... --graph-results ...]
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

from analysis import ranker_compare as rc

KEY_RE = re.compile(r"^[hnr](\d+)_a(\w+)_t(\w+)_s(\w+)_sk(\w+)_r(\w+)_sw(\w+)_c(\w+)$")
PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=6, min_data_in_leaf=15, feature_fraction=0.9, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=4, seed=0)
ROUNDS = 200


def shape_features(M: int, N: int, K: int) -> list[float]:
    from suites.generate import roofline_us
    roof = roofline_us(M, N, K)
    ctas = math.ceil(M / 128) * math.ceil(N / 128)
    return [math.log2(M), math.log2(N), math.log2(K), math.log(roof), math.log(ctas / 68 + 1e-9), float(M % 8 == 0), float(N % 8 == 0),
            float(K % 8 == 0), math.log(2 * M * N * K / (2 * (M * K + K * N + M * N)))]


def tune_targets(d: Path):
    cold = rc.load_stage2(d, ["results_lock1755_tune477.jsonl"], "cold")
    hot = rc.load_stage2(d, ["results_lock1755_tune477_graph.jsonl"], "hot")
    ref = rc.references(cold, hot)
    shapes = sorted(ref)
    from suites.generate import roofline_us
    X = np.array([shape_features(*s) for s in shapes])
    y = {"triton": np.array([math.log(ref[s]["tri_orc"] / roofline_us(*s)) for s in shapes]),
         "lt": np.array([math.log(ref[s]["b3p"] / roofline_us(*s)) for s in shapes])}
    return shapes, X, y


def fit_family_models(d: Path, verbose=True):
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    from analysis.lt_rank import groups_of
    shapes, X, y = tune_targets(d)
    grp = groups_of(shapes)
    models = {}
    for f, yf in y.items():
        oof = np.zeros(len(yf))
        for tr, te in GroupKFold(5).split(X, groups=grp):
            oof[te] = lgb.train(PARAMS, lgb.Dataset(X[tr], yf[tr]), ROUNDS).predict(X[te])
        if verbose:
            print(f"A_{f}: out-of-fold MAE {np.abs(yf - oof).mean():.3f} (constant {np.abs(yf - np.median(yf)).mean():.3f}), "
                  f"R2 {1 - ((yf - oof) ** 2).sum() / ((yf - yf.mean()) ** 2).sum():.3f}, sd {yf.std():.2f}")
        models[f] = lgb.train(PARAMS, lgb.Dataset(X, yf), ROUNDS)
    if verbose:
        diff = y["triton"] - y["lt"]
        pred_diff = np.zeros(len(diff))
        for tr, te in GroupKFold(5).split(X, groups=grp):
            pa = lgb.train(PARAMS, lgb.Dataset(X[tr], y["triton"][tr]), ROUNDS).predict(X[te])
            pb = lgb.train(PARAMS, lgb.Dataset(X[tr], y["lt"][tr]), ROUNDS).predict(X[te])
            pred_diff[te] = pa - pb
        print(f"A_triton - A_lt (who wins, log scale): out-of-fold corr {np.corrcoef(diff, pred_diff)[0, 1]:.3f}; sign agrees on "
              f"{100 * np.mean(np.sign(diff) == np.sign(pred_diff)):.0f}% of tune shapes")
    return models


def lt_scores(booster, M, N, K, keys):
    """Predicted log regret of cuBLASLt candidates given by their keys (any of the h / n / r prefixes)."""
    from baselines.cublaslt_rank import features
    cfgs = []
    for k in keys:
        m = KEY_RE.match(k)
        vals = [-1 if v == "None" else int(v) for v in m.groups()[1:]]
        cfgs.append(vals)
    return booster.predict(features((M, N, K), np.array(cfgs, dtype=np.int64)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--online", default="online_joint_diverse_v2_knn8_rank16.jsonl")
    ap.add_argument("--results", default="results_lock1755_rank16.jsonl")
    ap.add_argument("--graph-results", default="results_lock1755_rank16_graph.jsonl")
    ap.add_argument("--ranker", default="lgbm_rank_v2")
    a = ap.parse_args(argv)
    from analysis.learned import Ranker
    from analysis.lt_rank_ab import timeof
    from baselines.cublaslt_rank import LtRanker
    from tune import online
    from tune.env import default_data_dir
    d = a.data or default_data_dir()
    models = fit_family_models(d)
    tri_ranker = Ranker.load(d / a.ranker)
    lt_ranker = LtRanker.load(d / "lt_rank")
    cold, hot = rc.load_stage2(d, [a.results], "cold"), rc.load_stage2(d, [a.graph_results], "hot")
    ref = rc.references(cold, hot)
    shapes = sorted(ref)
    small = np.array([ref[s]["small"] for s in shapes]); eps = np.where(small, 1.05, 1.02)
    ORC, B3P = np.array([ref[s]["orc_x"] for s in shapes]), np.array([ref[s]["b3p"] for s in shapes])
    rows = {(r["M"], r["N"], r["K"]): r for r in map(json.loads, open(d / a.online))}

    scored = {}
    for s in shapes:
        c = rows[s]["candidates"]
        x = np.array([shape_features(*s)])
        a_tri, a_lt = float(models["triton"].predict(x)[0]), float(models["lt"].predict(x)[0])
        sc_tri = dict(zip(tri_ranker.pool_keys, tri_ranker.scores(*s)))
        lt_keys = [t["config_key"] for t in c if t.get("family", "triton") != "triton"]
        sc_lt = dict(zip(lt_keys, lt_scores(lt_ranker.booster, *s, lt_keys))) if lt_keys else {}
        for t in c:
            fam = t.get("family", "triton")
            t["_s"] = a_tri + float(sc_tri[t["config_key"]]) if fam == "triton" else a_lt + float(sc_lt[t["config_key"]])
        scored[s] = c

    def evaluate(poolfn):
        t, cost, nt, size = [], [], [], []
        for s in shapes:
            p = poolfn(scored[s])
            ch = online.choose(p, 99, None)
            t.append(timeof(s, ch["key"], ref, cold))
            tri_in = [x for x in p if x.get("family", "triton") == "triton"]
            cost.append(sum(x["compile_s"] + x.get("time_s", 0) for x in p) + (rows[s]["rank_s"] if tri_in else 0.0))
            nt.append(len(tri_in)); size.append(len(p))
        t = np.array(t)
        return t, np.array(cost), np.array(nt), np.array(size)

    def fixed(c, nt=3, h=16, r=5):
        out, ti, hi, ri = [], 0, 0, 0
        for x in c:
            f = x.get("family", "triton")
            ti += f == "triton"; hi += f == "cublaslt"; ri += f == "cublaslt_rank"
            if (f == "triton" and ti <= nt) or (f == "cublaslt" and hi <= h) or (f == "cublaslt_rank" and ri <= r):
                out.append(x)
        return out

    def topm(c, m, always_h1=False):
        pool = [x for x in c if x.get("family") != "cublaslt_knn"]
        pool = sorted(pool, key=lambda x: x["_s"])[:m]
        if always_h1 and not any(x.get("family") == "cublaslt" for x in pool):   # keep the library's own first choice as a safety net
            pool.append(next(x for x in c if x.get("family") == "cublaslt"))
        return pool

    print(f"\nheld-out simulation, n = {len(shapes)}; regret vs the joint oracle (median / p95 / p99 / mean), shapes > 5% / > 10% off, slower than B3+, "
          "mean pool size, cost (no bundle), shapes with no Triton candidate in the pool")
    rowsout = [("fixed: Triton top-3 + heuristic 16 + ranker 5", fixed)]
    for m in (4, 6, 8, 10, 12, 16, 24):
        rowsout.append((f"joint top-{m}", lambda c, m=m: topm(c, m)))
    for m in (4, 6, 8, 10):
        rowsout.append((f"joint top-{m} + heuristic #1", lambda c, m=m: topm(c, m, True)))
    for lab, fn in rowsout:
        t, cost, nt, size = evaluate(fn)
        g = t / ORC - 1
        print(f"  {lab:46s} {100 * np.median(g):4.1f}% / {100 * np.percentile(g, 95):5.1f}% / {100 * np.percentile(g, 99):5.1f}% / {100 * g.mean():4.1f}% | {(g > .05).sum():2d} / {(g > .10).sum():2d} | "
              f"{(t > B3P * eps).sum():2d} | {size.mean():4.1f} | {cost.mean():.3f}s | {(nt == 0).sum():2d}")


if __name__ == "__main__":
    main()
