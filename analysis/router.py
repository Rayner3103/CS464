"""Fit and score the cost-only family router (`tune/router.py`). CPU only.

fit:   train on the 477 TUNE shapes' full-protocol stage 2 (`results_lock1755_tune477[_graph].jsonl`: Triton screen top-8 vs B3+, one session).
       Grouped 5-fold out-of-fold scores (groups = shapes sharing two of M, N, K) choose the skip threshold by a rule fixed here, before any
       held-out shape is read: the largest P(win) threshold whose out-of-fold mean loss over ALL tune shapes stays <= LOSS_CAP (loss on a skipped
       shape = max(t_cublaslt / t_triton_oracle - 1, 0), the worst case: the real Triton pool is the ranker's top-3, not the oracle).
       Saves data/<hw>/router.txt/.json.
score: held-out simulation from the online run and its stage-2 session (reads held-out shapes, fits nothing): for each shape the router sees the best
       short-timed cuBLASLt candidate of the run's pool (heuristic top-16 + learned-ranker top-N), exactly what the search would have at decision
       time; a skipped shape keeps that candidate, any other shape keeps the run's full joint choice. Each pick is looked up at its stage-2 time.

    python -m analysis.router fit [--save data/<hw>/router]
    python -m analysis.router score [--online online_joint_diverse_v2_knn8_rank16.jsonl --results results_lock1755_rank16.jsonl --graph-results ..._graph.jsonl]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from analysis import ranker_compare as rc
from tune.router import FEATURES, features

LOSS_CAP = 0.001          # out-of-fold mean loss over all tune shapes the threshold may cost (0.10%)
GRID = np.round(np.arange(0.005, 0.2001, 0.005), 3)
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=8, min_data_in_leaf=15, feature_fraction=0.9, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=4, seed=0)
ROUNDS = 150


def tune_table(d: Path):
    cold = rc.load_stage2(d, ["results_lock1755_tune477.jsonl"], "cold")
    hot = rc.load_stage2(d, ["results_lock1755_tune477_graph.jsonl"], "hot")
    ref = rc.references(cold, hot)
    shapes = sorted(ref)
    lt = np.array([ref[s]["b3p"] for s in shapes]); tri = np.array([ref[s]["tri_orc"] for s in shapes])
    eps = np.array([1.05 if ref[s]["small"] else 1.02 for s in shapes])
    X = np.array([features(*s, t) for s, t in zip(shapes, lt)])
    return shapes, X, lt, tri, eps


def fit(d: Path, save: Path | None):
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from analysis.lt_rank import groups_of
    from tune.env import default_data_dir
    shapes, X, lt, tri, eps = tune_table(d)
    win = (tri * eps < lt).astype(int)
    print(f"{len(shapes)} tune shapes; Triton's screen-oracle beats B3+ by more than the tie margin on {win.sum()} ({100 * win.mean():.0f}%)")
    grp = groups_of(shapes)
    oof = np.zeros(len(shapes))
    for tr, te in GroupKFold(5).split(X, groups=grp):
        oof[te] = lgb.train(PARAMS, lgb.Dataset(X[tr], win[tr]), ROUNDS).predict(X[te])
    print(f"out-of-fold AUC {roc_auc_score(win, oof):.3f}")
    print("  skip if P(win) <  | skipped | harmed | mean loss (all) | mean loss (skipped) | worst single loss")
    chosen = None
    for p in GRID:
        sk = oof < p
        loss = np.maximum(lt / tri - 1, 0) * sk
        if loss.mean() <= LOSS_CAP:
            chosen = p
        if p in (0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2) or p == chosen:
            print(f"  {p:5.3f}            | {sk.sum():3d} ({100 * sk.mean():2.0f}%) | {(sk & (win == 1)).sum():3d}    | {100 * loss.mean():5.2f}%          | "
                  f"{100 * loss[sk].mean() if sk.any() else 0:5.2f}%              | {100 * loss.max():5.1f}%")
    print(f"threshold by the rule (largest with out-of-fold mean loss <= {100 * LOSS_CAP:.2f}%): {chosen}")
    if save:
        m = lgb.train(PARAMS, lgb.Dataset(X, win), ROUNDS)
        m.save_model(str(save) + ".txt")
        Path(str(save) + ".json").write_text(json.dumps({
            "hw_id": default_data_dir().name, "features": FEATURES, "threshold": float(chosen), "loss_cap": LOSS_CAP, "rounds": ROUNDS,
            "params": PARAMS, "n_tune_shapes": len(shapes), "oof_auc": float(roc_auc_score(win, oof)),
            "importance_gain": dict(zip(FEATURES, map(float, m.feature_importance("gain"))))}, indent=1))
        print("saved", save)


def score(d: Path, online_f: str, results: str, graph: str, stem: Path | None):
    from analysis.lt_rank_ab import K_TRITON, timeof
    from tune import online
    from tune.router import Router, default_stem
    router = Router.load(stem or default_stem())
    cold, hot = rc.load_stage2(d, [results], "cold"), rc.load_stage2(d, [graph], "hot")
    ref = rc.references(cold, hot)
    shapes = sorted(ref)
    small = np.array([ref[s]["small"] for s in shapes]); eps = np.where(small, 1.05, 1.02)
    B3P, ORC = np.array([ref[s]["b3p"] for s in shapes]), np.array([ref[s]["orc_x"] for s in shapes])
    rows = {(r["M"], r["N"], r["K"]): r for r in map(json.loads, open(d / online_f))}
    print(f"router threshold {router.threshold}, trained on {router.meta['n_tune_shapes']} tune shapes (OOF AUC {router.meta['oof_auc']:.3f}); n = {len(shapes)} held-out")

    def run(thr_scale=None, n_rank=5):
        t, cost, skipped, p_all = [], [], [], []
        for s in shapes:
            r = rows[s]; c = r["candidates"]
            tri = [x for x in c if x.get("family", "triton") == "triton"][:K_TRITON]
            lt = [x for x in c if x.get("family") == "cublaslt"] + [x for x in c if x.get("family") == "cublaslt_rank"][:n_rank]
            full = online.choose(tri + lt, K_TRITON, None)
            ltc = online.choose(lt, 0, None)
            ok = [x for x in lt if x["valid"] and x.get("cold_us") is not None]
            use_graph = ltc["mode"] == "graph"
            t_lt = ltc and min((x["graph_us"] if use_graph else x["cold_us"]) for x in ok)
            p = router.p_win(*s, t_lt)
            thr = router.threshold if thr_scale is None else thr_scale
            sk = p < thr
            ch = ltc if sk else full
            lt_cost = sum(x["compile_s"] + x.get("time_s", 0) for x in lt)
            tri_cost = sum(x["compile_s"] + x.get("time_s", 0) for x in tri) + r["rank_s"]
            t.append(timeof(s, ch["key"], ref, cold)); cost.append(lt_cost + (0 if sk else tri_cost)); skipped.append(sk); p_all.append(p)
        return np.array(t), np.array(cost), np.array(skipped), np.array(p_all)

    base_t, base_c, _, _ = run(thr_scale=0.0)       # never skip = the current default pool (Triton top-3 + LT16 + ranker top-5)
    print("\npool: v2 top-3 Triton + heuristic LT16 + learned-ranker top-5; regret vs the joint oracle (median / p95 / p99 / mean), shapes > 5% / > 10% off,")
    print("shapes slower than B3+ by the margin, first-use cost from the no-bundle online run (true compile), shapes skipped")
    for lab, thr in (("never skip (default)", 0.0), ("router @ rule threshold", None), ("router @ 2x threshold", 2 * router.threshold),
                     ("router @ 0.05", 0.05)):
        t, cost, sk, p = run(thr)
        g = t / ORC - 1
        print(f"  {lab:26s} {100 * np.median(g):4.1f}% / {100 * np.percentile(g, 95):5.1f}% / {100 * np.percentile(g, 99):5.1f}% / {100 * np.mean(g):4.1f}% | {(g > 0.05).sum():2d} / {(g > 0.10).sum():2d} | "
              f"{(t > B3P * eps).sum():2d} | {cost.mean():.3f}s (-{100 * (1 - cost.mean() / base_c.mean()):2.0f}%) | skipped {sk.sum():2d}"
              + (f" | worse than default by > margin on {(t > base_t * eps).sum()}, better on {(t < base_t / eps).sum()}" if thr != 0.0 else ""))
    t, cost, sk, p = run(None)
    harmed = np.where(t > base_t * eps)[0]
    for i in harmed:
        print(f"    harmed: {shapes[i]!s:24s} {base_t[i]:8.1f} -> {t[i]:8.1f} us  p_win {p[i]:.3f}")
    print(f"  skipped shapes: median p_win {np.median(p[sk]):.3f}; not skipped: {np.median(p[~sk]):.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit"); f.add_argument("--save", type=Path, default=None); f.add_argument("--data", type=Path, default=None)
    s = sub.add_parser("score"); s.add_argument("--data", type=Path, default=None)
    s.add_argument("--online", default="online_joint_diverse_v2_knn8_rank16.jsonl"); s.add_argument("--results", default="results_lock1755_rank16.jsonl")
    s.add_argument("--graph-results", default="results_lock1755_rank16_graph.jsonl"); s.add_argument("--model", type=Path, default=None)
    a = ap.parse_args(argv)
    from tune.env import default_data_dir
    d = a.data or default_data_dir()
    if a.cmd == "fit":
        fit(d, a.save)
    else:
        score(d, a.online, a.results, a.graph_results, a.model)


if __name__ == "__main__":
    main()
