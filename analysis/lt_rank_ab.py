"""Held-out A/B of cuBLASLt candidate sources in the joint online search, scored from ONE stage-2 session. CPU only.

One online run (`tune online --lt-n 16 --lt-knn 8 --lt-rank 16`) short-times the Triton top-k, the heuristic top-16, the k-NN proposals and
the learned ranker's top-16 in the same pool, each tagged with its family. Every pool below is a subset of that candidate list
(Triton top-3, always the heuristic top-16, plus the named proposals), re-chosen with the same `tune.online.choose` rule (short timing, graph
below 50 us), so the pools share the run's timing noise and differ only in what they contain. Each pick is looked up at its full-protocol
time in the stage-2 session that timed everything (`cublaslt_knn:8`, `cublaslt_rank:16`, `cublaslt_deep:20`, ...). A proposal's key carries its
rank among the kept proposals, which differs between the online run and stage 2, so it is matched on its config part.

Prints, per pool: regret vs the joint oracle (every Triton config of the session and B3+), speed vs B3+, shapes > 5% / > 10% off, the first-use
cost (compile + timing of the pool's candidates + ranking) and paired comparisons.

    python -m analysis.lt_rank_ab [--results results_lock1755_rank16.jsonl --graph-results ..._graph.jsonl --online online_joint_diverse_v2_knn8_rank16.jsonl]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from analysis import ranker_compare as rc
from tune import online

K_TRITON = 3
POOLS = [  # label, kNN proposals kept, ranker picks kept
    ("heuristic LT16 only", 0, 0),
    ("+ kNN8 proposals", 99, 0),
    ("+ ranker top-5", 0, 5),
    ("+ ranker top-10", 0, 10),
    ("+ ranker top-16", 0, 16),
    ("+ kNN8 + ranker top-5", 99, 5),
    ("+ kNN8 + ranker top-10", 99, 10),
]
PRIMARY = ("+ kNN8 proposals", "+ ranker top-10")


def pool_of(cands: list[dict], n_knn: int, n_rank: int) -> list[dict]:
    out, n_tri, nk, nr = [], 0, 0, 0
    for c in cands:
        fam = c.get("family", "triton")
        if fam == "triton":
            n_tri += 1
            if n_tri <= K_TRITON:
                out.append(c)
        elif fam == "cublaslt":
            out.append(c)
        elif fam == "cublaslt_knn":
            nk += 1
            if nk <= n_knn:
                out.append(c)
        elif fam == "cublaslt_rank":
            nr += 1
            if nr <= n_rank:
                out.append(c)
    return out


def _suffix(key: str) -> str:
    return key.split("_", 1)[1]


def timeof(shape, key, ref, cold):
    for src in (ref[shape]["src"], cold[shape]):
        for f in ("tri", "lth", "ltd"):
            if key in src[f]:
                return src[f][key]
        if re.match(r"[nr]\d+_a", key):
            for f in ("ltk", "ltr"):
                for k, t in src[f].items():
                    if _suffix(k) == _suffix(key):
                        return t
    raise KeyError((shape, key))


def st(x):
    return f"{100 * np.median(x):4.1f}% / {100 * np.percentile(x, 95):5.1f}% / {100 * np.mean(x):4.1f}%"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--results", default="results_lock1755_rank16.jsonl")
    ap.add_argument("--graph-results", default="results_lock1755_rank16_graph.jsonl")
    ap.add_argument("--online", default="online_joint_diverse_v2_knn8_rank16.jsonl")
    a = ap.parse_args(argv)
    if a.data is None:
        from tune.env import default_data_dir
        a.data = default_data_dir()
    D = a.data
    cold, hot = rc.load_stage2(D, [a.results], "cold"), rc.load_stage2(D, [a.graph_results], "hot")
    ref = rc.references(cold, hot)
    shapes = sorted(ref)
    small = np.array([ref[s]["small"] for s in shapes])
    eps = np.where(small, 1.05, 1.02)
    B3, B3P, ORC = (np.array([ref[s][k] for s in shapes]) for k in ("b3", "b3p", "orc"))
    rows = {(r["M"], r["N"], r["K"]): r for r in map(json.loads, open(D / a.online))}
    print(f"n = {len(shapes)} held-out shapes ({small.sum()} graph-amortized); joint oracle = best Triton of the session and B3+")

    # proposer ceilings: the best config each source proposes, full protocol
    for lab, fam in (("k-NN8 proposals", "ltk"), ("ranker top-16", "ltr")):
        best = np.array([min(ref[s]["src"][fam].values(), default=np.inf) for s in shapes])
        gap = B3 > B3P * eps
        print(f"ceiling {lab:16s}: best/B3+ median {np.median(best / B3P):.3f}; within the margin of B3+ on {(best <= B3P * eps).sum()}, "
              f"beats B3+ by > margin on {(best < B3P / eps).sum()}; of the {gap.sum()} shapes where B3+ beats the heuristic, closes "
              f"{(gap & (best <= B3P * eps)).sum()} fully and >= half way (log time) {(gap & (np.log(np.minimum(B3, best) / B3) <= 0.5 * np.log(B3P / B3))).sum()}")

    res = {}
    for label, nk, nr in POOLS:
        t, cost, fam = [], [], []
        for s in shapes:
            r = rows[s]
            pool = pool_of(r["candidates"], nk, nr)
            ch = online.choose(pool, K_TRITON, None)
            t.append(timeof(s, ch["key"], ref, cold))
            cost.append(ch["compile_s"] + ch["time_s"] + r["rank_s"])
            fam.append(ch.get("family", "triton"))
        res[label] = (np.array(t), np.array(cost), np.array(fam))

    print("\n== regret vs the joint oracle (median / p95 / mean); v2 top-3 Triton + heuristic LT16 + the named proposals ==")
    print(f"{'pool':26s} {'all':>22s} {'>= 50 us':>22s} {'< 50 us':>22s} | >5% >10% | vs B3+ faster/slower | cost  picks kNN/rank")
    for label, _, _ in POOLS:
        t, cost, fam = res[label]
        g = t / ORC - 1
        print(f"{label:26s} {st(g):>22s} {st(g[~small]):>22s} {st(g[small]):>22s} | {(g > 0.05).sum():3d} {(g > 0.10).sum():3d} | "
              f"{(t < B3P / eps).sum():3d} / {(t > B3P * eps).sum():2d}            | {cost.mean():.2f}s {100 * np.mean(fam == 'cublaslt_knn'):3.0f}% / {100 * np.mean(fam == 'cublaslt_rank'):3.0f}%")

    print("\n== paired: geometric-mean time ratio (a / b), 95% bootstrap interval, shapes a faster / slower than b by the margin ==")
    base = res["heuristic LT16 only"][0]
    rng = np.random.default_rng(0)

    def paired(x, y, lab):
        r = np.log(x / y)
        bi = rng.integers(0, len(r), (4000, len(r)))
        gm = np.exp(np.array([r[i].mean() for i in bi]))
        print(f"  {lab:48s} {np.exp(r.mean()):.4f} [{np.percentile(gm, 2.5):.4f}, {np.percentile(gm, 97.5):.4f}]  "
              f"{(x < y / eps).sum():2d} / {(x > y * eps).sum():2d}")
    for label, _, _ in POOLS[1:]:
        paired(res[label][0], base, f"{label} vs heuristic only")
    paired(res["+ ranker top-10"][0], res["+ kNN8 proposals"][0], "+ ranker top-10 vs + kNN8 (pool 10 vs ~6.6)")
    paired(res["+ kNN8 + ranker top-10"][0], res["+ kNN8 proposals"][0], "+ kNN8 + ranker top-10 vs + kNN8")
    paired(res["+ kNN8 + ranker top-5"][0], res["+ kNN8 proposals"][0], "+ kNN8 + ranker top-5 vs + kNN8")

    print("\n== shapes still > 10% off the joint oracle with + ranker top-10 ==")
    t = res["+ ranker top-10"][0]
    for i in np.argsort(-(t / ORC))[:6]:
        if t[i] / ORC[i] - 1 > 0.05:
            print(f"  {shapes[i]!s:24s} {t[i]:8.1f} us  regret {100 * (t[i] / ORC[i] - 1):5.1f}%  B3+/oracle {B3P[i] / ORC[i]:.2f}")


if __name__ == "__main__":
    main()
