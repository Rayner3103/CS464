"""Tune-only leave-one-out scan of the cuBLASLt k-NN proposer's (k nearest shapes, configs per neighbour). CPU only.

For each tune shape q, propose from the OTHER tune shapes' stored deep-search top-8, excluding neighbours that share two of M, N, K
with q (the grouped leave-one-out the selectors use). A proposal counts only if it is in q's own stored top-8 (its screened time is
then known; a miss counts as no gain, a lower bound). Baseline = the best heuristic-seeded config of q's search. captured =
log(best_seed / with_proposals) / log(best_seed / search winner) over shapes the search improves by > 5%. No held-out shape is read.
`python -m analysis.lt_knn_scan` (reports/week1.md 2026-11-04); the shipped default `tune.online.KNN_DEFAULT` = 8 (x 2 per neighbour).
"""
from __future__ import annotations

import glob
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from baselines import cublaslt_knn as kn

tab = kn.table()
shapes, tops = tab.shapes, tab.tops
search = {}
for f in glob.glob(str(kn.lt_search_dir() / "*_fp16.json")):
    d = json.loads(Path(f).read_text())
    search[(d["M"], d["N"], d["K"])] = d
rows = []
for s, top in zip(shapes, tops):
    d = search[s]
    us = {tuple(t["cfg"].get(c) for c in kn.CFG_KEYS): t["us"] for t in d["top"]}
    rows.append(dict(best=d["top"][0]["us"], seed=d["best_seed_us"], us=us))
n = len(shapes)
need = np.array([r["seed"] / r["best"] > 1.05 for r in rows])
print(f"{n} tune shapes with a search; {need.sum()} improved by >5% over the best heuristic seed")

X = tab.X
D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
share2 = np.array([[sum(a == b for a, b in zip(shapes[i], shapes[j])) >= 2 for j in range(n)] for i in range(n)])
order = [[j for j in np.argsort(D[i]) if j != i and not share2[i, j]] for i in range(n)]
order_plain = [[j for j in np.argsort(D[i]) if j != i] for i in range(n)]

KS = (1, 2, 3, 5, 8, 10, 15, 20)
PS = (1, 2, 3, 4)


def evaluate(order, k, per):
    cap, full, half, reg, size, hitn = [], 0, 0, [], [], 0
    for i, r in enumerate(rows):
        seen, pool = set(), []
        for j in order[i][:k]:
            for cfg in tops[j][:per]:
                key = tuple(cfg.get(c) for c in kn.CFG_KEYS)
                if key not in seen:
                    seen.add(key); pool.append(key)
        known = [r["us"][p] for p in pool if p in r["us"]]
        with_t = min([r["seed"]] + known)
        size.append(len(pool)); reg.append(with_t / r["best"] - 1)
        if need[i]:
            c = math.log(r["seed"] / with_t) / math.log(r["seed"] / r["best"])
            cap.append(c); full += with_t <= 1.02 * r["best"]; half += c >= 0.5
    m = need.sum()
    return dict(cap=np.mean(cap), full=full / m, half=half / m, reg=np.mean(reg), reg95=np.percentile(reg, 95), size=np.mean(size))


for label, od in (("GROUPED leave-one-out (no neighbour shares two of M,N,K)", order), ("plain leave-one-out (reference)", order_plain)):
    print(f"\n== {label} ==")
    print("  k  per | pool | captured(mean) | closes fully | >=half | mean regret vs search winner | p95")
    base = evaluate(od, 0, 1)
    print(f"  heuristic seeds only                 {base['cap']*100:5.1f}%        {base['full']*100:4.0f}%       {base['half']*100:4.0f}%        {base['reg']*100:5.1f}%     {base['reg95']*100:5.1f}%")
    for k, per in itertools.product(KS, PS):
        e = evaluate(od, k, per)
        mark = "  <== shipped" if (k, per) == (5, 2) else ""
        print(f"  {k:2d}  {per}  | {e['size']:4.1f} |    {e['cap']*100:5.1f}%     |    {e['full']*100:4.0f}%     | {e['half']*100:4.0f}% |          {e['reg']*100:5.1f}%        | {e['reg95']*100:5.1f}%{mark}")

# value per candidate: marginal captured fraction per extra pool member on the grouped scan
print("\n== marginal gain per added candidate (grouped) along per=2 ==")
prev = evaluate(order, 0, 2)
for k in KS:
    e = evaluate(order, k, 2)
    print(f"  k={k:2d}: captured {e['cap']*100:5.1f}%  (+{(e['cap']-prev['cap'])*100:4.1f} pts, +{e['size']-prev['size']:.1f} candidates)")
    prev = e

# stability: pick the best (k, per) at pool size <= 12 on random halves, how often is it (5,2)-like
rng = np.random.default_rng(0)
print("\n== which (k, per) wins on random halves of the tune shapes (pool <= 12 candidates, grouped)? ==")
wins = defaultdict(int)
idx = np.arange(n)
cache = {}
def per_shape(order, k, per):
    out = []
    for i, r in enumerate(rows):
        seen, pool = set(), []
        for j in order[i][:k]:
            for cfg in tops[j][:per]:
                key = tuple(cfg.get(c) for c in kn.CFG_KEYS)
                if key not in seen:
                    seen.add(key); pool.append(key)
        known = [r["us"][p] for p in pool if p in r["us"]]
        with_t = min([r["seed"]] + known)
        c = math.log(r["seed"] / with_t) / math.log(r["seed"] / r["best"]) if need[i] else np.nan
        out.append((c, len(pool)))
    return np.array(out)
for k, per in itertools.product(KS, PS):
    cache[(k, per)] = per_shape(order, k, per)
for _ in range(200):
    h = rng.random(n) < 0.5
    best, bs = None, -1
    for key, v in cache.items():
        if np.nanmean(np.where(h, v[:, 1], np.nan)) > 12:
            continue
        sc = np.nanmean(np.where(h & need, v[:, 0], np.nan))
        if sc > bs:
            bs, best = sc, key
    wins[best] += 1
print("  ", dict(sorted(wins.items(), key=lambda kv: -kv[1])))
