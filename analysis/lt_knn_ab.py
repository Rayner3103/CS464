"""A/B of the joint online search with and without the cuBLASLt k-NN proposals, scored from ONE stage-2 session. CPU only.

`choice_lt["<k>_lt16"]` of the online file is the heuristic-only search and `choice["<k>"]` the full pool (heuristic top-16 + proposals), both
looked up at their full-protocol time in the same cold / graph stage-2 session (`cublaslt_knn:<n>` timed every proposal; a proposal's key
carries its rank among the kept proposals, which differs between the online run and stage 2, so it is matched on its config part).
Prints the proposer ceiling, regret vs the joint oracle, speed vs B3 / B3+, the paired geometric-mean ratio with a bootstrap interval
and the first-use cost.

    python -m analysis.lt_knn_ab [--results results_lock1755_knn8.jsonl --graph-results ..._graph.jsonl --online online_joint_diverse_v2_knn8.jsonl]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from analysis import ranker_compare as rc

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--data", type=Path, default=None)
ap.add_argument("--results", default="results_lock1755_knn8.jsonl")
ap.add_argument("--graph-results", default="results_lock1755_knn8_graph.jsonl")
ap.add_argument("--online", default="online_joint_diverse_v2_knn8.jsonl")
args = ap.parse_args()
if args.data is None:
    from tune.env import default_data_dir
    args.data = default_data_dir()
D = args.data
cold = rc.load_stage2(D, [args.results], "cold")
hot = rc.load_stage2(D, [args.graph_results], "hot")
ref = rc.references(cold, hot)
shapes = sorted(ref)
small = np.array([ref[s]["small"] for s in shapes]); eps = np.where(small, 1.05, 1.02)
B3, B3P, ORC = (np.array([ref[s][k] for s in shapes]) for k in ("b3", "b3p", "orc"))
KNN = np.array([min(ref[s]["src"]["ltk"].values(), default=np.inf) for s in shapes])
ORC_X = np.minimum(ORC, KNN)
print(f"n={len(shapes)} ({small.sum()} graph-amortized); rows with knn timed on {np.isfinite(KNN).sum()} shapes")

def st(x):
    return f"median {100*np.median(x):5.1f}%  p90 {100*np.percentile(x,90):5.1f}%  p95 {100*np.percentile(x,95):5.1f}%  mean {100*np.mean(x):5.1f}%"

print("\n== proposer ceiling (best of the kNN proposals, full protocol) ==")
print(" vs B3 (heuristic top-16): faster by >margin on", int((KNN < B3 / eps).sum()), "shapes; slower by >margin on", int((KNN > B3 * eps).sum()))
print(" best-of-kNN / B3+ : median %.3f p10 %.3f ; within margin of B3+ on %d, beats B3+ by >margin on %d" % (
    np.median(KNN / B3P), np.percentile(KNN / B3P, 10), (KNN <= B3P * eps).sum(), (KNN < B3P / eps).sum()))
gap = B3 > B3P * eps       # shapes where the deep search beats the heuristic
print(f" shapes where B3+ beats B3 by >margin: {gap.sum()}; of those the proposals close it (within margin of B3+): {(gap & (KNN <= B3P*eps)).sum()}, "
      f"get >=half way in log-time: {(gap & (np.log(np.minimum(B3,KNN)/B3) <= 0.5*np.log(B3P/B3))).sum()}")

def _knn_lookup(d, key):
    # a proposal's key starts with its rank among the kept proposals; the online search drops those the heuristic already
    # returned, so ranks differ between the online run and stage 2: match on the config part of the key
    suffix = key.split("_", 1)[1]
    return next((t for k, t in d.items() if k.split("_", 1)[1] == suffix), None)

def timeof(s, key):
    for src in (ref[s]["src"], cold[s]):
        for f in ("tri", "lth", "ltd"):
            if key in src[f]:
                return src[f][key]
        if re.match(r"n\d+_a", key):
            t = _knn_lookup(src["ltk"], key)
            if t is not None:
                return t
    raise KeyError((s, key))

res = {}
for rk, fname in (("v2", args.online),):
    rows = {(r["M"], r["N"], r["K"]): r for r in map(json.loads, open(D / fname))}
    for k in (1, 3, 5):
        for lab, getter in (("heuristic LT16", lambda r, k=k: r["choice_lt"][f"{k}_lt16"]), ("LT16 + kNN", lambda r, k=k: r["choice"][str(k)])):
            t, fam, cost = [], [], []
            for s in shapes:
                r = rows[s]; ch = getter(r)
                t.append(timeof(s, ch["key"])); fam.append(ch.get("family", "triton"))
                cost.append(ch["compile_s"] + ch["time_s"] + r["rank_s"])
            res[(k, lab)] = (np.array(t), fam, np.array(cost))

def report(k):
    a, fa, ca = res[(k, "heuristic LT16")]; b, fb, cb = res[(k, "LT16 + kNN")]
    print(f"\n== v2 top-{k} Triton + LT16: heuristic-only vs + kNN proposals ==")
    for lab, (t, fam, cost) in (("heuristic LT16", res[(k, "heuristic LT16")]), ("LT16 + kNN   ", res[(k, "LT16 + kNN")])):
        print(f" {lab}: regret vs joint oracle (Triton + B3+)   {st(t/ORC-1)}  >5%: {(t/ORC-1>0.05).sum()}  >10%: {(t/ORC-1>0.10).sum()}")
        print(f"   {'':14s}  vs B3+ ratio median {np.median(t/B3P):.3f} p10 {np.percentile(t/B3P,10):.3f}; faster/slower than B3+ by margin: "
              f"{(t<B3P/eps).sum()}/{(t>B3P*eps).sum()}; vs B3: {(t<B3/eps).sum()}/{(t>B3*eps).sum()}; first-use cost {cost.mean():.2f}s; "
              f"picks LT {100*np.mean(np.array(fam)!='triton'):.0f}% (knn {100*np.mean(np.array(fam)=='cublaslt_knn'):.0f}%)")
    r = np.log(b / a)
    bi = np.random.default_rng(0).integers(0, len(r), (4000, len(r)))
    gm = np.exp(np.array([r[i].mean() for i in bi]))
    print(f" paired (kNN / heuristic): geomean {np.exp(r.mean()):.4f}  95% [{np.percentile(gm,2.5):.4f}, {np.percentile(gm,97.5):.4f}]; "
          f"kNN faster/slower by margin {(b<a/eps).sum()}/{(b>a*eps).sum()}; extra cost {cb.mean()-ca.mean():+.2f}s")
    for lab, m in (("< 50 us", small), (">= 50 us", ~small)):
        print(f"   {lab:8s} n={m.sum():2d}: regret heuristic {st(a[m]/ORC[m]-1)}")
        print(f"   {'':8s}       regret + kNN    {st(b[m]/ORC[m]-1)}")
    # extended oracle: regret vs oracle that also contains the proposals (harder bar)
    print(f" regret vs EXTENDED oracle (incl. kNN proposals): heuristic {st(a/ORC_X-1)} | + kNN {st(b/ORC_X-1)}")
    return a, b

for k in (3, 5):
    a, b = report(k)
    if k == 3:
        a3, b3 = a, b

print("\n== shapes where kNN changed the pick by more than the margin (top-3) ==")
for i in np.argsort(-(a3 / b3))[:8]:
    print(f"  {shapes[i]!s:24s} {a3[i]:8.1f} -> {b3[i]:8.1f} us ({a3[i]/b3[i]:.2f}x), B3+/B3 {B3P[i]/B3[i]:.2f}")
print("  ...slower:")
for i in np.argsort(a3 / b3)[:4]:
    print(f"  {shapes[i]!s:24s} {a3[i]:8.1f} -> {b3[i]:8.1f} us ({a3[i]/b3[i]:.2f}x)")
