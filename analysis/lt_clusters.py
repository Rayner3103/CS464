"""Structure of the cuBLASLt deep-search winners and how learnable they are from the shape (CPU only).

Reads `data/<hw>/lt_search/` (top-8 per shape of the 20 s enumerate-and-screen) and the suite splits; prints the winner components, the
regime breakdown, and a held-out check of static portfolios and k-NN proposals (pool hit in the shape's stored top-8, a lower bound).
It looked at held-out hit rates, so it is an analysis, not a fit: `analysis.lt_knn_scan` chooses (k, per-neighbour) on tune shapes only.
`python -m analysis.lt_clusters` (reports/week1.md 2026-11-04).
"""
from __future__ import annotations

import collections
import glob
import json
import math
from pathlib import Path

import numpy as np

D = Path("data/rtx3080_sm86_68sms_10gb")
split = {}
for f in glob.glob("suites/v1/*.json"):
    j = json.load(open(f))
    for s in (j["shapes"] if isinstance(j, dict) and "shapes" in j else j):
        split[(s["M"], s["N"], s["K"])] = s["split"]

rows = []
for f in glob.glob(str(D / "lt_search/*.json")):
    d = json.load(open(f))
    s = (d["M"], d["N"], d["K"])
    sp = split.get(s, "?")
    top = d["top"]
    cfgs = [tuple(t["cfg"][k] for k in ("algo_id", "tile_id", "stages_id", "splitk", "reduction", "swizzle", "custom")) for t in top]
    us = np.array([t["us"] for t in top])
    rows.append(dict(s=s, split=sp, cfgs=cfgs, us=us, seed=[t["seed"] for t in top], best_seed=d["best_seed_us"],
                     gain=d["best_seed_us"] / us[0], screened_all=d["screened_all"], nvalid=d["n_valid_configs"]))
print("lt_search entries:", len(rows), collections.Counter(r["split"] for r in rows))
print("top-8 spread (8th / best): median %.3f  p90 %.3f" % (np.median([r["us"][-1] / r["us"][0] for r in rows]), np.percentile([r["us"][-1] / r["us"][0] for r in rows], 90)))
print("screened all configs:", sum(r["screened_all"] for r in rows), "of", len(rows))

tune = [r for r in rows if r["split"] == "tune"]
held = [r for r in rows if r["split"].startswith("heldout")]
print(f"tune {len(tune)}  heldout {len(held)}")

# --- deep-needed shapes: the search winner beats the best heuristic-seeded config by > margin
def needy(r, m=1.05):
    return r["gain"] > m
for nm, g in (("tune", tune), ("heldout", held)):
    print(f"{nm}: winner beats best seed by >5%: {sum(needy(r) for r in g)}/{len(g)}, >10%: {sum(needy(r,1.10) for r in g)}, >20%: {sum(needy(r,1.2) for r in g)}; "
          f"winner is a seed: {sum(r['seed'][0] for r in g)}")

names = ("algo", "tile", "stages", "splitk", "reduct", "swizzle", "custom")
print("\n== winner components, all shapes with gain > 5% (deep-needed) ==")
dn = [r for r in rows if needy(r) and r["split"] != "aux"]
print("n =", len(dn))
for i, n in enumerate(names):
    c = collections.Counter(r["cfgs"][0][i] for r in dn)
    print(f"  {n:8s} distinct {len(c):3d}  top: {c.most_common(6)}")
sig = lambda c: (c[0], c[1], c[3] > 1)  # algo, tile, uses split-K
for nm, f in (("(algo,tile,splitK>1)", sig), ("(algo,tile,splitk)", lambda c: (c[0], c[1], c[3])), ("full cfg", lambda c: c)):
    c = collections.Counter(f(r["cfgs"][0]) for r in dn)
    print(f"  winner signature {nm}: {len(c)} distinct over {len(dn)} shapes; top5 cover {sum(v for _, v in c.most_common(5))}")

# --- regimes
def regime(s):
    M, N, K = s
    if min(M, N) <= 64 and K >= 1024: return "skinny+largeK (splitK region)"
    if min(M, N) <= 64: return "skinny"
    if M * N <= 256 * 256: return "small MN"
    if K >= 4096: return "largeK"
    return "regular"
print("\n== deep-needed share and winner features by regime (tune+heldout, non-aux) ==")
by = collections.defaultdict(list)
for r in rows:
    if r["split"] != "aux":
        by[regime(r["s"])].append(r)
for k, g in sorted(by.items(), key=lambda kv: -len(kv[1])):
    d5 = [r for r in g if needy(r)]
    sk = np.mean([r["cfgs"][0][3] > 1 for r in d5]) if d5 else float("nan")
    ca = collections.Counter(r["cfgs"][0][0] for r in d5).most_common(3)
    print(f"  {k:30s} n={len(g):3d} deep-needed {len(d5):3d} ({100*len(d5)/len(g):4.0f}%) median gain {np.median([r['gain'] for r in g]):.3f} "
          f"winner uses splitK>1: {100*sk:3.0f}%  algos {ca}")

# --- learnability: predict from tune shapes, score on held-out
def feat(s):
    M, N, K = s
    # shape features + 128x128 tile fill on 68 SMs
    ctas = math.ceil(M / 128) * math.ceil(N / 128)
    return np.array([math.log2(M), math.log2(N), math.log2(K), math.log2(ctas / 68 + 1e-9), M % 8 == 0 and N % 8 == 0 and K % 8 == 0])

Xt = np.array([feat(r["s"]) for r in tune]); sd = Xt.std(0) + 1e-9
def nn(r, k):
    d = np.sqrt((((Xt - feat(r["s"])) / sd) ** 2).sum(1))
    return np.argsort(d)[:k]

def rel_time(cfg, r):
    """time of `cfg` in r's top-8 relative to r's best; None if not in top-8"""
    return r["us"][r["cfgs"].index(cfg)] / r["us"][0] if cfg in r["cfgs"] else None

def eval_pool(pool_fn, grp, label):
    hit = []; ratio = []
    for r in grp:
        pool = pool_fn(r)
        ts = [t for t in (rel_time(c, r) for c in pool) if t is not None]
        hit.append(bool(ts)); ratio.append(min(ts) if ts else np.nan)
    hit = np.array(hit); ratio = np.array(ratio)
    ok2 = np.nanmean(ratio[hit] <= 1.02) if hit.any() else float("nan")
    print(f"  {label:44s} pool in top-8: {100*hit.mean():4.0f}%  (of those, within 2% of best: {100*ok2:3.0f}%)")
    return hit

print("\n== learnability: held-out shapes (non-trivial = deep-needed >5%) ==")
heldn = [r for r in held if needy(r)]
tunen = [r for r in tune if needy(r)]
print(f"held-out deep-needed shapes: {len(heldn)} of {len(held)};  tune deep-needed: {len(tunen)} of {len(tune)}")
freq = collections.Counter(r["cfgs"][0] for r in tunen)
for npool in (1, 3, 8, 16, 32):
    g = [c for c, _ in freq.most_common(npool)]
    eval_pool(lambda r, g=g: g, heldn, f"static portfolio of {npool} most frequent tune winners")
# portfolio from all top-8 membership (more generous)
mem = collections.Counter(c for r in tune for c in r["cfgs"][:3])
for npool in (8, 32):
    g = [c for c, _ in mem.most_common(npool)]
    eval_pool(lambda r, g=g: g, heldn, f"static: {npool} most common tune top-3 members")
for k in (1, 3, 5, 10):
    eval_pool(lambda r, k=k: [c for i in nn(r, k) for c in tune[i]["cfgs"][:2]], heldn, f"kNN k={k}: top-2 cfgs of neighbours")
# upper bound on what top-8 can show: random 8 from the same size pool is low; report pool sizes
print("  (a cfg may be valid and fast on a shape but absent from its stored top-8; hits are a lower bound)")

# winners that recur
cc = collections.Counter(r["cfgs"][0] for r in dn)
print("\n== most frequent deep-needed winners (cfg = algo,tile,stages,splitk,reduction,swizzle,custom) ==")
for c, v in cc.most_common(8):
    ex = [r["s"] for r in dn if r["cfgs"][0] == c][:3]
    print(f"  {v:3d}x {c}  e.g. {ex}")

print("\n== the 20 held-out deep-only shapes: winner cfg, gain, nearest-tune-neighbour's winner agrees on (algo,tile,splitK>1)? ==")
agree = 0
for r in sorted(heldn, key=lambda r: -r["gain"])[:24]:
    nbr = tune[nn(r, 1)[0]]
    a = sig(nbr["cfgs"][0]) == sig(r["cfgs"][0])
    agree += a
    print(f"  {str(r['s']):22s} gain {r['gain']:.2f} win {r['cfgs'][0]} nn {nbr['s']} {nbr['cfgs'][0][:2]}+sk{nbr['cfgs'][0][3]} {'AGREE' if a else ''}")
print("agree:", agree, "of", min(24, len(heldn)))
