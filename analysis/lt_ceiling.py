"""Ceiling of a joint Triton + cuBLASLt selector: where does the regret of the shipped joint search come from?

For the 86 held-out shapes it takes the stage-2 session (`results_lock1755_v2*.jsonl`) and the v1 / v2 online joint files, resolves
each shape's joint-oracle winner into {heuristic top-16, Triton, a cuBLASLt config only the 20 s deep search finds}, and recomputes the
regret with the oracle-best config of one family added to the pool ("+ best cuBLASLt deep config", "+ perfect Triton"). Ceilings, not a
selector. CPU only. `python -m analysis.lt_ceiling` (reports/week1.md 2026-11-04).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analysis import ranker_compare as rc

D = Path("data/rtx3080_sm86_68sms_10gb")
cold = rc.load_stage2(D, ["results_lock1755_v2.jsonl"], "cold")
hot = rc.load_stage2(D, ["results_lock1755_v2_graph.jsonl"], "hot")
ref = rc.references(cold, hot)
shapes = sorted(ref)
small = np.array([ref[s]["small"] for s in shapes])
eps = np.where(small, 1.05, 1.02)
R = {k: np.array([ref[s][k] for s in shapes]) for k in ("b2", "b3", "b3p", "tri_orc", "orc")}


def pct(x, name):
    print(f"  {name:38s} median {100*np.median(x):5.1f}%  p90 {100*np.percentile(x,90):5.1f}%  p95 {100*np.percentile(x,95):5.1f}%  mean {100*np.mean(x):5.1f}%  >5%: {(x>0.05).sum():2d}  >10%: {(x>0.10).sum():2d}")


# oracle winner family (ties within eps resolve toward the cheaper-to-reach family: heuristic LT, then Triton, then deep)
fam = []
for i, s in enumerate(shapes):
    o = R["orc"][i]
    if R["b3"][i] <= o * eps[i]:
        fam.append("lt_heuristic")      # heuristic top-16 is within tie margin of the oracle
    elif R["tri_orc"][i] <= o * eps[i]:
        fam.append("triton")
    else:
        fam.append("lt_deep_only")     # only a config outside the heuristic's list gets within the margin
fam = np.array(fam)
print(f"n = {len(shapes)} held-out shapes ({small.sum()} timed graph-amortized)")
print("oracle winner (tie-margin resolved):", {f: int((fam == f).sum()) for f in ("lt_heuristic", "triton", "lt_deep_only")})
print("shapes where B3+ beats heuristic top-16 by > margin:", int((R["b3p"] < R["b3"] / eps).sum()),
      "| by > 10%:", int((R["b3p"] < R["b3"] / 1.10).sum()), "| by > 20%:", int((R["b3p"] < R["b3"] / 1.20).sum()))
print("shapes where B3+ also beats best Triton by > margin:", int((R["b3p"] < R["tri_orc"] / eps).sum()))

for rk, f in (("v1", "online_joint_diverse.jsonl"), ("v2", "online_joint_diverse_v2.jsonl")):
    ct = rc.chosen_times(D / f, ref, cold)
    for name in ("top-3 + LT16", "top-5 + LT16"):
        order = ct[name]["shapes"]
        idx = [shapes.index(s) for s in order]
        t = np.array(ct[name]["t"])
        o = R["orc"][idx]
        print(f"\n=== {rk} {name}: regret vs joint oracle, and counterfactual pools ===")
        pct(t / o - 1, "current")
        pct(np.minimum(t, R["b3p"][idx]) / o - 1, "+ best cuBLASLt deep config (ceiling)")
        pct(np.minimum(t, R["tri_orc"][idx]) / o - 1, "+ perfect Triton (ceiling)")
        pct(np.minimum(t, np.minimum(R["tri_orc"][idx], R["b3p"][idx])) / o - 1, "+ both (=0 by construction)")
        # attribute shapes with regret > margin
        bad = t > o * eps[idx]
        f_ = fam[idx]
        print(f"  shapes with regret > tie margin: {bad.sum()}; by oracle-winner family:",
              {k: int((bad & (f_ == k)).sum()) for k in ("lt_heuristic", "triton", "lt_deep_only")})
        # of these, fixed by deep config alone / by Triton alone
        fixed_deep = bad & (np.minimum(t, R["b3p"][idx]) <= o * eps[idx])
        fixed_tri = bad & (np.minimum(t, R["tri_orc"][idx]) <= o * eps[idx])
        print(f"  bad shapes fixed by adding the deep-LT winner: {fixed_deep.sum()}, by perfect Triton: {fixed_tri.sum()}, "
              f"by neither: {(bad & ~fixed_deep & ~fixed_tri).sum()}")
        if name == "top-3 + LT16":
            for lab, m in (("< 50us", small[idx]), (">= 50us", ~small[idx])):
                pct((np.minimum(t, R["b3p"][idx]) / o - 1)[m], f"  deep-LT ceiling, {lab}")
            w = np.where(bad & ~np.isin(f_, ["lt_heuristic"]))[0]
            if rk == "v2":
                print("  worst shapes (v2): M,N,K | us | regret | oracle family | B3 / B3+ / tri_orc ratio to oracle")
                for j in np.argsort(-(t / o))[:10]:
                    s = order[j]; i = idx[j]
                    print(f"   {s}  {t[j]:8.1f} us  {100*(t[j]/o[j]-1):5.1f}%  {f_[j]:13s} B3 {R['b3'][i]/R['orc'][i]:.2f} B3+ {R['b3p'][i]/R['orc'][i]:.2f} tri {R['tri_orc'][i]/R['orc'][i]:.2f}")
