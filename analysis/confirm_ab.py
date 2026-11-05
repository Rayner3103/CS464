"""Score router / confirmation online runs against the Triton-first reference, from ONE stage-2 session. CPU only.

Every run short-times the same candidate pool (v2 diverse top-3 + heuristic top-16 + learned-ranker top-5) and differs only in the ORDER the
two halves are timed, whether the Triton half may be skipped (`--router`) and whether the top candidates are re-timed in reverse order
(`--confirm`). Each run's k = 3 choice is looked up at its full-protocol time in the stage-2 session that timed every candidate
(`results_lock1755_rank16*.jsonl`), so differences between runs are differences of choice, not of measurement. The reference is the earlier
Triton-first run's own k = 3 choice from the same pool (ranker top-5, no k-NN proposals).

Per run: regret vs the joint oracle (median / p95 / p99 / mean), shapes > 5% / > 10% off, shapes slower than B3+ by the tie margin, first-use cost
(no bundle: true compile), shapes the router skipped; then paired comparisons (geometric-mean time ratio with a 95% bootstrap interval and the
shapes worse / better by the tie margin) against the reference and between the named pairs.

    python -m analysis.confirm_ab [--run LABEL=FILE ...] [--reference FILE]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis import ranker_compare as rc
from analysis.lt_rank_ab import pool_of, timeof
from tune import online

RUNS = [
    ("router, no confirm", "online_joint_diverse_v2_rank5_router.jsonl"),
    ("LT first, never skip, no confirm", "online_joint_diverse_v2_rank5_router0.jsonl"),
    ("A: Triton first, confirm 3", "online_joint_diverse_v2_rank5_confirm3.jsonl"),
    ("B: LT first, never skip, confirm 3", "online_joint_diverse_v2_rank5_router0_confirm3.jsonl"),
    ("C: router, confirm 3", "online_joint_diverse_v2_rank5_router_confirm3.jsonl"),
]
REFERENCE = "online_joint_diverse_v2_knn8_rank16.jsonl"


def load(path: Path) -> dict:
    return {(r["M"], r["N"], r["K"]): r for r in map(json.loads, open(path))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--run", action="append", default=None, metavar="LABEL=FILE")
    ap.add_argument("--reference", default=REFERENCE)
    ap.add_argument("--results", default="results_lock1755_rank16.jsonl")
    ap.add_argument("--graph-results", default="results_lock1755_rank16_graph.jsonl")
    a = ap.parse_args(argv)
    if a.data is None:
        from tune.env import default_data_dir
        a.data = default_data_dir()
    d = a.data
    runs = [tuple(x.split("=", 1)) for x in a.run] if a.run else RUNS
    cold, hot = rc.load_stage2(d, [a.results], "cold"), rc.load_stage2(d, [a.graph_results], "hot")
    ref = rc.references(cold, hot)
    shapes = sorted(ref)
    small = np.array([ref[s]["small"] for s in shapes]); eps = np.where(small, 1.05, 1.02)
    ORC, B3P = np.array([ref[s]["orc_x"] for s in shapes]), np.array([ref[s]["b3p"] for s in shapes])
    base = load(d / a.reference)

    T, COST, SKIP = {}, {}, {}
    chs = [online.choose(pool_of(base[s]["candidates"], 0, 5), 3, None) for s in shapes]
    T["reference: Triton first, no router, no confirm"] = np.array([timeof(s, c["key"], ref, cold) for s, c in zip(shapes, chs)])
    COST["reference: Triton first, no router, no confirm"] = np.mean([c["compile_s"] + c["time_s"] + base[s]["rank_s"] for s, c in zip(shapes, chs)])
    SKIP["reference: Triton first, no router, no confirm"] = 0
    for lab, f in runs:
        rows = load(d / f)
        T[lab] = np.array([timeof(s, rows[s]["choice"]["3"]["key"], ref, cold) for s in shapes])
        COST[lab] = np.mean([rows[s]["choice"]["3"]["compile_s"] + rows[s]["choice"]["3"]["time_s"] + rows[s]["rank_s"] for s in shapes])
        SKIP[lab] = sum(bool((rows[s].get("router") or {}).get("skipped")) for s in shapes)
    labs = list(T)
    print(f"n = {len(shapes)} held-out shapes ({small.sum()} graph-amortized); picks looked up in {a.results} / {a.graph_results}")
    print(f"{'run':48s} {'regret med / p95 / p99 / mean':>32s} | >5% >10% | slow vs B3+ | cost (no bundle) | skipped")
    for lab in labs:
        g = T[lab] / ORC - 1
        print(f"{lab:48s} {100 * np.median(g):6.1f}% / {100 * np.percentile(g, 95):5.1f}% / {100 * np.percentile(g, 99):5.1f}% / {100 * g.mean():4.1f}% | {(g > .05).sum():3d} {(g > .10).sum():3d} | "
              f"{(T[lab] > B3P * eps).sum():5d}      | {COST[lab]:.3f} s ({100 * (COST[lab] / COST[labs[0]] - 1):+4.0f}%)   | {SKIP[lab]:3d}")

    rng = np.random.default_rng(0)

    def paired(x, y):
        r = np.log(x / y)
        bi = rng.integers(0, len(r), (4000, len(r)))
        gm = np.exp(np.array([r[i].mean() for i in bi]))
        return np.exp(r.mean()), np.percentile(gm, 2.5), np.percentile(gm, 97.5), int((x > y * eps).sum()), int((x < y / eps).sum())

    print("\npaired, run vs reference (geometric-mean time ratio [95% bootstrap], shapes worse / better by the tie margin)")
    for lab in labs[1:]:
        gm, lo, hi, w, b = paired(T[lab], T[labs[0]])
        print(f"  {lab:48s} {gm:.4f} [{lo:.4f}, {hi:.4f}]   {w:2d} / {b:2d}")
    byname = {l.split(":")[0]: l for l in labs if ":" in l[:3]}
    if {"A", "B", "C"} <= set(byname):
        print("\npaired between the confirmation runs (same pools; B vs A isolates the timing order, C vs B the skip decisions)")
        for x, y in (("B", "A"), ("C", "B"), ("C", "A")):
            gm, lo, hi, w, b = paired(T[byname[x]], T[byname[y]])
            print(f"  {x} vs {y}: {gm:.4f} [{lo:.4f}, {hi:.4f}]   {x} worse on {w}, better on {b}")
    if "C" in byname:
        c = load(d / dict(runs)[byname["C"]])
        sk = np.array([bool((c[s].get("router") or {}).get("skipped")) for s in shapes])
        p = np.array([(c[s].get("router") or {}).get("p_win") or np.nan for s in shapes])
        t_c, t_a = T[byname["C"]], T[byname["A"]]
        print(f"\nrouter run C: skipped {sk.sum()} shapes (median P(win) {np.nanmedian(p[sk]):.3f}; not skipped {np.nanmedian(p[~sk]):.3f}); "
              f"on the skipped shapes C / A time ratio: geomean {np.exp(np.log(t_c[sk] / t_a[sk]).mean()):.4f}, worst {np.max(t_c[sk] / t_a[sk]):.3f}, "
              f"best {np.min(t_c[sk] / t_a[sk]):.3f}")
        for i in np.where(sk & (t_c > t_a * eps))[0]:
            print(f"    skipped and slower: {shapes[i]} {t_a[i]:.1f} -> {t_c[i]:.1f} us (P(win) {p[i]:.3f})")
        print(f"  not skipped: identical pick time to A on {int((~sk & (np.abs(t_c / t_a - 1) < 1e-9)).sum())} of {int((~sk).sum())}")


if __name__ == "__main__":
    main()
