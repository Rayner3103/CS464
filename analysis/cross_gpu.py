"""Cross-GPU comparison of selectors on the held-out shapes two (or more) runs have in common. CPU only, reads existing results.

Why this exists: the held-out sets of different runs differ in size (the 3080 has 86 shapes, a `pipeline/run.sh --quick` run
27 of those), and a regret table over 86 shapes is not comparable to one over 27: the regime mix (shapes under / over 50 us)
changes with the GPU and the subset. So every side is restricted to the shapes ALL sides have, and each is scored with the
repo's own functions: `analysis.zero_shot.score_rows` (regret vs that GPU's own held-out screen oracle) and
`analysis.ranker_compare.load_stage2` / `references` (stage 2 vs the searched cuBLASLt B3+).

A side is `LABEL DIR SCREEN_ONLINE STAGE2_ONLINE RESULTS GRAPH` (files relative to DIR, a `data/<hw>` directory):
  SCREEN_ONLINE  `tune online` rows scored against DIR/sweep_screen_heldout (Triton-only choice; a joint row's "<k>_lt0")
  STAGE2_ONLINE  rows whose picks were timed in RESULTS (cold) / GRAPH (graph-amortized, shapes < 50 us), the same stage-2
                 session as B2 / B3 / B3+; the same file as SCREEN_ONLINE when one run did both
The first side is the reference of the paired bootstrap.

    python -m analysis.cross_gpu \\
      --side 3080-v1 data/rtx3080_sm86_68sms_10gb online_lgbm_rank_diverse.jsonl online_joint_diverse.jsonl results_lock1755_v2.jsonl results_lock1755_v2_graph.jsonl \\
      --side 3080-v2 data/rtx3080_sm86_68sms_10gb online_joint_diverse_v2.jsonl online_joint_diverse_v2.jsonl results_lock1755_v2.jsonl results_lock1755_v2_graph.jsonl \\
      --side 2060 data/rtx2060/<run>/data online_lgbm_rank_diverse.jsonl online_lgbm_rank_diverse.jsonl results_validate.jsonl results_validate_graph.jsonl

Speed vs B3+ is t_B3+ / t_pick (> 1 = faster than the searched cuBLASLt), the convention of reports/learned_topk.md (analysis.ranker_compare
uses the inverse). n is small by construction: read intervals, not point differences. Caveat of the `compiled kernels` line: it only counts
mma.sync in PTX files found under DIR/online_triton_cache.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

import numpy as np

from analysis import ranker_compare as rc
from analysis import zero_shot as zs

EPS = rc.EPS   # {"small": 1.05, "large": 1.02}: tie margins below / above 50 us


def heldout_shapes(data: Path) -> set:
    return {(s["M"], s["N"], s["K"]) for s in json.loads((data / "sweep_screen_heldout" / "subset.json").read_text())["shapes"]}


def load_rows(path: Path, only: set | None = None, min_k: int = 0) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return [r for r in rows if r.get("candidates") and r.get("k", 0) >= min_k
            and (only is None or (r["M"], r["N"], r["K"]) in only)]


def chosen(row: dict, k: int) -> dict:
    """The row's Triton-only choice for k candidates (a joint row stores it as "<k>_lt0")."""
    return (row.get("choice_lt") or {}).get(f"{k}_lt0") or row["choice"][str(k)]


def regret_by_shape(screen: dict, rows: list[dict], k: int) -> dict:
    """{shape: regret vs the screen oracle of the online choice at k}; shapes whose pick is not in the screen are dropped."""
    out = {}
    for r in rows:
        s = (r["M"], r["N"], r["K"])
        if s in screen and not r.get("skipped"):
            v = zs.regret(screen[s], chosen(r, k)["key"])
            if v is not None:
                out[s] = v
    return out


def screen_table(screen: dict, rows: list[dict], ks: tuple) -> dict:
    """{pick: {group: summarize()}} for top-1 and online-k, groups by the screen oracle (all / >= 50 us / < 50 us)."""
    sc = zs.score_rows(screen, rows, max(ks))
    oracle = np.array(sc.pop("_oracle_us"))
    sc.pop("_skipped", None)
    sc.pop("invalid_picks", None)
    masks = {"all": None, ">= 50 us": oracle >= zs.SMALL_US, "< 50 us": oracle < zs.SMALL_US}
    return {p: {g: zs.summarize(sc[p], m) for g, m in masks.items()} for p in ["top-1"] + [f"online-{k}" for k in ks]}


def paired_bootstrap(x, y, stat=np.mean, n: int = 5000, seed: int = 0) -> dict:
    """stat(y) - stat(x) over shapes paired by position, with a 95% bootstrap interval (resampling shapes)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    idx = np.random.default_rng(seed).integers(0, len(x), (n, len(x)))
    d = np.array([stat(y[i]) - stat(x[i]) for i in idx])
    return {"x": float(stat(x)), "y": float(stat(y)), "diff": float(stat(y) - stat(x)),
            "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5))}


def speed_vs_b3p(ref: dict, times: dict, shapes) -> dict:
    """Speed vs B3+ (t_B3+ / t_pick) per group: all / >= 50 us (cold) / < 50 us (graph), with shapes ahead / behind by more than
    the tie margin."""
    sh = [s for s in shapes if s in ref and times.get(s)]
    small = np.array([ref[s]["small"] for s in sh], bool)
    sp = np.array([ref[s]["b3p"] / times[s] for s in sh])
    eps = np.where(small, EPS["small"], EPS["large"])
    out = {}
    for g, m in (("all", np.ones(len(sh), bool)), (">= 50 us", ~small), ("< 50 us", small)):
        q = sp[m]
        out[g] = ({"n": 0} if not len(q) else
                  {"n": int(len(q)), "median": float(np.median(q)), "p10": float(np.percentile(q, 10)), "p90": float(np.percentile(q, 90)),
                   "ahead": int((q > eps[m]).sum()), "behind": int((q < 1 / eps[m]).sum())})
    return out


def family_ceiling(ref: dict, shapes) -> dict:
    """Best Triton row of the session vs B3+ (G2 / G4 on these shapes): t_B3+ / t_Triton-oracle."""
    sh = [s for s in shapes if s in ref]
    small = np.array([ref[s]["small"] for s in sh], bool)
    q = np.array([ref[s]["b3p"] / ref[s]["tri_orc"] for s in sh])
    eps = np.where(small, EPS["small"], EPS["large"])
    return {"n": len(sh), "median": float(np.median(q)), "p10": float(np.percentile(q, 10)), "p90": float(np.percentile(q, 90)),
            "ge_90pct": int((q >= 0.9).sum()), "ahead": int((q > eps).sum()), "behind": int((q < 1 / eps).sum()),
            "median_ge50": float(np.median(q[~small])) if (~small).any() else None,
            "median_lt50": float(np.median(q[small])) if small.any() else None}


def stage2_times(rows: list[dict], ref: dict, cold: dict, k: int, joint: bool = False) -> dict:
    """{shape: full-protocol time of the row's choice at k} (graph time for shapes timed graph-amortized). Triton-only choice by
    default; joint=True takes the joint choice (cuBLASLt top-N in the pool), whose cuBLASLt keys are deterministic and so are
    found in any stage-2 session that ran `cublaslt:16`. A pick that session did not time is absent: report the coverage."""
    out = {}
    for r in rows:
        s = (r["M"], r["N"], r["K"])
        if s not in ref:
            continue
        key = (r["choice"][str(k)] if joint else chosen(r, k))["key"]
        src = ref[s]["src"]
        t = next((src[f][key] for f in ("tri", "lth", "ltd") if key in src[f]), None)
        if t is None:  # not timed in this mode: the cold session
            t = next((cold[s][f][key] for f in ("tri", "lth", "ltd") if key in cold[s][f]), None)
        if t is not None:
            out[s] = t
    return out


def ptx_mma_share(data: Path) -> dict | None:
    """Compiled Triton kernels under DIR/online_triton_cache: how many have mma.sync / only fma.rn.f32, and the PTX targets."""
    files = sorted((data / "online_triton_cache").glob("*/*/_matmul_kernel.ptx"))
    if not files:
        return None
    c = collections.Counter()
    for p in files:
        t = p.read_text()
        c["n"] += 1
        has = "mma.sync" in t
        c["mma"] += has
        c["fma_only"] += (not has) and "fma.rn.f32" in t
        m = re.search(r"\.target (\w+)", t)
        c[m.group(1) if m else "?"] += 1
    return dict(c)


def joint_summary(rows: list[dict], ref: dict, cold: dict, k: int, shapes) -> dict | None:
    """Joint-search outcome for rows searched with `--lt-n`: share of shapes where cuBLASLt won, coverage of the stage-2
    session and speed vs B3+ (t_B3+ / t_pick) on the covered shapes."""
    rows = [r for r in rows if r.get("lt_n") and str(k) in r["choice"]]
    if not rows:
        return None
    t = stage2_times(rows, ref, cold, k, joint=True)
    sp = np.array([ref[s]["b3p"] / v for s, v in t.items() if s in shapes])
    fam = collections.Counter(r["choice"][str(k)].get("family", "triton") for r in rows)
    return {"n": len(rows), "cublaslt": int(fam.get("cublaslt", 0)), "covered": int(len(sp)),
            **({"median": float(np.median(sp)), "p10": float(np.percentile(sp, 10)), "within_5pct": int((sp >= 0.95).sum())} if len(sp) else {})}


def tile_diversity(rows: list[dict], k: int) -> tuple[int, list]:
    c = collections.Counter(chosen(r, k)["key"].split("_w")[0] for r in rows)
    return len(c), c.most_common(3)


def compare(sides: list[dict], k: int = 3, k_max: int = 5) -> dict:
    """sides: [{label, dir, screen_online, stage2_online, results, graph}] -> everything the report prints."""
    common = set.intersection(*(heldout_shapes(Path(s["dir"])) for s in sides))
    res = {"k": k, "k_max": k_max, "n_common": len(common), "sides": {}}
    paired = {}
    s2_cache = {}
    for s in sides:
        d = Path(s["dir"])
        screen = {sh: t for sh, t in zs.load_screen(d / "sweep_screen_heldout").items() if sh in common}
        rows = load_rows(d / s["screen_online"], common, k_max)
        r2 = load_rows(d / s["stage2_online"], common, k)
        key = (str(d), s["results"], s["graph"])
        if key not in s2_cache:
            cold, hot = rc.load_stage2(d, [s["results"]], "cold"), rc.load_stage2(d, [s["graph"]], "hot")
            s2_cache[key] = (cold, rc.references(cold, hot))
        cold, ref = s2_cache[key]
        shapes = sorted(common)
        paired[s["label"]] = regret_by_shape(screen, rows, k)
        n_t, top = tile_diversity(r2, k)
        res["sides"][s["label"]] = {
            "n_screen": len(screen), "regret": screen_table(screen, rows, (k, k_max)),
            "speed": speed_vs_b3p(ref, stage2_times(r2, ref, cold, k), shapes), "ceiling": family_ceiling(ref, shapes),
            "tiles": {"distinct": n_t, "top": top}, "dir": str(d), "ptx": ptx_mma_share(d),
            "joint": joint_summary(r2, ref, cold, k, set(shapes))}
    first = sides[0]["label"]
    res["bootstrap"] = {}
    for s in sides[1:]:
        both = sorted(set(paired[first]) & set(paired[s["label"]]))
        x, y = [paired[first][t] for t in both], [paired[s["label"]][t] for t in both]
        res["bootstrap"][s["label"]] = {"n": len(both), "vs": first,
                                        **{n: paired_bootstrap(x, y, f) for n, f in
                                           (("mean", np.mean), ("median", np.median), ("p90", lambda v: np.percentile(v, 90)))},
                                        "over_5pct": [int((np.array(x) > .05).sum()), int((np.array(y) > .05).sum())]}
    return res


def render(res: dict) -> str:
    k, km = res["k"], res["k_max"]
    labels = list(res["sides"])
    pct = lambda s: zs.fmt(s)
    lines = [f"# Cross-GPU comparison on the {res['n_common']} shared held-out shapes (auto-generated by `analysis.cross_gpu`)", "",
             "Regret vs each GPU's own held-out screen oracle (median / p95 / p99 (mean)); speed = t_B3+ / t_pick (> 1 = faster than the searched cuBLASLt).", "",
             "## Selection quality (screen oracle)", "", "| pick | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for p in ("top-1", f"online-{k}", f"online-{km}"):
        lines.append(f"| {p} | " + " | ".join(pct(res["sides"][l]["regret"][p]["all"]) for l in labels) + " |")
    lines.append(f"| regret > 50% (online-{k}) | " + " | ".join(
        f"{res['sides'][l]['regret'][f'online-{k}']['all'].get('over_50', 0)} of {res['sides'][l]['regret'][f'online-{k}']['all'].get('n', 0)}"
        f" (+{res['sides'][l]['regret'][f'online-{k}']['all'].get('unscored', 0)} unscored)" for l in labels) + " |")
    lines += ["", "Regime split by the screen oracle (n shapes >= 50 us / < 50 us): " + "; ".join(
        f"{l} {res['sides'][l]['regret'][f'online-{k}']['>= 50 us'].get('n', 0)} / {res['sides'][l]['regret'][f'online-{k}']['< 50 us'].get('n', 0)}" for l in labels), ""]
    for l, b in res["bootstrap"].items():
        lines.append(f"Paired bootstrap, online-{k}, {l} minus {b['vs']} (n = {b['n']}, 5,000 resamples): " + ", ".join(
            f"{n} {100 * b[n]['diff']:+.1f} points [{100 * b[n]['lo']:+.1f}, {100 * b[n]['hi']:+.1f}]" for n in ("mean", "median", "p90"))
            + f"; shapes over 5% regret {b['over_5pct'][1]} vs {b['over_5pct'][0]}.")
    lines += ["", f"## Stage 2: online top-{k} (Triton only) vs B3+", "", "| group | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for g in ("all", ">= 50 us", "< 50 us"):
        cell = lambda e: "n/a" if not e.get("n") else f"{e['median']:.3f} [{e['p10']:.3f}, {e['p90']:.3f}] (n={e['n']}; ahead {e['ahead']}, behind {e['behind']})"
        lines.append(f"| {g} | " + " | ".join(cell(res["sides"][l]["speed"][g]) for l in labels) + " |")
    if any(res["sides"][l]["speed"]["all"].get("n", 0) < res["n_common"] for l in labels):
        lines += ["", "n below the shared-shape count: the Triton picks of that side were not timed in its stage-2 session, so its Triton-only speed covers only a few shapes "
                      "and is not comparable (the joint table below is). Run `python -m analysis.zero_shot picks` and its stage-2 commands for a full comparison."]
    lines += ["", "## Family ceiling: best Triton row of the session vs B3+", "",
              "| | n | median [p10, p90] | >= 90% of B3+ | ahead / behind by > margin | median >= 50 us | median < 50 us |", "|---|---|---|---|---|---|---|"]
    for l in labels:
        c = res["sides"][l]["ceiling"]
        f = lambda v: "n/a" if v is None else f"{v:.3f}"
        lines.append(f"| {l} | {c['n']} | {c['median']:.3f} [{c['p10']:.3f}, {c['p90']:.3f}] | {c['ge_90pct']}/{c['n']} | {c['ahead']} / {c['behind']} | {f(c['median_ge50'])} | {f(c['median_lt50'])} |")
    if any(res["sides"][l]["joint"] for l in labels):
        lines += ["", f"## Joint search (top-{k} Triton + cuBLASLt top-16 in one pool) vs B3+", "",
                  "| | shapes | picks cuBLASLt | timed in the stage-2 session | speed vs B3+ median / p10 | within 5% of B3+ |", "|---|---|---|---|---|---|"]
        for l in labels:
            j = res["sides"][l]["joint"]
            if j:
                lines.append(f"| {l} | {j['n']} | {j['cublaslt']} | {j['covered']} | " + (
                    f"{j['median']:.3f} / {j['p10']:.3f} | {j['within_5pct']}/{j['covered']} |" if j["covered"] else "n/a | n/a |"))
        lines.append("\nA Triton pick that the stage-2 session did not time has no full-protocol time and is left out of the speed columns (see `timed in the stage-2 session`).")
    lines += ["", "## Choices and compiled kernels", ""]
    for l in labels:
        tl = res["sides"][l]["tiles"]
        lines.append(f"- {l}: {tl['distinct']} distinct tiles over the shapes (top {tl['top']}).")
    seen = {}
    for l in labels:
        seen.setdefault(res["sides"][l]["dir"], (l, res["sides"][l]["ptx"]))
    for d, (l, p) in seen.items():
        lines.append(f"- `{d}/online_triton_cache` (every run of that directory): " + (
            "no compiled-kernel cache found." if not p else f"{p.get('n', 0)} compiled kernels, {p.get('mma', 0)} with mma.sync, {p.get('fma_only', 0)} FMA-only "
            f"(targets: {', '.join(f'{a} {b}' for a, b in p.items() if a.startswith('sm_'))})."))
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="analysis.cross_gpu", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", nargs=6, action="append", required=True,
                    metavar=("LABEL", "DIR", "SCREEN_ONLINE", "STAGE2_ONLINE", "RESULTS", "GRAPH"))
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--k-max", type=int, default=5)
    ap.add_argument("--out", type=Path, default=Path("reports/cross_gpu.md"))
    a = ap.parse_args(argv)
    sides = [dict(zip(("label", "dir", "screen_online", "stage2_online", "results", "graph"), s)) for s in a.side]
    res = compare(sides, a.k, a.k_max)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(render(res))
    a.out.with_name(a.out.stem + "_summary.json").write_text(json.dumps(res, indent=1, default=str))
    print(render(res))
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
