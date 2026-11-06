"""Score rankers' joint online searches (Triton diverse top-k + cuBLASLt heuristic top-N, `tune online --lt-n`) side by side.

Every chosen config is looked up at its full-protocol time in ONE stage-2 session that timed the union of the rankers'
picks together with the baselines (torch, `triton:sweep:8`, `cublaslt:16`, `cublaslt_deep:20`), so the comparison is not
confounded by session noise: cold times for shapes whose fastest candidate is >= 50 us, CUDA-graph amortized times
below. References: B2 = cuBLASLt heuristic rank 0, B3 = best heuristic candidate, B3+ = best of B3 and the 20 s search;
the joint oracle = best of every Triton config timed in that session, B3+ and any k-NN / ranker proposal the session timed (`orc_x`). Ties are within EPS (5% below 50 us, 2% above).

    python -m analysis.ranker_compare --data data/<hw> --online v1=online_joint_diverse.jsonl v2=online_joint_diverse_v2.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

SMALL_US = 50.0
EPS = {"small": 1.05, "large": 1.02}
CONFIGS = (("top-3 Triton only", 3, 0), ("top-3 + LT16", 3, 16), ("top-5 + LT16", 5, 16), ("top-1 + LT16", 1, 16))
# Families a joint choice can come from: a search run with the learned cuBLASLt ranker / k-NN proposals (`lt_rank`, `lt_knn`) may pick one,
# so stage 2 must have timed them too (`--backends ...,cublaslt_rank:M`); without them in the lookup such a pick is a StopIteration.
PICK_FAMILIES = ("tri", "lth", "ltd", "ltk", "ltr")
FAMILY = {"triton_nn_v1": "tri", "cublaslt": "lth", "cublaslt_deep": "ltd", "cublaslt_knn": "ltk", "cublaslt_rank": "ltr", "torch_matmul": "torch"}


def load_stage2(data: Path, files: list[str], mode: str) -> dict:
    """{shape: {family: {config_key: median_us}}} for the valid rows of `files` with this cache mode."""
    d = collections.defaultdict(lambda: collections.defaultdict(dict))
    for f in files:
        for line in (Path(data) / f).read_text().splitlines():
            r = json.loads(line)
            if r["cache_mode"] == mode and r["valid"]:
                d[(r["M"], r["N"], r["K"])][FAMILY.get(r["kernel_family"], r["kernel_family"])][r["config_key"]] = r["time_median_us"]
    return d


def references(cold: dict, hot: dict) -> dict:
    """Per shape: whether it is timed graph-amortized, B2, B3, B3+, the Triton oracle and the joint oracle (same mode)."""
    out = {}
    for s, v in cold.items():
        small = min(min(v["tri"].values()), min(v["lth"].values())) < SMALL_US and s in hot and bool(hot[s]["tri"])
        src = hot[s] if small else v
        lth = min(src["lth"].values())
        ltd = min(src["ltd"].values(), default=lth)  # no cublaslt_deep rows (e.g. a new GPU): B3+ = B3
        out[s] = {"small": small, "b2": next(t for k, t in src["lth"].items() if k.startswith("h0_")), "b3": lth,
                  "b3p": min(lth, ltd), "tri_orc": min(src["tri"].values()), "src": src}
        out[s]["orc"] = min(out[s]["tri_orc"], out[s]["b3p"])
        # extended oracle: also every k-NN / learned-ranker proposal the session timed (empty in sessions that did not time them).
        # A pool that picks a proposal is otherwise scored against an oracle that cannot see it (negative regret).
        out[s]["orc_x"] = min(out[s]["orc"], min(src["ltk"].values(), default=np.inf), min(src["ltr"].values(), default=np.inf))
    return out


def chosen_times(online: Path, ref: dict, cold: dict) -> dict:
    """{config name: {"t": [us per shape], "cost": [first-use s], "family": [...], "shapes": [...]}} for one online file."""
    rows = [json.loads(line) for line in Path(online).read_text().splitlines()]
    rows = [r for r in rows if r.get("candidates")]
    # "+ LT16" = every cuBLASLt candidate of the row's search: only a joint run (lt_n > 0) has any
    has = lambda r, k, lt: (str(k) in r["choice"] and r.get("lt_n", 0) > 0) if lt == 16 else f"{k}_lt{lt}" in r.get("choice_lt", {})
    configs = [c for c in CONFIGS if all(has(r, c[1], c[2]) for r in rows)]  # a k = 3 run has no "top-5" selectors
    if not configs:
        raise ValueError(f"{online}: no joint selectors to score; run the search with --lt-n 16 (choice_lt missing)")
    out = {n: {"t": [], "cost": [], "family": [], "shapes": []} for n, _, _ in configs}
    for r in rows:
        s = (r["M"], r["N"], r["K"])
        for name, k, lt in configs:
            ch = r["choice"][str(k)] if lt == 16 else r["choice_lt"][f"{k}_lt{lt}"]
            src = ref[s]["src"]
            t = next((src[f][ch["key"]] for f in PICK_FAMILIES if ch["key"] in src[f]), None)
            if t is None:  # not timed in this mode (e.g. a pick only timed cold): fall back to the cold session
                t = next(cold[s][f][ch["key"]] for f in PICK_FAMILIES if ch["key"] in cold[s][f])
            out[name]["t"].append(t)
            out[name]["cost"].append(ch["compile_s"] + ch["time_s"] + r["rank_s"])
            out[name]["family"].append(ch.get("family", "triton"))
            out[name]["shapes"].append(s)
    return out


def stats(x: np.ndarray) -> dict:
    if not len(x):  # an empty group (e.g. no shape under 50 us on this GPU)
        return {k: float("nan") for k in ("median", "p10", "p90", "p95", "p99", "mean")}
    return {"median": float(np.median(x)), "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90)),
            "p95": float(np.percentile(x, 95)), "p99": float(np.percentile(x, 99)), "mean": float(np.mean(x))}


def score(online: Path, ref: dict, cold: dict) -> dict:
    """Per config and group (all / < 50 us / >= 50 us): cost, ratios to B2 / B3 / B3+, win-loss counts, regrets."""
    ct = chosen_times(online, ref, cold)
    names_ = list(ct)
    shapes = ct[names_[0]]["shapes"]
    small = np.array([ref[s]["small"] for s in shapes])
    R = {k: np.array([ref[s][k] for s in shapes]) for k in ("b2", "b3", "b3p", "orc", "orc_x", "tri_orc")}
    groups = {"all": (np.ones(len(small), bool), EPS["small"]), "< 50 us": (small, EPS["small"]), ">= 50 us": (~small, EPS["large"])}
    out = {}
    for name in names_:
        t = np.array(ct[name]["t"])
        out[name] = {}
        for g, (m, eps) in groups.items():
            e = {"n": int(m.sum()), "eps": eps, "cost_s": float(np.mean(np.array(ct[name]["cost"])[m])),
                 "pct_lt": float(100 * np.mean(np.array(ct[name]["family"])[m] == "cublaslt"))}
            for ref_name in ("b2", "b3", "b3p"):
                q = (t / R[ref_name])[m]
                e[ref_name] = {**stats(q), "faster": int((q < 1 / eps).sum()), "slower": int((q > eps).sum())}
            e["regret_joint"] = stats(t[m] / R["orc_x"][m] - 1)
            e["regret_triton"] = stats(t[m] / R["tri_orc"][m] - 1)
            out[name][g] = e
    return {"scores": out, "times": {n: ct[n]["t"] for n in names_}, "shapes": shapes, "small": small.tolist()}


def paired(a: dict, b: dict, name: str, seed: int = 0) -> dict:
    """b vs a (same shapes): geometric-mean time ratio with a bootstrap interval and shape counts (> EPS either way)."""
    ta, tb = np.array(a["times"][name]), np.array(b["times"][name])
    small = np.array(a["small"])
    r = np.log(tb / ta)
    bi = np.random.default_rng(seed).integers(0, len(r), (2000, len(r)))
    gm = np.array([r[i].mean() for i in bi])
    eps = np.where(small, EPS["small"], EPS["large"])
    return {"geomean_ratio": float(np.exp(r.mean())), "ci": [float(np.exp(np.percentile(gm, 2.5))), float(np.exp(np.percentile(gm, 97.5)))],
            "b_faster": int((tb < ta / eps).sum()), "b_slower": int((tb > ta * eps).sum()), "n": len(r)}


def fmt(e: dict, ref: str) -> str:
    return f"{e[ref]['median']:.3f} / {e[ref]['p10']:.3f} | {e[ref]['faster']} / {e[ref]['slower']}"


def build(data: Path, report: Path, online: dict[str, str], results: str, graph_results: str) -> tuple[Path, dict]:
    cold, hot = load_stage2(data, [results], "cold"), load_stage2(data, [graph_results], "hot")
    ref = references(cold, hot)
    sc = {n: score(Path(data) / f, ref, cold) for n, f in online.items()}
    names = list(online)
    cfgs = [c for c, _, _ in CONFIGS if all(c in sc[n]["scores"] for n in names)]  # selectors every file can score
    lines = [f"# Joint online search: {' vs '.join(names)} (auto-generated)", "",
             f"Stage 2 (one session for every pick and baseline): `{results}` (cold), `{graph_results}` (graph, < 50 us). "
             f"{len(ref)} held-out shapes, {sum(r['small'] for r in ref.values())} timed graph-amortized. "
             "Ratio < 1 = faster than the reference; counts are shapes faster / slower than the reference by more than the tie "
             "margin (5% below 50 us, 2% above). Regret is vs the joint oracle (every Triton config timed in the session and B3+).", ""]
    for g in ("all", "< 50 us", ">= 50 us"):
        lines += [f"## {g} (n = {sc[names[0]]['scores'][cfgs[0]][g]['n']})", "",
                  "| ranker | selector | first-use cost | vs B2 median / p10 | faster / slower | vs B3 median / p10 | faster / slower | "
                  "vs B3+ median / p10 | faster / slower | regret vs joint oracle median / p95 / p99 (mean) | picks cuBLASLt |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for n in names:
            for c in cfgs:
                e = sc[n]["scores"][c][g]
                rg = e["regret_joint"]
                lines.append(f"| {n} | {c} | {e['cost_s']:.2f} s | " + " | ".join(fmt(e, r) for r in ("b2", "b3", "b3p"))
                             + f" | {100 * rg['median']:.1f}% / {100 * rg['p95']:.1f}% / {100 * rg['p99']:.1f}% ({100 * rg['mean']:.1f}%) | {e['pct_lt']:.0f}% |")
        lines.append("")
    lines += ["## Triton only: regret vs the Triton oracle of the session (median / p95 / p99 (mean))", "",
              "| selector | " + " | ".join(f"{n} {g}" for n in names for g in ("all", "< 50 us", ">= 50 us")) + " |",
              "|---|" + "---|" * (3 * len(names))]
    for c in [c for c in ("top-3 Triton only",) if c in cfgs]:
        cells = []
        for n in names:
            for g in ("all", "< 50 us", ">= 50 us"):
                rg = sc[n]["scores"][c][g]["regret_triton"]
                cells.append(f"{100 * rg['median']:.1f}% / {100 * rg['p95']:.1f}% / {100 * rg['p99']:.1f}% ({100 * rg['mean']:.1f}%)")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    pr = {}
    if len(names) == 2:
        a, b = sc[names[0]], sc[names[1]]
        pr = {c: paired(a, b, c) for c in cfgs}
        lines += ["", f"## Paired: {names[1]} vs {names[0]} (chosen config's stage-2 time; geometric-mean ratio, 95% bootstrap interval)", "",
                  "| selector | geomean ratio | 95% interval | " + f"{names[1]} faster / slower (> tie margin) |", "|---|---|---|---|"]
        for c in cfgs:
            p = pr[c]
            lines.append(f"| {c} | {p['geomean_ratio']:.4f} | [{p['ci'][0]:.4f}, {p['ci'][1]:.4f}] | {p['b_faster']} / {p['b_slower']} of {p['n']} |")
    out = Path(report) / "ranker_compare.md"
    out.write_text("\n".join(lines) + "\n")
    summary = {"online": online, "results": results, "graph_results": graph_results, "scores": {n: sc[n]["scores"] for n in names}, "paired": pr}
    (Path(report) / "ranker_compare_summary.json").write_text(json.dumps(summary, indent=1))
    return out, summary


def main(argv=None):
    from tune.env import default_data_dir
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=Path("reports"))
    ap.add_argument("--online", nargs="+", required=True, help="name=online_jsonl pairs, e.g. v1=online_joint_diverse.jsonl")
    ap.add_argument("--results", default="results_lock1755_v2.jsonl")
    ap.add_argument("--graph-results", default="results_lock1755_v2_graph.jsonl")
    args = ap.parse_args(argv)
    out, _ = build(args.data or default_data_dir(), args.report, dict(x.split("=", 1) for x in args.online), args.results, args.graph_results)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
