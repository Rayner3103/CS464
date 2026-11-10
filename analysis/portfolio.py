"""Config-space reduction from the stage-1 screen (plan 6.3; gates G3 and G5).

Input: data/<hw>/sweep_screen/*.json (one file per shape, every pruned config screened).
Regret of config c on shape s = t(c, s) / min_c t(c, s). A config that was pruned, invalid or failed on a shape has
infinite regret there; one that was cut early as "slow" keeps its single-sample time (always >= 1.5x the best).

  * B6 over the full space: the single config with the lowest geomean regret (inf capped at REGRET_CAP).
  * Greedy portfolio: repeatedly add the config that most reduces mean regret (each shape takes the best member).
    Reports portfolio size vs mean / p95 regret, and the smallest size reaching p95 < 5% (G5).
  * Set cover at eps: the greedy cover where every shape has a member within eps of the oracle (plan 6.3 step 2).

Caveat: screen times are a few samples each, so the per-shape oracle is the min over many noisy values and is biased
slightly low (winner's curse). Regrets here are therefore a little pessimistic; stage-2 data re-measures the top tiles.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REGRET_CAP = 10.0
PARAMS = ("BLOCK_M", "BLOCK_N", "BLOCK_K", "num_warps", "num_stages", "GROUP_M", "SPLIT_K")


def load_screen(sweep_dir: Path) -> tuple[list[dict], list[str], np.ndarray]:
    """Shapes, config keys, and a [shape, config] time matrix (inf where not timed)."""
    files = sorted(p for p in Path(sweep_dir).glob("*_fp16.json"))
    docs = [json.loads(p.read_text()) for p in files]
    docs = [d for d in docs if d.get("complete")]
    keys = sorted({r["key"] for d in docs for r in d["rows"]})
    col = {k: i for i, k in enumerate(keys)}
    T = np.full((len(docs), len(keys)), np.inf)
    shapes = []
    for i, d in enumerate(docs):
        shapes.append({"M": d["M"], "N": d["N"], "K": d["K"], "n_configs": d["n_configs"]})
        for r in d["rows"]:
            if r["status"] in ("ok", "slow") and r["us"] is not None:
                T[i, col[r["key"]]] = r["us"]
    return shapes, keys, T


def regret_matrix(T: np.ndarray) -> np.ndarray:
    return T / T.min(axis=1, keepdims=True)


def greedy_portfolio(R: np.ndarray, max_size: int = 100) -> list[int]:
    """Greedy on mean regret (capped): each step adds the config that lowers the portfolio's mean regret most."""
    Rc = np.minimum(R, REGRET_CAP)
    cur = np.full(R.shape[0], REGRET_CAP)
    chosen: list[int] = []
    for _ in range(min(max_size, R.shape[1])):
        gain = (cur[:, None] - np.minimum(cur[:, None], Rc)).sum(axis=0)
        gain[chosen] = -1
        j = int(gain.argmax())
        if gain[j] <= 0:
            break
        chosen.append(j)
        cur = np.minimum(cur, Rc[:, j])
    return chosen


def set_cover(R: np.ndarray, eps: float) -> list[int]:
    """Greedy set cover: every shape needs a member within (1 + eps) of its oracle."""
    ok = R <= 1 + eps
    uncovered = np.ones(R.shape[0], bool)
    chosen: list[int] = []
    while uncovered.any():
        j = int((ok & uncovered[:, None]).sum(axis=0).argmax())
        chosen.append(j)
        uncovered &= ~ok[:, j]
    return chosen


def curve(R: np.ndarray, order: list[int]) -> list[dict]:
    cur = np.full(R.shape[0], np.inf)
    out = []
    for n, j in enumerate(order, 1):
        cur = np.minimum(cur, R[:, j])
        out.append({"size": n, "mean": float(np.minimum(cur, REGRET_CAP).mean() - 1),
                    "p50": float(np.percentile(cur, 50) - 1), "p95": float(np.percentile(cur, 95) - 1),
                    "p99": float(np.percentile(cur, 99) - 1), "max": float(cur.max() - 1)})
    return out


def param_importance(keys: list[str], R: np.ndarray) -> dict:
    """For each parameter: mean regret when that parameter is forced to its worst value on each shape (among values
    that were timed there) with all other parameters chosen freely. High = getting this parameter right matters."""
    from tune.sweep import parse_key
    cfgs = [parse_key(k) for k in keys]
    logR = np.where(np.isfinite(R), np.log(np.minimum(R, REGRET_CAP)), np.nan)  # untimed (pruned/invalid) -> nan
    out = {}
    with np.errstate(all="ignore"):
        import warnings
        warnings.simplefilter("ignore", RuntimeWarning)
        for p in PARAMS:
            vals = np.array([getattr(c, p) for c in cfgs])
            uniq = np.unique(vals)
            # Best config per shape given p = v; nan where no config with p = v was timed on that shape.
            stack = np.stack([np.nanmin(logR[:, vals == v], axis=1) for v in uniq], axis=1)
            worst = np.nanmax(stack, axis=1)
            best_v = uniq[np.nanargmin(np.where(np.isnan(stack), np.inf, stack), axis=1)]
            out[p] = {"mean_regret_if_forced_worst_value": float(np.nanmean(np.exp(worst)) - 1),
                      "best_value_counts": {str(int(k)): int(c) for k, c in zip(*np.unique(best_v, return_counts=True))}}
    return out


def build(data: Path, report: Path, eps_list=(0.02, 0.03, 0.05)) -> tuple[Path, dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shapes, keys, T = load_screen(data / "sweep_screen")
    R = regret_matrix(T)
    logR = np.log(np.minimum(R, REGRET_CAP))
    b6 = int(logR.mean(axis=0).argmin())
    b6_r = R[:, b6]
    order = greedy_portfolio(R)
    cv = curve(R, order)
    g5 = next((c["size"] for c in cv if c["p95"] < 0.05), None)
    covers = {eps: len(set_cover(R, eps)) for eps in eps_list}
    imp = param_importance(keys, R)
    timed = np.isfinite(T)

    summary = {"n_shapes": len(shapes), "n_configs": len(keys), "screened_cells": int(timed.sum()),
               "b6": keys[b6], "b6_regret_median": float(np.median(b6_r) - 1), "b6_regret_p90": float(np.percentile(b6_r, 90) - 1),
               "g5_portfolio_size_p95_lt_5pct": g5, "set_cover_sizes": {str(k): v for k, v in covers.items()},
               "curve": cv[:60], "portfolio": [keys[j] for j in order[:60]], "param_importance": imp}
    report.mkdir(parents=True, exist_ok=True)
    (report / "sweep_portfolio_summary.json").write_text(json.dumps(summary, indent=1))

    figs = report / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = [c["size"] for c in cv]
    ax.plot(xs, [100 * c["mean"] for c in cv], label="mean regret")
    ax.plot(xs, [100 * c["p95"] for c in cv], label="p95 regret")
    ax.axhline(5, ls="--", c="gray", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("portfolio size (greedy)")
    ax.set_ylabel("regret vs per-shape oracle (%)")
    ax.set_title(f"Portfolio size vs regret ({len(shapes)} shapes, screen data)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs / "portfolio_regret.png", dpi=120)
    plt.close(fig)

    md = [f"# Triton config sweep: portfolio study (auto-generated)\n",
          f"Source: `{data / 'sweep_screen'}` | {len(shapes)} shapes x {len(keys)} configs "
          f"({timed.sum()} timed cells; pruned / invalid cells count as infinite regret).\n",
          "Regret = t / per-shape oracle - 1, from **screen** times (few samples; slightly pessimistic, see module doc).\n",
          "## G3: value of per-shape selection over the best single config (B6, full space)\n",
          f"B6 = `{keys[b6]}`. Regret of B6 vs oracle: median {100 * (np.median(b6_r) - 1):.1f}%, "
          f"p90 {100 * (np.percentile(b6_r, 90) - 1):.1f}%, max {100 * (min(b6_r.max(), REGRET_CAP) - 1):.1f}%.\n",
          "## G5: portfolio size vs regret (greedy on mean regret)\n",
          "| size | mean | p50 | p95 | p99 | max |", "|---|---|---|---|---|---|"]
    for c in cv:
        if c["size"] in (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 30, 40, 50, 60, 80, 100):
            md.append(f"| {c['size']} | {100 * c['mean']:.2f}% | {100 * c['p50']:.2f}% | {100 * c['p95']:.2f}% | "
                      f"{100 * c['p99']:.2f}% | {100 * min(c['max'], REGRET_CAP - 1):.1f}% |")
    md.append(f"\n**Smallest greedy portfolio with p95 regret < 5%: {g5 if g5 else '> ' + str(len(cv))}** "
              f"(G5 asks for <= ~50).\n")
    md.append("Set cover (every shape has a member within eps of its oracle): "
              + ", ".join(f"eps={int(100 * e)}%: {n} configs" for e, n in covers.items()) + "\n")
    md.append("First 20 portfolio members: " + ", ".join(f"`{keys[j]}`" for j in order[:20]) + "\n")
    md.append("## Which parameters matter\n")
    md.append("Mean regret if a parameter is forced to its worst value (others chosen freely per shape), and how often "
              "each value is the best choice:\n")
    md.append("| parameter | regret at worst value | best value counts (shapes) |")
    md.append("|---|---|---|")
    for p, v in sorted(imp.items(), key=lambda kv: -kv[1]["mean_regret_if_forced_worst_value"]):
        md.append(f"| {p} | {100 * v['mean_regret_if_forced_worst_value']:.0f}% | "
                  + ", ".join(f"{k}: {c}" for k, c in v["best_value_counts"].items()) + " |")
    md.append("\n![portfolio](figures/portfolio_regret.png)\n")
    out = report / "sweep_portfolio.md"
    out.write_text("\n".join(md) + "\n")
    return out, summary


SMALL_US = 50.0  # below this the oracle is a ~1 us-tick event time (see methodology); report those shapes separately


def _split_info(screen_dir: Path, shapes: list[dict]) -> list[dict]:
    """Attach suite / split from the screen dir's subset.json to each loaded shape."""
    p = Path(screen_dir) / "subset.json"
    meta = {(s["M"], s["N"], s["K"]): s for s in json.loads(p.read_text())["shapes"]} if p.exists() else {}
    return [dict(s, **{k: meta.get((s["M"], s["N"], s["K"]), {}).get(k) for k in ("suite", "split")}) for s in shapes]


def align(keys_from: list[str], keys_to: list[str], T_to: np.ndarray) -> np.ndarray:
    """Columns of T_to in the order of keys_from; inf for a config never timed in T_to's screen."""
    col = {k: i for i, k in enumerate(keys_to)}
    out = np.full((T_to.shape[0], len(keys_from)), np.inf)
    for j, k in enumerate(keys_from):
        if k in col:
            out[:, j] = T_to[:, col[k]]
    return out


def _stats(r: np.ndarray) -> dict:
    """Regret summary (r = t / oracle) over shapes."""
    if not len(r):
        return {"n": 0}
    return {"n": int(len(r)), "mean": float(np.minimum(r, REGRET_CAP).mean() - 1), "p50": float(np.percentile(r, 50) - 1),
            "p95": float(np.percentile(r, 95) - 1), "p99": float(np.percentile(r, 99) - 1), "max": float(r.max() - 1), "frac_le_5pct": float((r <= 1.05).mean())}


def stage2_regret(rows: list[dict], portfolio: list[str], sizes) -> tuple[list[dict], dict[int, np.ndarray]]:
    """Stage-2 (re-measured) regret of portfolio prefixes. Per shape, oracle = fastest valid row (the screen's top-8
    tiles plus the portfolio members); prefix n = fastest valid row among the first n portfolio keys."""
    by_shape: dict[tuple, dict[str, float]] = {}
    for r in rows:
        if r.get("valid") and r.get("kernel_family", "").startswith("triton") and r.get("time_median_us"):
            d = by_shape.setdefault((r["M"], r["N"], r["K"]), {})
            d[r["config_key"]] = min(d.get(r["config_key"], np.inf), r["time_median_us"])
    shapes = sorted(by_shape)
    oracle = np.array([min(by_shape[s].values()) for s in shapes])
    regrets = {}
    for n in sizes:
        keys = portfolio[:n]
        regrets[n] = np.array([min((by_shape[s].get(k, np.inf) for k in keys)) for s in shapes]) / oracle
    meta = [{"M": s[0], "N": s[1], "K": s[2], "oracle_us": float(o)} for s, o in zip(shapes, oracle)]
    return meta, regrets


def build_heldout(data: Path, report: Path, sizes=(1, 5, 10, 16, 20, 25, 29, 30, 40, 50, 60, 100),
                  eps_list=(0.02, 0.03, 0.05), results: str = "results_lock1755_heldout.jsonl") -> tuple[Path, dict]:
    """G5 out of sample: fit B6, the greedy portfolio and set covers on the tune-split screen, then score them on the
    held-out screen, where the per-shape oracle is the best of the full space on that held-out shape."""
    shapes, keys, T = load_screen(data / "sweep_screen")
    R = regret_matrix(T)
    order = greedy_portfolio(R)
    cv_in = curve(R, order)
    g5_in = next((c["size"] for c in cv_in if c["p95"] < 0.05), None)
    b6 = int(np.log(np.minimum(R, REGRET_CAP)).mean(axis=0).argmin())
    covers = {eps: set_cover(R, eps) for eps in eps_list}

    hdir = data / "sweep_screen_heldout"
    hshapes, hkeys, hT = load_screen(hdir)
    hshapes = _split_info(hdir, hshapes)
    oracle = hT.min(axis=1)
    Rh = align(keys, hkeys, hT) / oracle[:, None]  # tune-fitted configs scored against the held-out oracle
    small = oracle < SMALL_US
    groups = {"all": np.ones(len(hshapes), bool), f">= {SMALL_US:.0f} us": ~small, f"< {SMALL_US:.0f} us": small}
    for sp in sorted({s["split"] for s in hshapes if s["split"]}):
        groups[sp] = np.array([s["split"] == sp for s in hshapes])
    # In-sample reference split the same way.
    small_in = T.min(axis=1) < SMALL_US

    def best_of(Rx, members):
        return Rx[:, members].min(axis=1)

    rows = []
    for n in sizes:
        if n > len(order):
            continue
        r_h, r_i = best_of(Rh, order[:n]), best_of(R, order[:n])
        rows.append({"size": n, "in_sample": _stats(r_i), "in_sample_ge50": _stats(r_i[~small_in]),
                     **{f"heldout[{g}]": _stats(r_h[m]) for g, m in groups.items()}})
    cv_h = curve(Rh, order)
    g5_h = next((c["size"] for c in cv_h if c["p95"] < 0.05), None)
    # Re-fit on the held-out shapes themselves: how large a portfolio would they need if we could tune on them?
    cv_refit = curve(regret_matrix(hT), greedy_portfolio(regret_matrix(hT)))
    g5_refit = next((c["size"] for c in cv_refit if c["p95"] < 0.05), None)
    g5_size = g5_in or 29
    r_g5 = best_of(Rh, order[:g5_size])
    worst = np.argsort(-r_g5)[:12]
    hk = {k: i for i, k in enumerate(hkeys)}
    worst_rows = []
    for i in worst:
        j = int(np.argmin(Rh[i, order[:g5_size]]))
        worst_rows.append({**{k: hshapes[i][k] for k in ("M", "N", "K", "suite", "split")},
                           "oracle_us": float(oracle[i]), "oracle_cfg": hkeys[int(hT[i].argmin())],
                           "portfolio_us": float(oracle[i] * r_g5[i]), "portfolio_cfg": keys[order[:g5_size][j]],
                           "regret": float(r_g5[i] - 1)})
    cover_rows = {str(eps): {"size": len(c), **_stats(best_of(Rh, c)),
                             "ge50": _stats(best_of(Rh, c)[~small])} for eps, c in covers.items()}
    b6_h = Rh[:, b6]
    summary = {"n_tune_shapes": len(shapes), "n_heldout_shapes": len(hshapes),
               "heldout_split_counts": {g: int(m.sum()) for g, m in groups.items()},
               "g5_size_in_sample": g5_in, "g5_size_heldout_tune_order": g5_h, "g5_size_heldout_refit": g5_refit,
               "b6": keys[b6], "b6_heldout": _stats(b6_h), "rows": rows, "set_covers": cover_rows, "worst": worst_rows,
               "missing_configs": int(sum(k not in hk for k in (keys[j] for j in order[:g5_size])))}
    report.mkdir(parents=True, exist_ok=True)
    (report / "sweep_portfolio_heldout_summary.json").write_text(json.dumps(summary, indent=1))

    pct = lambda x: f"{100 * x:.1f}%"
    md = ["# Triton config sweep: held-out G5 check (auto-generated)\n",
          f"Portfolio, B6 and set covers are fit on the tune-split screen (`{data / 'sweep_screen'}`, {len(shapes)} shapes) "
          f"and scored on the held-out screen (`{hdir}`, {len(hshapes)} shapes: "
          + ", ".join(f"{g} {int(m.sum())}" for g, m in groups.items() if g != "all") + "). "
          "Held-out oracle = best of the full pruned space on that shape. Regret = t / oracle - 1, screen times.\n",
          "## G5: tune-fitted greedy portfolio on held-out shapes (p95 regret)\n",
          "| size | in-sample | in-sample >= 50 us | " + " | ".join(f"held-out {g}" for g in groups) + " |",
          "|---|---|---|" + "---|" * len(groups)]
    for r in rows:
        md.append(f"| {r['size']} | {pct(r['in_sample']['p95'])} | {pct(r['in_sample_ge50']['p95'])} | "
                  + " | ".join(pct(r[f'heldout[{g}]']['p95']) if r[f'heldout[{g}]']['n'] else "-" for g in groups) + " |")
    md.append("\nSame table, mean regret:\n")
    md.append("| size | in-sample | " + " | ".join(f"held-out {g}" for g in groups) + " |")
    md.append("|---|---|" + "---|" * len(groups))
    for r in rows:
        md.append(f"| {r['size']} | {pct(r['in_sample']['mean'])} | "
                  + " | ".join(pct(r[f'heldout[{g}]']['mean']) if r[f'heldout[{g}]']['n'] else "-" for g in groups) + " |")
    md.append("\nSame table, p99 regret (on 86 held-out shapes this is essentially the worst shape):\n")
    md.append("| size | in-sample | in-sample >= 50 us | " + " | ".join(f"held-out {g}" for g in groups) + " |")
    md.append("|---|---|---|" + "---|" * len(groups))
    for r in rows:
        md.append(f"| {r['size']} | {pct(r['in_sample']['p99'])} | {pct(r['in_sample_ge50']['p99'])} | "
                  + " | ".join(pct(r[f'heldout[{g}]']['p99']) if r[f'heldout[{g}]']['n'] else "-" for g in groups) + " |")
    md.append(f"\n**Smallest tune-order prefix with held-out p95 < 5%: {g5_h if g5_h else '> ' + str(len(cv_h))}** "
              f"(in-sample: {g5_in}; a greedy portfolio re-fit on the held-out shapes themselves would need {g5_refit}).\n")
    md.append(f"B6 (`{keys[b6]}`) on held-out shapes: median regret {pct(np.median(b6_h) - 1)}, "
              f"p90 {pct(np.percentile(b6_h, 90) - 1)} (G3 out of sample).\n")
    md.append("Tune-fitted set covers on held-out shapes:\n")
    md.append("| eps | size | held-out p95 | held-out p99 | held-out max | shapes within 5% | >= 50 us p95 |")
    md.append("|---|---|---|---|---|---|---|")
    for eps, c in cover_rows.items():
        md.append(f"| {pct(float(eps))} | {c['size']} | {pct(c['p95'])} | {pct(c['p99'])} | {pct(c['max'])} | {pct(c['frac_le_5pct'])} | "
                  f"{pct(c['ge50']['p95']) if c['ge50']['n'] else '-'} |")
    md.append(f"\n## Worst held-out shapes for the {g5_size}-config portfolio\n")
    md.append("| shape | suite / split | oracle us | oracle config | portfolio us | portfolio config | regret |")
    md.append("|---|---|---|---|---|---|---|")
    for w in worst_rows:
        md.append(f"| {w['M']}x{w['N']}x{w['K']} | {w['suite']} / {w['split']} | {w['oracle_us']:.2f} | `{w['oracle_cfg']}` | "
                  f"{w['portfolio_us']:.2f} | `{w['portfolio_cfg']}` | {pct(w['regret'])} |")
    res = data / results
    if res.exists():
        rows2 = [json.loads(l) for l in res.read_text().splitlines() if l.strip()]
        # The portfolio summary that stage 2 actually ran with: <results stem>.portfolio.json next to the results,
        # else the first run's copy in the held-out screen dir, else the current summary.
        used = next((p for p in (data / f"{Path(results).stem}.portfolio.json", hdir / "portfolio_summary_used.json")
                     if p.exists()), report / "sweep_portfolio_summary.json")
        pkeys = json.loads(used.read_text())["portfolio"]
        n_meas = max(len({r["config_key"] for r in rows2 if r["config_key"] in pkeys}), 1)
        s2_sizes = [n for n in (1, 5, 10, 16, 20, 25, 29, 40, 48, 60) if n <= n_meas]
        meta, reg = stage2_regret(rows2, pkeys, s2_sizes)
        small2 = np.array([m["oracle_us"] < SMALL_US for m in meta])
        split_of = {(s["M"], s["N"], s["K"]): s["split"] for s in hshapes}
        sp2 = np.array([split_of.get((m["M"], m["N"], m["K"])) for m in meta])
        g2 = {"all": np.ones(len(meta), bool), f">= {SMALL_US:.0f} us": ~small2, f"< {SMALL_US:.0f} us": small2,
              **{sp: sp2 == sp for sp in sorted(set(sp2) - {None})}}
        summary["stage2"] = {"results": str(res), "n_shapes": len(meta), "n_invalid_rows": sum(not r.get("valid") for r in rows2),
                             "rows": [{"size": n, **{g: _stats(reg[n][m]) for g, m in g2.items()}} for n in s2_sizes]}
        md.append(f"\n## Stage-2 confirmation (re-measured, `{results}`)\n")
        md.append(f"Each held-out shape re-timed cold (adaptive repeats, 3 input placements): the screen's 8 fastest tiles "
                  f"for that shape plus the first {s2_sizes[-1]} portfolio members. Oracle = fastest of those. "
                  f"{len(meta)} shapes, {summary['stage2']['n_invalid_rows']} invalid rows. "
                  "Sub-50 us cold times carry ~1 us event-tick quantization.\n")
        md.append("| size | " + " | ".join(f"{g} p95 / p99 / mean" for g in g2) + " |")
        md.append("|---|" + "---|" * len(g2))
        for n in s2_sizes:
            md.append(f"| {n} | " + " | ".join(f"{pct(_stats(reg[n][m])['p95'])} / {pct(_stats(reg[n][m])['p99'])} / {pct(_stats(reg[n][m])['mean'])}"
                                               if m.any() else "-" for m in g2.values()) + " |")
        n = s2_sizes[-1]
        worst2 = np.argsort(-reg[n])[:8]
        md.append(f"\nWorst shapes at size {n} (stage 2): " + ", ".join(
            f"{meta[i]['M']}x{meta[i]['N']}x{meta[i]['K']} {pct(reg[n][i] - 1)} ({meta[i]['oracle_us']:.1f} us)" for i in worst2) + "\n")
        (report / "sweep_portfolio_heldout_summary.json").write_text(json.dumps(summary, indent=1))
    out = report / "sweep_portfolio_heldout.md"
    out.write_text("\n".join(md) + "\n")
    return out, summary
