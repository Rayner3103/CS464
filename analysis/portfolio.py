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
                    "max": float(cur.max() - 1)})
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
          "| size | mean | p50 | p95 | max |", "|---|---|---|---|---|"]
    for c in cv:
        if c["size"] in (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 30, 40, 50, 60, 80, 100):
            md.append(f"| {c['size']} | {100 * c['mean']:.2f}% | {100 * c['p50']:.2f}% | {100 * c['p95']:.2f}% | "
                      f"{100 * min(c['max'], REGRET_CAP - 1):.1f}% |")
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
