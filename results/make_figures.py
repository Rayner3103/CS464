"""Build the figures for results/*.md from the repo's existing result files (CPU only, never touches the GPU).

    python results/make_figures.py

Every number is read from reports/, data/<hw>/ or suites/v1/, except a few small tables that only exist as
text in reports/week1.md / reports/ranker_compare.md / reports/cross_gpu.md; those are typed in below as
HAND-COPIED blocks with the source named next to them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
D = ROOT / "data" / "rtx3080_sm86_68sms_10gb"
R = ROOT / "reports"

# validated categorical palette (dataviz skill, light mode), fixed order
BLUE, ORANGE, AQUA, MAGENTA, VIOLET, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#6250d6", "#9a9a95"
INK, INK2, BG = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": "#c9c8c2", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e8e7e2", "grid.linewidth": 0.7,
    "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 160,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def jload(p):
    return json.loads(Path(p).read_text())


# ------------------------------------------------------------------------------------------------ 01 background
def fig_cross_matrix():
    """Best config of shape i, run on shape j: how much slower than j's own best (from the stage-1 screens)."""
    shapes = [(1, 4096, 4096), (16, 4096, 4096), (512, 4096, 4096), (4096, 4096, 4096), (1000, 1000, 1000)]
    screens = {}
    for f in (D / "sweep_screen").glob("*.json"):
        if f.name == "subset.json":
            continue
        s = jload(f)
        if "rows" not in s:
            continue
        screens[(s["M"], s["N"], s["K"])] = {r["key"]: r["us"] for r in s["rows"] if r["status"] == "ok"}
    use = [s for s in shapes if s in screens]
    if len(use) < 4:  # fall back to whichever tune shapes exist, spread over the size range
        keys = sorted(screens, key=lambda k: k[0] * k[1] * k[2])
        use = [keys[i] for i in np.linspace(0, len(keys) - 1, 6).astype(int)]
    best = {s: min(screens[s], key=screens[s].get) for s in use}
    mat = np.full((len(use), len(use)), np.nan)
    for i, si in enumerate(use):
        for j, sj in enumerate(use):
            t = screens[sj].get(best[si])
            if t:
                mat[i, j] = t / min(screens[sj].values())
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.grid(False)
    im = ax.imshow(np.log2(mat), cmap="Blues", vmin=0, vmax=np.log2(np.nanmax(mat)))
    lab = [f"{m}×{n}×{k}" for m, n, k in use]
    ax.set_xticks(range(len(use)), lab, rotation=30, ha="right")
    ax.set_yticks(range(len(use)), lab)
    ax.set_xlabel("run on this shape")
    ax.set_ylabel("config that is best for this shape")
    for i in range(len(use)):
        for j in range(len(use)):
            v = mat[i, j]
            ax.text(j, i, "n/a" if np.isnan(v) else f"{v:.1f}×", ha="center", va="center",
                    color="white" if (not np.isnan(v) and np.log2(v) > np.log2(np.nanmax(mat)) / 2) else INK, fontsize=9)
    ax.set_title("A config tuned for one shape is slow on others\n(1.0× = as fast as that shape's own best config)", fontsize=10)
    save(fig, "01_config_cross_matrix.png")


def fig_config_spread():
    """For a few shapes: how much slower than the best is the k-th fastest config? Shows how much a choice matters."""
    pick = [(16, 4096, 4096), (512, 4096, 4096), (4096, 4096, 4096)]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    cols = [BLUE, ORANGE, AQUA]
    n_drawn = 0
    for (m, n, k), c in zip(pick, cols):
        f = D / "sweep_screen" / f"{m}x{n}x{k}_fp16.json"
        if not f.exists():
            continue
        s = jload(f)
        t = np.sort([r["us"] for r in s["rows"] if r["status"] in ("ok", "slow")])
        ax.plot(np.arange(1, len(t) + 1), t / t[0], color=c, lw=2, label=f"{m}×{n}×{k}  ({len(t)} configs)")
        n_drawn += 1
    ax.set_yscale("log")
    ax.set_xscale("log")
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.set_yticks([1, 2, 5, 10])
    ax.set_xlabel("config rank (1 = fastest, log scale)")
    ax.set_ylabel("time vs the fastest config")
    ax.set_title("Most configs are far slower than the best one\n(early-stopped timings for the slow ones, so rough)", fontsize=10)
    if n_drawn:
        ax.legend(loc="upper left")
    save(fig, "01_config_spread.png")


# ------------------------------------------------------------------------------------------------ 02 baselines
def _cold():
    t = pd.read_csv(R / "baseline_table_cold_lock1755_sweep.csv", usecols=lambda c: not c.startswith("ours:"))
    return t


def fig_baseline_bars():
    t = _cold()
    cols = [("B1 torch.matmul", "B1  torch.matmul (default cuBLAS)", GREY), ("B5 triton tutorial", "B5  Triton tutorial kernel", MAGENTA),
            ("oracle(ours)", "Triton family, best config per shape", BLUE)]
    order = ["S1 square", "S2 decode (M<=16)", "S2 small batch (32<=M<=128)", "S2 prefill (M>=512)", "S3 skinny/tall", "S4 large-K",
             "S5 unaligned", "S6 wave-boundary", "S7 random"]
    nice = {"S1 square": "square", "S2 decode (M<=16)": "LLM decode\n(M≤16)", "S2 small batch (32<=M<=128)": "LLM small\nbatch",
            "S2 prefill (M>=512)": "LLM prefill\n(M≥512)", "S3 skinny/tall": "skinny/tall", "S4 large-K": "large K",
            "S5 unaligned": "unaligned", "S6 wave-boundary": "wave\nboundary", "S7 random": "random"}
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    w = 0.26
    x = np.arange(len(order))
    for i, (c, lab, col) in enumerate(cols):
        med = [np.median(t.loc[t.regime == r, "B3+ cublasLt deep"] / t.loc[t.regime == r, c]) for r in order]
        ax.bar(x + (i - 1) * w, med, w * 0.92, color=col, label=lab)
    ax.axhline(1, color=INK, lw=1.2)
    ax.text(len(order) - 0.45, 1.03, "= searched cuBLASLt (B3+)", ha="right", fontsize=9, color=INK2)
    ax.set_xticks(x, [nice[r] for r in order])
    ax.set_ylabel("speed relative to B3+\n(higher = faster)")
    ax.set_title("Speed of each baseline vs the searched cuBLASLt (median per shape group, 160 shapes)")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.set_ylim(0, 1.35)
    save(fig, "02_baseline_speed_by_regime.png")


def fig_square_tflops():
    t = _cold()
    s = t[t.regime == "S1 square"].sort_values("M")
    fl = lambda col: 2.0 * s.M * s.N * s.K / s[col] / 1e6
    fig, ax = plt.subplots(figsize=(7, 4))
    for col, lab, c, mk in [("B5 triton tutorial", "B5 Triton tutorial", MAGENTA, "^"), ("B1 torch.matmul", "B1 torch.matmul", GREY, "o"),
                            ("B3+ cublasLt deep", "B3+ searched cuBLASLt", ORANGE, "s"), ("oracle(ours)", "Triton family, best per shape", BLUE, "D")]:
        ax.plot(s.M, fl(col), marker=mk, ms=4.5, lw=1.8, color=c, label=lab)
    ax.axhline(62.9, color=INK2, ls="--", lw=1)
    ax.text(70, 64, "measured hardware peak 62.9 TFLOP/s", fontsize=8.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("matrix size n  (n×n times n×n)")
    ax.set_ylabel("TFLOP/s (higher = faster)")
    ax.set_ylim(0, 70)
    ax.set_title("Square matrices: big sizes reach the hardware limit")
    ax.legend(loc="lower right", fontsize=8.5)
    save(fig, "02_square_tflops.png")


def fig_b6():
    """Why per-shape selection matters: best single config (B6) vs per-shape best, held-out (reports/sweep_portfolio_heldout_summary.json)."""
    d = jload(R / "sweep_portfolio_heldout_summary.json")["rows"]
    r1 = [r for r in d if r["size"] == 1][0]["heldout[all]"]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    names = ["median", "mean", "95th percentile"]
    vals = [r1["p50"], r1["mean"], r1["p95"]]
    b = ax.bar(names, [v * 100 for v in vals], color=ORANGE, width=0.55)
    for rect, v in zip(b, vals):
        ax.text(rect.get_x() + rect.get_width() / 2, v * 100 + 8, f"+{v * 100:.0f}%", ha="center", fontsize=10)
    ax.set_ylabel("extra time vs best config for the shape")
    ax.set_title("One fixed config for every shape\n(unseen shapes, 86 of them)", fontsize=10)
    ax.set_ylim(0, max(vals) * 100 * 1.18)
    save(fig, "02_single_config_regret.png")


def fig_noise():
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    # HAND-COPIED from reports/week1.md "Noise floor": across-round range of medians, cold
    ax = axs[0]
    x = np.arange(2)
    ax.bar(x - 0.18, [0.67, 0.29], 0.34, color=BLUE, label="typical (median)")
    ax.bar(x + 0.18, [5.4, 2.6], 0.34, color=ORANGE, label="unlucky (95th percentile)")
    for xi, (a, b) in zip(x, [(0.67, 5.4), (0.29, 2.6)]):
        ax.text(xi - 0.18, a + 0.1, f"{a}%", ha="center", fontsize=9)
        ax.text(xi + 0.18, b + 0.1, f"{b}%", ha="center", fontsize=9)
    ax.set_xticks(x, ["1 input placement", "3 placements pooled"])
    ax.set_ylabel("run-to-run spread of the same timing")
    ax.set_title("Pooling 3 memory placements\nhalves the noise")
    ax.legend(fontsize=8.5)
    # HAND-COPIED from reports/results_overview.md "Allocation placement"
    ax = axs[1]
    shp = ["8192×64×8192\n(memory-bound)", "4096³\n(compute-bound)"]
    ax.bar(np.arange(2) - 0.18, [0.0, 0.46], 0.34, color=BLUE, label="same memory block, repeated")
    ax.bar(np.arange(2) + 0.18, [3.73, 0.84], 0.34, color=ORANGE, label="different memory blocks")
    ax.set_xticks(np.arange(2), shp)
    ax.set_ylabel("spread of median time (%)")
    ax.set_title("Where the data sits in memory\nchanges the time")
    ax.set_ylim(0, 5.2)
    ax.legend(fontsize=8.5, loc="upper right")
    save(fig, "02_noise.png")


# ------------------------------------------------------------------------------------------------ 03 models
def fig_selectors():
    from analysis.plots import selector_rows
    rows = selector_rows(R, "cold", "all")
    want = ["B6 best single config", "hand-written rule", "5-NN on (M, N, K)", "k-NN + GPU-fill features", "decision tree, classify",
            "random forest, rank", "LightGBM, classify", "LightGBM top-1", "plain online top-3", "diverse online top-3 (bar)", "diverse online top-5"]
    d = {lab: st for lab, fam, st in rows}
    fam = {lab: f for lab, f, st in rows}
    names = [w for w in want if w in d]
    short = {"B6 best single config": "one fixed config", "hand-written rule": "hand-written rules", "5-NN on (M, N, K)": "5 nearest shapes",
             "k-NN + GPU-fill features": "nearest shapes\n+ GPU-fill features", "decision tree, classify": "decision tree",
             "random forest, rank": "random forest", "LightGBM, classify": "LightGBM (pick 1)", "LightGBM top-1": "LightGBM ranker\n(pick 1)",
             "plain online top-3": "ranker top-3\n+ quick race", "diverse online top-3 (bar)": "ranker diverse top-3\n+ quick race", "diverse online top-5": "ranker diverse top-5\n+ quick race"}
    colr = [GREY, VIOLET, BLUE][0:3]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    y = np.arange(len(names))[::-1]
    for ax, key, ttl in [(axs[0], "p50", "Typical shape (median)"), (axs[1], "p95", "Unlucky shapes (95th percentile)")]:
        vals = [d[n][key] * 100 for n in names]
        ax.barh(y, vals, color=[[GREY, BLUE, ORANGE][fam[n]] for n in names], height=0.62)
        for yi, v in zip(y, vals):
            ax.text(v * 1.08 + 0.1, yi, f"{v:.1f}%", va="center", fontsize=8.5)
        ax.set_xscale("log")
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("extra time vs the best possible config (log scale, lower = better)")
    axs[0].set_yticks(y, [short[n] for n in names], fontsize=8.5)
    from matplotlib.patches import Patch
    axs[1].legend(handles=[Patch(color=GREY, label="simple / static"), Patch(color=BLUE, label="one learned pick"), Patch(color=ORANGE, label="learned ranker + quick race")],
                  loc="lower right", fontsize=8.5)
    fig.suptitle("Selectors on 86 unseen shapes (never used for training)", x=0.01, ha="left", fontweight="bold", fontsize=11)
    save(fig, "03_selector_comparison.png")


def fig_portfolio():
    d = jload(R / "sweep_portfolio_heldout_summary.json")["rows"]
    keep = [r for r in d if "heldout[all]" in r and "in_sample" in r]
    fig, ax = plt.subplots(figsize=(6.6, 4))
    sz = [r["size"] for r in keep]
    ax.plot(sz, [r["in_sample"]["p95"] * 100 for r in keep], color=BLUE, lw=2, label="shapes used to build the list (95th pct)")
    ax.plot(sz, [r["heldout[all]"]["p95"] * 100 for r in keep], color=ORANGE, lw=2, label="unseen shapes (95th pct)")
    ax.plot(sz, [r["heldout[all]"]["mean"] * 100 for r in keep], color=ORANGE, lw=2, ls="--", label="unseen shapes (average)")
    ax.axhline(5, color=INK2, lw=1, ls=":")
    ax.text(sz[-1], 6.5, "5% goal", ha="right", fontsize=8.5, color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of configs kept in a fixed 'menu'")
    ax.set_ylabel("extra time vs best config")
    ax.set_title("A fixed menu of configs stops improving on unseen shapes")
    ax.legend(fontsize=8.5)
    save(fig, "03_portfolio.png")


def fig_topk():
    t = jload(R / "learned_topk_summary.json")["stage2_cold"]["online"]
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=False)
    for ax, key, ttl in [(axs[0], "p50", "Typical shape (median)"), (axs[1], "p95", "Unlucky shapes (95th percentile)")]:
        for order, c, lab in [("plain", GREY, "plain top-k"), ("diverse", BLUE, "diverse top-k")]:
            ks = [k for k in range(1, 6) if f"{order}@{k}" in t]
            ax.plot(ks, [t[f"{order}@{k}"]["all"][key] * 100 for k in ks], marker="o", color=c, lw=2, label=lab)
        ax.set_xticks(range(1, 6))
        ax.set_xlabel("k = how many candidates get compiled and raced")
        ax.set_title(ttl, fontsize=10)
    axs[0].set_ylabel("extra time vs best config (%)")
    axs[0].legend(fontsize=8.5)
    fig.suptitle("Racing a few candidates beats trusting the first pick", x=0.01, ha="left", fontweight="bold", fontsize=11)
    save(fig, "03_topk.png")


def fig_v1_v2():
    # HAND-COPIED from reports/ranker_compare.md "Triton only: regret vs the Triton oracle" (top-3, 86 held-out shapes)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    labels = ["median", "average", "95th percentile"]
    v1, v2 = [0.4, 2.3, 9.5], [0.2, 1.2, 4.6]
    x = np.arange(3)
    ax.bar(x - 0.18, v1, 0.34, color=GREY, label="ranker v1")
    ax.bar(x + 0.18, v2, 0.34, color=BLUE, label="ranker v2 (+ roofline features, + 150 extra shapes)")
    for xi, a, b in zip(x, v1, v2):
        ax.text(xi - 0.18, a + 0.15, f"{a}%", ha="center", fontsize=9)
        ax.text(xi + 0.18, b + 0.15, f"{b}%", ha="center", fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylabel("extra time vs best Triton config")
    ax.set_title("v2 halves the bad tail (Triton only, top-3)")
    ax.legend(fontsize=8.5, loc="upper left")
    save(fig, "03_ranker_v1_v2.png")


def fig_joint():
    # HAND-COPIED from reports/ranker_compare.md "all (n = 86)", rows v2 'top-3 Triton only' and 'top-3 + LT16' (faster / slower than the reference)
    refs = ["B2: cuBLASLt\ndefault pick", "B3: cuBLASLt\nbest of its top list", "B3+: cuBLASLt\n20 s offline search"]
    tri = [(43, 15), (30, 23), (23, 35)]
    jnt = [(49, 0), (30, 0), (23, 14)]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, data, ttl in [(axs[0], tri, "Triton candidates only (top-3)"), (axs[1], jnt, "Triton top-3 + cuBLASLt top-16 in the same race")]:
        y = np.arange(3)[::-1]
        fa = np.array([a for a, b in data])
        sl = np.array([b for a, b in data])
        ti = 86 - fa - sl
        ax.barh(y, fa, color=AQUA, label="our pick faster", height=0.6)
        ax.barh(y, ti, left=fa, color="#d5d4ce", label="tie (within noise)", height=0.6)
        ax.barh(y, sl, left=fa + ti, color=ORANGE, label="our pick slower", height=0.6)
        for yi, a, b, c in zip(y, fa, ti, sl):
            if a:
                ax.text(a / 2, yi, str(a), ha="center", va="center", color="white", fontsize=9)
            if c:
                ax.text(a + b + c / 2, yi, str(c), ha="center", va="center", color="white", fontsize=9)
            ax.text(a + b / 2, yi, str(b), ha="center", va="center", fontsize=9)
        ax.set_xlim(0, 86)
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("number of the 86 unseen shapes")
        ax.grid(False)
    axs[0].set_yticks(np.arange(3)[::-1], refs, fontsize=8.5)
    axs[1].legend(loc="upper center", bbox_to_anchor=(0.0, -0.22), ncol=3, fontsize=8.5)
    save(fig, "03_joint_vs_cublaslt.png")


def fig_cross_gpu():
    # HAND-COPIED from reports/cross_gpu.md (27 shared held-out shapes, online top-3)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axs[0]
    names = ["RTX 3080\nranker v1", "RTX 3080\nranker v2", "RTX 2060\n(retrained)"]
    med, p95 = [1.5, 0.8, 0.7], [7.6, 6.9, 7.5]
    x = np.arange(3)
    ax.bar(x - 0.18, med, 0.34, color=BLUE, label="median")
    ax.bar(x + 0.18, p95, 0.34, color=ORANGE, label="95th percentile")
    for xi, a, b in zip(x, med, p95):
        ax.text(xi - 0.18, a + 0.15, f"{a}", ha="center", fontsize=9)
        ax.text(xi + 0.18, b + 0.15, f"{b}", ha="center", fontsize=9)
    ax.set_xticks(x, names)
    ax.set_ylabel("extra time vs best config (%)")
    ax.set_title("Picking the best config works on both GPUs", fontsize=10)
    ax.set_ylim(0, 10)
    ax.legend(fontsize=8.5, loc="upper left", ncol=2)
    ax = axs[1]
    sp = [1.005, 1.005, 0.434]
    ax.bar(x, sp, 0.5, color=[BLUE, BLUE, ORANGE])
    for xi, v in zip(x, sp):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax.axhline(1, color=INK, lw=1.2)
    ax.set_xticks(x, names)
    ax.set_ylabel("speed vs searched cuBLASLt (B3+)")
    ax.set_title("…but on the 2060 Triton itself is too slow", fontsize=10)
    ax.set_ylim(0, 1.25)
    save(fig, "03_cross_gpu.png")


# ------------------------------------------------------------------------------------------------ 05 transferability
def fig_transfer_quality():
    """Same 27 held-out shapes on three GPUs: regret of the online top-3 pick, and picks that are > 50% off."""
    # HAND-COPIED from reports/cross_gpu.md and reports/cross_gpu_3090_retrained.md ("Selection quality", online-3,
    # regret vs each GPU's own screen oracle, 27 shared held-out shapes) and the "regret > 50%" row.
    # (label, GPU colour, trained on that GPU?, median %, p95 %, picks > 50% off)
    arms = [
        ("RTX 3080, v1 (home)", BLUE, True, 1.5, 7.6, 0),
        ("RTX 3080, v2 (home)", BLUE, True, 0.8, 6.9, 0),
        ("RTX 3090, v1, 3080 model unchanged", AQUA, False, 5.4, 14.5, 0),
        ("RTX 3090, v2, 3080 model unchanged", AQUA, False, 0.9, 14.1, 0),
        ("RTX 3090, v1, retrained on the 3090", AQUA, True, 0.2, 8.4, 0),
        ("RTX 3090, v2, retrained on the 3090", AQUA, True, 0.9, 11.9, 0),
        ("RTX 2060, retrained on the 2060", ORANGE, True, 0.7, 7.5, 0),
        ("RTX 2060, v1, 3080 model unchanged", ORANGE, False, 79.8, 658.0, 16),
        ("RTX 2060, v2, 3080 model unchanged", ORANGE, False, 58.4, 635.8, 13),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True, gridspec_kw={"width_ratios": [1, 1, 0.8]})
    y = np.arange(len(arms))[::-1]
    for ax, idx, ttl in [(axs[0], 3, "Typical shape (median)"), (axs[1], 4, "Unlucky shapes (95th percentile)")]:
        for yi, a in zip(y, arms):
            ax.barh(yi, a[idx], color=a[1], height=0.62, hatch=None if a[2] else "//", edgecolor=BG if a[2] else "white", linewidth=0)
            ax.text(a[idx] * 1.12, yi, f"{a[idx]:.1f}%", va="center", fontsize=8.5)
        ax.set_xscale("log")
        ax.set_xlim(0.1, 3000)
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("extra time vs best config (log scale)")
    axs[0].set_yticks(y, [a[0] for a in arms], fontsize=8.5)
    ax = axs[2]
    for yi, a in zip(y, arms):
        ax.barh(yi, a[5], color=a[1], height=0.62, hatch=None if a[2] else "//", edgecolor=BG if a[2] else "white", linewidth=0)
        ax.text(a[5] + 0.4, yi, str(a[5]), va="center", fontsize=9)
    ax.set_xlim(0, 20)
    ax.set_title("Picks more than 50% off (of 27)", fontsize=10)
    ax.set_xlabel("number of shapes")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=GREY, label="model trained on that GPU"),
                        Patch(facecolor=GREY, hatch="//", edgecolor="white", label="3080 model, zero-shot")],
               loc="upper center", bbox_to_anchor=(0.5, 0.01), ncol=2, fontsize=9)
    fig.suptitle("Zero-shot works on the 3090 and fails on the 2060 (27 shared unseen shapes, online top-3 pick)", x=0.01, ha="left",
                 fontweight="bold", fontsize=11)
    save(fig, "05_transfer_quality.png")


def fig_transfer_ceiling():
    """Why the joint search is a safety net: how good Triton can be on each GPU, and who wins the joint race."""
    # HAND-COPIED from reports/cross_gpu.md ("Family ceiling", 3080 v2 / 3090 zero-shot v2 / 2060 rows, median) and
    # "Joint search" (picks cuBLASLt, of 27 shapes; 3080 v2, 3090 zero-shot v2 as_is, 2060 zero-shot v2 as_is).
    names = ["RTX 3080", "RTX 3090", "RTX 2060"]
    cols = [BLUE, AQUA, ORANGE]
    ceil = [1.007, 1.008, 0.440]
    lt = [9, 8, 24]
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.8))
    ax = axs[0]
    ax.bar(names, ceil, 0.5, color=cols)
    for i, v in enumerate(ceil):
        ax.text(i, v + 0.025, f"{v:.2f}×", ha="center", fontsize=10)
    ax.axhline(1, color=INK, lw=1.2)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("best Triton config per shape,\nspeed vs searched cuBLASLt (B3+)")
    ax.set_title("How good can our Triton kernels get?", fontsize=10)
    ax = axs[1]
    ax.bar(names, lt, 0.5, color=cols)
    for i, v in enumerate(lt):
        ax.text(i, v + 0.5, f"{v} of 27", ha="center", fontsize=10)
    ax.set_ylim(0, 30)
    ax.set_ylabel("shapes where the joint race\nends on a cuBLASLt kernel")
    ax.set_title("Who wins the Triton + cuBLASLt race?", fontsize=10)
    save(fig, "05_transfer_ceiling.png")


def fig_noise_3090():
    """Noise floor of the measuring stick on the two GPUs (>= 50 us kernels)."""
    # HAND-COPIED from reports/week1.md 2026-11-09 "Noise floor of the 3090": range p95 over 3 rounds, >= 50 us.
    # 3080: its two complete locked-1755 sessions (cold 1.4-1.8%, hot 1.8-1.9%); 3090: one unlocked session (cold 4.26%, hot 3.93%).
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(2)
    a3080 = [(1.4, 1.8), (1.8, 1.9)]
    for k, (lo, hi) in enumerate(a3080):
        ax.bar(k - 0.2, lo, 0.18, color=BLUE)
        ax.bar(k, hi, 0.18, color=BLUE, alpha=0.6)
        ax.text(k - 0.2, lo + 0.08, f"{lo}", ha="center", fontsize=8.5)
        ax.text(k, hi + 0.08, f"{hi}", ha="center", fontsize=8.5)
    ax.bar(x + 0.22, [4.26, 3.93], 0.18, color=AQUA)
    for k, v in enumerate([4.26, 3.93]):
        ax.text(k + 0.22, v + 0.08, f"{v}", ha="center", fontsize=8.5)
    ax.axhline(2, color=INK2, ls=":", lw=1)
    ax.text(-0.48, 2.2, "2% tie margin (3080)", ha="left", fontsize=8.5, color=INK2)
    ax.set_xticks(x, ["cold L2", "hot L2"])
    ax.set_xlim(-0.5, 1.6)
    ax.set_ylim(0, 6.4)
    ax.set_ylabel("run-to-run spread, unlucky 5% (%)")
    ax.set_title("The 3090's measuring stick is about 2× wider\n(kernels of 50 µs or more)", fontsize=10)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="3080, session 1 (locked clock)"), Patch(color=BLUE, alpha=0.6, label="3080, session 2"),
                       Patch(color=AQUA, label="3090 (clock not locked)")], fontsize=8.5, loc="upper left")
    save(fig, "05_noise_3090.png")


# ------------------------------------------------------------------------------------------------ 06 shipping
def fig_bundle():
    """First-use cost with and without the prebuilt kernels, and how small the shipped archive is."""
    # HAND-COPIED from docs/kernel_bundle.md "What was built and measured" (10 held-out shapes, rank v2, k = 3 + cuBLASLt top-16,
    # empty Triton cache, one process per shape): wall per shape and Triton compile + validate.
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axs[0]
    names = ["compile every\ncandidate\n(no bundle)", "load prebuilt\nkernels\n(bundle)"]
    comp, rest = [0.480, 0.005], [0.618 - 0.480, 0.150 - 0.005]
    ax.bar(names, comp, 0.5, color=ORANGE, label="compile + check Triton candidates")
    ax.bar(names, rest, 0.5, bottom=comp, color=BLUE, label="rank, time the race, cuBLASLt")
    for i, t in enumerate([0.618, 0.150]):
        ax.text(i, t + 0.015, f"{t:.2f} s", ha="center", fontsize=10)
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("time to pick a kernel for a new shape")
    ax.set_title("First use is about 4× faster", fontsize=10)
    ax.legend(fontsize=8.5, loc="upper right")
    ax = axs[1]
    # HAND-COPIED from docs/kernel_bundle.md "Shipping size" (MiB)
    lab = ["unpacked on disk", "cubin + json files", "plain tar | xz", "zstd -19", "our archive\n(committed)"]
    val = [986, 799, 51, 25, 19.2]
    cols = [GREY, GREY, GREY, GREY, BLUE]
    y = np.arange(len(lab))[::-1]
    ax.barh(y, val, color=cols, height=0.6)
    for yi, v in zip(y, val):
        ax.text(v * 1.1, yi, f"{v:g} MiB", va="center", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(8, 4000)
    ax.set_yticks(y, lab, fontsize=9)
    ax.set_xlabel("size of the 21,187 compiled kernels (log scale)")
    ax.set_title("…and fits in one 19 MiB file", fontsize=10)
    save(fig, "06_bundle.png")


# ------------------------------------------------------------------------------------------------ 04 shapes
def fig_shapes():
    suites = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "tail"]
    allsh = []
    for s in suites:
        for sh in jload(ROOT / "suites" / "v1" / f"{s}.json")["shapes"]:
            if sh.get("out_of_range"):
                continue
            allsh.append({"suite": s, "split": sh["split"], "M": sh["M"], "N": sh["N"], "K": sh["K"]})
    df = pd.DataFrame(allsh)
    nice = {"tune": "tune (used to train)", "heldout_region": "held-out region", "heldout_random": "held-out random"}
    colr = {"tune": BLUE, "heldout_region": ORANGE, "heldout_random": MAGENTA}
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    bottom = np.zeros(len(suites))
    for sp in ["tune", "heldout_region", "heldout_random"]:
        v = np.array([((df.suite == s) & (df.split == sp)).sum() for s in suites])
        ax.bar(suites, v, bottom=bottom, color=colr[sp], label=nice[sp], width=0.65, edgecolor=BG, linewidth=1.5)
        bottom += v
    for i, b in enumerate(bottom):
        ax.text(i, b + 3, int(b), ha="center", fontsize=9)
    ax.set_ylabel("number of shapes")
    ax.set_title("Shapes per suite (shapes with a side > 16384 removed)")
    ax.legend(fontsize=8.5)
    save(fig, "04_shapes_per_suite.png")

    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, (xc, yc) in zip(axs, [("M", "K"), ("M", "N")]):
        for sp in ["tune", "heldout_region", "heldout_random"]:
            q = df[df.split == sp]
            ax.scatter(q[xc], q[yc], s=11, alpha=0.7 if sp == "tune" else 0.95, color=colr[sp], label=nice[sp], lw=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"{xc} (rows of A and C)")
        ax.set_ylabel({"K": "K (shared inner size)", "N": "N (columns of B and C)"}[yc])
    axs[0].axvspan(300, 450, color=ORANGE, alpha=0.12, lw=0)
    axs[0].axhspan(9000, 12000, color=ORANGE, alpha=0.12, lw=0)
    axs[0].text(470, 1.3, "M in 300–450 held out", fontsize=8, color=INK2)
    axs[0].text(1.2, 13000, "K in 9000–12000 held out", fontsize=8, color=INK2)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.01), ncol=3, fontsize=9, markerscale=2)
    fig.suptitle("Where the shapes sit (each dot = one matrix multiplication)", x=0.01, ha="left", fontweight="bold", fontsize=11)
    save(fig, "04_shape_scatter.png")


def main():
    for f in (fig_config_spread, fig_baseline_bars, fig_square_tflops, fig_b6, fig_noise, fig_selectors, fig_portfolio,
              fig_topk, fig_v1_v2, fig_joint, fig_cross_gpu, fig_transfer_quality, fig_transfer_ceiling, fig_noise_3090,
              fig_bundle, fig_shapes):
        try:
            f()
        except Exception as e:  # keep going; report which figure failed
            print("FAILED", f.__name__, type(e).__name__, e)


if __name__ == "__main__":
    main()
