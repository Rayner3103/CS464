"""Figures and a results overview for the current RTX 3080 state: baselines, noise, selector performance, kernel selection.

    python -m analysis.plots [--data data/<hw_id>] [--report reports] [--theme light|dark] [--only baselines,noise,selectors,selection]

Reads files that already exist (benchmark JSONL, screens, summary JSON, picks); it never touches the GPU, so it is safe
to run next to a measurement (deliberately not a `python -m tune` subcommand: that entry point imports torch and queries the device). Writes reports/figures/overview/*.png and reports/results_overview.md (every number a figure
shows is also in that file, as a table). The older `tune analyze` figures are left alone.

What each group draws, and where the data comes from:
  baselines  B1 / B3+ / Triton oracle / B5 on the square sweep and against the roofline; Triton oracle vs B3+ per regime.
             Tables: reports/baseline_table_*_lock1755_sweep*.csv (regenerated from the JSONL when missing).
  noise      across-round and cross-session spread (data/<hw>/noise.jsonl, standard-conditions sessions only), the
             allocation-placement experiment, thermal drift.
  selectors  held-out regret of every selector, per-shape CDFs, best-of-k, and the online top-3 against B3+. Aggregates come
             from reports/*_summary.json; per-shape numbers are rebuilt from the stage-2 JSONL the same way `selectors` does.
  selection  portfolio curves, parameter importance, which config wins where, oracle vs picked parameters, online search cost.

Style: one palette (the validated reference categorical order, fixed slots per entity), sequential blue for ordered
quantities, recessive grid, direct labels where there is room, and marker / line-style changes so identity is never colour alone.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from analysis.portfolio import load_screen, _split_info  # noqa: E402
from tune.records import read_records  # noqa: E402

SMALL_US = 50.0
EPS_BIG, EPS_SMALL = 0.02, 0.05  # tie margins at >= 50 us and below (AGENTS.md section 6)
LT_FAMILIES = {"cublaslt", "cublaslt_deep"}
REGIMES = ["S1 square", "S2 decode (M<=16)", "S2 small batch (32<=M<=128)", "S2 prefill (M>=512)", "S3 skinny/tall",
           "S4 large-K", "S5 unaligned", "S6 wave-boundary", "S7 random"]
KEY_RE = re.compile(r"bm(\d+)_bn(\d+)_bk(\d+)_w(\d+)_s(\d+)_g(\d+)_sk(\d+)")
PARAMS = ("BLOCK_M", "BLOCK_N", "BLOCK_K", "num_warps", "num_stages", "GROUP_M", "SPLIT_K")

# Result files behind each figure group (all under data/<hw_id>/).
BASE_COLD, BASE_GRAPH = "results_lock1755_sweep.jsonl", "results_lock1755_sweep_graph.jsonl"
SEL_COLD, SEL_GRAPH = "results_lock1755_selectors3.jsonl", "results_lock1755_selectors3_graph.jsonl"
TOPK_COLD, TOPK_GRAPH = "results_lock1755_topki.jsonl", "results_lock1755_topki_graph.jsonl"
ONLINE = {"plain": "online_lgbm_rank.jsonl", "diverse": "online_lgbm_rank_diverse.jsonl", "inner": "online_lgbm_rank_inner.jsonl"}

# ----------------------------------------------------------------------------------------------- theme
THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e", "grid": "#e4e3df", "rule": "#9a9993",
              "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
              "seq": ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"],
              "cmap": ["#fcfcfb", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"]},
    "dark": {"surface": "#1a1a19", "text": "#f0efec", "text2": "#c3c2b7", "grid": "#33332f", "rule": "#8a8985",
             "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
             "seq": ["#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4"],
             "cmap": ["#1a1a19", "#184f95", "#256abf", "#3987e5", "#9ec5f4"]},
}


class Theme:
    def __init__(self, name: str):
        self.name = name
        t = THEMES[name]
        self.__dict__.update(t)
        plt.rcdefaults()
        plt.rcParams.update({
            "figure.facecolor": self.surface, "axes.facecolor": self.surface, "savefig.facecolor": self.surface,
            "axes.edgecolor": self.rule, "axes.labelcolor": self.text2, "text.color": self.text,
            "xtick.color": self.text2, "ytick.color": self.text2, "axes.titlecolor": self.text,
            "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": self.grid,
            "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.titlelocation": "left", "axes.titlesize": 11,
            "axes.titleweight": "regular", "axes.titlepad": 8, "font.size": 9.5, "lines.linewidth": 2,
            "lines.markersize": 6, "legend.frameon": False, "legend.fontsize": 9, "savefig.dpi": 150,
            "figure.dpi": 100,
        })
        self.cmap_seq = LinearSegmentedColormap.from_list("seq", t["cmap"])

    def ink(self, color: str) -> str:
        """Readable text colour on top of a heatmap cell of this colour."""
        r, g, b = (int(color[i:i + 2], 16) / 255 for i in (1, 3, 5))
        return "#0b0b0b" if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.5 else "#f0efec"


def _save(fig, out: Path, name: str) -> str:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return name


def _title(fig, th: Theme, title: str, sub: str = ""):
    fig.text(0.012, 0.985, title, ha="left", va="top", fontsize=13, color=th.text, fontweight="bold")
    if sub:
        fig.text(0.012, 0.985 - 0.045 * (5.0 / fig.get_figheight()), sub, ha="left", va="top", fontsize=9, color=th.text2)


def _quiet_minor(ax):
    """Log axes with hand-set ticks: drop the minor tick labels (they print as 3x10^-1 over the real ones)."""
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def _pct_axis(ax, axis: str = "x"):
    f = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}%")
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(f)


def _read(path: Path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def _tag(s: dict) -> str:
    return f"{s['M']}x{s['N']}x{s['K']}"


def _parse(key: str) -> dict:
    m = KEY_RE.fullmatch(key)
    return dict(zip(PARAMS, map(int, m.groups())))


def _md_table(df: pd.DataFrame, fmt: str = "{:.2f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in row) + " |")
    return "\n".join(lines)


class Ctx:
    def __init__(self, data: Path, report: Path, theme: Theme):
        self.data, self.report, self.th = Path(data), Path(report), theme
        self.out = self.report / "figures" / "overview"
        self.md: dict[str, list[str]] = {}   # section -> markdown blocks
        self.figs: dict[str, list[tuple[str, str]]] = {}  # section -> (file, caption)

    def add(self, section: str, fig_name: str, caption: str):
        self.figs.setdefault(section, []).append((fig_name, caption))

    def text(self, section: str, block: str):
        self.md.setdefault(section, []).append(block)


# ----------------------------------------------------------------------------------------------- baselines
def _baseline_table(ctx: Ctx, results: str, cm: str) -> pd.DataFrame | None:
    """The per-shape table `tune analyze` writes (B1, B2, B3, B3+, B5, oracle, ...); regenerated from the JSONL if missing."""
    stem = Path(results).stem.removeprefix("results_")
    path = ctx.report / f"baseline_table_{cm}_{stem}.csv"
    if not path.exists() and (ctx.data / results).exists():
        from analysis.report import baseline_analysis
        baseline_analysis(ctx.data, ctx.report / "figures", results)
    if not path.exists():
        return None
    return pd.read_csv(path, usecols=lambda c: not c.startswith("ours:"))


def _tflops(t: pd.DataFrame, col: str) -> pd.Series:
    return 2.0 * t["M"] * t["N"] * t["K"] / t[col] / 1e6


ENT = {"B1 torch.matmul": ("B1 torch.matmul", 0, "o"), "B3+ cublasLt deep": ("B3+ cuBLASLt (searched)", 1, "s"),
       "oracle(ours)": ("Triton oracle", 2, "D"), "B5 triton tutorial": ("B5 Triton tutorial", 3, "^")}


def fig_baselines(ctx: Ctx):
    th, S = ctx.th, ctx.th.series
    cold = _baseline_table(ctx, BASE_COLD, "cold")
    graph = _baseline_table(ctx, BASE_GRAPH, "hot")
    roof = _read(ctx.data / "roofline.json")
    if cold is None:
        ctx.text("baselines", "_no baseline benchmark found_")
        return
    cold = cold.dropna(subset=["B3+ cublasLt deep", "oracle(ours)"])

    # --- 1. square sweep: absolute throughput and time relative to B3+
    sq = cold[(cold["regime"] == "S1 square") & (cold["M"] == cold["N"]) & (cold["N"] == cold["K"])].sort_values("M")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    # S1 holds powers of two (aligned) plus 1000/2000/3000/6000 and +-1 probes of the powers (unaligned strides):
    # lines join the powers of two only, the other sizes are hollow markers at their own x.
    pw = (sq["M"] & (sq["M"] - 1)) == 0
    for col, (lab, slot, mk) in ENT.items():
        for a_, y_ in ((ax[0], _tflops(sq, col)), (ax[1], sq[col] / sq["B3+ cublasLt deep"])):
            a_.plot(sq["M"][pw], y_[pw], marker=mk, color=S[slot], label=lab, ms=5, lw=1.8)
            a_.scatter(sq["M"][~pw], y_[~pw], marker=mk, facecolor=th.surface, edgecolor=S[slot], s=30, lw=1.3, zorder=3)
    if roof:
        peak = roof["compute"]["fp16_f32acc_tflops_measured"]
        ax[0].axhline(peak, color=th.rule, lw=1.2, ls="--")
        ax[0].text(sq["M"].min(), peak * 1.03, f"measured peak {peak:.1f} TFLOP/s (locked 1755 MHz)", color=th.text2, fontsize=8.5, va="bottom")
    ax[1].axhline(1.0, color=th.rule, lw=1.2)
    ax[1].axhspan(1 - EPS_BIG, 1 + EPS_BIG, color=th.grid, alpha=0.7, lw=0)
    for a in ax:
        a.set_xscale("log", base=2)
        a.set_xlabel("M = N = K")
    ax[0].set_ylabel("achieved TFLOP/s (cold L2)")
    ax[0].set_title("Throughput on the square sweep")
    ax[1].set_yscale("log")
    ax[1].set_ylabel("time relative to B3+ (lower is faster)")
    ax[1].set_title("Time relative to the searched cuBLASLt (shaded: 2% tie band)")
    ax[1].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}x"))
    ax[1].set_yticks([0.9, 1, 1.25, 1.5, 2, 3])
    h_, l_ = ax[0].get_legend_handles_labels()
    h_.append(Line2D([], [], marker="o", ls="", mfc=th.surface, mec=th.text2, label="hollow: 1000/2000/3000/6000, 2^n +- 1"))
    ax[0].legend(handles=h_, loc="upper left", bbox_to_anchor=(0.0, 0.88))
    for a in ax:
        _quiet_minor(a)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _title(fig, th, "Baselines on the square sweep (S1, RTX 3080)", f"{len(sq)} shapes, median of 3 input placements; B3+ = best of cuBLASLt heuristic top-N and a 20 s/shape search")
    ctx.add("baselines", _save(fig, ctx.out, "baselines_square.png"), "Square sweep: TFLOP/s and time relative to B3+.")

    # --- 2. Triton oracle vs B3+ per regime
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), sharey=True)
    rng = np.random.default_rng(464)
    stats = []
    for ax, (title, t, mode) in zip(axes, [("cold L2, per launch", cold, "cold"),
                                          ("hot L2, CUDA-graph amortized (shapes < 50 us)", graph, "graph")]):
        if t is None:
            ax.set_visible(False)
            continue
        t = t.dropna(subset=["B3+ cublasLt deep", "oracle(ours)"]).copy()
        t["ratio"] = t["B3+ cublasLt deep"] / t["oracle(ours)"]
        big = t["B3+ cublasLt deep"] >= SMALL_US
        for yi, reg in enumerate(REGIMES):
            g = t[t["regime"] == reg]
            if g.empty:
                continue
            y = yi + rng.uniform(-0.22, 0.22, len(g))
            ax.scatter(g["ratio"][big[g.index]], y[big[g.index].to_numpy()], s=26, color=S[0], alpha=0.85, lw=0, zorder=3, label="B3+ >= 50 us" if yi == 0 else None)
            ax.scatter(g["ratio"][~big[g.index]], y[(~big[g.index]).to_numpy()], s=26, color=S[1], alpha=0.85, lw=0, marker="s", zorder=3, label="B3+ < 50 us" if yi == 0 else None)
            med = g["ratio"].median()
            ax.plot([med, med], [yi - 0.38, yi + 0.38], color=th.text, lw=2.2, solid_capstyle="butt", zorder=4)
            stats.append({"mode": mode, "regime": reg, "n": len(g), "median": med,
                          "oracle > B3+ by 5%": int((g["ratio"] > 1.05).sum()), "within 5%": int((abs(g["ratio"] - 1) <= 0.05).sum()),
                          "oracle < B3+ by 5%": int((g["ratio"] < 1 / 1.05).sum())})
        ax.axvline(1.0, color=th.rule, lw=1.2)
        ax.axvspan(1 / (1 + EPS_SMALL), 1 + EPS_SMALL, color=th.grid, alpha=0.7, lw=0, zorder=0)
        ax.set_xscale("log")
        ax.set_xticks([0.25, 0.5, 0.8, 1, 1.25, 2, 4])
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}x"))
        _quiet_minor(ax)
        ax.grid(axis="y", visible=False)
        ax.set_title(title)
        ax.set_xlabel("B3+ time / Triton-oracle time  (> 1: the Triton family is faster)")
        ax.legend(loc="lower right")
    axes[0].set_yticks(range(len(REGIMES)))
    axes[0].set_yticklabels(REGIMES)
    axes[0].set_ylim(len(REGIMES) - 0.4, -0.7)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    n_all = len(cold)
    _title(fig, th, "Per-shape Triton oracle vs the searched cuBLASLt (B3+)", f"each dot is a shape; black tick = regime median; shaded = 5% tie band; {n_all} cold shapes")
    ctx.add("baselines", _save(fig, ctx.out, "baselines_vs_b3plus.png"), "Triton oracle against B3+ by regime (cold and graph-amortized).")
    st = pd.DataFrame(stats)
    ctx.text("baselines", "### Triton oracle vs B3+ (B3+ time / oracle time)\n\n" + _md_table(st, "{:.3f}") + "\n")

    # --- 3. roofline
    if roof:
        t = cold.copy()
        ai = 2.0 * t["M"] * t["N"] * t["K"] / (2.0 * (t["M"] * t["K"] + t["K"] * t["N"] + t["M"] * t["N"]))
        peak_s, peak_m = roof["spec"]["fp16_f32acc_tflops_at_boost"], roof["compute"]["fp16_f32acc_tflops_measured"]
        bw_s, bw_m = roof["spec"]["dram_gbps_spec"], roof["bandwidth"]["dram_gbps_measured"]
        x = np.logspace(np.log10(max(ai.min() / 2, 0.3)), np.log10(ai.max() * 2), 200)
        fig, ax = plt.subplots(figsize=(9.2, 5.8))
        ax.plot(x, np.minimum(peak_s, bw_s * x / 1e3), color=th.rule, lw=1.3, label=f"spec ({peak_s:.0f} TFLOP/s, {bw_s:.0f} GB/s)")
        ax.plot(x, np.minimum(peak_m, bw_m * x / 1e3), color=th.text2, lw=1.3, ls="--", label=f"measured ({peak_m:.1f} TFLOP/s, {bw_m:.0f} GB/s)")
        ax.scatter(ai, _tflops(t, "B3+ cublasLt deep"), s=22, color=S[1], marker="s", alpha=0.8, lw=0, label="B3+ cuBLASLt (searched)")
        ax.scatter(ai, _tflops(t, "oracle(ours)"), s=26, color=S[2], marker="D", alpha=0.7, lw=0, label="Triton oracle")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-3)
        ax.set_xlabel("arithmetic intensity (FLOP per compulsory DRAM byte)")
        ax.set_ylabel("achieved TFLOP/s (cold L2)")
        ax.legend(loc="lower right")
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        _title(fig, th, "Roofline: fp16 GEMM, fp32 accumulate", f"{len(t)} shapes from all suites; points left of the ridge are memory-bound")
        ctx.add("baselines", _save(fig, ctx.out, "baselines_roofline.png"), "Roofline with B3+ and the Triton oracle.")
        ctx.text("baselines", f"Measured roofline (`data/<hw>/roofline.json`): {peak_m:.1f} TFLOP/s fp16 tensor (spec at boost {peak_s:.1f}), "
                              f"{bw_m:.0f} GB/s DRAM (spec {bw_s:.0f}). Best B3+ in the table: {_tflops(t, 'B3+ cublasLt deep').max():.1f} TFLOP/s; "
                              f"best Triton oracle: {_tflops(t, 'oracle(ours)').max():.1f} TFLOP/s.\n")


# ----------------------------------------------------------------------------------------------- noise
def _locked_sessions(data: Path, df: pd.DataFrame) -> dict[str, str]:
    """session -> 'locked@MHz' for sessions run under the standard clock lock (rows predating provenance use the session env)."""
    out = {}
    for sess, d in df.groupby("session_id"):
        state = d["clock_lock_state"].dropna().iloc[0] if d["clock_lock_state"].notna().any() else None
        mhz = d["clock_lock_mhz"].dropna().iloc[0] if d["clock_lock_mhz"].notna().any() else None
        env = _read(data / "env" / f"{sess}.json")
        if state is None and env:
            cl = env.get("clock_lock", {})
            state, mhz = cl.get("state"), cl.get("lock_mhz")
        if state == "locked" and str(mhz) == "1755" and d["round"].nunique() >= 3:
            out[sess] = f"locked@{mhz}"
    return out


def fig_noise(ctx: Ctx):
    th, S = ctx.th, ctx.th.series
    rows = read_records(ctx.data / "noise.jsonl")
    if rows:
        df = pd.DataFrame(rows)
        sessions = _locked_sessions(ctx.data, df)
        df = df[df["session_id"].isin(sessions)].copy()
        df["shape"] = df["M"].astype(str) + "x" + df["N"].astype(str) + "x" + df["K"].astype(str)
        g = df.groupby(["session_id", "shape", "kernel_family", "cache_mode"])["time_median_us"]
        st = g.agg(median_us="median", rng=lambda s: (s.max() - s.min()) / s.median()).reset_index()
        order = sorted(sessions)
        marker = {s: m for s, m in zip(order, ["o", "^", "s"])}
        FLOOR = 1e-4  # ranges of exactly 0 (timer grid) are drawn on this line
        modes = {"cold": S[0], "hot": S[1]}
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.7))
        for s in order:
            for cm, col in modes.items():
                d = st[(st["session_id"] == s) & (st["cache_mode"] == cm)]
                ax[0].scatter(d["median_us"], 100 * np.maximum(d["rng"], FLOOR), s=20, marker=marker[s], color=col, alpha=0.7, lw=0)
        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        ax[0].set_ylim(FLOOR * 100 * 0.7, 20)
        ax[0].set_yticks([0.01, 0.1, 1, 10])
        _quiet_minor(ax[0])
        ax[0].axhline(100 * EPS_BIG, color=th.rule, ls="--", lw=1)
        ax[0].axvline(SMALL_US, color=th.rule, ls=":", lw=1)
        ax[0].set_xlabel("median kernel time (us)")
        ax[0].set_ylabel("range of 3 round medians / median (0 drawn at 0.01%)")
        _pct_axis(ax[0], "y")
        ax[0].set_title("Across-round spread within a session")
        h = [Line2D([], [], marker="o", ls="", color=c, label=f"{cm} L2") for cm, c in modes.items()]
        h += [Line2D([], [], marker=marker[s], ls="", color=th.text2, label=s.replace("noise-", "")[-28:]) for s in order]
        h += [Line2D([], [], ls="--", color=th.rule, label="2% tie margin"), Line2D([], [], ls=":", color=th.rule, label="50 us")]
        ax[0].legend(handles=h, loc="upper right", fontsize=8, ncol=2, columnspacing=1.0)

        # ECDF of the within-session range (>= 50 us) per mode, and the cross-session difference
        big = st[st["median_us"] >= SMALL_US]
        summary = []
        for cm, col in modes.items():
            for s in order:
                d = big[(big["session_id"] == s) & (big["cache_mode"] == cm)]["rng"].sort_values().to_numpy()
                if len(d):
                    ax[1].step(100 * d, np.arange(1, len(d) + 1) / len(d), where="post", color=col, ls="-" if s == order[0] else "--", lw=1.8)
                    summary.append({"quantity": "across-round range", "session": s, "mode": cm, "n": len(d),
                                    "p50": np.median(d), "p95": np.percentile(d, 95), "p99": np.percentile(d, 99), "max": d.max()})
        if len(order) >= 2:
            piv = st.pivot_table(index=["shape", "kernel_family", "cache_mode"], columns="session_id", values="median_us").dropna()
            piv = piv[piv[order].min(axis=1) >= SMALL_US]
            a, b = piv[order[0]], piv[order[-1]]
            diff = (abs(a - b) / ((a + b) / 2)).rename("diff").reset_index()
            for cm, col in modes.items():
                d = diff[diff["cache_mode"] == cm]["diff"].sort_values().to_numpy()
                if len(d):
                    ax[2].step(100 * d, np.arange(1, len(d) + 1) / len(d), where="post", color=col, lw=1.8)
                    summary.append({"quantity": "cross-session |difference|", "session": f"{order[0][-12:]} vs {order[-1][-12:]}",
                                    "mode": cm, "n": len(d), "p50": np.median(d), "p95": np.percentile(d, 95), "p99": np.percentile(d, 99), "max": d.max()})
        for a_, t_ in ((ax[1], "Across-round range, kernels >= 50 us"), (ax[2], "Cross-session difference, kernels >= 50 us")):
            a_.axvline(100 * EPS_BIG, color=th.rule, ls="--", lw=1)
            a_.set_xlim(0, 6)
            a_.set_ylim(0, 1.02)
            a_.set_xlabel("relative spread")
            a_.set_ylabel("fraction of (shape, kernel) pairs")
            _pct_axis(a_, "x")
            a_.set_title(t_)
        ax[1].legend(handles=[Line2D([], [], color=c, label=f"{cm} L2") for cm, c in modes.items()] +
                     [Line2D([], [], color=th.text2, ls="-", label="session 1"), Line2D([], [], color=th.text2, ls="--", label="session 2")],
                     loc="lower right", fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        _title(fig, th, "Measurement noise floor (30 noise shapes, standard conditions)",
               f"only sessions under the 1755 MHz lock with 3 rounds: {', '.join(s.replace('noise-', '') for s in order)}")
        ctx.add("noise", _save(fig, ctx.out, "noise_floor.png"), "Noise floor: scatter, within-session ECDF, cross-session ECDF.")
        ctx.text("noise", "### Noise floor, kernels >= 50 us (relative spread, 0.01 = 1%)\n\n" + _md_table(pd.DataFrame(summary), "{:.4f}") + "\n")
        skipped = sorted(set(pd.DataFrame(rows)["session_id"]) - set(order))
        if skipped:
            ctx.text("noise", f"Sessions excluded (unlocked, different driver, or fewer than 3 rounds): {', '.join(skipped)}.\n")

    # allocation placement
    av = _read(ctx.data / "alloc_variance.json")
    if av:
        items = sorted(av.items(), key=lambda kv: kv[1]["across_alloc_range"])
        fig, ax = plt.subplots(figsize=(8.6, 0.55 * len(items) + 2.2))
        y = np.arange(len(items))
        ax.barh(y + 0.19, [100 * v["within_alloc_range"] for _, v in items], height=0.34, color=S[0], label="repeat within one allocation")
        ax.barh(y - 0.19, [100 * v["across_alloc_range"] for _, v in items], height=0.34, color=S[1], label="across allocation placements")
        for yi, (_, v) in zip(y, items):
            for off, key in ((0.19, "within_alloc_range"), (-0.19, "across_alloc_range")):
                ax.text(100 * v[key] + 0.03, yi + off, f"{100 * v[key]:.2f}%", va="center", fontsize=8.5, color=th.text2)
        ax.set_xlim(0, 100 * max(v["across_alloc_range"] for _, v in items) * 1.15)
        ax.set_yticks(y)
        ax.set_yticklabels([k for k, _ in items])
        ax.set_xlabel("range of medians / median")
        _pct_axis(ax)
        ax.grid(axis="y", visible=False)
        ax.legend(loc="lower right")
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        _title(fig, th, "Allocation placement is the dominant noise source", "same kernel, same shape, inputs allocated at different address offsets (memory-bound shapes move most)")
        ctx.add("noise", _save(fig, ctx.out, "noise_alloc_placement.png"), "Within-allocation vs across-allocation spread.")
        ctx.text("noise", "### Allocation placement (range of medians / median)\n\n" + _md_table(pd.DataFrame(
            [{"shape": k, "within allocation": v["within_alloc_range"], "across allocations": v["across_alloc_range"]} for k, v in items[::-1]]), "{:.4f}") + "\n")

    # thermal drift
    files = sorted(ctx.data.glob("drift-*.csv"))
    if files:
        d = pd.read_csv(files[-1])
        cap = (d["throttle_reasons"].fillna(0).astype(int) & 0x4) != 0
        fig, ax = plt.subplots(4, 1, figsize=(10, 7.4), sharex=True)
        for a, col, lab, c, scale in ((ax[0], "median_us", "8192^3 time (ms)", S[0], 1e-3), (ax[1], "clock_mhz", "SM clock (MHz)", S[2], 1),
                                      (ax[2], "temp_c", "temperature (C)", S[1], 1), (ax[3], "power_w", "power (W)", S[3], 1)):
            a.plot(d["t_s"], d[col] * scale, color=c, lw=1.4)
            a.set_ylabel(lab)
        t = d["t_s"].to_numpy()
        dt = np.median(np.diff(t)) if len(t) > 1 else 1
        if cap.mean() < 0.9:  # shading a run that is power-capped throughout says nothing; the subtitle carries it
            for a in ax:
                for x in t[cap.to_numpy()]:
                    a.axvspan(x - dt / 2, x + dt / 2, color=th.rule, alpha=0.18, lw=0)
        ax[2].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax[3].set_xlabel("time (s)")
        first, last = d[d["t_s"] <= 60]["median_us"].median(), d[d["t_s"] >= d["t_s"].max() - 60]["median_us"].median()
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        _title(fig, th, "Thermal drift under continuous 8192^3 fp16 GEMM",
               f"{d['t_s'].max() / 60:.0f} min; time {first / 1e3:.2f} -> {last / 1e3:.2f} ms ({100 * (last / first - 1):+.1f}%); power-capped in {cap.mean():.0%} of intervals"
               + ("; clock above 1755 MHz: run predates the lock" if d["clock_mhz"].max() > 1765 else ""))
        ctx.add("noise", _save(fig, ctx.out, "noise_thermal_drift.png"), "Thermal drift, clock, temperature and power.")
        ctx.text("noise", f"Thermal drift (`{files[-1].name}`): 8192^3 median {first:.0f} -> {last:.0f} us ({100 * (last / first - 1):+.2f}%), "
                          f"clock {d['clock_mhz'].min():.0f}-{d['clock_mhz'].max():.0f} MHz, temp {d['temp_c'].iloc[0]:.0f}->{d['temp_c'].max():.0f} C, "
                          f"median power {d['power_w'].median():.0f} W, power-capped in {cap.mean():.0%} of intervals."
                          + (" **The clock moved above 1755 MHz, so this run predates the clock lock (not standard conditions).**" if d["clock_mhz"].max() > 1765 else "") + "\n")


# ----------------------------------------------------------------------------------------------- selectors
PRETTY = {"b6": "B6 best single config", "rule": "hand-written rule", "nn1": "1-NN on (M, N, K)", "knn5": "5-NN on (M, N, K)",
          "nn1f": "1-NN + GPU-fill features", "knnf": "k-NN + GPU-fill features", "knnc": "k-NN per-candidate features",
          "dt_cls": "decision tree, classify", "svm_cls": "linear SVM, classify", "lgbm_cls": "LightGBM, classify",
          "rf_cls": "random forest, classify", "logreg_cls": "logistic regression, classify", "dt_rank": "decision tree, rank",
          "svm_rank": "linear SVM, rank", "lgbm_rank": "LightGBM, rank (top-1)", "rf_rank": "random forest, rank",
          "logreg_rank": "logistic regression, rank"}
FAMILY = ["static / k-NN", "learned, single pick", "LightGBM ranker, top-k + online search"]


def selector_rows(report: Path, mode: str = "cold", group: str = "all") -> list[tuple[str, int, dict]]:
    """(label, family index, regret stats) for every selector scored in stage 2. `group` is the stats key (all, >= 50 us, ...)."""
    s = _read(report / "selector_baselines_summary.json")
    l = _read(report / "learned_selectors_summary.json")
    t = _read(report / "learned_topk_summary.json")
    rows: list[tuple[str, int, dict]] = []
    key = f"stage2_{mode}"
    if s and s.get(key):
        g = group if mode == "cold" else "all"
        rows += [(PRETTY[n], 0, s[key]["regret"][n][g]) for n in ("b6", "rule", "nn1", "knn5", "nn1f", "knnf", "knnc") if n in s[key]["regret"]]
    if l and l.get(key):
        g = group if mode == "cold" else "all"
        rows += [(PRETTY[n], 1, l[key]["regret"][n][g]) for n in PRETTY if n in l[key]["regret"] and n.endswith(("_cls", "_rank")) and n != "lgbm_rank"]
    if t and t.get(key):
        g = group if mode == "cold" else "all (< 50 us shapes)"
        for lab, sect, k in (("LightGBM top-1", "perfect", "plain@1"), ("plain online top-3", "online", "plain@3"),
                             ("diverse online top-3 (bar)", "online", "diverse@3"), ("diverse online top-5", "online", "diverse@5")):
            if k in t[key][sect] and g in t[key][sect][k]:
                rows.append((lab, 2, t[key][sect][k][g]))
    return rows


def _stage2(rows: list[dict], hshapes: list[dict], held_keys: dict[str, list[str | None]]) -> dict[str, np.ndarray]:
    """Per held-out shape from one benchmark session: Triton oracle (fastest valid Triton row), B3+ (fastest cuBLASLt row),
    B1, and each selector's pick time (same rules as analysis.selectors.stage2; missing = nan)."""
    tri: dict[tuple, dict[str, float]] = {}
    lt: dict[tuple, float] = {}
    b1: dict[tuple, float] = {}
    for r in rows:
        if not (r.get("valid") and r.get("time_median_us")):
            continue
        s, t = (r["M"], r["N"], r["K"]), r["time_median_us"]
        fam = r.get("kernel_family", "")
        if fam.startswith("triton_nn"):
            d = tri.setdefault(s, {})
            d[r["config_key"]] = min(d.get(r["config_key"], np.inf), t)
        elif fam in LT_FAMILIES:
            lt[s] = min(lt.get(s, np.inf), t)
        elif fam == "torch_matmul":
            b1[s] = min(b1.get(s, np.inf), t)
    sk = [(s["M"], s["N"], s["K"]) for s in hshapes]
    out = {"oracle": np.array([min(tri[s].values()) if s in tri else np.nan for s in sk]),
           "b3plus": np.array([lt.get(s, np.nan) for s in sk]), "b1": np.array([b1.get(s, np.nan) for s in sk])}
    for name, ks in held_keys.items():
        out[name] = np.array([tri.get(s, {}).get(k, np.nan) if k else np.nan for s, k in zip(sk, ks)])
    return out


def _heldout_shapes(data: Path) -> list[dict]:
    return _read(data / "sweep_screen_heldout" / "subset.json")["shapes"]


def _online_keys(data: Path, hshapes: list[dict], order: str, k: int) -> list[str | None]:
    path = data / ONLINE[order]
    by = {}
    for r in read_records(path) if path.exists() else []:
        if r.get("k") == 5 and r.get("order", "plain") == order and r.get("choice"):
            by[(r["M"], r["N"], r["K"])] = r  # later rows win
    return [by.get((s["M"], s["N"], s["K"]), {}).get("choice", {}).get(str(k), {}).get("key") for s in hshapes]


def _rows(path: Path) -> list[dict]:
    return read_records(path) if path.exists() else []


def selector_arrays(ctx: Ctx) -> dict[str, dict[str, np.ndarray]]:
    """Per-shape stage-2 arrays for the selectors the CDF / B3+ figures use, per session (cold and graph)."""
    data = ctx.data
    hs = _heldout_shapes(data)
    tags = [_tag(s) for s in hs]
    sel = _read(data / "selector_picks.json")["by_selector"]
    tk = _read(data / "learned_topk_picks.json")["by_selector"]
    out: dict[str, dict[str, np.ndarray]] = {}
    for mode, f_sel, f_topk in (("cold", SEL_COLD, TOPK_COLD), ("graph", SEL_GRAPH, TOPK_GRAPH)):
        a = _stage2(_rows(data / f_sel), hs, {n: [sel[n].get(t) for t in tags] for n in ("b6", "rule", "knnf")}) if (data / f_sel).exists() else {}
        held = {"knnf": [sel["knnf"].get(t) for t in tags], "lgbm@1": [tk["lgbm@1"].get(t) for t in tags],
                "online-diverse@3": _online_keys(data, hs, "diverse", 3), "online-plain@3": _online_keys(data, hs, "plain", 3)}
        b = _stage2(_rows(data / f_topk), hs, held) if (data / f_topk).exists() else {}
        out[mode] = {"sel": a, "topk": b}
    return out


def fig_selectors(ctx: Ctx):
    th, S = ctx.th, ctx.th.series
    rows = selector_rows(ctx.report, "cold")
    if rows:
        fig, ax = plt.subplots(figsize=(10.5, 0.36 * len(rows) + 2.6))
        # best (lowest p95) on top inside each family
        order = sorted(range(len(rows)), key=lambda i: (rows[i][1], rows[i][2]["p95"]))
        rows = [rows[i] for i in order]
        fam_col = {0: S[0], 1: S[1], 2: S[2]}
        for yi, (lab, fam, d) in enumerate(rows):
            lo, mid, hi = (100 * max(d[k], 0.003) for k in ("p50", "mean", "p95"))
            ax.plot([lo, hi], [yi, yi], color=fam_col[fam], lw=2, alpha=0.5, solid_capstyle="round")
            ax.scatter([lo], [yi], s=46, color=fam_col[fam], zorder=3, lw=0)
            ax.scatter([hi], [yi], s=46, facecolor=th.surface, edgecolor=fam_col[fam], lw=1.8, zorder=3)
            ax.scatter([mid], [yi], s=34, marker="D", color=fam_col[fam], zorder=3, lw=0.8, edgecolor=th.surface)
        ax.set_xscale("log")
        ax.set_xlim(0.3, 1000)
        ax.set_xticks([0.5, 1, 2, 5, 10, 25, 100, 500])
        _pct_axis(ax)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows])
        ax.set_ylim(len(rows) - 0.4, -0.8)
        ax.axvline(100 * EPS_BIG, color=th.rule, ls="--", lw=1)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("regret vs the per-shape Triton oracle  (t / oracle - 1; log scale)")
        h = [Line2D([], [], marker="o", ls="", color=th.text2, label="median"), Line2D([], [], marker="D", ls="", color=th.text2, label="mean"),
             Line2D([], [], marker="o", ls="", mfc=th.surface, mec=th.text2, mew=1.8, label="p95")]
        h += [Line2D([], [], marker="s", ls="", color=fam_col[i], label=FAMILY[i]) for i in range(3)]
        ax.legend(handles=h, loc="lower right", ncol=1)
        fig.tight_layout(rect=(0, 0, 1, 0.91))
        _title(fig, th, "Selector regret on the 86 held-out shapes (cold L2, stage-2 re-timing)",
               "no held-out shape was used to fit anything; dashed line = 2% tie margin; each selector is scored against its own session's oracle")
        ctx.add("selectors", _save(fig, ctx.out, "selectors_heldout_regret.png"), "Median / mean / p95 regret of every selector on held-out shapes (p99 is in the table below).")
        df = pd.DataFrame([{"selector": r[0], "family": FAMILY[r[1]], "median": r[2]["p50"], "mean": r[2]["mean"], "p95": r[2]["p95"],
                            "p99": r[2].get("p99", float("nan")), "max": r[2]["max"], "within 5%": r[2]["frac_le_5pct"]} for r in rows])
        ctx.text("selectors", "### Held-out regret, cold L2, all 86 shapes (t / oracle - 1)\n\n" + _md_table(df, "{:.3f}") + "\n")
        for grp, mode in ((">= 50 us", "cold"), ("< 50 us", "cold"), ("all", "graph")):
            rr = selector_rows(ctx.report, mode, grp)
            if rr:
                sub = pd.DataFrame([{"selector": r[0], "median": r[2]["p50"], "mean": r[2]["mean"], "p95": r[2]["p95"], "p99": r[2].get("p99", float("nan")), "n": r[2]["n"]} for r in rr
                                    if r[0] in ("B6 best single config", "k-NN + GPU-fill features", "LightGBM top-1", "plain online top-3", "diverse online top-3 (bar)")])
                ctx.text("selectors", f"### {'graph-amortized, shapes < 50 us' if mode == 'graph' else 'cold, ' + grp}\n\n" + _md_table(sub, "{:.3f}") + "\n")

    # per-shape: CDF of slowdown, and vs B3+
    try:
        arr = selector_arrays(ctx)
    except (FileNotFoundError, KeyError, TypeError) as e:
        ctx.text("selectors", f"_per-shape figures skipped: {e!r}_\n")
        return
    cold = arr["cold"]
    series = [("B6 best single config", cold["sel"], "b6", 0, "-"), ("hand-written rule", cold["sel"], "rule", 1, "--"),
              ("k-NN + fill features", cold["topk"], "knnf", 2, "-"), ("LightGBM top-1", cold["topk"], "lgbm@1", 3, "--"),
              ("diverse online top-3", cold["topk"], "online-diverse@3", 4, "-")]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for ai, (xmax, ttl, sub) in enumerate(((16, "All held-out shapes", None), (1.5, "Zoom: the common case", None))):
        for lab, st, name, slot, ls in series:
            if name not in st:
                continue
            r = (st[name] / st["oracle"])
            r = np.sort(r[np.isfinite(r)])
            ax[ai].step(r, np.arange(1, len(r) + 1) / len(r), where="post", color=S[slot], ls=ls, lw=2, label=lab)
        ax[ai].set_xscale("log")
        ax[ai].set_xlim(0.995, xmax)
        ax[ai].set_ylim(0, 1.02)
        ax[ai].axvline(1 + EPS_BIG, color=th.rule, ls=":", lw=1)
        ax[ai].set_xlabel("slowdown vs the per-shape Triton oracle (x)")
        ax[ai].set_ylabel("fraction of shapes at or below")
        ax[ai].set_title(ttl)
    ax[0].set_xticks([1, 1.1, 1.25, 1.5, 2, 4, 8, 16])
    ax[1].set_xticks([1, 1.02, 1.05, 1.1, 1.25, 1.5])
    for a in ax:
        a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}x"))
        _quiet_minor(a)
    ax[0].legend(loc="lower right")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _title(fig, th, "How often each selector lands near the oracle (held-out, cold L2)",
           "step curves; further up and left is better; dotted = 2% tie margin; B6 and rule from the selectors3 session, the others from topki")
    ctx.add("selectors", _save(fig, ctx.out, "selectors_regret_cdf.png"), "Per-shape slowdown CDF for five selectors.")

    # best-of-k curves (online)
    t = _read(ctx.report / "learned_topk_summary.json")
    if t and t.get("stage2_cold"):
        on, refs = t["stage2_cold"]["online"], t["stage2_cold"].get("refs", {})
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
        for si, (stat, ttl) in enumerate((("p50", "Median regret"), ("p95", "p95 regret"))):
            for slot, (v, lab, ls) in enumerate((("plain", "plain order", "--"), ("diverse", "diverse order (rule fit on tune CV)", "-"), ("inner", "inner pass (diagnostic)", ":"))):
                ks = [k for k in range(1, 6) if f"{v}@{k}" in on]
                ax[si].plot(ks, [100 * on[f"{v}@{k}"]["all"][stat] for k in ks], marker="o", ms=5, ls=ls, color=S[[0, 1, 2][slot]], label=lab)
            if "knnf" in refs:
                y = 100 * refs["knnf"]["all"][stat]
                ax[si].axhline(y, color=th.rule, ls="--", lw=1.2)
                ax[si].text(5, y, "k-NN + fill features", ha="right", va="bottom", fontsize=8.5, color=th.text2)
            ax[si].set_xticks(range(1, 6))
            ax[si].set_xlabel("k: candidates compiled, validated and short-timed online")
            ax[si].set_ylabel(ttl)
            _pct_axis(ax[si], "y")
            ax[si].set_title(ttl + " vs k")
            ax[si].set_ylim(bottom=0)
        ax[0].legend(loc="lower left")
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        _title(fig, th, "Top-k + short online search (held-out, cold L2, 86 shapes)", "the first use of a shape compiles k configs, checks and times them briefly, then keeps the fastest")
        ctx.add("selectors", _save(fig, ctx.out, "selectors_topk_curve.png"), "Regret as a function of k for plain / diverse / inner orders.")

    # vs B3+: the diverse online top-3 choice against the searched cuBLASLt, per shape
    ck, gk = arr["cold"]["topk"], arr["graph"]["topk"]
    if "online-diverse@3" in ck and np.isfinite(ck["oracle"]).any():
        small = ck["oracle"] < SMALL_US
        if gk and "online-diverse@3" in gk:
            t_pick = np.where(small, gk["online-diverse@3"], ck["online-diverse@3"])
            t_bar = np.where(small, gk["b3plus"], ck["b3plus"])
        else:
            t_pick, t_bar = ck["online-diverse@3"], ck["b3plus"]
        ok = np.isfinite(t_pick) & np.isfinite(t_bar)
        x, y = t_bar[ok], t_bar[ok] / t_pick[ok]
        eps = np.where(x >= SMALL_US, EPS_BIG, EPS_SMALL)
        fig, ax = plt.subplots(figsize=(9.6, 5.4))
        ax.axhline(1, color=th.rule, lw=1.2)
        xs = np.logspace(np.log10(x.min() / 1.3), np.log10(x.max() * 1.3), 200)
        ax.fill_between(xs, 1 / (1 + np.where(xs >= SMALL_US, EPS_BIG, EPS_SMALL)), 1 + np.where(xs >= SMALL_US, EPS_BIG, EPS_SMALL), color=th.grid, alpha=0.7, lw=0)
        ahead, behind = y > 1 + eps, y < 1 / (1 + eps)
        ax.scatter(x[ahead], y[ahead], s=34, color=S[2], marker="D", lw=0, label=f"Triton pick faster ({ahead.sum()})")
        ax.scatter(x[~ahead & ~behind], y[~ahead & ~behind], s=34, color=th.text2, lw=0, label=f"tie ({(~ahead & ~behind).sum()})")
        ax.scatter(x[behind], y[behind], s=34, color=S[1], marker="s", lw=0, label=f"B3+ faster ({behind.sum()})")
        ax.axvline(SMALL_US, color=th.rule, ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_yticks([0.25, 0.5, 0.8, 1, 1.25, 2])
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}x"))
        ax.set_xlabel("B3+ time (us): cold per-launch >= 50 us, graph-amortized below")
        ax.set_ylabel("B3+ time / selected-Triton time  (> 1: Triton faster)")
        _quiet_minor(ax)
        ax.legend(loc="lower left")
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        _title(fig, th, "Selected Triton kernel vs the searched cuBLASLt, per held-out shape",
               f"{ok.sum()} shapes; shaded = tie margin (2% at >= 50 us, 5% below); dotted = 50 us")
        ctx.add("selectors", _save(fig, ctx.out, "selectors_vs_b3plus.png"), "Diverse online top-3 choice vs B3+ per shape.")
        big = x >= SMALL_US
        tab = []
        for lab, m, eps_ in (("B3+ >= 50 us (cold, 2% margin)", big, EPS_BIG), ("B3+ < 50 us (graph, 5% margin)", ~big, EPS_SMALL)):
            tab.append({"regime": lab, "n": int(m.sum()), "Triton faster": int((ahead & m).sum()), "ties": int((~ahead & ~behind & m).sum()),
                        "B3+ faster": int((behind & m).sum()), "median B3+/pick": float(np.median(y[m])) if m.any() else np.nan})
        ctx.text("selectors", "### Diverse online top-3 vs B3+ (held-out)\n\n" + _md_table(pd.DataFrame(tab), "{:.3f}") + "\n")


# ----------------------------------------------------------------------------------------------- kernel selection stats
def _screens(ctx: Ctx):
    """(shapes, keys, T) for the tune and held-out screens, split info attached; held-out last."""
    out = []
    for name in ("sweep_screen", "sweep_screen_heldout"):
        d = ctx.data / name
        if not d.exists():
            continue
        shapes, keys, T = load_screen(d)
        out.append((name, _split_info(d, shapes), keys, T))
    return out


def fig_selection(ctx: Ctx):
    th, S = ctx.th, ctx.th.series
    seq = th.seq
    # --- portfolio
    p = _read(ctx.report / "sweep_portfolio_summary.json")
    hp = _read(ctx.report / "sweep_portfolio_heldout_summary.json")
    screens = _screens(ctx)
    if p and hp:
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
        sizes = [r["size"] for r in hp["rows"]]
        ins = [r["in_sample"] for r in hp["rows"]]
        held = [r["heldout[all]"] for r in hp["rows"]]
        for a, stat, ttl in ((ax[0], "p95", "p95 regret"), (ax[1], "mean", "mean regret")):
            a.plot(sizes, [100 * d[stat] for d in ins], marker="o", ms=5, color=S[0], label="tune shapes (in-sample, 477)")
            a.plot(sizes, [100 * d[stat] for d in held], marker="s", ms=5, color=S[1], ls="--", label="held-out shapes (86)")
            a.set_xscale("log")
            a.set_yscale("log")
            a.axhline(5, color=th.rule, ls=":", lw=1)
            a.set_xlabel("portfolio size (configs, greedy order fit on tune shapes)")
            a.set_ylabel(ttl)
            _pct_axis(a, "y")
            a.set_title(f"{ttl} vs portfolio size")
        ax[0].legend(loc="upper right")
        ax[0].text(sizes[0], 5.4, "5% target", fontsize=8.5, color=th.text2)
        # CDF per portfolio size on the tune screen
        tune = next((s for s in screens if s[0] == "sweep_screen"), None)
        if tune:
            _, shapes, keys, T = tune
            col = {k: i for i, k in enumerate(keys)}
            R = T / T.min(axis=1, keepdims=True)
            port = p["portfolio"]
            picks = [(1, seq[0]), (5, seq[1]), (16, seq[2]), (29, seq[3]), (60, seq[4])]
            for n, c in picks:
                idx = [col[k] for k in port[:n] if k in col]
                r = np.sort(R[:, idx].min(axis=1))
                ax[2].step(r, np.arange(1, len(r) + 1) / len(r), where="post", color=c, lw=2, label=f"{n} config{'s' if n > 1 else ''}")
            ax[2].set_xscale("log")
            ax[2].set_xlim(0.995, 12)
            ax[2].set_xticks([1, 1.1, 1.5, 2, 4, 10])
            ax[2].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}x"))
            _quiet_minor(ax[2])
            ax[2].set_xlabel("slowdown vs the per-shape oracle (x)")
            ax[2].set_ylabel("fraction of tune shapes")
            ax[2].set_title("Static portfolio: regret CDF by size (tune shapes)")
            ax[2].legend(loc="lower right")
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        _title(fig, th, "How many configs does a static portfolio need?",
               f"B6 alone: median regret {100 * p['b6_regret_median']:.0f}%; in-sample p95 < 5% needs {p['g5_portfolio_size_p95_lt_5pct']} configs, held-out plateaus (G5 fails out of sample)")
        ctx.add("selection", _save(fig, ctx.out, "selection_portfolio.png"), "Portfolio size vs regret (in-sample and held-out) and regret CDFs.")
        sub = [r for r in hp["rows"] if r["size"] in (1, 5, 10, 20, 30, 60, 100)]
        ctx.text("selection", "### Static portfolio (greedy, fit on tune shapes)\n\n" + _md_table(pd.DataFrame([
            {"size": str(r["size"]), "tune mean": r["in_sample"]["mean"], "tune p95": r["in_sample"]["p95"], "tune p99": r["in_sample"].get("p99", float("nan")),
             "held-out mean": r["heldout[all]"]["mean"], "held-out p95": r["heldout[all]"]["p95"],
             "held-out p99": r["heldout[all]"].get("p99", float("nan"))} for r in sub]), "{:.3f}") + "\n")

    # --- parameter importance
    if p:
        pi = p["param_importance"]
        order = sorted(pi, key=lambda k: pi[k]["mean_regret_if_forced_worst_value"])
        fig, ax = plt.subplots(figsize=(8, 4.2))
        v = [100 * pi[k]["mean_regret_if_forced_worst_value"] for k in order]
        bars = ax.barh(range(len(order)), v, color=S[0], height=0.62)
        for b_, val in zip(bars, v):
            ax.text(val + max(v) * 0.01, b_.get_y() + b_.get_height() / 2, f"{val:.0f}%", va="center", fontsize=9, color=th.text2)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order)
        ax.set_xlabel("mean regret if the parameter is forced to its worst value (others at their best)")
        _pct_axis(ax)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(v) * 1.12)
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        _title(fig, th, "Which kernel parameters matter", "tune-screen sensitivity over 477 shapes; split-K and the M tile dominate")
        ctx.add("selection", _save(fig, ctx.out, "selection_param_importance.png"), "Parameter importance.")

    # --- which config wins where
    if screens:
        recs = []
        for name, shapes, keys, T in screens:
            best = T.argmin(axis=1)
            for s, j, t in zip(shapes, best, T.min(axis=1)):
                recs.append({"set": "held-out" if "heldout" in name else "tune", "M": s["M"], "N": s["N"], "K": s["K"], "oracle_us": t,
                             "key": keys[j], **_parse(keys[j])})
        W = pd.DataFrame(recs)
        W["regime"] = np.where(W["oracle_us"] >= SMALL_US, ">= 50 us", "< 50 us")
        fig, ax = plt.subplots(1, 7, figsize=(17, 3.9))
        for a, prm in zip(ax, PARAMS):
            vals = sorted(W[prm].unique())
            for gi, (reg, col) in enumerate(((">= 50 us", S[0]), ("< 50 us", S[1]))):
                g = W[W["regime"] == reg]
                share = [100 * (g[prm] == v).mean() for v in vals]
                a.bar(np.arange(len(vals)) + (gi - 0.5) * 0.4, share, width=0.38, color=col, label=f"oracle {reg} (n={len(g)})")
            a.set_xticks(range(len(vals)))
            a.set_xticklabels(vals)
            a.set_title(prm, fontsize=10)
            a.grid(axis="x", visible=False)
            _pct_axis(a, "y")
        ax[0].set_ylabel("share of shapes whose oracle uses the value")
        h_, l_ = ax[0].get_legend_handles_labels()
        fig.legend(h_, l_, loc="upper right", ncol=2, fontsize=9, bbox_to_anchor=(0.995, 0.985))
        fig.tight_layout(rect=(0, 0, 1, 0.84))
        _title(fig, th, "Which parameter values win", f"oracle config per shape from the stage-1 screen ({len(W)} shapes: tune + held-out); tiny GEMMs want small tiles and split-K")
        ctx.add("selection", _save(fig, ctx.out, "selection_oracle_winners.png"), "Share of shapes won by each parameter value, by oracle time regime.")
        top = W["key"].value_counts()
        ctx.text("selection", f"Oracle winners: {len(top)} distinct configs win at least one of {len(W)} screened shapes; the most common are "
                              + ", ".join(f"`{k}` ({v})" for k, v in top.head(5).items()) + ".\n")

        # where the winners are: composition of the winning BLOCK_M by M, and of the winning SPLIT_K by output size
        W["MN"] = W["M"] * W["N"]
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.9))
        panels = ((ax[0], "M", [0, 16, 64, 256, 1024, 4096, np.inf], ["<= 16", "17-64", "65-256", "257-1k", "1k-4k", "> 4k"], "BLOCK_M",
                   "Winning BLOCK_M by M", "M"),
                  (ax[1], "MN", [0, 1e4, 1e5, 1e6, 1e7, np.inf], ["< 1e4", "1e4-1e5", "1e5-1e6", "1e6-1e7", ">= 1e7"], "SPLIT_K",
                   "Winning SPLIT_K by output size", "M x N (output elements)"))
        for a_, col, edges, labels, prm, ttl, xl in panels:
            vals = sorted(W[prm].unique())
            cols = [seq[i] for i in np.linspace(0, len(seq) - 1, len(vals)).round().astype(int)]
            bins = pd.cut(W[col], edges, labels=labels, include_lowest=True)
            share = pd.crosstab(bins, W[prm], normalize="index").reindex(columns=vals).fillna(0) * 100
            n = bins.value_counts().reindex(labels)
            bottom = np.zeros(len(labels))
            for v, c in zip(vals, cols):
                h = share[v].reindex(labels).fillna(0).to_numpy()
                a_.bar(range(len(labels)), h, bottom=bottom, color=c, edgecolor=th.surface, lw=1.5, width=0.72, label=str(v))
                for xi, (hh, bb) in enumerate(zip(h, bottom)):
                    if hh >= 9:
                        a_.text(xi, bb + hh / 2, f"{hh:.0f}", ha="center", va="center", fontsize=8.5, color=th.ink(c))
                bottom += h
            a_.set_xticks(range(len(labels)))
            a_.set_xticklabels([f"{l}\n(n={int(n[l])})" for l in labels])
            a_.set_xlabel(xl)
            a_.set_ylabel("share of shapes (%)")
            a_.set_ylim(0, 100)
            a_.grid(axis="x", visible=False)
            a_.set_title(ttl)
            a_.legend(title=prm, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8.5, title_fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        _title(fig, th, "Where each tile size and split-K wins", "stacked shares of the oracle config's parameter; one blue ramp for ordered values (light = small, dark = large; reversed in dark mode)")
        ctx.add("selection", _save(fig, ctx.out, "selection_winner_map.png"), "Winning BLOCK_M by M and SPLIT_K by output size.")

        # oracle vs picked parameters, held-out
        tk = _read(ctx.data / "learned_topk_picks.json")
        held = next((s for s in screens if s[0] == "sweep_screen_heldout"), None)
        if tk and held:
            hs = held[1]
            choice = _online_keys(ctx.data, hs, "diverse", 3)
            orc = W[W["set"] == "held-out"].set_index(["M", "N", "K"])
            rows_ = []
            for s, ck in zip(hs, choice):
                if ck is None or (s["M"], s["N"], s["K"]) not in orc.index:
                    continue
                o = orc.loc[(s["M"], s["N"], s["K"])]
                rows_.append({**{f"o_{k}": o[k] for k in PARAMS}, **{f"p_{k}": v for k, v in _parse(ck).items()}})
            C = pd.DataFrame(rows_)
            if len(C):
                fig, ax = plt.subplots(1, 3, figsize=(14.5, 5.6))
                for a, prm in zip(ax, ("BLOCK_M", "BLOCK_N", "SPLIT_K")):
                    vals = sorted(set(C[f"o_{prm}"]) | set(C[f"p_{prm}"]))
                    M_ = np.array([[((C[f"o_{prm}"] == o) & (C[f"p_{prm}"] == q)).sum() for q in vals] for o in vals])
                    a.imshow(M_, cmap=th.cmap_seq, vmin=0, vmax=max(M_.max(), 1))
                    for i in range(len(vals)):
                        for j in range(len(vals)):
                            if M_[i, j]:
                                shade = th.cmap_seq(M_[i, j] / max(M_.max(), 1))
                                a.text(j, i, str(M_[i, j]), ha="center", va="center", fontsize=9,
                                       color=th.ink("#%02x%02x%02x" % tuple(int(255 * c) for c in shade[:3])))
                    a.set_xticks(range(len(vals)))
                    a.set_xticklabels(vals)
                    a.set_yticks(range(len(vals)))
                    a.set_yticklabels(vals)
                    a.set_xlabel(f"chosen {prm}")
                    a.set_ylabel(f"oracle {prm}")
                    a.grid(False)
                    exact = 100 * np.trace(M_) / M_.sum()
                    a.set_title(f"{prm}: {exact:.0f}% match")
                fig.tight_layout(rect=(0, 0, 1, 0.86))
                _title(fig, th, "Chosen vs oracle parameters (held-out, diverse online top-3)", f"{len(C)} shapes; diagonal = same parameter value as the oracle; off-diagonal cells can still be fast (regret is on the selector charts)")
                ctx.add("selection", _save(fig, ctx.out, "selection_pick_vs_oracle.png"), "Confusion of oracle and chosen parameters.")

    # --- online search: which candidate wins, and its cost
    t = _read(ctx.report / "learned_topk_summary.json")
    if t:
        fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
        for vi, (v, lab, ls) in enumerate((("plain", "plain order", "-"), ("diverse", "diverse order", "-"))):
            c = t.get(f"online_cost_{v}", {})
            if "3" in c:
                cnt = c["3"]["picked_rank_counts"]
                ax[0].bar(np.arange(len(cnt)) + (vi - 0.5) * 0.38, cnt, width=0.36, color=S[[0, 1][vi]], label=lab)
        ax[0].set_xticks(range(3))
        ax[0].set_xticklabels(["rank 1 (the top pick)", "rank 2", "rank 3"])
        ax[0].set_ylabel("held-out shapes")
        ax[0].set_title("Which of the top-3 the online search keeps")
        ax[0].grid(axis="x", visible=False)
        ax[0].legend(loc="upper right")
        for vi, (v, lab) in enumerate((("plain", "plain order"), ("diverse", "diverse order"))):
            c = t.get(f"online_cost_{v}", {})
            ks = sorted(int(k) for k in c)
            if ks:
                ax[1].plot(ks, [c[str(k)]["compile_s_median"] + c[str(k)]["time_s_median"] for k in ks], marker="o", ms=5, color=S[[0, 1][vi]], label=lab)
        ax[1].set_xticks(range(1, 6))
        ax[1].set_xlabel("k candidates")
        ax[1].set_ylabel("median seconds per shape")
        ax[1].set_title("First-use cost (JIT + validate + short timing, empty cache)")
        ax[1].set_ylim(bottom=0)
        ax[1].legend(loc="upper left")
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        _title(fig, th, "Online search: what it keeps and what it costs", "how often the search keeps the ranker's top pick versus its 2nd or 3rd candidate, and the wall-clock cost per shape")
        ctx.add("selection", _save(fig, ctx.out, "selection_online_search.png"), "Picked rank counts and first-use cost.")
        lines = []
        for v in ("plain", "diverse"):
            c = t.get(f"online_cost_{v}", {}).get("3")
            if c:
                lines.append(f"- {v} top-3: picked rank counts {c['picked_rank_counts']}, median {c['compile_s_median'] + c['time_s_median']:.2f} s per shape, "
                             f"{c['total_s_sum']:.0f} s over {t['n_heldout_shapes']} shapes")
        if lines:
            ctx.text("selection", "Online search cost:\n\n" + "\n".join(lines) + "\n")


# ----------------------------------------------------------------------------------------------- overview markdown
SECTIONS = [("baselines", "Baselines", "B1 `torch.matmul`, B3+ searched cuBLASLt, B5 Triton tutorial and the per-shape Triton oracle."),
            ("noise", "Noise floor", "How small a difference can be trusted (ties: 2% at >= 50 us, 5% below)."),
            ("selectors", "Selector performance", "Held-out regret against the per-shape Triton oracle, and the comparison with B3+."),
            ("selection", "Kernel selection statistics", "Portfolio size, parameter importance, which configs win where, and what the online search does.")]


def write_overview(ctx: Ctx, which: list[str], env: dict | None):
    head = ["# RTX 3080 results overview (auto-generated)\n",
            "Generated by `python -m analysis.plots`; every figure is built from existing result files, the tables below carry the same numbers."]
    if env:
        head.append(f"\nhw_id `{env['hw_id']}` | driver {env['driver']} | torch {env['versions']['torch']} | triton {env['versions']['triton']}; "
                    "standard conditions (SM clock locked at 1755 MHz, 3 input placements pooled, cold L2 primary).\n")
    parts = head
    rel = Path("figures") / "overview"
    for sec, title, blurb in SECTIONS:
        if sec not in which:
            continue
        parts.append(f"\n## {title}\n\n{blurb}\n")
        for name, cap in ctx.figs.get(sec, []):
            parts.append(f"![{cap}]({rel / name})\n")
        parts += ctx.md.get(sec, [])
    path = ctx.report / "results_overview.md"
    path.write_text("\n".join(parts) + "\n")
    return path


def build(data: Path, report: Path, theme: str = "light", only: list[str] | None = None) -> Path:
    which = only or [s[0] for s in SECTIONS]
    ctx = Ctx(data, report, Theme(theme))
    fns = {"baselines": fig_baselines, "noise": fig_noise, "selectors": fig_selectors, "selection": fig_selection}
    for name in which:
        fns[name](ctx)
        print(f"[plots] {name}: {', '.join(n for n, _ in ctx.figs.get(name, []))}")
    env = _read(Path(data) / "env.json")
    path = write_overview(ctx, which, env)
    print(f"[plots] wrote {path} and figures in {ctx.out}")
    return path


def default_data_dir() -> Path:
    """The only data/<hw_id> directory in the repo. tune.env.default_data_dir would ask torch for the device, which creates a
    CUDA context; this script must not touch the GPU, so it looks at the filesystem and asks for --data when it is ambiguous."""
    dirs = sorted(p for p in (Path(__file__).resolve().parent.parent / "data").glob("*_sm*") if p.is_dir())
    if len(dirs) != 1:
        raise SystemExit(f"found {len(dirs)} data/<hw_id> directories ({[d.name for d in dirs]}); pass --data")
    return dirs[0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=None, help="default data/<hw_id> of this machine")
    ap.add_argument("--report", type=Path, default=Path("reports"))
    ap.add_argument("--theme", choices=list(THEMES), default="light")
    ap.add_argument("--only", default=None, help="comma-separated subset of baselines,noise,selectors,selection")
    a = ap.parse_args(argv)
    only = a.only.split(",") if a.only else None
    if only and set(only) - {s[0] for s in SECTIONS}:
        ap.error(f"--only takes {','.join(s[0] for s in SECTIONS)}")
    build(a.data or default_data_dir(), a.report, a.theme, only)


if __name__ == "__main__":
    main()
