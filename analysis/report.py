"""Week-1 analysis: noise floor, thermal drift, roofline, and baseline comparison (plan 3.3, 4, 13).

    python -m tune analyze [--data data/<hw_id>] [--report reports]

Writes reports/figures/*.png and reports/week1_results.md. The numbers go into the hand-written
reports/week1.md.
"""
from __future__ import annotations

import datetime as _dt
import glob
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tune.records import read_raw, read_records  # noqa: E402


def _label(r) -> str:
    if r["kernel_family"] == "torch_matmul":
        return "B1 torch.matmul"
    if r["kernel_family"] == "triton_tutorial_default":
        return "B5 triton tutorial"
    return "ours:" + r["config_key"]


def _regime(r) -> str:
    if r["suite"] == "S1":
        return "S1 square"
    m = r["M"]
    if m <= 16:
        return "S2 decode (M<=16)"
    if m <= 128:
        return "S2 small batch (32<=M<=128)"
    return "S2 prefill (M>=512)"


def _dist(x: pd.Series) -> dict:
    x = x.dropna()
    return {"n": len(x), "median": x.median(), "p10": x.quantile(0.1), "p90": x.quantile(0.9),
            "min": x.min(), "max": x.max()}


def _fmt_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------- noise
def noise_analysis(data: Path, figs: Path) -> tuple[str, dict]:
    rows = read_records(data / "noise.jsonl")
    if not rows:
        return "_no noise data_\n", {}
    raw = read_raw(data / "noise.raw.jsonl")
    df = pd.DataFrame(rows)
    df["kernel"] = df.apply(_label, axis=1)
    df["shape"] = df.apply(lambda r: f"{r['M']}x{r['N']}x{r['K']}", axis=1)
    # Within-run spread: robust CV = IQR / median of the 100 raw samples.
    df["within_iqr_rel"] = df["time_iqr_us"] / df["time_median_us"]
    df["within_cv"] = [np.std(raw[i]) / np.mean(raw[i]) if i in raw else np.nan for i in df["record_id"]]

    if "n_allocs" not in df:
        df["n_allocs"] = 1
    df["n_allocs"] = df["n_allocs"].fillna(1).astype(int)
    keys = ["session_id", "shape", "kernel", "cache_mode"]
    g = df.groupby(keys)
    across = g["time_median_us"].agg(lambda s: (s.max() - s.min()) / s.median()).rename("across_round_range_rel")
    stats = g.agg(median_us=("time_median_us", "median"), within_iqr_rel=("within_iqr_rel", "median"),
                  within_cv=("within_cv", "median"), n_rounds=("round", "nunique"),
                  n_allocs=("n_allocs", "first")).join(across).reset_index()
    # Paired noise: how much the ratio between the two kernels moves across rounds (same inputs per round).
    kernels = sorted(df["kernel"].unique())
    paired = None
    if len(kernels) == 2:
        pv = df.pivot_table(index=["session_id", "shape", "cache_mode", "round"], columns="kernel",
                            values="time_median_us")
        ratio = (pv[kernels[0]] / pv[kernels[1]]).rename("ratio")
        paired = ratio.groupby(level=[0, 1, 2]).agg(lambda s: (s.max() - s.min()) / s.median())
        stats = stats.merge(paired.rename("paired_ratio_range_rel").reset_index(),
                            on=["session_id", "shape", "cache_mode"], how="left")

    out = {}
    md = []
    # Session context: rows from before provenance fields existed have no recorded lock state, so the
    # observed clock range is shown alongside; complete = every (shape, kernel, mode) has all rounds.
    ctx = {}
    for sess, d in df.groupby("session_id"):
        lock = d["clock_lock_state"].dropna().iloc[0] if "clock_lock_state" in d and d["clock_lock_state"].notna().any() else "unrecorded"
        mhz = d["clock_lock_mhz"].dropna().iloc[0] if "clock_lock_mhz" in d and d["clock_lock_mhz"].notna().any() else None
        n_rounds = d["round"].nunique()
        complete = d.groupby(["shape", "kernel", "cache_mode"])["round"].nunique().eq(n_rounds).all() and len(d) == d[["shape", "kernel", "cache_mode"]].drop_duplicates().shape[0] * n_rounds
        ctx[sess] = {"date": _dt.datetime.fromtimestamp(d["timestamp"].min()).strftime("%m-%d %H:%M"),
                     "driver": d["driver"].iloc[0], "clock": f"{lock}" + (f"@{mhz}" if mhz else ""),
                     "clk_obs": f"{int(d['clock_mhz'].min())}-{int(d['clock_mhz'].max())}",
                     "rounds": n_rounds, "complete": "yes" if complete and n_rounds >= 3 else "NO"}
    md.append("### Sessions\n")
    md.append(_fmt_table(pd.DataFrame([{"session": k, **v} for k, v in ctx.items()])))
    md.append("\nOnly compare sessions with the same driver and clock state; ignore sessions with complete = NO "
              "(fewer rounds understate the range).\n")
    summ = []
    for (sess, cm), s in stats.groupby(["session_id", "cache_mode"]):
        big = s[s["median_us"] >= 50]
        small = s[s["median_us"] < 50]
        d = {"session": sess, "cache_mode": cm, "clock": ctx[sess]["clock"], "complete": ctx[sess]["complete"],
             "n_allocs": int(s["n_allocs"].iloc[0]),
             "range_p50": s["across_round_range_rel"].median(),
             "range_p95": s["across_round_range_rel"].quantile(0.95),
             "range_max": s["across_round_range_rel"].max(),
             "range_p95_ge50us": big["across_round_range_rel"].quantile(0.95) if len(big) else np.nan,
             "range_p95_lt50us": small["across_round_range_rel"].quantile(0.95) if len(small) else np.nan,
             "within_iqr_p50": s["within_iqr_rel"].median()}
        if paired is not None:
            d["paired_p50"] = s["paired_ratio_range_rel"].median()
            d["paired_p95"] = s["paired_ratio_range_rel"].quantile(0.95)
        summ.append(d)
        out[f"{sess}/{cm}"] = d
    summary = pd.DataFrame(summ)
    md.append("### Noise summary per session (relative; 0.01 = 1%)\n")
    md.append(_fmt_table(summary, "{:.4f}"))
    md.append("\n`range` = (max - min) / median of the per-round medians for the same (shape, kernel, cache mode), "
              "over 3 rounds. `paired` = the same statistic for the ratio of the two kernels' times "
              "(the quantity that decides which kernel wins). `within_iqr` = IQR / median of one run's samples.\n")
    worst = stats.sort_values("across_round_range_rel", ascending=False).head(10)
    md.append("### Noisiest 10 (session, shape, kernel, mode)\n")
    md.append(_fmt_table(worst[["session_id", "shape", "kernel", "cache_mode", "median_us", "within_iqr_rel",
                                "across_round_range_rel"]], "{:.4f}"))
    stats = stats[stats["session_id"] == stats["session_id"].iloc[0]] if stats["session_id"].nunique() == 1 else stats

    # cold vs hot ratio
    piv = stats.pivot_table(index=["session_id", "shape", "kernel"], columns="cache_mode", values="median_us").dropna()
    if {"cold", "hot"} <= set(piv.columns):
        piv["cold_over_hot"] = piv["cold"] / piv["hot"]
        out["cold_over_hot"] = _dist(piv["cold_over_hot"])
        md.append("\n### Cold / hot L2 median-time ratio over the noise set\n")
        md.append(_fmt_table(pd.DataFrame([out["cold_over_hot"]]), "{:.3f}"))
        r = piv.reset_index()
        r["delta_us"] = r["cold"] - r["hot"]
        md.append("\nLargest ratios:\n")
        md.append(_fmt_table(r.sort_values("cold_over_hot", ascending=False).head(8)
                             [["session_id", "shape", "kernel", "hot", "cold", "delta_us", "cold_over_hot"]], "{:.3f}"))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for (sess, cm), s in stats.groupby(["session_id", "cache_mode"]):
        mk = "o" if cm == "cold" else "^"
        lab = f"{cm}, n_allocs={int(s['n_allocs'].iloc[0])}"
        ax[0].scatter(s["median_us"], 100 * s["across_round_range_rel"], marker=mk, alpha=0.7, label=lab)
        ax[1].scatter(s["median_us"], 100 * s["within_iqr_rel"], marker=mk, alpha=0.7, label=lab)
    for a, t in zip(ax, ["Across-round range of medians", "Within-run IQR / median"]):
        a.set_xscale("log")
        a.set_xlabel("median kernel time (us)")
        a.set_ylabel("%")
        a.set_title(t)
        a.axhline(2, color="k", lw=0.8, ls="--")
        a.legend()
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figs / "noise_floor.png", dpi=120)
    plt.close(fig)
    stats.to_csv(figs.parent / "noise_stats.csv", index=False)
    return "\n".join(md) + "\n", out


# --------------------------------------------------------------------------------------------- drift
def drift_analysis(data: Path, figs: Path) -> tuple[str, dict]:
    files = sorted(glob.glob(str(data / "drift-*.csv")))
    if not files:
        return "_no drift data_\n", {}
    df = pd.read_csv(files[-1])
    first = df[df["t_s"] <= 60]["median_us"].median()
    last = df[df["t_s"] >= df["t_s"].max() - 60]["median_us"].median()
    out = {"file": Path(files[-1]).name, "minutes": df["t_s"].max() / 60, "first_min_median_us": first,
           "last_min_median_us": last, "drift_rel": last / first - 1,
           "clock_min": df["clock_mhz"].min(), "clock_max": df["clock_mhz"].max(),
           "temp_start": df["temp_c"].iloc[0], "temp_max": df["temp_c"].max(),
           "power_median": df["power_w"].median(),
           "throttled_intervals": int((df["throttle_reasons"].fillna(0).astype(int) & ~0x1).astype(bool).sum())}
    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(df["t_s"], df["median_us"] / 1e3)
    ax[0].set_ylabel("8192^3 time (ms)")
    ax[1].plot(df["t_s"], df["clock_mhz"])
    ax[1].set_ylabel("SM clock (MHz)")
    ax[2].plot(df["t_s"], df["temp_c"], label="temp C")
    ax2 = ax[2].twinx()
    ax2.plot(df["t_s"], df["power_w"], color="tab:red", alpha=0.6, label="power W")
    ax[2].set_ylabel("temp (C)")
    ax2.set_ylabel("power (W)")
    ax[2].set_xlabel("time (s)")
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle("Thermal drift under continuous 8192^3 fp16 GEMM")
    fig.tight_layout()
    fig.savefig(figs / "thermal_drift.png", dpi=120)
    plt.close(fig)
    md = "### Thermal drift\n\n" + _fmt_table(pd.DataFrame([out]), "{:.4g}") + "\n"
    return md, out


# --------------------------------------------------------------------------------------------- baselines
def baseline_analysis(data: Path, figs: Path) -> tuple[str, dict]:
    rows = [r for r in read_records(data / "results.jsonl")]
    if not rows:
        return "_no benchmark data_\n", {}
    df = pd.DataFrame(rows)
    invalid = df[df["valid"] != True]  # noqa: E712
    df = df[df["valid"] == True].copy()  # noqa: E712
    df["kernel"] = df.apply(_label, axis=1)
    df["regime"] = df.apply(_regime, axis=1)
    roof = json.loads((data / "roofline.json").read_text()) if (data / "roofline.json").exists() else None

    md, out = [], {"n_invalid": len(invalid)}
    if len(invalid):
        md.append(f"**{len(invalid)} invalid measurements** (excluded): "
                  f"{invalid['failure_reason'].value_counts().to_dict()}\n")

    for cm in ["cold", "hot"]:
        d = df[df["cache_mode"] == cm]
        if d.empty:
            continue
        piv = d.pivot_table(index=["M", "N", "K", "regime"], columns="kernel", values="time_median_us")
        ours = [c for c in piv.columns if c.startswith("ours:")]
        piv["oracle(ours)"] = piv[ours].min(axis=1)
        piv["oracle_cfg"] = piv[ours].idxmin(axis=1).str.replace("ours:", "")
        # B6 restricted to the hand-picked set: the single config with best geomean regret over all shapes.
        regret = piv[ours].div(piv["oracle(ours)"], axis=0)
        b6 = np.exp(np.log(regret).mean()).idxmin()
        piv["B6(ours,fixed)"] = piv[b6]
        piv["ours/B5"] = piv["B5 triton tutorial"] / piv["oracle(ours)"]
        piv["ours/B1"] = piv["B1 torch.matmul"] / piv["oracle(ours)"]
        piv["B6/B1"] = piv["B1 torch.matmul"] / piv["B6(ours,fixed)"]
        piv["oracle/B6"] = piv["B6(ours,fixed)"] / piv["oracle(ours)"]
        piv["B5/B1"] = piv["B1 torch.matmul"] / piv["B5 triton tutorial"]
        flops = 2 * piv.index.get_level_values("M") * piv.index.get_level_values("N") * piv.index.get_level_values("K")
        piv["B1_tflops"] = flops / piv["B1 torch.matmul"] / 1e6
        piv["oracle_tflops"] = flops / piv["oracle(ours)"] / 1e6
        piv.reset_index().to_csv(figs.parent / f"baseline_table_{cm}.csv", index=False)

        o = out.setdefault(cm, {})
        o["b6_config"] = b6.replace("ours:", "")
        o["n_shapes"] = len(piv)
        o["n_shapes_ours_beats_b5"] = int((piv["ours/B5"] > 1.0).sum())
        o["n_shapes_ours_beats_b1"] = int((piv["ours/B1"] > 1.0).sum())
        md.append(f"\n## Cache mode: {cm}\n")
        md.append(f"Shapes: {len(piv)}. B6 (best single hand-picked config by geomean regret): `{o['b6_config']}`.\n")
        md.append("Speedup columns are time_baseline / time_ours (>1 means ours is faster).\n")
        tab = []
        for reg, grp in piv.groupby(level="regime"):
            for col in ["ours/B5", "ours/B1", "B6/B1", "oracle/B6", "B5/B1"]:
                tab.append({"regime": reg, "metric": col, **_dist(grp[col])})
        for col in ["ours/B5", "ours/B1", "B6/B1", "oracle/B6", "B5/B1"]:
            tab.append({"regime": "ALL", "metric": col, **_dist(piv[col])})
            o[col] = _dist(piv[col])
        md.append(_fmt_table(pd.DataFrame(tab), "{:.3f}"))
        wins = piv["oracle_cfg"].value_counts()
        md.append("\nOracle (hand-picked) winner counts: " + ", ".join(f"`{k}`: {v}" for k, v in wins.items()) + "\n")

        # Per-shape table for S1 (small enough to read in full).
        s1 = piv.xs("S1 square", level="regime", drop_level=False).reset_index() if "S1 square" in piv.index.get_level_values("regime") else None
        if s1 is not None:
            md.append("\n### S1 square sweep (median us)\n")
            md.append(_fmt_table(s1[["M", "B1 torch.matmul", "B5 triton tutorial", "oracle(ours)", "oracle_cfg",
                                     "B1_tflops", "oracle_tflops", "ours/B1", "ours/B5"]], "{:.2f}"))

        if roof is not None and cm == "cold":
            _roofline_plot(piv, roof, figs / "roofline.png")
        _speedup_plot(piv, cm, figs / f"speedup_{cm}.png")
    return "\n".join(md) + "\n", out


def _roofline_plot(piv: pd.DataFrame, roof: dict, path: Path):
    M = piv.index.get_level_values("M").to_numpy()
    N = piv.index.get_level_values("N").to_numpy()
    K = piv.index.get_level_values("K").to_numpy()
    regime = piv.index.get_level_values("regime").to_numpy()
    flops = 2.0 * M * N * K
    bytes_ = 2.0 * (M * K + K * N + M * N)
    ai = flops / bytes_
    peak_spec = roof["spec"]["fp16_f32acc_tflops_at_boost"]
    peak_meas = roof["compute"]["fp16_f32acc_tflops_measured"]
    bw_spec = roof["spec"]["dram_gbps_spec"]
    bw_meas = roof["bandwidth"]["dram_gbps_measured"]
    x = np.logspace(np.log10(max(ai.min() / 2, 0.3)), np.log10(ai.max() * 2), 200)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x, np.minimum(peak_spec, bw_spec * x / 1e3), "k-", lw=1, label=f"spec ({peak_spec:.0f} TF, {bw_spec:.0f} GB/s)")
    ax.plot(x, np.minimum(peak_meas, bw_meas * x / 1e3), "k--", lw=1, label=f"measured ({peak_meas:.0f} TF, {bw_meas:.0f} GB/s)")
    for reg in sorted(set(regime)):
        m = regime == reg
        ax.scatter(ai[m], piv["B1_tflops"].to_numpy()[m], s=18, alpha=0.8, label=f"B1 {reg}")
    ax.scatter(ai, piv["oracle_tflops"], s=10, marker="x", c="k", alpha=0.5, label="ours oracle (hand-picked)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP / compulsory DRAM byte)")
    ax.set_ylabel("achieved TFLOP/s (cold L2)")
    ax.set_title("Roofline: fp16 GEMM, fp32 accumulate, S1 + S2")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _speedup_plot(piv: pd.DataFrame, cm: str, path: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    flops = 2.0 * piv.index.get_level_values("M") * piv.index.get_level_values("N") * piv.index.get_level_values("K")
    regime = piv.index.get_level_values("regime")
    for reg in sorted(set(regime)):
        m = regime == reg
        ax.scatter(flops[m], piv["ours/B1"][m], s=16, alpha=0.8, label=f"oracle(ours) vs B1, {reg}")
    ax.scatter(flops, piv["ours/B5"], s=10, marker="x", c="gray", alpha=0.6, label="oracle(ours) vs B5")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("GEMM FLOPs")
    ax.set_ylabel("speedup (>1: ours faster)")
    ax.set_title(f"Hand-picked Triton oracle vs baselines ({cm} L2)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def build_reports(data: Path, report: Path):
    figs = report / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    env = json.loads((data / "env.json").read_text())
    roof = json.loads((data / "roofline.json").read_text()) if (data / "roofline.json").exists() else {}
    parts = [f"# Week-1 results (auto-generated)\n\nhw_id `{env['hw_id']}` | driver {env['driver']} | "
             f"torch {env['versions']['torch']} | triton {env['versions']['triton']} | harness "
             f"`{(env['harness_git']['commit'] or '')[:8]}`\n"]
    if roof:
        parts.append("## B0 roofline\n")
        parts.append(_fmt_table(pd.DataFrame([{
            "spec TFLOP/s @boost": roof["spec"]["fp16_f32acc_tflops_at_boost"],
            "spec TFLOP/s @max clk": roof["spec"]["fp16_f32acc_tflops_at_max_clock"],
            "measured TFLOP/s": roof["compute"]["fp16_f32acc_tflops_measured"],
            "spec GB/s": roof["spec"]["dram_gbps_spec"], "measured GB/s": roof["bandwidth"]["dram_gbps_measured"]}]),
            "{:.1f}") + "\n")
    summary = {"roofline": roof}
    for name, fn in [("noise", noise_analysis), ("drift", drift_analysis), ("baselines", baseline_analysis)]:
        md, s = fn(data, figs)
        parts.append(f"## {name}\n\n{md}")
        summary[name] = s
    (report / "week1_results.md").write_text("\n".join(parts))
    (report / "week1_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"wrote {report / 'week1_results.md'} and figures in {figs}")
