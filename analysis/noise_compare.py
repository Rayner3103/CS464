"""Noise floor of one group of noise sessions vs another (e.g. locked vs unlocked SM clock), and the online protocol.

`tune noise` writes one row per (session, round, shape, kernel, cache mode) with the median of 3 placements x `repeats`
samples; `tune analyze` reports the across-round spread per session. The tie margin epsilon (docs/methodology.md) is
a *cross-session* statistic: for each (shape, kernel, mode), the range of the per-session medians over a group of
sessions, (max - min) / median, read at its p95. This module computes that statistic for each named group of
sessions with the same definitions, so a new condition (unlocked clock, another driver) is compared like with like:

  * within:  per session, the range of the per-round medians (what `tune analyze` calls `range`);
  * cross:   per group, the range of the per-session medians (median of that session's round medians);
  * paired:  the same cross-session range for the ratio of the two kernels' times (what decides selection);
  * within_paired: the across-round range of that ratio inside one session;
  * single:  one full-protocol measurement (one round's 3 x `repeats` samples) vs the group's cross-session median,
             |t / ref - 1|, per kernel and (`single_paired`) as the ratio. `cross` compares per-session medians of
             3 rounds, which damps a bad round; a benchmark row is one measurement, so `single` is its error bar;
  * per_round: per (session, round) at >= 50 us, the geometric-mean speed shift vs the session's median and the p95
             of the per-kernel deviation: a shift that is the same for every kernel cancels in a ratio, a spread
             does not;
  * online:  the shipped selector's short timing (tune/online.py: SHORT_REPS = 10 cold samples, one placement)
             emulated from the stored raw samples: each placement's first 10 samples (raw samples are stored per
             placement, `repeats` at a time, in placement order). `single` = |short / ref - 1| for one kernel,
             `paired` = |short ratio / ref ratio - 1| for the two kernels timed on the same placement, where ref is
             the group's cross-session median. The reference is cross-session on purpose: the winner cache keeps a
             short-timed pick indefinitely, so the pick has to hold in later sessions, not only in the one it was
             timed in. The noise study warms up 25 ms per measurement, the online search 2 ms, so this is a slight
             underestimate. Under 50 us the online search ranks by graph-amortized timing, which the noise study
             does not run, so only the >= 50 us cold row describes what the selector does.

The margin read-off follows the locked convention (p95 1.75% -> epsilon 2%): the p95, rounded up to a whole percent.

    python -m analysis.noise_compare --group locked=642a,lock1755,6985 --group unlocked=<s1>,<s2>,<s3> \
        [--data data/<hw>] [--json reports/noise_compare_summary.json]

Session names match by suffix. CPU only.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from tune.records import read_raw, read_records

SMALL_US = 50.0
SHORT_REPS = 10   # tune/online.py SHORT_REPS


def _rel_range(s: pd.Series) -> float:
    return float((s.max() - s.min()) / s.median())


def _q(x: pd.Series) -> dict:
    x = pd.Series(x).dropna()
    if not len(x):
        return {"n": 0, "p50": np.nan, "p95": np.nan, "p99": np.nan, "max": np.nan}
    return {"n": int(len(x)), "p50": float(x.median()), "p95": float(x.quantile(0.95)),
            "p99": float(x.quantile(0.99)), "max": float(x.max())}


# The statistics whose p95 sets a margin; their bootstrap interval resamples shapes (a shape's kernels, modes,
# sessions and rounds move together), 2000 draws, seed 0.
BOOT = (("cross", "ge50", "cold"), ("cross", "ge50", "hot"), ("paired", "ge50", "cold"), ("paired", "ge50", "hot"),
        ("single", "ge50", "cold"), ("cross", "lt50", "cold"), ("cross", "lt50", "hot"),
        ("online_paired", "ge50", "cold"))


def boot_p95(d: pd.DataFrame, col: str, regime: str, mode: str, n: int = 2000, seed: int = 0) -> list[float]:
    d = d[(d["regime"] == regime) & (d["cache_mode"] == mode)]
    if d.empty:
        return [np.nan, np.nan]
    groups = [x[col].to_numpy() for _, x in d.groupby("shape")]
    rng = np.random.default_rng(seed)
    qs = [np.quantile(np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]), 0.95)
          for _ in range(n)]
    return [float(np.quantile(qs, 0.025)), float(np.quantile(qs, 0.975))]


def margin(p95: float) -> int:
    """The locked rule: p95 rounded up to a whole percent (1.75% -> 2)."""
    return int(math.ceil(round(100 * p95, 6)))


def load(data: Path, sessions: dict[str, list[str]]) -> pd.DataFrame:
    rows = read_records(data / "noise.jsonl")
    all_ids = sorted({r["session_id"] for r in rows})
    sid_group = {}
    for g, pats in sessions.items():
        for p in pats:
            hit = [s for s in all_ids if s.endswith(p)]
            if len(hit) != 1:
                raise SystemExit(f"session pattern {p!r} matches {hit or 'nothing'} (known: {all_ids})")
            sid_group[hit[0]] = g
    rows = [r for r in rows if r["session_id"] in sid_group]
    raw = read_raw(data / "noise.raw.jsonl")
    df = pd.DataFrame(rows)
    df["group"] = df["session_id"].map(sid_group)
    df["kernel"] = np.where(df["kernel_family"] == "torch_matmul", "torch", "triton:" + df["config_key"])
    df["shape"] = df["M"].astype(str) + "x" + df["N"].astype(str) + "x" + df["K"].astype(str)
    df["n_allocs"] = df["n_allocs"].fillna(1).astype(int)
    # Short-protocol medians: the first SHORT_REPS samples of each placement block.
    shorts = []
    for rid, n_al, n in zip(df["record_id"], df["n_allocs"], df["n_samples"]):
        s = raw.get(rid)
        if s is None or len(s) != n:
            shorts.append(None)
            continue
        per = len(s) // n_al
        shorts.append([float(np.median(s[i * per:i * per + SHORT_REPS])) for i in range(n_al)])
    df["short_us"] = shorts
    return df


def _harness(d: pd.DataFrame) -> str:
    c = d["harness_commit"].iloc[0] if "harness_commit" in d else None
    if not isinstance(c, str) or not c:
        return "unrecorded"
    return c[:7] + (" dirty" if d["harness_dirty"].iloc[0] is True else "")


def _lock(d: pd.DataFrame, data: Path | None) -> str:
    """Recorded lock state; rows predating per-row provenance fall back to the session env (as `tune analyze`)."""
    if "clock_lock_state" in d and d["clock_lock_state"].notna().any():
        st, mhz = d["clock_lock_state"].dropna().iloc[0], d["clock_lock_mhz"].dropna()
    else:
        env = data / "env" / f"{d['session_id'].iloc[0]}.json" if data else None
        if not env or not env.exists():
            return "unrecorded"
        cl = json.loads(env.read_text()).get("clock_lock", {})
        st, mhz = cl.get("state", "unrecorded"), pd.Series([cl.get("lock_mhz")]).dropna()
    return f"{st}@{mhz.iloc[0]}" if len(mhz) else str(st)


def session_context(df: pd.DataFrame, data: Path | None = None) -> pd.DataFrame:
    out = []
    for sid, d in df.groupby("session_id"):
        thr = d["throttle_reasons"].fillna(0).astype(int)
        complete = d.groupby(["shape", "kernel", "cache_mode"])["round"].nunique().eq(d["round"].nunique()).all()
        out.append({"group": d["group"].iloc[0], "session": sid, "driver": d["driver"].iloc[0],
                    "lock": _lock(d, data),
                    "rows": len(d), "rounds": int(d["round"].nunique()), "complete": bool(complete),
                    "clock_mhz": f"{int(d['clock_mhz'].min())}-{int(d['clock_mhz'].max())}",
                    "clock_p50": int(d["clock_mhz"].median()),
                    "power_capped": float((thr & 0x4).astype(bool).mean()),
                    "temp_c": f"{int(d['temp_c'].min())}-{int(d['temp_c'].max())}",
                    "harness": _harness(d)})
    return pd.DataFrame(out)


def analyse(df: pd.DataFrame) -> dict:
    res = {}
    # Per-session medians and the size regime (by the group's median time of that kernel).
    sess = (df.groupby(["group", "session_id", "shape", "kernel", "cache_mode"])["time_median_us"]
            .median().rename("t").reset_index())
    big = sess.groupby(["group", "shape", "kernel", "cache_mode"])["t"].median().rename("tg").reset_index()
    big["regime"] = np.where(big["tg"] >= SMALL_US, "ge50", "lt50")
    reg = big.set_index(["group", "shape", "kernel", "cache_mode"])["regime"]
    for g, dg in df.groupby("group"):
        res[g] = {}
        # within: per session, across rounds
        w = (dg.groupby(["session_id", "shape", "kernel", "cache_mode"])["time_median_us"].agg(_rel_range)
             .rename("r").reset_index())
        w["regime"] = [reg[(g, s, k, c)] for s, k, c in zip(w["shape"], w["kernel"], w["cache_mode"])]
        # cross: per group, across sessions
        sg = sess[sess["group"] == g]
        c = sg.groupby(["shape", "kernel", "cache_mode"])["t"].agg(_rel_range).rename("r").reset_index()
        c["regime"] = [reg[(g, s, k, m)] for s, k, m in zip(c["shape"], c["kernel"], c["cache_mode"])]
        # paired: ratio torch / triton per session, then across sessions
        pv = sg.pivot_table(index=["session_id", "shape", "cache_mode"], columns="kernel", values="t")
        k0, k1 = sorted(pv.columns)
        ratio = (pv[k0] / pv[k1]).rename("ratio").reset_index()
        p = ratio.groupby(["shape", "cache_mode"])["ratio"].agg(_rel_range).rename("r").reset_index()
        # A pair's regime: the faster kernel's (the one a selection decision hinges on).
        p["regime"] = [("ge50" if min(big.set_index(["group", "shape", "kernel", "cache_mode"])["tg"][(g, s, k, m)]
                                      for k in (k0, k1)) >= SMALL_US else "lt50")
                       for s, m in zip(p["shape"], p["cache_mode"])]
        # online emulation: short medians vs the group's cross-session median
        ref = sg.groupby(["shape", "kernel", "cache_mode"])["t"].median()
        sh = dg[dg["short_us"].notna()][["session_id", "round", "shape", "kernel", "cache_mode", "short_us"]]
        sh = sh.explode("short_us").assign(alloc=lambda x: x.groupby(level=0).cumcount()).reset_index(drop=True)
        sh["short_us"] = sh["short_us"].astype(float)
        sh["dev"] = [abs(v / ref[(s, k, m)] - 1) for v, s, k, m in zip(sh["short_us"], sh["shape"], sh["kernel"], sh["cache_mode"])]
        sh["regime"] = [reg[(g, s, k, m)] for s, k, m in zip(sh["shape"], sh["kernel"], sh["cache_mode"])]
        spv = sh.pivot_table(index=["session_id", "round", "alloc", "shape", "cache_mode"], columns="kernel",
                             values="short_us").reset_index()
        refr = (ref.xs(k0, level="kernel") / ref.xs(k1, level="kernel"))
        spv["dev"] = [abs((a / b) / refr[(s, m)] - 1) for a, b, s, m in zip(spv[k0], spv[k1], spv["shape"], spv["cache_mode"])]
        preg = p.set_index(["shape", "cache_mode"])["regime"]
        spv["regime"] = [preg[(s, m)] for s, m in zip(spv["shape"], spv["cache_mode"])]
        # single: one full-protocol measurement (one round) vs the group reference, per kernel and as a ratio;
        # within_paired: the across-round range of the kernel ratio inside one session.
        one = dg[["session_id", "round", "shape", "kernel", "cache_mode", "time_median_us", "clock_mhz", "throttle_reasons"]].copy()
        one["dev"] = [abs(v / ref[(s, k, m)] - 1) for v, s, k, m in zip(one["time_median_us"], one["shape"], one["kernel"], one["cache_mode"])]
        one["regime"] = [reg[(g, s, k, m)] for s, k, m in zip(one["shape"], one["kernel"], one["cache_mode"])]
        opv = one.pivot_table(index=["session_id", "round", "shape", "cache_mode"], columns="kernel",
                              values="time_median_us").reset_index()
        opv["dev"] = [abs((a / b) / refr[(s, m)] - 1) for a, b, s, m in zip(opv[k0], opv[k1], opv["shape"], opv["cache_mode"])]
        opv["regime"] = [preg[(s, m)] for s, m in zip(opv["shape"], opv["cache_mode"])]
        opv["ratio"] = opv[k0] / opv[k1]
        wp = opv.groupby(["session_id", "shape", "cache_mode"])["ratio"].agg(_rel_range).rename("r").reset_index()
        wp["regime"] = [preg[(s, m)] for s, m in zip(wp["shape"], wp["cache_mode"])]
        # Per round (>= 50 us): geometric-mean shift vs the session's median of each kernel, the p95 of the
        # per-kernel deviation, and the share of power-capped rows: is the spread a drift, or per kernel?
        smed = one.groupby(["session_id", "shape", "kernel", "cache_mode"])["time_median_us"].transform("median")
        one["rel"] = one["time_median_us"] / smed
        res[g]["per_round"] = {
            f"{sid}/r{rd}": {"shift": float(np.expm1(np.log(x["rel"]).mean())),
                            "spread_p95": float((x["rel"] - 1).abs().quantile(0.95)),
                            "clock_p50": float(x["clock_mhz"].median()),
                            "power_capped": float((x["throttle_reasons"].fillna(0).astype(int) & 0x4).astype(bool).mean())}
            for (sid, rd), x in one[one["regime"] == "ge50"].groupby(["session_id", "round"])}
        tables = {"within": w, "cross": c, "paired": p, "within_paired": wp, "single": one,
                  "single_paired": opv, "online_single": sh, "online_paired": spv}
        res[g]["p95_ci"] = {f"{n}/{rg}/{cm}": boot_p95(tables[n], "dev" if n.startswith(("online", "single")) else "r", rg, cm)
                            for n, rg, cm in BOOT}
        for name, d in tables.items():
            col = "dev" if name.startswith(("online", "single")) else "r"
            for (rg, cm), x in d.groupby(["regime", "cache_mode"]):
                res[g][f"{name}/{rg}/{cm}"] = _q(x[col])
        res[g]["within_per_session"] = {f"{sid}/{rg}/{cm}": _q(x["r"])
                                        for (sid, rg, cm), x in w.groupby(["session_id", "regime", "cache_mode"])}
        # Median speed of each session relative to the group's cross-session median (does the session shift?).
        # Geometric mean of t / (the geometric mean over the group's sessions), per session.
        lg = sg.assign(lt=np.log(sg["t"]))
        lg = lg.join(lg.groupby(["shape", "kernel", "cache_mode"])["lt"].mean().rename("lm"), on=["shape", "kernel", "cache_mode"])
        res[g]["session_shift"] = {sid: float(np.expm1((x["lt"] - x["lm"]).mean())) for sid, x in lg.groupby("session_id")}
        res[g]["noisiest_cross"] = (c.sort_values("r", ascending=False).head(5)
                                    .assign(r=lambda x: x["r"].round(4)).to_dict("records"))
    # Speed of each group relative to the first (same shape, kernel, mode).
    gm = sess.groupby(["group", "shape", "kernel", "cache_mode"])["t"].median().unstack("group")
    gs = list(res)
    res["speed_vs_" + gs[0]] = {g: _q(gm[g] / gm[gs[0]]) for g in gs[1:]}
    return res


def _pct(x: float) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.2f}%"


def markdown(ctx: pd.DataFrame, res: dict) -> str:
    groups = [g for g in res if not g.startswith("speed_vs_")]
    out = ["### Sessions\n", "| group | session | driver | lock | rows | complete | SM clock MHz (p50) | power-capped | temp °C | harness |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in ctx.iterrows():
        out.append(f"| {r['group']} | `{r['session']}` | {r['driver']} | {r['lock']} | {r['rows']} | {'yes' if r['complete'] else 'NO'} | "
                   f"{r['clock_mhz']} ({r['clock_p50']}) | {100 * r['power_capped']:.0f}% | {r['temp_c']} | {r['harness']} |")
    out.append("\n### Noise floor, p50 / p95 / p99 / max (relative range; n = (shape, kernel) or shape pairs)\n")
    out.append("| statistic | regime | mode | " + " | ".join(groups) + " |")
    out.append("|---|---|---|" + "---|" * len(groups))
    for name in ("cross", "paired", "within", "within_paired", "single", "single_paired", "online_single", "online_paired"):
        for rg in ("ge50", "lt50"):
            for cm in ("cold", "hot"):
                cells = []
                for g in groups:
                    q = res[g].get(f"{name}/{rg}/{cm}")
                    cells.append("–" if not q or not q["n"] else
                                 f"{_pct(q['p50'])} / **{_pct(q['p95'])}** / {_pct(q['p99'])} / {_pct(q['max'])} (n {q['n']})")
                if any(c != "–" for c in cells):
                    out.append(f"| {name} | {'≥ 50 µs' if rg == 'ge50' else '< 50 µs'} | {cm} | " + " | ".join(cells) + " |")
    out.append("\n### Per session: range across rounds, p95 / p99 (≈ the worst kernel), cold / hot\n")
    out.append("| group | session | ≥ 50 µs p95 | ≥ 50 µs p99 | < 50 µs p95 | < 50 µs p99 |")
    out.append("|---|---|---|---|---|---|")
    for g in groups:
        sids = sorted({k.split("/")[0] for k in res[g]["within_per_session"]})
        for sid in sids:
            q = lambda rg, cm, st: res[g]["within_per_session"].get(f"{sid}/{rg}/{cm}", {}).get(st, np.nan)
            cell = lambda rg, st: f"{_pct(q(rg, 'cold', st))} / {_pct(q(rg, 'hot', st))}"
            out.append(f"| {g} | `{sid}` | {cell('ge50', 'p95')} | {cell('ge50', 'p99')} | {cell('lt50', 'p95')} | {cell('lt50', 'p99')} |")
    out.append("\n### Margin read-off (p95 rounded up to a whole percent, worse of cold / hot)\n")
    out.append("| margin | " + " | ".join(groups) + " |")
    out.append("|---|" + "---|" * len(groups))
    for label, name, rg, modes in (("ε ≥ 50 µs (cross-session, per kernel)", "cross", "ge50", ("cold", "hot")),
                                   ("ε ≥ 50 µs (cross-session, kernel ratio)", "paired", "ge50", ("cold", "hot")),
                                   ("ε < 50 µs per-launch (cross-session, per kernel)", "cross", "lt50", ("cold", "hot")),
                                   ("ε < 50 µs per-launch (within session, per kernel)", "within", "lt50", ("cold", "hot")),
                                   ("online short timing ≥ 50 µs, cold (kernel ratio)", "online_paired", "ge50", ("cold",))):
        cells = []
        for g in groups:
            p95s = [res[g][f"{name}/{rg}/{cm}"]["p95"] for cm in modes if f"{name}/{rg}/{cm}" in res[g]]
            cells.append(f"{margin(max(p95s))}% (p95 {_pct(max(p95s))})" if p95s else "–")
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    out.append("\n### p95 with a 95% bootstrap interval over shapes\n")
    out.append("| statistic | " + " | ".join(groups) + " |")
    out.append("|---|" + "---|" * len(groups))
    for n_, rg, cm in BOOT:
        k = f"{n_}/{rg}/{cm}"
        cells = [f"{_pct(res[g][k]['p95'])} [{_pct(res[g]['p95_ci'][k][0])}, {_pct(res[g]['p95_ci'][k][1])}]"
                 if k in res[g] else "–" for g in groups]
        out.append(f"| {n_} {'≥ 50 µs' if rg == 'ge50' else '< 50 µs'} {cm} | " + " | ".join(cells) + " |")
    out.append("\n### Per round (≥ 50 µs): speed shift vs the session median, p95 per-kernel deviation, SM clock p50, power-capped\n")
    out.append("| group | session / round | shift | spread p95 | clock MHz | power-capped |")
    out.append("|---|---|---|---|---|---|")
    for g in groups:
        for k, v in res[g]["per_round"].items():
            out.append(f"| {g} | `{k}` | {100 * v['shift']:+.2f}% | {_pct(v['spread_p95'])} | {v['clock_p50']:.0f} | {100 * v['power_capped']:.0f}% |")
    out.append("\n### Session speed shift (geometric mean of t over the group's cross-session geometric mean)\n")
    for g in groups:
        out.append(f"- {g}: " + ", ".join(f"`{s}` {100 * v:+.2f}%" for s, v in res[g]["session_shift"].items()))
    for k, v in res.items():
        if k.startswith("speed_vs_"):
            out.append(f"\n### Time ratio vs {k[9:]} (same shape, kernel, mode)\n")
            for g, q in v.items():
                out.append(f"- {g}: p50 {q['p50']:.3f}, p95 {q['p95']:.3f}, p99 {q['p99']:.3f}, max {q['max']:.3f} (n {q['n']})")
    return "\n".join(out) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, default=Path("data/rtx3080_sm86_68sms_10gb"))
    ap.add_argument("--group", action="append", required=True, help="NAME=sess1,sess2,... (session id suffixes)")
    ap.add_argument("--json", type=Path, default=None, help="write the summary here")
    a = ap.parse_args(argv)
    sessions = {}
    for g in a.group:
        name, _, pats = g.partition("=")
        sessions[name] = [p for p in pats.split(",") if p]
    df = load(a.data, sessions)
    ctx = session_context(df, a.data)
    res = analyse(df)
    print(markdown(ctx, res))
    if a.json:
        a.json.write_text(json.dumps({"sessions": ctx.to_dict("records"), "groups": sessions, **res}, indent=1, default=str))


if __name__ == "__main__":
    main()
