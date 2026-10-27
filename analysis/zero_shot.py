"""Zero-shot transfer experiment: how far does a selector fitted on one GPU get on another, with no refitting?

The saved lgbm_rank (analysis/learned.py `Ranker`) was fit on RTX 3080 screens: its candidate pool, its learned regrets
and the GPU numbers behind its fill features are all 3080's. This runs it unchanged on a different GPU and scores the
picks against that GPU's own screen oracle, in the same units as the 3080 numbers in reports/learned_topk.md
("Screen" table: median / p95 (mean) regret vs the held-out screen oracle), so the two are comparable.

Two variants separate what the model gets wrong from what the features get wrong:
  as_is   the ranker exactly as saved: the 3080's SM count, shared memory and register limits feed the fill features
          and the `runnable` filter (picks can exceed the new GPU's limits and fail to compile);
  native  same booster and pool, but the GPU numbers are replaced by the device it runs on, so the fill features and
          pruning limits describe the real hardware. This is what a deployment would do.
Both use the diversified top-k (`order="diverse"`, the saved CV-chosen rule) with the short online search, and store the
choice for every k' <= k, so top-1 (no online search) and best-of-k come from one run.

Runbook on the target machine (GPU idle, standard conditions, see AGENTS.md section 7):
  1. screen the held-out shapes there (the oracle; hours):   python -m tune --clock-lock MHZ sweep --split heldout
  2. copy the source model next to this machine's data:      data/<src_hw>/lgbm_rank.{txt,json} -> any directory
  3. python -m analysis.zero_shot run   --ranker <dir>/lgbm_rank --fresh-cache
  4. python -m analysis.zero_shot score

`run` is resumable. Scoring needs no GPU: pass --data to score on another machine. Never fit anything on these shapes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np

CAP = 10.0       # regret for "no valid pick", the same convention as `portfolio` (inf capped at 10 = 1000%)
EPS = 0.02       # tie margin at >= 50 us (AGENTS.md section 6, G1)
SMALL_US = 50.0
VARIANTS = ("as_is", "native")
# The 3080 reference rows: reports/learned_topk.md, "Screen" table, orders plain / diverse (median, p95, mean; all shapes).
REFERENCE_3080 = {"top-1": (0.036, 0.181, 0.055), "best-of-3 (diverse)": (0.020, 0.106, 0.034)}
# B6 on the 3080 (AGENTS.md section 6): the "just reuse the single best 3080 config" baseline.
FIXED_3080 = "bm32_bn32_bk32_w4_s5_g1_sk1"


def out_name(variant: str) -> str:
    return f"zero_shot_{variant}.jsonl"


def source_gpu_name(stem: Path) -> str | None:
    """Name of the GPU the saved model was fit on (lgbm_rank.json["gpu"])."""
    return (json.loads(Path(f"{stem}.json").read_text()).get("gpu") or {}).get("name")


def variant_ranker(stem: Path, variant: str, native_gpu: dict | None):
    from analysis.learned import Ranker
    r = Ranker.load(stem)
    if variant == "as_is":
        return r
    return Ranker(r.booster, r.pool_keys, r.hp, native_gpu, r.meta)


def run(shapes: list[dict], ranker_stem: Path, out_dir: Path, *, variants=VARIANTS, k: int = 3,
        session_id: str | None = None, log=print) -> list[Path]:
    from tune.env import provenance, write_env
    from tune.online import choose, search_shape
    from tune.records import RecordWriter
    import torch

    session_id = session_id or time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    env = write_env(out_dir, session_id=session_id, kind="zero_shot")
    paths, source_gpu = [], source_gpu_name(ranker_stem)
    for variant in variants:
        ranker = variant_ranker(ranker_stem, variant, env["gpu"])
        rule = ranker.rule("diverse")
        path = out_dir / out_name(variant)
        paths.append(path)
        done = set()
        if path.exists():
            done = {(r["M"], r["N"], r["K"]) for r in map(json.loads, path.read_text().splitlines())
                    if r.get("k") == k and r.get("rule") == rule}
        todo = [s for s in shapes if (s["M"], s["N"], s["K"]) not in done]
        t0 = time.perf_counter()
        with RecordWriter(path) as w:
            for i, s in enumerate(todo):
                M, N, K = s["M"], s["N"], s["K"]
                t = time.perf_counter()
                r = search_shape(ranker, M, N, K, k, torch.float16, "diverse")
                row = {"M": M, "N": N, "K": K, "dtype": "fp16", "k": k, "order": "diverse", "rule": rule,
                       "variant": variant, "split": s.get("split"), "suite": s.get("suite"),
                       "session_id": session_id, "hw_id": env["hw_id"], "driver": env["driver"], **provenance(env),
                       "source_gpu": source_gpu,
                       "ranker": str(ranker_stem), "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
                       "wall_s": time.perf_counter() - t, **r,
                       "choice": {str(j): choose(r["candidates"], j) for j in range(1, k + 1)},
                       "timestamp": time.time()}
                w.write(row)
                el = time.perf_counter() - t0
                log(f"[{variant} {i + 1}/{len(todo)}] {M}x{N}x{K} -> {row['choice'][str(k)]['key']} "
                    f"({row['wall_s']:.1f}s) | {el:.0f}s elapsed, ETA {el / (i + 1) * (len(todo) - i - 1):.0f}s")
    return paths


# ---------------------------------------------------------------- scoring (no GPU)

def load_screen(screen_dir: Path) -> dict:
    """{(M, N, K): {config_key: us}} over every complete screen file; only status == ok rows with a time."""
    out = {}
    for p in sorted(screen_dir.glob("*_fp16.json")):
        d = json.loads(p.read_text())
        if not d.get("complete"):
            continue
        t = {r["key"]: r["us"] for r in d["rows"] if r["status"] == "ok" and r["us"] is not None}
        if t:
            out[(d["M"], d["N"], d["K"])] = t
    return out


def regret(times: dict, key: str | None) -> float | None:
    """t(key) / min t - 1 against the screen oracle; CAP when there is no pick, None when the pick is not in the
    screen (so it cannot be scored there; counted and reported, not guessed)."""
    if key is None:
        return CAP
    if key not in times:
        return None
    return times[key] / min(times.values()) - 1.0


def score_rows(screen: dict, rows: list[dict], k: int, fixed: str | None = None) -> dict:
    """Per-shape regret lists for one variant's rows: top-1 / best-of-k' / online-k' (k' <= k) and a fixed config.
    best-of-k' = the fastest of the first k' candidates that compiled and validated, by screen time (the ranker's
    quality, free of online-timing noise); online-k' = what the short search actually chose."""
    out: dict[str, list] = {}
    meta = []   # (shape oracle us) for the regime split, aligned with every list
    for r in rows:
        t = screen.get((r["M"], r["N"], r["K"]))
        if not t:
            continue
        meta.append(min(t.values()))
        cands = r["candidates"]
        out.setdefault("top-1", []).append(regret(t, cands[0]["config_key"] if cands and cands[0]["valid"] else None))
        for j in range(1, k + 1):
            ok = [c["config_key"] for c in cands[:j] if c["valid"] and c["config_key"] in t]
            out.setdefault(f"best-of-{j}", []).append(
                regret(t, min(ok, key=t.get)) if ok else (CAP if not any(c["valid"] for c in cands[:j]) else None))
            out.setdefault(f"online-{j}", []).append(regret(t, r["choice"][str(j)]["key"]))
        out["invalid_picks"] = out.get("invalid_picks", []) + [sum(not c["valid"] for c in cands[:k])]
    if fixed is not None:
        shapes = [(r["M"], r["N"], r["K"]) for r in rows if screen.get((r["M"], r["N"], r["K"]))]
        out["fixed"] = [regret(screen[s], fixed) for s in shapes]
    out["_oracle_us"] = meta
    return out


def summarize(vals: list, mask: np.ndarray | None = None) -> dict:
    """median / p90 / p95 / mean / worst and the share within EPS over the scorable shapes; `unscored` counts picks that
    are absent from the screen (excluded) and `no_pick` counts shapes with no valid pick (counted at CAP)."""
    v = np.array([np.nan if x is None else x for x in vals], float)
    keep = np.ones(len(v), bool) if mask is None else mask
    unscored = int(np.isnan(v[keep]).sum())
    x = v[keep & ~np.isnan(v)]
    if not len(x):
        return {"n": 0, "unscored": unscored, "no_pick": 0}
    return {"n": int(len(x)), "median": float(np.median(x)), "p90": float(np.percentile(x, 90)),
            "p95": float(np.percentile(x, 95)), "mean": float(x.mean()), "worst": float(x.max()),
            "within_eps": float((x <= EPS).mean()), "unscored": unscored, "no_pick": int((x == CAP).sum())}


def fmt(s: dict) -> str:
    if not s.get("n"):
        return "n/a"
    return f"{100 * s['median']:.1f}% / {100 * s['p95']:.1f}% ({100 * s['mean']:.1f}%)"


def build(data: Path, report: Path, *, screen_name: str = "sweep_screen_heldout", k: int = 3,
          fixed: str | None = FIXED_3080, variants=VARIANTS) -> tuple[Path, dict]:
    screen = load_screen(data / screen_name)
    summary: dict = {"hw_id": data.name, "n_screen_shapes": len(screen), "k": k, "fixed_config": fixed, "variants": {}}
    for v in variants:
        p = data / out_name(v)
        if not p.exists():
            continue
        rows = [json.loads(x) for x in p.read_text().splitlines()]
        rows = [r for r in rows if r.get("k") == k]
        sc = score_rows(screen, rows, k, fixed)
        oracle = np.array(sc.pop("_oracle_us"))
        inv = sc.pop("invalid_picks", [])
        masks = {"all": None, ">= 50 us": oracle >= SMALL_US, "< 50 us": oracle < SMALL_US}
        summary["variants"][v] = {
            "n_shapes": len(oracle), "invalid_picks_per_shape_mean": float(np.mean(inv)) if inv else None,
            "scores": {name: {g: summarize(vals, m) for g, m in masks.items()} for name, vals in sc.items()}}
    lines = [f"# Zero-shot selector transfer: `{data.name}`", "",
             f"Source model: lgbm_rank fit on another GPU, no refit. Oracle: this GPU's held-out screen "
             f"(`{screen_name}`, {len(screen)} complete shapes). Cells are median / p95 (mean) regret vs the screen oracle, "
             f"the format of the 3080 `Screen` table in `reports/learned_topk.md`. A shape with no valid pick counts as "
             f"{100 * CAP:.0f}% regret.", ""]
    for v, d in summary["variants"].items():
        inv = d["invalid_picks_per_shape_mean"]
        lines += [f"## Variant `{v}` ({d['n_shapes']} shapes, {'n/a' if inv is None else f'{inv:.2f}'} invalid picks per shape "
                  f"out of {k})", "", "| pick | all | >= 50 us | < 50 us | within 2% | no pick | unscored |", "|---|---|---|---|---|---|---|"]
        for name, g in d["scores"].items():
            lines.append(f"| {name} | {fmt(g['all'])} | {fmt(g['>= 50 us'])} | {fmt(g['< 50 us'])} | "
                         f"{100 * g['all'].get('within_eps', 0):.0f}% | {g['all'].get('no_pick', 0)} | {g['all'].get('unscored', 0)} |")
        lines.append("")
    lines += ["## RTX 3080 reference (same method, its own screen)", "", "| pick | all |", "|---|---|"]
    lines += [f"| {n} | {100 * a:.1f}% / {100 * b:.1f}% ({100 * c:.1f}%) |" for n, (a, b, c) in REFERENCE_3080.items()]
    lines += ["", f"`fixed` row: the 3080's B6 config `{fixed}` used as-is on every shape.",
              "`unscored`: a pick that is not in this GPU's screen (pruned there), excluded from the stats.",
              "`online-k` is what the short search chose; `best-of-k` is the best of the same k candidates by screen time."]
    report.mkdir(parents=True, exist_ok=True)
    out = report / f"zero_shot_{data.name}.md"
    out.write_text("\n".join(lines) + "\n")
    (report / f"zero_shot_{data.name}_summary.json").write_text(json.dumps(summary, indent=1))
    return out, summary


def _main(argv=None):
    ap = argparse.ArgumentParser(prog="analysis.zero_shot", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clock-lock", default=None, help="declared SM clock lock in MHz, like `tune --clock-lock`")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="rank + short online search on this GPU for each variant (needs the GPU, idle)")
    p.add_argument("--ranker", type=Path, required=True, help="saved model stem, e.g. <dir>/lgbm_rank (.txt + .json)")
    p.add_argument("--shapes-file", type=Path, default=None,
                   help="default <data>/sweep_screen_heldout/subset.json (the shapes the oracle covers)")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--out", type=Path, default=None, help="default data/<hw_id>")
    p.add_argument("--fresh-cache", action="store_true", help="compile into an empty TRITON_CACHE_DIR under --out")
    p.add_argument("--no-pin", action="store_true")
    p.add_argument("--session", default=None)
    p = sub.add_parser("score", help="score the runs against this GPU's screen (no GPU needed with --data)")
    p.add_argument("--data", type=Path, default=None, help="default data/<hw_id> of this machine")
    p.add_argument("--report", type=Path, default=Path("reports"))
    p.add_argument("--screen", default="sweep_screen_heldout")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--fixed-config", default=FIXED_3080, help="fixed-config baseline row ('' to skip)")
    args = ap.parse_args(argv)
    if args.clock_lock is not None:
        os.environ["TUNE_CLOCK_LOCK"] = args.clock_lock
    if args.cmd == "score":
        from tune.env import default_data_dir
        data = args.data or default_data_dir()
        out, s = build(data, args.report, screen_name=args.screen, k=args.k, fixed=args.fixed_config or None)
        for v, d in s["variants"].items():
            g = d["scores"][f"online-{args.k}"]["all"]
            print(f"{v}: {d['n_shapes']} shapes, online-{args.k} {fmt(g)}")
        print(f"-> {out}")
        return 0
    from tune.env import default_data_dir, pin_cpu
    out_dir = args.out or default_data_dir()
    if args.fresh_cache:
        cache = out_dir / "online_triton_cache" / time.strftime("%Y%m%d-%H%M%S")
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = str(cache)  # before any Triton compile in this process
    if not args.no_pin:
        pin_cpu()
    shapes_file = args.shapes_file or out_dir / "sweep_screen_heldout" / "subset.json"
    shapes = json.loads(shapes_file.read_text())["shapes"]
    paths = run(shapes, args.ranker, out_dir, variants=tuple(args.variants.split(",")), k=args.k, session_id=args.session)
    print("zero-shot rows -> " + ", ".join(map(str, paths)))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
