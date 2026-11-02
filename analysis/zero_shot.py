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

More than one model, the joint selector and the k policy (added 2026-11-01):
  * `--ranker v1=<dir>/lgbm_rank v2=<dir>/lgbm_rank_v2` runs several saved models (rank v1 and rank v2, reports/week1.md
    2026-11-01) in one invocation: `name=stem` pairs write zero_shot_<name>_<variant>.jsonl; a bare path keeps the original
    zero_shot_<variant>.jsonl names, so results already produced with the old version still score. v2 needs its
    lgbm_rank_v2.txt/.json copied by hand like v1 (its constants are fixed model units, not this GPU's roofline: do not change them).
    `native` rebuilds v2's pruning space from this GPU's limits, which is what a deployment would do.
  * `--lt-n 16` also times cuBLASLt's heuristic top 16 in the same short-timing pool (joint selection). The screen oracle is
    Triton-only, so the screen table scores the Triton-only choice (`choice_lt` "<k>_lt0"); the joint choice is scored in a
    small stage-2 session on this GPU against B2 / B3 (see the `picks` subcommand and analysis.ranker_compare).
  * `--k 3 --k-frequent 5 --frequent-calls 1000000` applies tune.online.KPolicy: a shape in the shapes file with `calls` >= the
    threshold (or `frequent: true`) gets k = 5, every other shape k = 3. Held-out shapes carry no call counts, so for the
    evaluation just run `--k 5` and read `online-3` (default tier) and `online-5` (frequent tier) from the same rows.

Runbook on the target machine (e.g. an RTX 2060; GPU idle, one GPU job at a time, AGENTS.md section 7):
  0. check out the commit with this module (the saved model is NOT in git: data/ is gitignored, lgbm_rank.txt is ~35 MB, so
     copy data/<src_hw>/lgbm_rank.txt and lgbm_rank.json to this machine by hand); pip versions must match requirements.txt
  1. pytest -q tests/test_correctness.py tests/test_zero_shot.py     and     python -m tune env
  2. lock the SM clock to a value this GPU holds under load (sudo nvidia-smi -lgc X,X) and use the same X everywhere below;
     the 3080's 1755 MHz does not apply, and numbers are only comparable at equal lock state and driver
  3. the oracle, this GPU's own screen of the held-out shapes (hours; ~1.3 h on the 3080, expect longer on a 2060):
        python -m tune --clock-lock X sweep --split heldout --per-suite 1000
     --per-suite 1000 gives the same 86 shapes as the 3080 run (the default of 24 per suite is a smaller subset)
  4. python -m analysis.zero_shot --clock-lock X run --ranker v1=<dir>/lgbm_rank v2=<dir>/lgbm_rank_v2 --k 5 --lt-n 16 --fresh-cache
     (--clock-lock goes before the subcommand; only start this after step 3 has finished: never two GPU jobs at once;
      plain `--ranker <dir>/lgbm_rank --k 3` is the original Triton-only experiment)
  5. python -m analysis.zero_shot --clock-lock X score --k 3    -> reports/zero_shot_<hw_id>.md (+ _summary.json);
     `score --k 5` for the frequent tier
  6. (joint selector, optional, ~0.5 h) python -m analysis.zero_shot picks   writes the picks file and prints the two
     `tune benchmark` commands (cold, and graph for the shapes under 50 us) and the final `analysis.ranker_compare` command

Reading the report: online-3 and best-of-3 of `native` vs `as_is` (the gap is what the source GPU's numbers in the features
cost); the 3080 reference rows at the bottom; the `fixed` row (if the ranker barely beats one reused config the transfer is
weak); and the no-pick / unscored counts, the `regret > 50%` count and invalid picks per shape (a high rate for as_is means the source GPU's limits
produce configs this GPU cannot compile). Caveats: the `native` pruner assumes Ampere-style stage buffers, so on Turing it
may drop tiles that would run (conservative, a lower bound); 6 GB cards record a too-large shape as a failed shape; a hard
CUDA fault poisons the process, so rerun `run` (it resumes). Afterwards add a dated update to reports/week1.md (AGENTS.md
section 7) with the session ids, row counts and the table.

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
OVER = 0.5       # a screen row cut as `slow` was > 1.5x the best so far at its first sample, so its regret is > 50%
SMALL_US = 50.0
VARIANTS = ("as_is", "native")
# The 3080 reference rows (median, p95, mean; all 86 held-out shapes, regret vs the held-out screen oracle, diverse picks):
# rank v1 from reports/learned_topk.md "Screen" table; rank v2 from reports/week1.md 2026-11-01 (same method).
REFERENCE_3080 = {"top-1": (0.036, 0.181, 0.055), "best-of-3 (diverse)": (0.020, 0.106, 0.034)}
REFERENCE_3080_V2 = {"top-1": (0.028, 0.130, 0.046), "best-of-3 (diverse)": (0.010, 0.093, 0.021),
                     "best-of-5 (diverse)": (0.005, 0.078, 0.017)}
# B6 on the 3080 (AGENTS.md section 6): the "just reuse the single best 3080 config" baseline.
FIXED_3080 = "bm32_bn32_bk32_w4_s5_g1_sk1"


def out_name(variant: str, model: str = "") -> str:
    """Rows file of one (model, variant); model "" keeps the original zero_shot_<variant>.jsonl name."""
    return f"zero_shot_{model}_{variant}.jsonl" if model else f"zero_shot_{variant}.jsonl"


def parse_rankers(items: list[str]) -> list[tuple[str, Path]]:
    """`--ranker` values: `name=stem` (named model) or a bare stem (the original, unnamed model)."""
    out = []
    for x in items:
        name, _, stem = x.partition("=") if "=" in x else ("", "", x)
        if name and not name.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"model name {name!r} must be alphanumeric (it is part of a file name)")
        out.append((name, Path(stem)))
    if len({n for n, _ in out}) != len(out):
        raise ValueError("duplicate model names in --ranker")
    return out


def find_models(data: Path, variants=VARIANTS) -> list[str]:
    """Models with rows in `data`: "" for the original file names, then every <name> of zero_shot_<name>_<variant>.jsonl."""
    names = [""] if any((data / out_name(v)).exists() for v in variants) else []
    for v in variants:
        for p in sorted(data.glob(f"zero_shot_*_{v}.jsonl")):
            name = p.name[len("zero_shot_"):-len(f"_{v}.jsonl")]
            if name and name not in names:
                names.append(name)
    return names


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
        session_id: str | None = None, log=print, name: str = "", lt_n: int = 0, policy=None) -> list[Path]:
    """Rank + short online search for each variant of one saved model. name: output file tag (see out_name); lt_n > 0:
    also time cuBLASLt's heuristic top lt_n (joint selection); policy: a tune.online.KPolicy making k per shape from
    the shape's `calls` / `frequent` fields (k is then the default tier)."""
    from tune.env import provenance, write_env
    from tune.online import all_choices, search_shape, shape_k
    from tune.records import RecordWriter
    import torch

    session_id = session_id or time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    env = write_env(out_dir, session_id=session_id, kind="zero_shot")
    paths, source_gpu = [], source_gpu_name(ranker_stem)
    for variant in variants:
        ranker = variant_ranker(ranker_stem, variant, env["gpu"])
        rule = ranker.rule("diverse")
        path = out_dir / out_name(variant, name)
        paths.append(path)
        done = set()
        if path.exists():
            done = {(r["M"], r["N"], r["K"], r.get("k")) for r in map(json.loads, path.read_text().splitlines())
                    if r.get("rule") == rule and r.get("lt_n", 0) == lt_n}
        todo = [dict(s, _k=shape_k(s, k, policy)) for s in shapes]
        todo = [s for s in todo if (s["M"], s["N"], s["K"], s["_k"]) not in done]
        t0 = time.perf_counter()
        with RecordWriter(path) as w:
            for i, s in enumerate(todo):
                M, N, K, ks = s["M"], s["N"], s["K"], s["_k"]
                t = time.perf_counter()
                r = search_shape(ranker, M, N, K, ks, torch.float16, "diverse", lt_n)
                row = {"M": M, "N": N, "K": K, "dtype": "fp16", "k": ks, "order": "diverse", "rule": rule, "lt_n": lt_n,
                       "model": name or "lgbm_rank", "variant": variant, "split": s.get("split"), "suite": s.get("suite"),
                       **({"calls": s.get("calls"), "frequent": s.get("frequent")} if policy else {}),
                       "session_id": session_id, "hw_id": env["hw_id"], "driver": env["driver"], **provenance(env),
                       "source_gpu": source_gpu,
                       "ranker": str(ranker_stem), "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
                       "wall_s": time.perf_counter() - t, **r, **all_choices(r["candidates"], ks, lt_n),
                       "timestamp": time.time()}
                w.write(row)
                el = time.perf_counter() - t0
                log(f"[{name or 'lgbm_rank'} {variant} {i + 1}/{len(todo)}] {M}x{N}x{K} k={ks} -> {row['choice'][str(ks)]['key']} "
                    f"({row['wall_s']:.1f}s) | {el:.0f}s elapsed, ETA {el / (i + 1) * (len(todo) - i - 1):.0f}s")
    return paths


# ---------------------------------------------------------------- scoring (no GPU)

def load_screen(screen_dir: Path, include_slow: bool = True) -> dict:
    """{(M, N, K): {config_key: us}} over every complete screen file. The screen cuts a config as `slow` after one sample once it is
    > 1.5x the best so far: that one-sample time is a real measurement and proves regret > 50%, so it is kept (`include_slow`).
    Dropping those rows (the first version of this function did) silently removed exactly the bad picks from every statistic:
    on the 2060 12-16 of 27 zero-shot picks were `slow` and the table looked 5-10x better than it was. The oracle is the fastest
    `ok` row (a `slow` row cannot be below it); a shape with no `ok` row is skipped."""
    keep = ("ok", "slow") if include_slow else ("ok",)
    out = {}
    for p in sorted(screen_dir.glob("*_fp16.json")):
        d = json.loads(p.read_text())
        if not d.get("complete"):
            continue
        if not any(r["status"] == "ok" and r["us"] is not None for r in d["rows"]):
            continue
        out[(d["M"], d["N"], d["K"])] = {r["key"]: r["us"] for r in d["rows"] if r["status"] in keep and r["us"] is not None}
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
    n_skipped = 0
    for r in rows:
        t = screen.get((r["M"], r["N"], r["K"]))
        if not t:
            continue
        if r.get("skipped"):  # tensors did not fit in VRAM (tune.online.search_shape): not scorable, not a bad pick
            n_skipped += 1
            continue
        meta.append(min(t.values()))
        cands = r["candidates"]
        out.setdefault("top-1", []).append(regret(t, cands[0]["config_key"] if cands and cands[0]["valid"] else None))
        for j in range(1, k + 1):
            ok = [c["config_key"] for c in cands[:j] if c["valid"] and c["config_key"] in t]
            out.setdefault(f"best-of-{j}", []).append(
                regret(t, min(ok, key=t.get)) if ok else (CAP if not any(c["valid"] for c in cands[:j]) else None))
            # The screen holds Triton configs only: a joint row's Triton-only choice ("<j>_lt0") is what it can score.
            ch = (r.get("choice_lt") or {}).get(f"{j}_lt0") or r["choice"][str(j)]
            out.setdefault(f"online-{j}", []).append(regret(t, ch["key"]))
        out["invalid_picks"] = out.get("invalid_picks", []) + [sum(not c["valid"] for c in cands[:k])]
    if fixed is not None:
        shapes = [(r["M"], r["N"], r["K"]) for r in rows if screen.get((r["M"], r["N"], r["K"])) and not r.get("skipped")]
        out["fixed"] = [regret(screen[s], fixed) for s in shapes]
    out["_oracle_us"] = meta
    out["_skipped"] = n_skipped
    return out


def summarize(vals: list, mask: np.ndarray | None = None) -> dict:
    """median / p90 / p95 / mean / worst, the share within EPS and `over_50` (picks with regret > 50%) over the scorable shapes;
    `unscored` counts picks that are absent from the screen (pruned or failed there: excluded, so read it) and `no_pick` counts
    shapes with no valid pick (counted at CAP)."""
    v = np.array([np.nan if x is None else x for x in vals], float)
    keep = np.ones(len(v), bool) if mask is None else mask
    unscored = int(np.isnan(v[keep]).sum())
    x = v[keep & ~np.isnan(v)]
    if not len(x):
        return {"n": 0, "unscored": unscored, "no_pick": 0}
    return {"n": int(len(x)), "median": float(np.median(x)), "p90": float(np.percentile(x, 90)),
            "p95": float(np.percentile(x, 95)), "mean": float(x.mean()), "worst": float(x.max()),
            "within_eps": float((x <= EPS).mean()), "over_50": int((x > OVER).sum()), "unscored": unscored, "no_pick": int((x == CAP).sum())}


def fmt(s: dict) -> str:
    if not s.get("n"):
        return "n/a"
    return f"{100 * s['median']:.1f}% / {100 * s['p95']:.1f}% ({100 * s['mean']:.1f}%)"


def build(data: Path, report: Path, *, screen_name: str = "sweep_screen_heldout", k: int = 3,
          fixed: str | None = FIXED_3080, variants=VARIANTS, models: list[str] | None = None) -> tuple[Path, dict]:
    """Score every (model, variant) rows file in `data` against the screen. Rows searched with a larger k (a k = 5 run, or
    a KPolicy run) also score at this k, since they store the choice for every k' <= k. models: names as in out_name
    ("" = the original file names); default: every model found."""
    screen = load_screen(data / screen_name)
    models = find_models(data, variants) if models is None else models
    summary: dict = {"hw_id": data.name, "n_screen_shapes": len(screen), "k": k, "fixed_config": fixed, "variants": {}}
    for model in models:
        for v in variants:
            p = data / out_name(v, model)
            if not p.exists():
                continue
            rows = [r for r in map(json.loads, p.read_text().splitlines()) if r.get("k", 0) >= k]
            if not rows:
                continue
            sc = score_rows(screen, rows, k, fixed)
            oracle = np.array(sc.pop("_oracle_us"))
            n_skipped = sc.pop("_skipped")
            inv = sc.pop("invalid_picks", [])
            masks = {"all": None, ">= 50 us": oracle >= SMALL_US, "< 50 us": oracle < SMALL_US}
            summary["variants"][f"{model}/{v}" if model else v] = {
                "model": model or "lgbm_rank", "variant": v, "joint_rows": bool(rows[0].get("lt_n")),
                "n_shapes": len(oracle), "n_skipped_oom": n_skipped,
                "invalid_picks_per_shape_mean": float(np.mean(inv)) if inv else None,
                "scores": {name: {g: summarize(vals, m) for g, m in masks.items()} for name, vals in sc.items()}}
    lines = [f"# Zero-shot selector transfer: `{data.name}`", "",
             f"Source models: lgbm_rank (v1) and, when present, lgbm_rank_v2, fit on another GPU, no refit. Oracle: this GPU's held-out screen "
             f"(`{screen_name}`, {len(screen)} complete shapes). Cells are median / p95 (mean) regret vs the screen oracle, "
             f"the format of the 3080 `Screen` table in `reports/learned_topk.md`. A shape with no valid pick counts as "
             f"{100 * CAP:.0f}% regret. Scored at k = {k}; the screen holds Triton configs only, so joint rows are scored on their "
             f"Triton-only choice (the joint choice is scored in a stage-2 session: `picks`, then analysis.ranker_compare).", ""]
    for key, d in summary["variants"].items():
        inv = d["invalid_picks_per_shape_mean"]
        lines += [f"## Model `{d['model']}`, variant `{d['variant']}` ({d['n_shapes']} shapes, "
                  f"{'n/a' if inv is None else f'{inv:.2f}'} invalid picks per shape out of {k}"
                  + (f"; {d['n_skipped_oom']} shapes skipped, out of VRAM" if d["n_skipped_oom"] else "") + ")", "",
                  "| pick | all | >= 50 us | < 50 us | within 2% | regret > 50% | no pick | unscored |", "|---|---|---|---|---|---|---|---|"]
        for name, g in d["scores"].items():
            lines.append(f"| {name} | {fmt(g['all'])} | {fmt(g['>= 50 us'])} | {fmt(g['< 50 us'])} | "
                         f"{100 * g['all'].get('within_eps', 0):.0f}% | {g['all'].get('over_50', 0)} of {g['all'].get('n', 0)} | "
                         f"{g['all'].get('no_pick', 0)} | {g['all'].get('unscored', 0)} |")
        lines.append("")
    lines += ["## RTX 3080 reference (same method, its own screen)", "", "| pick | rank v1 | rank v2 |", "|---|---|---|"]
    fm = lambda t: f"{100 * t[0]:.1f}% / {100 * t[1]:.1f}% ({100 * t[2]:.1f}%)"
    names = list(dict.fromkeys([*REFERENCE_3080, *REFERENCE_3080_V2]))
    lines += [f"| {n} | {fm(REFERENCE_3080[n]) if n in REFERENCE_3080 else 'n/a'} | "
              f"{fm(REFERENCE_3080_V2[n]) if n in REFERENCE_3080_V2 else 'n/a'} |" for n in names]
    lines += ["", f"`fixed` row: the 3080's B6 config `{fixed}` used as-is on every shape.",
              "`regret > 50%`: scored picks at least 50% slower than the oracle, including picks the screen cut as `slow` (their one-sample time is kept).",
              "`unscored`: a pick that is not in this GPU's screen at all (pruned or failed there), excluded from the stats: read it before trusting a row.",
              "The 3080 reference is over its full 86-shape held-out set; for a run on a subset, compare on the shared shapes with `python -m analysis.cross_gpu`.",
              "`online-k` is what the short search chose; `best-of-k` is the best of the same k candidates by screen time."]
    report.mkdir(parents=True, exist_ok=True)
    out = report / f"zero_shot_{data.name}.md"   # one fixed name per data directory: score k = 3 and k = 5 into different --report folders
    out.write_text("\n".join(lines) + "\n")
    (report / f"zero_shot_{data.name}_summary.json").write_text(json.dumps(summary, indent=1))
    return out, summary


def write_stage2_picks(data: Path, screen_name: str = "sweep_screen_heldout") -> tuple[Path, Path, int]:
    """Inputs of the joint-selector stage-2 session on this GPU: the union of every Triton candidate the zero-shot runs
    searched, per shape (<data>/zero_shot_stage2_picks.json, the format of TUNE_PICKS), and the shapes whose screen
    oracle is < 50 us (<data>/zero_shot_lt50us.json) for the graph-amortized pass."""
    picks: dict[str, list[str]] = {}
    for p in sorted(data.glob("zero_shot*_*.jsonl")):
        if p.name.startswith("zero_shot_stage2"):
            continue
        for line in p.read_text().splitlines():
            r = json.loads(line)
            key = f"{r['M']}x{r['N']}x{r['K']}"
            for c in r.get("candidates", []):
                if c.get("family", "triton") == "triton" and c["config_key"] not in picks.setdefault(key, []):
                    picks[key].append(c["config_key"])
    path = data / "zero_shot_stage2_picks.json"
    path.write_text(json.dumps({"picks": picks, "by_selector": {"zero_shot": {}},
                                "note": "union of the Triton candidates of every zero-shot run"}))
    screen = load_screen(data / screen_name)
    base = json.loads((data / screen_name / "subset.json").read_text())
    small = [s for s in base["shapes"] if screen.get((s["M"], s["N"], s["K"])) and min(screen[(s["M"], s["N"], s["K"])].values()) < SMALL_US]
    lt = data / "zero_shot_lt50us.json"
    lt.write_text(json.dumps({**base, "note": f"shapes whose screen oracle is < {SMALL_US:.0f} us", "n": len(small), "shapes": small}))
    return path, lt, len(picks)


def _main(argv=None):
    ap = argparse.ArgumentParser(prog="analysis.zero_shot", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clock-lock", default=None, help="declared SM clock lock in MHz, like `tune --clock-lock`")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="rank + short online search on this GPU for each variant (needs the GPU, idle)")
    p.add_argument("--ranker", nargs="+", required=True,
                   help="saved model stems, bare (<dir>/lgbm_rank, original file names) or name=stem (v2=<dir>/lgbm_rank_v2)")
    p.add_argument("--shapes-file", type=Path, default=None,
                   help="default <data>/sweep_screen_heldout/subset.json (the shapes the oracle covers)")
    p.add_argument("--k", type=int, default=3, help="Triton candidates per shape (default tier with --k-frequent)")
    p.add_argument("--k-frequent", type=int, default=0, help="k for shapes marked `frequent` / with calls >= --frequent-calls; 0 = off")
    p.add_argument("--frequent-calls", type=int, default=1_000_000)
    p.add_argument("--lt-n", type=int, default=0, help="also time cuBLASLt's heuristic top N (joint selection); 0 = Triton only")
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--out", type=Path, default=None, help="default data/<hw_id>")
    p.add_argument("--fresh-cache", action="store_true", help="compile into an empty TRITON_CACHE_DIR under --out")
    p.add_argument("--no-pin", action="store_true")
    p.add_argument("--session", default=None)
    p = sub.add_parser("score", help="score the runs against this GPU's screen (no GPU needed with --data)")
    p.add_argument("--data", type=Path, default=None, help="default data/<hw_id> of this machine")
    p.add_argument("--report", type=Path, default=Path("reports"))
    p.add_argument("--screen", default="sweep_screen_heldout")
    p.add_argument("--k", type=int, default=3, help="score at this k (rows searched with a larger k also score)")
    p.add_argument("--fixed-config", default=FIXED_3080, help="fixed-config baseline row ('' to skip)")
    p = sub.add_parser("picks", help="write the joint-selector stage-2 inputs and print the commands (no GPU)")
    p.add_argument("--data", type=Path, default=None)
    p.add_argument("--screen", default="sweep_screen_heldout")
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
    if args.cmd == "picks":
        from tune.env import default_data_dir
        data = args.data or default_data_dir()
        picks, lt, n = write_stage2_picks(data, args.screen)
        lock = f"--clock-lock {args.clock_lock} " if args.clock_lock else "--clock-lock X "
        files = [f"{(m + '_' if m else '') + v}={out_name(v, m)}" for m in find_models(data) for v in VARIANTS if (data / out_name(v, m)).exists()]
        be = "torch,triton:sweep:8,triton:picks,cublaslt:16"
        print(f"{n} shapes -> {picks}\n{lt}\n\nRun on the idle GPU (never alongside another GPU job), then score:\n"
              f"  export TUNE_SWEEP_DIR={data / args.screen} TUNE_PICKS={picks}\n"
              f"  python -m tune {lock}benchmark --shapes-file {data / args.screen / 'subset.json'} --backends {be} --cache-modes cold --out-name results_zero_shot.jsonl\n"
              f"  python -m tune {lock}benchmark --shapes-file {lt} --backends {be} --cache-modes hot --graph-batch 50 --out-name results_zero_shot_graph.jsonl\n"
              f"  python -m analysis.ranker_compare --data {data} --report reports/zero_shot_{data.name} --results results_zero_shot.jsonl "
              f"--graph-results results_zero_shot_graph.jsonl --online " + " ".join(files)
              + "\n(add cublaslt_deep:20 to --backends for B3+, 20 s per shape; --report keeps ranker_compare from overwriting reports/ranker_compare.md, the 3080's)")
        return 0
    from tune.env import default_data_dir, pin_cpu
    from tune.online import KPolicy
    out_dir = args.out or default_data_dir()
    if args.fresh_cache:
        cache = out_dir / "online_triton_cache" / time.strftime("%Y%m%d-%H%M%S")
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = str(cache)  # before any Triton compile in this process
    if not args.no_pin:
        pin_cpu()
    shapes_file = args.shapes_file or out_dir / "sweep_screen_heldout" / "subset.json"
    shapes = json.loads(shapes_file.read_text())["shapes"]
    policy = KPolicy(args.k, args.k_frequent, args.frequent_calls) if args.k_frequent else None
    paths = []
    for name, stem in parse_rankers(args.ranker):
        paths += run(shapes, stem, out_dir, variants=tuple(args.variants.split(",")), k=args.k, session_id=args.session,
                     name=name, lt_n=args.lt_n, policy=policy)
    print("zero-shot rows -> " + ", ".join(map(str, paths)))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
