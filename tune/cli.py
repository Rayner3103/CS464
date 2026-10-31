"""Command-line entry point: `python -m tune <command>` (or `tune <command>` once pip-installed).

  env        write data/<hw_id>/env.json
  check      correctness gate for backends over a suite (default: edge)
  roofline   B0: measure peak TFLOP/s and DRAM bandwidth -> data/<hw_id>/roofline.json
  benchmark  validate + time backends over suites -> data/<hw_id>/<out>.jsonl  (resumable)
  sweep      stage-1 screen of the full pruned Triton config space on a coarse shape subset -> data/<hw_id>/sweep_screen/
  baseline   alias of benchmark with --backends defaulting to the vendor/reference baselines
  noise      noise-floor study -> data/<hw_id>/noise.jsonl
  drift      thermal drift run -> data/<hw_id>/drift-*.csv
  analyze    build plots + markdown summaries into reports/
  portfolio  config-space reduction from the sweep screen (B6 over the full space, G3, G5) -> reports/sweep_portfolio.md
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from tune.env import default_data_dir, pin_cpu, write_env


def _parse_budget(s: str | None) -> float | None:
    if not s:
        return None
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([smh]?)", s)
    if not m:
        raise argparse.ArgumentTypeError(f"bad budget {s!r}; use e.g. 90s, 30m, 2h")
    return float(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600}[m.group(2)]


def _shapes(suites: str, shapes_file: str | None = None) -> list[dict]:
    if shapes_file:
        return json.loads(Path(shapes_file).read_text())["shapes"]
    from tune.runner import load_suite
    names = ["S1", "S2", "S3", "S4", "S5", "S6", "S7"] if suites == "all" else suites.split(",")
    return [s for n in names for s in load_suite(n)]


def cmd_env(args):
    env = write_env(args.out)
    print(json.dumps({k: env[k] for k in ("hw_id", "driver", "versions", "clock_lock", "display_active",
                                          "harness_git")}, indent=1))
    print(f"wrote {args.out / 'env.json'}")


def cmd_check(args):
    import torch
    from tune.correctness import validate
    from tune.runner import DTYPES, candidates_for

    from tune.env import provenance
    env = write_env(args.out, kind="check")
    shapes = _shapes(args.suite)
    fails, total = [], 0
    per_cfg: dict[str, list[int]] = {}
    for s in shapes:
        M, N, K = s["M"], s["N"], s["K"]
        for spec in args.backends.split(","):
            for c in candidates_for(spec, M, N, K):
                r = validate(c.backend, M, N, K, DTYPES[args.dtype])
                total += 1
                tag = f"{c.family}/{c.config_key}"
                per_cfg.setdefault(tag, [0, 0])[0 if r.valid else 1] += 1
                if not r.valid:
                    fails.append({"M": M, "N": N, "K": K, "backend": tag, **r.to_dict()})
        torch.cuda.empty_cache()
    print(f"{total - len(fails)}/{total} passed")
    for tag, (ok, bad) in sorted(per_cfg.items()):
        print(f"  {tag:60s} pass={ok} fail={bad}")
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"check-{args.suite}.json"
    out.write_text(json.dumps({"suite": args.suite, "dtype": args.dtype, "total": total, "failures": fails,
                               "per_config": per_cfg, "timestamp": time.time(),
                               "session_id": env["session_id"], **provenance(env)}, indent=1, default=str))
    print(f"wrote {out}")
    for f in fails[:20]:
        print("  FAIL", f)
    return 1 if fails else 0


def cmd_roofline(args):
    from baselines.roofline import measure_bandwidth_gbps, measure_peak_tflops
    from tune.env import provenance, spec_peaks
    env = write_env(args.out, kind="roofline")
    res = {"spec": spec_peaks(), "bandwidth": measure_bandwidth_gbps(), "compute": measure_peak_tflops(),
           "timestamp": time.time(), "session_id": env["session_id"], **provenance(env)}
    (args.out / "roofline.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def cmd_benchmark(args):
    from tune.runner import run
    timing = {}
    if args.max_time_s is not None:
        timing["max_time_s"] = args.max_time_s
    if args.target_rse is not None:
        timing["target_rse"] = args.target_rse
    if args.graph_batch:
        # Amortized per-launch time for us-scale kernels (methodology: per-launch events tick in ~1.02 us steps).
        if set(args.cache_modes.split(",")) != {"hot"}:
            raise SystemExit("--graph-batch needs --cache-modes hot (cold flushes L2 before every launch)")
        timing.update(batch=args.graph_batch, use_graph=True)
    path = run(_shapes(args.suite, args.shapes_file), args.backends.split(","), args.out, dtype=args.dtype,
               cache_modes=tuple(args.cache_modes.split(",")), timing_overrides=timing,
               budget_s=_parse_budget(args.budget), session_id=args.session, prune_shapes=args.prune_shapes,
               out_name=args.out_name, n_allocs=args.n_allocs)
    print(f"results -> {path}")


def cmd_sweep(args):
    from tune.sweep import HELDOUT_SPLITS, coarse_subset, run_sweep
    splits = HELDOUT_SPLITS if args.split == "heldout" else ("tune",)
    screen_dir = args.screen_dir or ("sweep_screen_heldout" if args.split == "heldout" else "sweep_screen")
    shapes = coarse_subset(_shapes(args.suite), args.per_suite, args.max_gflop, splits)
    sweep_dir = args.out / screen_dir
    sweep_dir.mkdir(parents=True, exist_ok=True)
    subset = sweep_dir / "subset.json"
    subset.write_text(json.dumps({"suite": args.suite, "split": args.split, "per_suite": args.per_suite, "max_gflop": args.max_gflop,
                                  "n": len(shapes), "shapes": shapes}, indent=1))
    print(f"{len(shapes)} shapes -> {subset}")
    if args.list_only:
        return
    print(f"screen -> {run_sweep(shapes, args.out, workers=args.workers, session_id=args.session, screen_dir=screen_dir)}")


def cmd_noise(args):
    from tune.noise import run_noise
    shapes = json.loads(Path(args.shapes).read_text())["shapes"]
    path = run_noise(shapes, args.out, backends=tuple(args.backends.split(",")), repeats=args.repeats,
                     rounds=args.rounds, cache_modes=tuple(args.cache_modes.split(",")), session_id=args.session,
                     n_allocs=args.n_allocs)
    print(f"noise -> {path}")


def cmd_drift(args):
    from tune.noise import run_drift
    print(f"drift -> {run_drift(args.out, minutes=args.minutes, n=args.n)}")


def cmd_portfolio(args):
    from analysis.portfolio import build, build_heldout
    if args.heldout:
        out, s = build_heldout(args.data, args.report, results=args.results)
        print(f"{s['n_heldout_shapes']} held-out shapes; G5 size in-sample {s['g5_size_in_sample']}, held-out "
              f"{s['g5_size_heldout_tune_order']} (re-fit {s['g5_size_heldout_refit']}) -> {out}")
        return
    out, s = build(args.data, args.report)
    print(f"{s['n_shapes']} shapes x {s['n_configs']} configs; B6 {s['b6']} median regret {100 * s['b6_regret_median']:.1f}%; "
          f"G5 size {s['g5_portfolio_size_p95_lt_5pct']} -> {out}")


def cmd_selectors(args):
    from analysis.selectors import build
    out, s = build(args.data, args.report, results=args.results, graph_results=args.graph_results)
    print(f"{s['n_heldout_shapes']} held-out shapes; held-out screen p95 regret: "
          + ", ".join(f"{n} {100 * v['all']['p95']:.1f}%" for n, v in s["screen"].items()) + f" -> {out}")


def cmd_learned(args):
    from analysis.learned import build
    out, s = build(args.data, args.report, results=args.results, graph_results=args.graph_results)
    print(f"{s['n_heldout_shapes']} held-out shapes; held-out screen median / p95 regret: "
          + ", ".join(f"{n} {100 * v['all']['p50']:.1f}% / {100 * v['all']['p95']:.1f}%" for n, v in s["screen"].items())
          + f" -> {out}")


def cmd_topk(args):
    from analysis.learned import build_topk
    out, s = build_topk(args.data, args.report, results=args.results, graph_results=args.graph_results,
                        online=args.online, online_diverse=args.online_diverse, online_inner=args.online_inner)
    print(f"diverse rule {s['diverse_rule']}; held-out screen best-of-k median / p95: " + "; ".join(
        f"{o} " + ", ".join(f"k={k} {100 * v['p50']:.1f}% / {100 * v['p95']:.1f}%" for k, v in sc.items())
        for o, sc in s["screen_best_of_k"].items()) + f" -> {out}")


def cmd_ranker_v2(args):
    from analysis.learned import train_ranker_v2
    rk, info = train_ranker_v2(args.data, tuple(args.extra_screens), args.stem)
    r = info["rules"][0]
    print(f"saved {args.data / args.stem}.txt/.json: {len(rk.pool_keys)} pool configs, {rk.meta['n_tune_shapes']} tune shapes; "
          f"diverse rule {r['rule']} (tune OOF best-of-3 mean {100 * r['k3']['mean']:.2f}%, p95 {100 * r['k3']['p95']:.1f}%)")


def cmd_online(args):
    if args.fresh_cache:
        # Before any Triton compile in this process: compile cost must be a first use, not a cache hit.
        cache = args.out / "online_triton_cache" / time.strftime("%Y%m%d-%H%M%S")
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = str(cache)
    from tune.online import run
    out_name = args.out_name or ("online_lgbm_rank.jsonl" if args.order == "plain" else f"online_lgbm_rank_{args.order}.jsonl")
    if args.lt_n and not args.out_name:
        out_name = out_name.replace("online_lgbm_rank", "online_joint")
    from tune.online import KPolicy
    policy = KPolicy(args.k, args.k_frequent, args.frequent_calls) if args.k_frequent else None
    path = run(_shapes(args.suite, args.shapes_file), Path(args.ranker or args.out / "lgbm_rank"), args.out, k=args.k,
               dtype=args.dtype, order=args.order, out_name=out_name, session_id=args.session, lt_n=args.lt_n, policy=policy)
    print(f"online -> {path}")


def cmd_analyze(args):
    from analysis.report import build_reports
    build_reports(args.data, args.report, args.results)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tune", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-pin", action="store_true", help="do not pin the process to one CPU core")
    ap.add_argument("--clock-lock", default=None,
                    help="declare the SM clock lock in effect, e.g. 1755 (after nvidia-smi -lgc 1755,1755) or 'none'. "
                         "Recorded in env.json and on every row; default: env var TUNE_CLOCK_LOCK, else probed")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, **kw):
        p = sub.add_parser(name, **kw)
        p.set_defaults(fn=fn)
        p.add_argument("--out", type=Path, default=None, help="default: data/<hw_id>/")
        return p

    add("env", cmd_env)
    p = add("check", cmd_check)
    p.add_argument("--suite", default="edge")
    p.add_argument("--backends", default="triton:handpicked,triton_tutorial,torch")
    p.add_argument("--dtype", default="fp16")
    add("roofline", cmd_roofline)
    for name, default_backends in (("benchmark", "triton:handpicked"), ("baseline", "torch,triton_tutorial")):
        p = add(name, cmd_benchmark)
        p.add_argument("--suite", default="S1,S2")
        p.add_argument("--backends", "--which", dest="backends", default=default_backends)
        p.add_argument("--dtype", default="fp16")
        p.add_argument("--cache-modes", default="cold")
        p.add_argument("--budget", default=None, help="e.g. 10m, 30m, 2h")
        p.add_argument("--session", default=None)
        p.add_argument("--prune-shapes", action="store_true")
        p.add_argument("--max-time-s", type=float, default=None)
        p.add_argument("--target-rse", type=float, default=None)
        p.add_argument("--out-name", default="results.jsonl")
        p.add_argument("--n-allocs", type=int, default=3, help="input placements pooled per measurement")
        p.add_argument("--graph-batch", type=int, default=0,
                       help="time N launches captured in a CUDA graph and report time/N (hot only)")
        p.add_argument("--shapes-file", default=None, help="JSON with a 'shapes' list (e.g. sweep_screen/subset.json); "
                                                           "overrides --suite")
    p = add("sweep", cmd_sweep)
    p.add_argument("--suite", default="all")
    p.add_argument("--per-suite", type=int, default=24, help="shapes per suite in the coarse subset")
    p.add_argument("--max-gflop", type=float, default=1200.0, help="skip shapes above this many GFLOP")
    p.add_argument("--workers", type=int, default=0, help="parallel compile processes; 0 = auto (up to 10). Either way capped by free host RAM and VRAM (~1 GiB and ~450 MiB per worker), see tune/sweep.py")
    p.add_argument("--session", default=None)
    p.add_argument("--list-only", action="store_true", help="write the subset and stop")
    p.add_argument("--split", choices=["tune", "heldout"], default="tune",
                   help="heldout = heldout_region + heldout_random shapes (G5 out-of-sample check)")
    p.add_argument("--screen-dir", default=None,
                   help="directory under --out; default sweep_screen (tune) or sweep_screen_heldout (heldout)")
    p = add("noise", cmd_noise)
    p.add_argument("--shapes", default="suites/v1/noise.json")
    p.add_argument("--backends", default="torch,triton:bm128_bn128_bk32_w4_s4_g8_sk1")
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--cache-modes", default="cold,hot")
    p.add_argument("--session", default=None)
    p.add_argument("--n-allocs", type=int, default=3)
    p = add("drift", cmd_drift)
    p.add_argument("--minutes", type=float, default=10.0)
    p.add_argument("--n", type=int, default=8192)
    p = sub.add_parser("analyze")
    p.set_defaults(fn=cmd_analyze)
    p.add_argument("--data", type=Path, default=None)
    p.add_argument("--report", type=Path, default=Path("reports"))
    p.add_argument("--results", default="results.jsonl",
                   help="benchmark file under --data, e.g. results_lock1755.jsonl (outputs get a _lock1755 suffix)")

    p = sub.add_parser("portfolio")
    p.set_defaults(fn=cmd_portfolio)
    p.add_argument("--data", type=Path, default=None)
    p.add_argument("--report", type=Path, default=Path("reports"))
    p.add_argument("--heldout", action="store_true",
                   help="score the tune-fitted portfolio on data/<hw>/sweep_screen_heldout (G5 out of sample)")
    p.add_argument("--results", default="results_lock1755_heldout.jsonl",
                   help="stage-2 held-out benchmark file under --data (with --heldout); the portfolio it ran with is "
                        "read from <stem>.portfolio.json beside it if present")

    p = sub.add_parser("selectors", help="trivial selector baselines (NN, k-NN, hand rule) on held-out shapes; also "
                                         "writes data/<hw>/selector_picks.json for the triton:picks backend")
    p.set_defaults(fn=cmd_selectors)
    p.add_argument("--data", type=Path, default=None)
    p.add_argument("--report", type=Path, default=Path("reports"))
    p.add_argument("--results", default="results_lock1755_selectors3.jsonl", help="cold stage-2 file under --data")
    p.add_argument("--graph-results", default="results_lock1755_selectors3_graph.jsonl",
                   help="graph-amortized stage-2 file under --data (< 50 us shapes)")

    p = sub.add_parser("learned", help="learned selectors v0 (decision tree, linear SVM, LightGBM, random forest, logistic "
                                       "regression; cls and rank forms) on held-out shapes; writes data/<hw>/learned_picks.json")
    p.set_defaults(fn=cmd_learned)
    p.add_argument("--data", type=Path, default=None)
    p.add_argument("--report", type=Path, default=Path("reports"))
    p.add_argument("--results", default="results_lock1755_learned.jsonl", help="cold stage-2 file under --data")
    p.add_argument("--graph-results", default="results_lock1755_learned_graph.jsonl",
                   help="graph-amortized stage-2 file under --data (< 50 us shapes)")

    p = sub.add_parser("topk", help="refit lgbm_rank on all tune shapes, save it (data/<hw>/lgbm_rank.*), write its top-5 "
                                    "picks per held-out shape (learned_topk_picks.json) and score best-of-k / online search")
    p.set_defaults(fn=cmd_topk)
    p.add_argument("--data", type=Path, default=None)
    p.add_argument("--report", type=Path, default=Path("reports"))
    p.add_argument("--results", default="results_lock1755_topk.jsonl", help="cold stage-2 file under --data")
    p.add_argument("--graph-results", default="results_lock1755_topk_graph.jsonl", help="graph stage-2 file (< 50 us)")
    p.add_argument("--online", default="online_lgbm_rank.jsonl", help="`tune online` output under --data")
    p.add_argument("--online-diverse", default="online_lgbm_rank_diverse.jsonl", help="`tune online --order diverse` output")
    p.add_argument("--online-inner", default="online_lgbm_rank_inner.jsonl", help="`tune online --order inner` output")
    p = add("online", cmd_online, help="top-k + short online search with the saved learned ranker")
    p.add_argument("--suite", default="S1,S2")
    p.add_argument("--shapes-file", default=None)
    p.add_argument("--ranker", default=None, help="model stem; default data/<hw>/lgbm_rank")
    p.add_argument("--k", type=int, default=5, help="candidates compiled and timed per shape")
    p.add_argument("--dtype", default="fp16")
    p.add_argument("--session", default=None)
    p.add_argument("--order", choices=["plain", "diverse", "inner"], default="plain",
                   help="top-k order: score order, the ranker's CV-chosen diversification rule, or the diagnostic "
                        "rule that also varies warps / stages of the top tile")
    p.add_argument("--k-frequent", type=int, default=0,
                   help="call-count-aware k: shapes in --shapes-file with `calls` >= --frequent-calls (or `frequent: true`) "
                        "get this many candidates, the rest --k (e.g. --k 3 --k-frequent 5); 0 = same k for every shape")
    p.add_argument("--frequent-calls", type=int, default=1_000_000,
                   help="launches at which a shape counts as frequently called (default: the held-out 3 -> 5 break-even)")
    p.add_argument("--lt-n", type=int, default=0,
                   help="also time cuBLASLt's heuristic top N in the same pool (joint selection); default 0 = Triton only")
    p.add_argument("--out-name", default=None, help="default online_lgbm_rank[_diverse].jsonl (online_joint_* with --lt-n)")
    p.add_argument("--fresh-cache", action="store_true", help="compile into an empty TRITON_CACHE_DIR under --out")

    p = sub.add_parser("ranker-v2", help="fit rank v2 (analytic features, near-top weights, extra tune screens) on all tune "
                                         "shapes and save data/<hw>/lgbm_rank_v2.*; CPU only, reads no held-out shape")
    p.set_defaults(fn=cmd_ranker_v2)
    p.add_argument("--data", type=Path, default=None)
    p.add_argument("--extra-screens", nargs="*", default=["sweep_screen_tail"], help="extra tune screen dirs under --data")
    p.add_argument("--stem", default="lgbm_rank_v2")

    args = ap.parse_args(argv)
    if args.clock_lock is not None:
        os.environ["TUNE_CLOCK_LOCK"] = args.clock_lock
    if args.cmd in ("analyze", "portfolio", "selectors", "learned", "topk", "ranker-v2"):
        args.data = args.data or default_data_dir()
    else:
        args.out = args.out or default_data_dir()
        if not args.no_pin:
            pin_cpu()
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
