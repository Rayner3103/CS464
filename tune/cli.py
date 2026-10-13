"""Command-line entry point: `python -m tune <command>` (or `tune <command>` once pip-installed).

  env        write data/<hw_id>/env.json
  check      correctness gate for backends over a suite (default: edge)
  roofline   B0: measure peak TFLOP/s and DRAM bandwidth -> data/<hw_id>/roofline.json
  benchmark  validate + time backends over suites -> data/<hw_id>/<out>.jsonl  (resumable)
  baseline   alias of benchmark with --backends defaulting to the vendor/reference baselines
  noise      noise-floor study -> data/<hw_id>/noise.jsonl
  drift      thermal drift run -> data/<hw_id>/drift-*.csv
  analyze    build plots + markdown summaries into reports/
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


def _shapes(suites: str) -> list[dict]:
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
    path = run(_shapes(args.suite), args.backends.split(","), args.out, dtype=args.dtype,
               cache_modes=tuple(args.cache_modes.split(",")), timing_overrides=timing,
               budget_s=_parse_budget(args.budget), session_id=args.session, prune_shapes=args.prune_shapes,
               out_name=args.out_name, n_allocs=args.n_allocs)
    print(f"results -> {path}")


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


def cmd_analyze(args):
    from analysis.report import build_reports
    build_reports(args.data, args.report)


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

    args = ap.parse_args(argv)
    if args.clock_lock is not None:
        os.environ["TUNE_CLOCK_LOCK"] = args.clock_lock
    if args.cmd == "analyze":
        args.data = args.data or default_data_dir()
    else:
        args.out = args.out or default_data_dir()
        if not args.no_pin:
            pin_cpu()
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
