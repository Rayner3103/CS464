"""Screen the whole cuBLASLt configuration space on the tune shapes and keep every config's time.

`cublaslt_search` keeps only the top 8 per shape, which is enough to propose neighbours' winners (`cublaslt_knn`) but not to
train a ranker (it needs the slow configs too) or to score a proposer exactly (a proposal outside a shape's top 8 has no
time). This runs the same enumerate-and-screen per TUNE-split shape (never a held-out or aux shape), in a subprocess per
shape (a faulting config cannot take the run down), with a larger per-shape budget than the 20 s of B3+ so that nearly every
shape is screened completely, and writes one gzip JSON per shape: rows [algo, tile, stages, splitk, reduction, swizzle,
custom, us, seed] (tile / stages None for tile-less algorithms) in `data/<hw>/lt_screen/`. Resumable (a finished shape is
skipped). Times are the search's own screen timing (hot L2, 1 warmup + 3 launches, 1.0 us grid): noisy per config, a label
for ranking, not a measurement. GPU idle, one job at a time, `--clock-lock` as for every other run (recorded in meta.json).

    python -m baselines.cublaslt_screen [--budget-s 40] [--limit N [--spread]] [--out DIR]   (TUNE_CLOCK_LOCK=1755 in the environment)

`--limit N` alone takes the N SMALLEST unfinished shapes (a smoke test); with `--spread` it takes N shapes evenly spaced over the
whole size-sorted tune set instead, so a reduced screen (pipeline/zero_shot.sh --quick) still sees small and large GEMMs.
"""
from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
import time
from pathlib import Path

from baselines.cublaslt_knn import CFG_KEYS, tune_shapes

ROOT = Path(__file__).resolve().parent.parent


def out_dir() -> Path:
    from tune.env import default_data_dir
    return default_data_dir() / "lt_screen"


def path_for(d: Path, M: int, N: int, K: int, dtype: str = "fp16") -> Path:
    return d / f"{M}x{N}x{K}_{dtype}.json.gz"


def load(path: Path) -> dict:
    """{"M","N","K",... "rows": [(cfg tuple in CFG_KEYS order, us, seed)]}"""
    d = json.loads(gzip.open(path, "rt").read())
    d["rows"] = [(tuple(r[:7]), r[7], bool(r[8])) for r in d.pop("rows")]
    return d


def screen_one(M: int, N: int, K: int, dest: Path, budget_s: float, dtype: str = "fp16", timeout_s: float = 900.0) -> str:
    tmp = dest.with_suffix(".tmp.json")
    cmd = [sys.executable, "-m", "baselines.cublaslt_search", str(M), str(N), str(K), "--dtype", dtype,
           "--budget-s", str(budget_s), "--top-k", "8", "--keep-all", "--out", str(tmp)]
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return f"timeout after {timeout_s}s"
    if p.returncode != 0 or not tmp.exists():
        return (p.stderr.strip().splitlines() or ["?"])[-1][:200]
    d = json.loads(tmp.read_text())
    tmp.unlink()
    rows = [[*(s["cfg"].get(k) for k in CFG_KEYS), s["us"], int(s["seed"])] for s in d["screened"]]
    meta = {k: d[k] for k in ("M", "N", "K", "dtype", "budget_s", "n_valid_configs", "n_screened", "n_screen_failed",
                              "screened_all", "enumerate_s", "screen_s", "best_seed_us")}
    with gzip.open(dest, "wt") as f:
        f.write(json.dumps({**meta, "rows": rows}))
    return "ok"


def limit_shapes(todo: list, limit: int, spread: bool) -> list:
    """The first `limit` shapes of the size-sorted list, or (spread) `limit` shapes evenly spaced over all of it, ends included."""
    if not limit or limit >= len(todo):
        return todo
    if spread and limit > 1:
        return [todo[round(i * (len(todo) - 1) / (limit - 1))] for i in range(limit)]
    return todo[:limit]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--budget-s", type=float, default=40.0)
    ap.add_argument("--limit", type=int, default=0, help="screen only the first N unfinished shapes (smoke test)")
    ap.add_argument("--spread", action="store_true", help="with --limit: N shapes evenly spaced over the size-sorted tune set, not the N smallest")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    d = a.out or out_dir()
    d.mkdir(parents=True, exist_ok=True)
    shapes = sorted(tune_shapes(), key=lambda s: s[0] * s[1] * s[2])
    todo = [s for s in shapes if not path_for(d, *s).exists()]
    todo = limit_shapes(todo, a.limit, a.spread)
    import os
    (d / "meta.json").write_text(json.dumps({
        "started": time.strftime("%Y-%m-%d %H:%M:%S"), "budget_s": a.budget_s, "clock_lock_mhz": os.environ.get("TUNE_CLOCK_LOCK"),
        "note": "hot-L2 screen timing (1 warmup + 3 launches); tune-split shapes only"}, indent=1))
    t0 = time.perf_counter()
    for i, (M, N, K) in enumerate(todo):
        t = time.perf_counter()
        st = screen_one(M, N, K, path_for(d, M, N, K), a.budget_s)
        el = time.perf_counter() - t0
        print(f"[{i + 1}/{len(todo)}] {M}x{N}x{K} {st} ({time.perf_counter() - t:.1f}s) | {el:.0f}s elapsed, "
              f"ETA {el / (i + 1) * (len(todo) - i - 1):.0f}s", flush=True)


if __name__ == "__main__":
    main()
