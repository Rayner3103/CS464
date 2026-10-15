"""Full Triton config sweep, stage 1: screen the whole pruned config space per shape (plan 6.2, 6.3; M3/M4).

Stage 1 (this module, `python -m tune sweep`): for every shape in a coarse subset, every config that survives the
static + shape pruning rules is compiled, validated against the fp32 reference, and given a cheap cold-L2 timing.
Stage 2 is the ordinary benchmark: backend spec `triton:sweep[:k]` expands to the k fastest screened configs per shape,
which then go through the full protocol next to the baselines.

Screen timing: 1 sample (L2 flushed before it); if it is more than SLOW_FACTOR x the best so far on this shape, stop
there, otherwise take SCREEN_SAMPLES samples and keep the median (SMALL_SAMPLES and a trimmed mean for kernels under
SMALL_US, since CUDA events tick in ~1.02 us steps). Rankings within a few % are refined in stage 2.

Isolation: each shape runs in its own subprocess, which appends one line per config to a partial file. If it dies
(a config faulted the CUDA context), the parent marks the config after the last written line as crashed and restarts
from there. Compilation runs first in parallel worker processes (`warmup`, no launch) to fill Triton's disk cache.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

SCREEN_VERSION = 1
SCREEN_SAMPLES = 5
SMALL_US, SMALL_SAMPLES = 50.0, 15  # events tick in ~1.02 us steps: average more samples for short kernels
SLOW_FACTOR = 1.5
ROOT = Path(__file__).resolve().parent.parent


HELDOUT_SPLITS = ("heldout_region", "heldout_random")


def coarse_subset(shapes: list[dict], per_suite: int, max_gflop: float, splits=("tune",)) -> list[dict]:
    """Deterministic stratified subset: in-range shapes of the given splits only, evenly spaced by FLOPs within each
    suite."""
    out, seen = [], set()
    by_suite: dict[str, list[dict]] = {}
    for s in shapes:
        if s.get("split") not in splits or s.get("out_of_range") or 2e-9 * s["M"] * s["N"] * s["K"] > max_gflop:
            continue
        by_suite.setdefault(s["suite"], []).append(s)
    for suite in sorted(by_suite):
        ss = sorted(by_suite[suite], key=lambda s: (s["M"] * s["N"] * s["K"], s["M"], s["N"], s["K"]))
        n = min(per_suite, len(ss))
        idx = sorted({round(i * (len(ss) - 1) / max(n - 1, 1)) for i in range(n)})
        for i in idx:
            key = (ss[i]["M"], ss[i]["N"], ss[i]["K"])
            if key not in seen:
                seen.add(key)
                out.append(ss[i])
    return out


def shape_configs(M: int, N: int, K: int):
    """Configs to screen on this shape, hand-picked first (they set a good early 'best' for the slow cut-off)."""
    from kernels.config_space import HANDPICKED, HwLimits, full_space, prune
    kept, dropped = prune(full_space(), HwLimits.from_device(), (M, N, K))
    first = [c for c in HANDPICKED.values() if c in set(kept)]
    rest = sorted((c for c in kept if c not in set(first)), key=lambda c: c.key())
    return first + rest, dropped


def _warm_worker(args):
    """Compile (no launch) a chunk of configs for one shape's specialization. Runs in a spawned process."""
    keys, M, N, K = args
    os.sched_setaffinity(0, range(os.cpu_count()))
    import torch
    from kernels.triton_matmul import MatmulConfig, _matmul_kernel, even_k
    t = torch.empty(16, 16, device="cuda", dtype=torch.float16)
    ws = torch.empty(16, 16, device="cuda", dtype=torch.float32)  # split-K writes fp32 partials
    out = {}
    for k in keys:
        cfg = MatmulConfig(**k)
        t0 = time.perf_counter()
        try:
            s_cs, s_cm = (M * N, N) if cfg.SPLIT_K > 1 else (0, N)
            _matmul_kernel.warmup(t, t, ws if cfg.SPLIT_K > 1 else t, M, N, K, K, 1, N, 1, s_cs, s_cm, 1, BLOCK_M=cfg.BLOCK_M, BLOCK_N=cfg.BLOCK_N,
                                  BLOCK_K=cfg.BLOCK_K, GROUP_M=cfg.GROUP_M, SPLIT_K=cfg.SPLIT_K,
                                  EVEN_K=even_k(K, cfg), num_warps=cfg.num_warps, num_stages=cfg.num_stages, grid=(1,))
            out[cfg.key()] = (round(time.perf_counter() - t0, 3), None)
        except Exception as e:
            out[cfg.key()] = (round(time.perf_counter() - t0, 3), f"compile: {type(e).__name__}: {str(e)[:150]}")
    return out


def prewarm(cfgs, M: int, N: int, K: int, workers: int) -> dict:
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    dicts = [c.to_dict() for c in cfgs]
    chunks = [(dicts[i::workers * 4], M, N, K) for i in range(workers * 4)]
    res: dict = {}
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        for d in ex.map(_warm_worker, chunks):
            res.update(d)
    return res


def _center(samples: list[float]) -> float:
    """Median for a few samples; for many, the mean of the middle 60%, which resolves below the event tick."""
    x = sorted(samples)
    if len(x) < SMALL_SAMPLES:
        return x[len(x) // 2]
    cut = len(x) // 5
    mid = x[cut:len(x) - cut]
    return sum(mid) / len(mid)


EXIT_POISONED = 3  # the screen process recorded a context-killing config itself; just restart it


def _context_poisoned(reason: str | None) -> bool:
    r = (reason or "").lower()
    return any(x in r for x in ("illegal", "misaligned", "unspecified launch failure", "device-side assert",
                                "an illegal instruction"))


def screen_shape(M: int, N: int, K: int, partial: Path, skip: set[str], dtype: str = "fp16"):
    """Validate + screen-time every config on one shape, appending a JSON line per config to `partial`."""
    import torch
    from kernels.triton_matmul import TritonMatmul
    from tune.correctness import placed_inputs, reference, validate
    from tune.runner import DTYPES
    from tune.timing import _backlog, _L2Flusher

    tdtype = DTYPES[dtype]
    cfgs, _ = shape_configs(M, N, K)
    a, b, _pad = placed_inputs(M, N, K, tdtype, seed=0)
    ref = reference(a, b)
    flush = _L2Flusher(None, 0)
    best = float("inf")
    for r in read_partial(partial):
        if r.get("us") is not None:
            best = min(best, r["us"])
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(SMALL_SAMPLES)]
    with partial.open("a") as f:
        for cfg in cfgs:
            key = cfg.key()
            if key in skip:
                continue
            row = {"key": key, "us": None, "n": 0, "status": "ok"}
            chk = validate(TritonMatmul(cfg), M, N, K, tdtype, ref=ref, inputs=(a, b))
            if not chk.valid:
                row["status"] = f"invalid: {chk.failure_reason}"
                if _context_poisoned(chk.failure_reason):
                    row["status"] = f"invalid: crashed: {chk.failure_reason}"
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    sys.exit(EXIT_POISONED)
            else:
                try:
                    run = TritonMatmul(cfg).make_runner(a, b)
                    samples, n_target = [], SCREEN_SAMPLES
                    while len(samples) < n_target:
                        i = len(samples)
                        _backlog(0.0005)
                        flush()
                        ev[i][0].record()
                        run()
                        ev[i][1].record()
                        ev[i][1].synchronize()
                        samples.append(ev[i][0].elapsed_time(ev[i][1]) * 1e3)
                        if i == 0 and samples[0] > SLOW_FACTOR * best:
                            row["status"] = "slow"
                            break
                        if i == 0 and samples[0] < SMALL_US:
                            n_target = SMALL_SAMPLES
                    row["us"], row["n"] = round(_center(samples), 3), len(samples)
                    if row["status"] == "ok":
                        best = min(best, row["us"])
                    del run
                except torch.OutOfMemoryError:
                    row["status"] = "invalid: oom_timing"
                    torch.cuda.empty_cache()
            f.write(json.dumps(row) + "\n")
            f.flush()


def read_partial(partial: Path) -> list[dict]:
    """Rows of a partial screen file; a torn last line (process killed mid-write) is dropped."""
    rows = []
    for line in partial.read_text().splitlines() if partial.exists() else []:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def screen_path(sweep_dir: Path, M: int, N: int, K: int, dtype: str = "fp16") -> Path:
    return Path(sweep_dir) / f"{M}x{N}x{K}_{dtype}.json"


def run_shape_isolated(M: int, N: int, K: int, sweep_dir: Path, workers: int, dtype: str = "fp16",
                       log=print, max_restarts: int = 20) -> dict:
    final = screen_path(sweep_dir, M, N, K, dtype)
    if final.exists():
        d = json.loads(final.read_text())
        if d.get("screen_version") == SCREEN_VERSION:
            return d
    sweep_dir.mkdir(parents=True, exist_ok=True)
    partial = final.with_suffix(".partial.jsonl")
    cfgs, dropped = shape_configs(M, N, K)
    order = [c.key() for c in cfgs]
    t0 = time.perf_counter()
    compile_info = prewarm(cfgs, M, N, K, workers)
    t_compile = time.perf_counter() - t0
    crashed: list[str] = []
    for attempt in range(max_restarts + 1):
        p = subprocess.run([sys.executable, "-m", "tune.sweep", "_screen", str(M), str(N), str(K), "--dtype", dtype,
                            "--partial", str(partial), "--crashed", ",".join(crashed)],
                           cwd=ROOT, capture_output=True, text=True)
        done = [r["key"] for r in read_partial(partial)]
        if p.returncode == 0:
            break
        if p.returncode == EXIT_POISONED:
            log(f"  screen process restarted after a context-killing config ({done[-1] if done else '?'})")
            continue
        nxt = next((k for k in order if k not in set(done) and k not in set(crashed)), None)
        err = (p.stderr.strip().splitlines() or ["?"])[-1][:200]
        log(f"  screen process died ({err}); marking {nxt} as crashed and resuming")
        if nxt is None:
            break
        crashed.append(nxt)
    rows = read_partial(partial)
    rows += [{"key": k, "us": None, "n": 0, "status": "invalid: crashed"} for k in crashed]
    for r in rows:
        ct, cerr = compile_info.get(r["key"], (None, None))
        r["compile_s"] = ct
        if cerr and r["status"] == "ok":
            r["status"] = f"invalid: {cerr}"
    d = {"screen_version": SCREEN_VERSION, "M": M, "N": N, "K": K, "dtype": dtype, "n_configs": len(order),
         "pruned_by_rule": dropped, "n_done": len(rows), "complete": len(rows) >= len(order),
         "compile_wall_s": round(t_compile, 1), "screen_wall_s": round(time.perf_counter() - t0 - t_compile, 1),
         "screen_samples": SCREEN_SAMPLES, "slow_factor": SLOW_FACTOR, "rows": rows}
    final.write_text(json.dumps(d))
    partial.unlink(missing_ok=True)
    return d


_KEY_RE = re.compile(r"bm(\d+)_bn(\d+)_bk(\d+)_w(\d+)_s(\d+)_g(\d+)_sk(\d+)")


def parse_key(key: str):
    """Inverse of MatmulConfig.key()."""
    from kernels.triton_matmul import MatmulConfig
    m = _KEY_RE.fullmatch(key)
    if not m:
        raise ValueError(f"bad config key {key!r}")
    return MatmulConfig(*map(int, m.groups()))


def top_configs(sweep_dir: Path, M: int, N: int, K: int, k: int, dtype: str = "fp16") -> list[dict]:
    """The k fastest valid screened configs for a shape, or [] if not screened.

    Only the fastest variant of each tile shape (BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K) is kept: variants differing only in
    warps / stages / GROUP_M usually screen within noise of each other, and k distinct tiles cover more ground.
    """
    path = screen_path(sweep_dir, M, N, K, dtype)
    if not path.exists():
        return []
    rows = [r for r in json.loads(path.read_text())["rows"] if r["status"] == "ok" and r["us"] is not None]
    rows.sort(key=lambda r: r["us"])
    out, tiles = [], set()
    for r in rows:
        cfg = parse_key(r["key"])
        tile = (cfg.BLOCK_M, cfg.BLOCK_N, cfg.BLOCK_K, cfg.SPLIT_K)
        if tile in tiles:
            continue
        tiles.add(tile)
        out.append({"cfg": cfg, "screen_us": r["us"], "screen_rank": len(out)})
        if len(out) == k:
            break
    return out


def run_sweep(shapes: list[dict], out_dir: Path, *, workers: int, session_id: str | None = None, log=print,
              screen_dir: str = "sweep_screen") -> Path:
    from tune.env import write_env
    sweep_dir = out_dir / screen_dir
    env = write_env(out_dir, session_id=session_id, kind="sweep")
    t0 = time.perf_counter()
    manifest = {"session_id": env["session_id"], "screen_version": SCREEN_VERSION, "shapes": []}
    for i, s in enumerate(shapes):
        M, N, K = s["M"], s["N"], s["K"]
        d = run_shape_isolated(M, N, K, sweep_dir, workers, log=log)
        ok = [r for r in d["rows"] if r["status"] == "ok"]
        best = min((r["us"] for r in ok), default=None)
        manifest["shapes"].append({"M": M, "N": N, "K": K, "suite": s.get("suite"), "split": s.get("split"),
                                   "n_configs": d["n_configs"], "n_ok": len(ok), "best_us": best})
        el = time.perf_counter() - t0
        log(f"[{i + 1}/{len(shapes)}] {s.get('suite')} {M}x{N}x{K}: {d['n_configs']} cfgs, {len(ok)} ok, "
            f"best {best} us | compile {d['compile_wall_s']}s screen {d['screen_wall_s']}s | "
            f"{el:.0f}s elapsed, ETA {el / (i + 1) * (len(shapes) - i - 1):.0f}s")
    (sweep_dir / f"manifest-{env['session_id']}.json").write_text(json.dumps(manifest, indent=1))
    return sweep_dir


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["_screen"])
    ap.add_argument("M", type=int)
    ap.add_argument("N", type=int)
    ap.add_argument("K", type=int)
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--partial", required=True)
    ap.add_argument("--crashed", default="")
    a = ap.parse_args()
    from tune.env import pin_cpu
    pin_cpu()
    p = Path(a.partial)
    done = {r["key"] for r in read_partial(p)}
    screen_shape(a.M, a.N, a.K, p, done | set(filter(None, a.crashed.split(","))), a.dtype)


if __name__ == "__main__":
    _main()
