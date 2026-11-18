"""Does the prebuilt kernel bundle change which kernel the selector picks, or only how long the pick takes? (GPU)

    python results/bundle_ab.py run  [--shapes 30] [--clock-lock 1755]   # ~15 min, GPU idle, one fresh process per run
    python results/bundle_ab.py score [--session <id>]                   # CPU only: pick agreement, timings, binaries

Why. The first bundle measurement (docs/kernel_bundle.md, 2026-11-04) found that 9 of 10 shapes picked the same kernel
with and without the bundle and did not record why the tenth differed, nor which 10 shapes it used. A bundle hit loads a
cubin instead of compiling it; it should change the cost of a search, never its outcome. Two other things can flip a
pick between two runs: the cubin itself (if the bundled binary differed from a fresh compile) and the short online
timing (10 cold samples on one placement, `tune.online.SHORT_REPS`), which ranks near-tied candidates arbitrarily.
This script separates the three:

  * four runs per shape: bundle, no bundle, and a repeat of each (`bundle2`, `nobundle2`). Each run is a fresh process
    with an empty Triton cache (`TRITON_CACHE_DIR` = a new directory), no winner cache, the same search as the first
    measurement (rank v2, k = 3 + cuBLASLt heuristic top-16, no cuBLASLt proposals, no router), after one warm-up
    selection on another shape that takes the one-time CUDA / cuBLASLt / launcher setup out of the timed call. The
    run order rotates per shape so drift does not line up with an arm.
  * the repeats are the control: if bundle vs no-bundle disagrees about as often as two identical runs do, the bundle
    is not the cause.
  * every candidate's short time is stored, so each flip shows the two picks' gap in both runs.
  * the no-bundle runs keep their Triton cache; `score` compares each freshly compiled cubin with the bundle's file for
    the same Triton cache key (same key = same source, options and compiler).

Shapes: `--shapes` held-out shapes evenly spaced by FLOPs over `sweep_screen_heldout/subset.json` (deterministic).
Output: data/<hw>/bundle_ab/<session>.jsonl (+ caches/ for the no-bundle runs), and the score printed as markdown.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HW = "rtx3080_sm86_68sms_10gb"
D = ROOT / "data" / HW
OUT = D / "bundle_ab"
ARMS = ("bundle", "nobundle", "bundle2", "nobundle2")
WARMUP_SHAPE = (1000, 1000, 1000)          # a tune shape (S1); its candidates are recorded so overlap can be checked
SEARCH = dict(lt_n=16, lt_knn=0, lt_rank=0, router=False, cache=None)   # the 2026-11-04 measurement's pool


def pick_shapes(n: int) -> list[tuple[int, int, int]]:
    sub = json.loads((D / "sweep_screen_heldout" / "subset.json").read_text())["shapes"]
    ss = sorted(sub, key=lambda s: (s["M"] * s["N"] * s["K"], s["M"], s["N"], s["K"]))
    idx = sorted({round(i * (len(ss) - 1) / max(n - 1, 1)) for i in range(n)})
    return [(ss[i]["M"], ss[i]["N"], ss[i]["K"]) for i in idx]


# ------------------------------------------------------------------------------------------------ child: one run
def child(M: int, N: int, K: int, arm: str) -> dict:
    """One selection in this (fresh) process. TRITON_CACHE_DIR is set by the parent to an empty directory."""
    sys.path.insert(0, str(ROOT))
    import torch
    from analysis.learned import Ranker
    from kernels import bundle
    from tune.online import select_kernel

    use_bundle = arm.startswith("bundle")
    ranker = Ranker.load(D / "lgbm_rank_v2")
    t0 = time.perf_counter()
    w = select_kernel(ranker, *WARMUP_SHAPE, use_bundle=use_bundle, **SEARCH)
    torch.cuda.synchronize()
    warm_s = time.perf_counter() - t0
    bundle.reset_stats()
    t0 = time.perf_counter()
    r = select_kernel(ranker, M, N, K, use_bundle=use_bundle, **SEARCH)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    keep = ("config_key", "family", "valid", "compile_s", "time_s", "cold_us", "graph_us", "rank")
    cands = [{k: c.get(k) for k in keep} for c in r["candidates"]]
    tri = [c for c in cands if c["family"] == "triton"]
    return {"M": M, "N": N, "K": K, "arm": arm, "wall_s": wall, "warmup_s": warm_s, "rank_s": r["rank_s"],
            "triton_compile_s": sum(c["compile_s"] or 0.0 for c in tri), "choice": r["choice"], "candidates": cands,
            "warmup_triton": [c["config_key"] for c in w["candidates"] if c.get("family") == "triton"],
            "bundle_stats": bundle.stats(), "triton_cache": os.environ.get("TRITON_CACHE_DIR")}


# ------------------------------------------------------------------------------------------------ parent: all runs
def run(n: int, clock_lock: str | None) -> None:
    probe = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True)
    if probe.stdout.strip():
        sys.exit(f"GPU not idle (compute processes: {probe.stdout.split()}); refusing to measure")
    session = time.strftime("%Y%m%d-%H%M%S") + f"-lock{clock_lock or 'none'}"
    caches = OUT / "caches" / session
    caches.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{session}.jsonl"
    shapes = pick_shapes(n)
    print(f"session {session}: {len(shapes)} shapes x {len(ARMS)} runs -> {path}", flush=True)
    for i, (M, N, K) in enumerate(shapes):
        order = ARMS[i % 4:] + ARMS[:i % 4]
        for arm in order:
            cdir = Path(tempfile.mkdtemp(prefix=f"{M}x{N}x{K}-{arm}-", dir=caches))
            env = {k: v for k, v in os.environ.items() if not k.startswith(("TRITON_CACHE", "TUNE_BUNDLE"))}
            env.update(TRITON_CACHE_DIR=str(cdir), TUNE_WINNER_CACHE="off", PYTHONHASHSEED="0")
            if clock_lock:
                env["TUNE_CLOCK_LOCK"] = clock_lock
            p = subprocess.run([sys.executable, __file__, "child", str(M), str(N), str(K), arm], env=env, cwd=ROOT,
                               capture_output=True, text=True)
            line = next((ln for ln in reversed(p.stdout.splitlines()) if ln.startswith("{")), None)
            if p.returncode or line is None:
                rec = {"M": M, "N": N, "K": K, "arm": arm, "error": (p.stderr or p.stdout)[-2000:]}
            else:
                rec = json.loads(line)
            rec["session"] = session
            with path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            ch = (rec.get("choice") or {}).get("key")
            print(f"[{i + 1}/{len(shapes)}] {M}x{N}x{K} {arm:9s} pick={ch} wall={rec.get('wall_s', float('nan')):.3f}s"
                  + (" ERROR" if "error" in rec else ""), flush=True)
            if "arm" in rec and not rec["arm"].startswith("nobundle"):
                subprocess.run(["rm", "-rf", str(cdir)])          # only the no-bundle caches are needed by `score`


# ------------------------------------------------------------------------------------------------ score (CPU only)
def _t(c: dict, mode: str) -> float | None:
    return c.get("graph_us") if mode == "graph" else c.get("cold_us")


def score(session: str | None) -> None:
    import numpy as np
    files = sorted(OUT.glob("*.jsonl"))
    path = OUT / f"{session}.jsonl" if session else files[-1]
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    errs = [r for r in rows if "error" in r]
    by = {}
    for r in rows:
        if "error" not in r:
            by.setdefault((r["M"], r["N"], r["K"]), {})[r["arm"]] = r
    shapes = [s for s, a in by.items() if all(x in a for x in ARMS)]
    print(f"# Bundle A/B, session `{path.stem}`\n\n{len(shapes)} shapes with all four runs; {len(errs)} failed runs.\n")

    def key(r):
        return (r["choice"] or {}).get("key")

    pairs = [("bundle", "nobundle", "bundle vs no bundle"), ("bundle2", "nobundle2", "bundle vs no bundle (repeat)"),
             ("bundle", "bundle2", "bundle vs bundle (control)"), ("nobundle", "nobundle2", "no bundle vs no bundle (control)")]
    print("| comparison | same pick | different pick |\n|---|---|---|")
    for a, b, lab in pairs:
        same = sum(key(by[s][a]) == key(by[s][b]) for s in shapes)
        print(f"| {lab} | {same} | {len(shapes) - same} |")

    walls = {a: np.array([by[s][a]["wall_s"] for s in shapes]) for a in ARMS}
    comp = {a: np.array([by[s][a]["triton_compile_s"] for s in shapes]) for a in ARMS}
    hits = sum(by[s][a]["bundle_stats"]["bundle_hits"] for s in shapes for a in ("bundle", "bundle2"))
    miss = sum(by[s][a]["bundle_stats"]["misses"] for s in shapes for a in ("bundle", "bundle2"))
    print("\n| run | wall per shape, median (mean) | Triton compile + validate, median (mean) |\n|---|---|---|")
    for a in ARMS:
        print(f"| {a} | {np.median(walls[a]):.3f} s ({walls[a].mean():.3f}) | {np.median(comp[a]):.3f} s ({comp[a].mean():.3f}) |")
    print(f"\nBundle runs: {hits} bundle hits, {miss} misses.")
    overlap = [s for s in shapes for a in ARMS
               if set(by[s][a]["warmup_triton"]) & {c["config_key"] for c in by[s][a]["candidates"] if c["family"] == "triton"}]
    print(f"Shapes whose Triton candidates overlap the warm-up shape's: {len(set(overlap))} (their compile cost is understated).")

    # every flip: the two picks' times in each run of the pair
    print("\n| shape | runs | pick A | pick B | gap between the two picks in run A / run B | timing mode |\n|---|---|---|---|---|---|")
    gaps = []
    for a, b, lab in pairs:
        for s in shapes:
            ra, rb = by[s][a], by[s][b]
            ka, kb = key(ra), key(rb)
            if ka == kb:
                continue
            mode = ra["choice"]["mode"]

            def rel(r):
                t = {c["config_key"]: _t(c, mode) for c in r["candidates"] if c["valid"]}
                if t.get(ka) is None or t.get(kb) is None:
                    return None
                return t[kb] / t[ka] - 1           # > 0: A's pick is faster in this run

            ga, gb = rel(ra), rel(rb)
            gaps += [abs(g) for g in (ga, gb) if g is not None]
            fa = "n/a" if ga is None else f"{ga:+.1%}"
            fb = "n/a" if gb is None else f"{gb:+.1%}"
            print(f"| {s[0]}×{s[1]}×{s[2]} | {lab} | `{ka}` | `{kb}` | {fa} / {fb} | {mode} |")
    if gaps:
        print(f"\nAcross flips the two picks are within {max(gaps):.1%} of each other (median {np.median(gaps):.1%}).")

    # same binary? a no-bundle run's freshly compiled cubin vs the bundle's file for the same cache key
    bdir = ROOT / "kernels" / "bundle"
    same = diff = absent = 0
    for s in shapes:
        for a in ("nobundle", "nobundle2"):
            cdir = Path(by[s][a]["triton_cache"] or "")
            for cub in cdir.glob("*/*.cubin"):
                ref = bdir / cub.parent.name / cub.name
                if not ref.exists():
                    absent += 1
                    continue
                if hashlib.sha256(cub.read_bytes()).digest() == hashlib.sha256(ref.read_bytes()).digest():
                    same += 1
                else:
                    diff += 1
    print(f"\nFresh compiles vs the bundle (same Triton cache key): {same} cubins byte-identical, {diff} different, "
          f"{absent} not in the bundle (e.g. kernels of the warm-up shape outside the pool, or launcher-only entries).")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--shapes", type=int, default=30)
    r.add_argument("--clock-lock", default=None, help="declared SM clock lock in MHz (set it yourself with nvidia-smi -lgc)")
    c = sub.add_parser("child")
    c.add_argument("M", type=int)
    c.add_argument("N", type=int)
    c.add_argument("K", type=int)
    c.add_argument("arm", choices=ARMS)
    s = sub.add_parser("score")
    s.add_argument("--session", default=None)
    a = p.parse_args()
    if a.cmd == "run":
        run(a.shapes, a.clock_lock)
    elif a.cmd == "child":
        print(json.dumps(child(a.M, a.N, a.K, a.arm)))
    else:
        score(a.session)


if __name__ == "__main__":
    main()
