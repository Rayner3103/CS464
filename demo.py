"""Demo: run ~10 s of GEMM work through each baseline and through the learned kernel selector, and compare durations.

    python -m demo [--target-seconds 10] [--clock-lock 1755]

The workload is a fixed list of held-out shapes (LLM decode/prefill layers plus awkward irregular ones; the ranker never
saw them) with a call count per shape. Counts are calibrated once so that `torch.matmul` (B1) spends the same GPU time
on every shape and ~`--target-seconds` in total; then *every* backend runs the identical (shape, calls) sequence on the
same inputs, so the totals compare directly.

Contenders:
  B1  torch.matmul                          what a typical user gets (default cuBLAS)
  B2  cuBLASLt heuristic top-1              the vendor library's own first pick
  B3  cuBLASLt best of heuristic top-16     OFFLINE oracle: every candidate timed first, free to the clock below
  B5  Triton tutorial matmul                fixed 128x256x64 default config
  B6  one fixed Triton config               the best single config of the family (bm32_bn32_bk32_w4_s5_g1_sk1)
  S-T selector, Triton only                 saved lgbm_rank, diversified top-k, compile + validate + short-time, keep the fastest
  S-J selector, Triton + cuBLASLt top-16    the same short-timed pool with cuBLASLt's heuristic top-16 added (joint)

Selection is a first-use cost, so the table reports it separately and adds it to the end-to-end column: the selector
is only a win when the calls amortize its search (k = 5 for shapes with >= 1M calls, else 3: `tune.online.KPolicy`).
Run it with an empty Triton cache (default; `--warm-cache` keeps the user cache) so the compile cost is a true first use.

Timing: each (shape, backend) is captured into a CUDA graph of `batch` launches and replayed `reps` times, hot L2,
measured with CUDA events around the whole replay loop (no Python launch overhead; the methodology's graph mode).
Every output is checked against the fp32 reference first; a backend that fails a shape is flagged and left out of that
shape's totals. Standard conditions (`--clock-lock 1755` with the clock actually locked, GPU idle) as in AGENTS.md.
B3 stays an offline bar (see AGENTS.md section 6: the family does not beat B3+ at >= 50 us); do not read a win over B1
as a win over the vendor library.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# DEMO_SHAPES: all from the held-out split (checked at start-up). M, N, K plus a short label.
DEMO_SHAPES = [
    (4, 4096, 11008, "llama2-7b mlp_down, decode"),
    (32, 4096, 11008, "llama2-7b mlp_down, small batch"),
    (128, 4096, 11008, "llama2-7b mlp_down, batch 128"),
    (512, 4096, 11008, "llama2-7b mlp_down, prefill"),
    (2048, 4096, 11008, "llama2-7b mlp_down, long prefill"),
    (384, 2176, 4096, "region: M in 300..450"),
    (448, 1088, 4096, "region: M in 300..450"),
    (342, 88, 6772, "region: skinny N, deep K"),
    (443, 4723, 393, "random: unaligned"),
    (3402, 796, 408, "random: unaligned"),
    (226, 2141, 1139, "random: odd N, K"),
    (15, 3363, 2632, "random: tiny M"),
]

FIXED_B6 = "bm32_bn32_bk32_w4_s5_g1_sk1"
LT_N = 16
BATCH_US = 2000.0       # target device time of one graph (sets `batch` for short kernels)
MAX_BATCH = 500
NAMES = [("B1", "torch.matmul"), ("B2", "cuBLASLt heuristic top-1"), ("B3", "cuBLASLt best of top-16 (offline)"),
         ("B5", "Triton tutorial default"), ("B6", "Triton fixed best single config"),
         ("S-T", "selector: Triton only"), ("S-J", "selector: Triton + cuBLASLt top-16")]


def _args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--target-seconds", type=float, default=10.0, help="GPU time torch.matmul (B1) spends on the workload")
    ap.add_argument("--ranker", default=None, help="ranker stem (default data/<hw>/lgbm_rank; use lgbm_rank_v2 for rank v2)")
    ap.add_argument("--shapes-file", default=None, help="JSON list of {M, N, K[, label]} instead of the built-in shapes")
    ap.add_argument("--clock-lock", default=None, help="MHz the SM clock is locked at (declared, as in `tune --clock-lock`)")
    ap.add_argument("--warm-cache", action="store_true", help="keep the user Triton cache: compile cost is then not a first use")
    ap.add_argument("--no-pin", action="store_true", help="do not pin the process to one CPU core")
    ap.add_argument("--out", default=None, help="result JSON (default data/<hw>/demo-<timestamp>.json)")
    return ap.parse_args(argv)


def graph_of(run, batch: int):
    """`batch` back-to-back launches of `run` captured in one CUDA graph (warmed up first: JIT, heuristics, allocator)."""
    import torch
    run()
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        run()
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(batch):
            run()
    torch.cuda.synchronize()
    return g


def replay_s(g, reps: int, warm: int = 2) -> float:
    """Device seconds for `reps` replays of graph `g` (CUDA events around the loop)."""
    import torch
    for _ in range(warm):
        g.replay()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        g.replay()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) / 1e3


def quick_us(run, batch: int = 20, reps: int = 5) -> float:
    """Median per-launch microseconds from `reps` graph replays of `batch` launches (used for calibration and B3)."""
    import numpy as np
    import torch
    g = graph_of(run, batch)
    g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e) * 1e3 / batch)
    del g
    return float(np.median(ts))


def load_shapes(path: str | None) -> list[dict]:
    if path:
        return [{"label": "", **s} for s in json.loads(Path(path).read_text())]
    return [{"M": m, "N": n, "K": k, "label": lab} for m, n, k, lab in DEMO_SHAPES]


def check_heldout(shapes: list[dict]) -> list[str]:
    """Shapes of the suites that are not held-out (anything the ranker or portfolio could have seen)."""
    split = {}
    for f in (Path(__file__).parent / "suites" / "v1").glob("*.json"):
        for s in json.loads(f.read_text()).get("shapes", []):
            split[(s["M"], s["N"], s["K"])] = s.get("split")
    return [f"{s['M']}x{s['N']}x{s['K']}" for s in shapes
            if not str(split.get((s["M"], s["N"], s["K"]), "unknown")).startswith("heldout")]


def main(argv=None) -> int:
    args = _args(argv)
    if not args.warm_cache:  # before any Triton compile in this process: selection cost must be a first use
        os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix="demo_triton_cache_")
    if args.clock_lock is not None:
        os.environ["TUNE_CLOCK_LOCK"] = args.clock_lock

    import numpy as np
    import torch
    from analysis.learned import Ranker
    from baselines.cublaslt import candidates as lt_candidates
    from baselines.torch_mm import TorchMatmul
    from baselines.triton_tutorial import TritonTutorialMatmul
    from kernels.triton_matmul import TritonMatmul
    from tune.correctness import placed_inputs, reference, validate
    from tune.env import default_data_dir, gpu_processes, hw_id, pin_cpu, probe_clock_lock
    from tune.online import KPolicy, choose, select_kernel
    from tune.sweep import parse_key

    if not torch.cuda.is_available():
        print("no CUDA device", file=sys.stderr)
        return 2
    busy = [p for p in gpu_processes() if p.get("kind") == "compute"]  # the desktop's graphics clients are always there
    if busy:
        print(f"GPU is not idle ({busy}); another job would contaminate the timings. Stop it first.", file=sys.stderr)
        return 2
    if not args.no_pin:
        pin_cpu()
    data = default_data_dir()
    stem = Path(args.ranker) if args.ranker else data / "lgbm_rank"
    if not stem.with_suffix(".json").exists():
        print(f"ranker not found: {stem}.json/.txt (train with `python -m tune topk`, or pass --ranker)", file=sys.stderr)
        return 2
    shapes = load_shapes(args.shapes_file)
    seen = check_heldout(shapes)
    if seen:
        print(f"WARNING: not held-out shapes (the ranker may have been fit on them): {', '.join(seen)}")
    clk = probe_clock_lock(declared=args.clock_lock)
    print(f"GPU {hw_id()} | clock: {clk['state']}" + (f" @ {clk['lock_mhz']} MHz" if clk.get("lock_mhz") else "")
          + (" | WARNING: not the standard conditions, numbers are not comparable to the study" if clk["state"] not in
             ("locked",) else ""))
    print(f"ranker {stem.name} | {len(shapes)} shapes | target {args.target_seconds:.0f} s of torch.matmul work | "
          f"Triton cache {'user (warm)' if args.warm_cache else 'fresh (first-use compile cost)'}\n")

    dtype = torch.float16
    ranker = Ranker.load(stem)
    policy = KPolicy()
    share_s = args.target_seconds / len(shapes)
    fixed_b6 = TritonMatmul(parse_key(FIXED_B6))
    t_session = time.perf_counter()
    rows = []
    for si, sh in enumerate(shapes):
        M, N, K = sh["M"], sh["N"], sh["K"]
        a, b, _ = placed_inputs(M, N, K, dtype, seed=0)
        ref = reference(a, b)
        out = torch.empty((M, N), device="cuda", dtype=dtype)
        row = {"M": M, "N": N, "K": K, "label": sh["label"], "backends": {}}

        # Calibrate: per-shape batch (graph length) and call count from torch.matmul, shared by every backend.
        t1_us = quick_us(TorchMatmul().make_runner(a, b, out))
        batch = int(min(max(round(BATCH_US / t1_us), 1), MAX_BATCH))
        reps = max(int(round(share_s * 1e6 / t1_us / batch)), 1)
        calls = batch * reps
        row.update(calls=calls, batch=batch, reps=reps, torch_us=t1_us)

        # First-use selection (cold Triton cache): one joint search of k Triton + LT_N cuBLASLt candidates; the
        # Triton-only choice and its cost are read back from the same candidate records (`choose` with lt_n = 0).
        sel = select_kernel(ranker, M, N, K, calls=calls, policy=policy, lt_n=LT_N, order="diverse", dtype=dtype,
                           cache=None, use_bundle=False)  # measure the search itself: no winner cache, no prebuilt kernels
        picks = {}
        if not sel.get("skipped"):
            k = sel["k"]
            for tag, lt in (("S-T", 0), ("S-J", LT_N)):
                ch = choose(sel["candidates"], k, lt)
                picks[tag] = (ch, sel["rank_s"] + ch["compile_s"] + ch["time_s"])
        row["selector_k"] = sel["k"]

        lts = lt_candidates(M, N, K, dtype, LT_N)
        backends = {"B1": TorchMatmul(), "B2": lts[0] if lts else None, "B5": TritonTutorialMatmul(), "B6": fixed_b6}
        if lts:  # B3: time every heuristic candidate first (offline), keep the fastest
            ok = [c for c in lts if validate(c, M, N, K, dtype, ref=ref, inputs=(a, b)).valid]
            backends["B3"] = min(ok, key=lambda c: quick_us(c.make_runner(a, b, out)), default=None)
        for tag, (ch, _) in picks.items():
            if ch["key"] is None:
                backends[tag] = None
            elif ch.get("family", "triton") == "triton":
                backends[tag] = TritonMatmul(parse_key(ch["key"]))
            else:
                backends[tag] = next(c for c in lts if c.key == ch["key"])
            row["backends"].setdefault(tag, {})["choice"] = ch["key"]
            row["backends"][tag]["search_s"] = picks[tag][1]

        for tag, _ in NAMES:
            be = backends.get(tag)
            info = row["backends"].setdefault(tag, {})
            if be is None:
                info["error"] = "no valid candidate" if tag in picks or tag == "B3" else "unavailable"
                continue
            chk = validate(be, M, N, K, dtype, ref=ref, inputs=(a, b))
            if not chk.valid:
                info["error"] = f"failed check: {chk.failure_reason}"
                continue
            try:
                g = graph_of(be.make_runner(a, b, out), batch)
                info["run_s"] = replay_s(g, reps)
                del g
            except Exception as e:  # a graph-capture / launch fault must not kill the demo
                info["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"
        rows.append(row)
        line = "  ".join(f"{t}={row['backends'][t]['run_s'] * 1e3:7.0f}ms" if "run_s" in row["backends"][t] else f"{t}=   FAIL"
                         for t, _ in NAMES)
        print(f"[{si + 1:2d}/{len(shapes)}] {M}x{N}x{K} ({calls:,} calls, k={sel['k']})  {line}", flush=True)
        del a, b, ref, out
        torch.cuda.empty_cache()

    summary = {}
    complete = [r for r in rows if all("run_s" in r["backends"][t] for t, _ in NAMES)]
    for tag, name in NAMES:
        ok = [r for r in rows if "run_s" in r["backends"][tag]]
        run_s = sum(r["backends"][tag]["run_s"] for r in ok)
        search_s = sum(r["backends"][tag].get("search_s", 0.0) for r in ok)
        summary[tag] = {"name": name, "workload_s": run_s, "search_s": search_s, "end_to_end_s": run_s + search_s,
                        "shapes_ok": len(ok)}
    base = sum(r["backends"]["B1"]["run_s"] for r in complete)
    print(f"\nWorkload: {len(shapes)} held-out shapes, {sum(r['calls'] for r in rows):,} GEMM calls; "
          f"all-backends-valid shapes: {len(complete)}/{len(shapes)} (speedups use these only)\n")
    print(f"{'':4} {'backend':36} {'workload':>9} {'vs B1':>7} {'search':>8} {'end-to-end':>11} {'vs B1':>7}")
    for tag, name in NAMES:
        s = summary[tag]
        w = sum(r["backends"][tag]["run_s"] for r in complete)
        e2e = w + sum(r["backends"][tag].get("search_s", 0.0) for r in complete)
        summary[tag].update(common_workload_s=w, common_end_to_end_s=e2e)
        print(f"{tag:4} {name:36} {w:8.2f}s {base / w:6.2f}x {s['search_s']:7.2f}s {e2e:10.2f}s {base / e2e:6.2f}x"
              + ("" if s["shapes_ok"] == len(shapes) else f"   ({len(shapes) - s['shapes_ok']} shape(s) failed)"))
    sj, b2 = summary["S-J"], summary["B2"]
    if complete and sj["common_workload_s"] > 0:
        saved = b2["common_workload_s"] - sj["common_workload_s"]
        pay = (f"pays back after ~{sj['search_s'] / saved:.0f} repeats of this workload (~{sj['search_s'] / saved * b2['common_workload_s']:.0f} s of GEMM time)"
               if saved > 0 else "never pays back on this workload (no saving over B2)")
        print(f"\nS-J vs B2 (vendor first pick): workload {b2['common_workload_s'] / sj['common_workload_s']:.2f}x "
              f"({saved:+.2f} s per run); one-time search {sj['search_s']:.1f} s {pay}")
        wins = sum(r["backends"]["S-J"]["run_s"] < 0.98 * r["backends"]["B2"]["run_s"] for r in complete)
        print(f"S-J > 2% faster than B2 on {wins}/{len(complete)} shapes; slower by > 2% on "
              f"{sum(r['backends']['S-J']['run_s'] > 1.02 * r['backends']['B2']['run_s'] for r in complete)}")
    print("\nPer shape (ms; S picks in brackets):")
    for r in rows:
        cells = []
        for tag, _ in NAMES:
            i = r["backends"][tag]
            cells.append(f"{tag}={i['run_s'] * 1e3:.0f}" if "run_s" in i else f"{tag}=FAIL")
        print(f"  {r['M']}x{r['N']}x{r['K']:<6} " + " ".join(cells)
              + f"  [T:{r['backends']['S-T'].get('choice')} | J:{r['backends']['S-J'].get('choice')}]")

    out_path = Path(args.out) if args.out else data / f"demo-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"hw_id": hw_id(), "clock": clk, "ranker": str(stem), "target_seconds": args.target_seconds,
                                    "wall_s": time.perf_counter() - t_session, "summary": summary, "shapes": rows}, indent=1, default=str))
    print(f"\ntotal wall {time.perf_counter() - t_session:.0f} s | results -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
