"""Top-k + short online search (AGENTS.md next step 2): what the learned ranker would do at a shape's first use.

Per shape: the saved lgbm_rank model (analysis.learned.Ranker, fit on the tune screen only) ranks its candidate pool;
the top k runnable configs (plain score order; --order diverse: the CV-chosen diversification rule, which spreads
the k picks over distinct tile x BLOCK_K x split choices; --order inner: the diagnostic rule that also spends a slot on
the top tile's warps / stages) are compiled and validated (the correctness gate, so a broken pick is never chosen), then
timed with a short protocol on one input placement:

  * cold, per-launch: SHORT_REPS samples after a SHORT_WARMUP_MS warmup;
  * if the fastest cold median is under SMALL_US, every valid candidate is re-timed hot, CUDA-graph amortized
    (GRAPH_BATCH launches x GRAPH_REPS samples), and the pick uses those times: per-launch events tick in ~1 us
    steps, which cannot separate us-scale kernels (methodology).

The row stores every candidate's short times and wall-clock costs (compile + validate, timing), so the choice and
cost for any k' <= k is recomputed offline (`choose`) without re-running. Run it with an empty Triton cache
(TRITON_CACHE_DIR, or `tune online --fresh-cache`) so compile cost is a true first-use cost, not a cache hit
from an earlier stage-2 session that timed the same configs. With lt_n > 0 (`tune online --lt-n 16`) cuBLASLt's heuristic top lt_n joins the same pool (joint selection): they need
no compile, so they cost only validation + short timing; `choice_lt` stores the choice for every k' and lt_n' in
{0, 1, 4, lt_n} (lt_n' = 0 is Triton only, 1 is B2 + Triton). Picks are scored against a full-protocol stage-2
session in analysis/learned.py (`build_topk`).
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Callable, Iterable

import torch

from tune.correctness import placed_inputs, reference, validate
from tune.env import provenance, write_env
from tune.records import RecordWriter

SHORT_REPS = 10
SHORT_WARMUP_MS = 2.0
GRAPH_BATCH = 20
GRAPH_REPS = 5
SMALL_US = 50.0


def short_configs():
    from tune.timing import TimingConfig
    cold = TimingConfig(cache_mode="cold", warmup_ms=SHORT_WARMUP_MS, min_warmup_iters=1, min_reps=SHORT_REPS,
                        max_reps=SHORT_REPS, target_rse=1.0, max_time_s=1.0, keep_raw=False)
    graph = TimingConfig(cache_mode="hot", warmup_ms=SHORT_WARMUP_MS, min_warmup_iters=1, min_reps=GRAPH_REPS,
                         max_reps=GRAPH_REPS, target_rse=1.0, max_time_s=1.0, batch=GRAPH_BATCH, use_graph=True,
                         keep_raw=False)
    return cold, graph


def _is_lt(x: dict) -> bool:
    return x.get("family", "triton") != "triton"


def _subset(cands: list[dict], k: int, lt_n: int | None) -> list[dict]:
    """The first k Triton candidates plus the first lt_n cuBLASLt ones (all of them for None); rows written before
    cuBLASLt candidates existed have no "family" and are all Triton, so this is cands[:k] for them."""
    out, n_tri, n_lt = [], 0, 0
    for x in cands:
        if _is_lt(x):
            n_lt += 1
            if lt_n is None or n_lt <= lt_n:
                out.append(x)
        else:
            n_tri += 1
            if n_tri <= k:
                out.append(x)
    return out


def choose(cands: list[dict], k: int, lt_n: int | None = None) -> dict:
    """Online choice among the first k ranked Triton candidates of a row, plus its first lt_n cuBLASLt heuristic
    candidates (None = all, 0 = Triton only): fastest valid by short graph time when the fastest short cold time is
    < SMALL_US, else by short cold time. Cost = compile + timing wall time of those candidates. The cuBLASLt problem
    setup (descriptor + heuristic query) is booked on its first candidate's compile_s."""
    c = _subset(cands, k, lt_n)
    ok = [x for x in c if x["valid"] and x.get("cold_us") is not None]
    cost = {"compile_s": sum(x["compile_s"] for x in c), "time_s": sum(x.get("time_s", 0.0) for x in c)}
    if not ok:
        return {"key": None, "mode": None, **cost}
    small = min(x["cold_us"] for x in ok) < SMALL_US and all(x.get("graph_us") is not None for x in ok)
    best = min(ok, key=lambda x: x["graph_us"] if small else x["cold_us"])
    return {"key": best["config_key"], "rank": c.index(best), "mode": "graph" if small else "cold",
            "family": best.get("family", "triton"), **cost}


def search_shape(ranker, M: int, N: int, K: int, k: int, dtype=torch.float16, order: str = "plain",
                 lt_n: int = 0) -> dict:
    """Rank (plain or diverse top-k, Ranker.rank), then compile + validate + short-time the top k on one placement.
    lt_n > 0 also times cuBLASLt's heuristic top lt_n (no compile; same validate + short timing) in the same pool, so
    the choice is joint over both families. Returns the per-candidate record."""
    from kernels.triton_matmul import TritonMatmul
    from tune.timing import benchmark
    cold_cfg, graph_cfg = short_configs()
    t0 = time.perf_counter()
    keys = ranker.rank(M, N, K, k, order=order)
    rank_s = time.perf_counter() - t0
    try:
        a, b, _ = placed_inputs(M, N, K, dtype, seed=0)
        ref = reference(a, b)
    except torch.OutOfMemoryError as e:  # the shape's tensors + fp32 reference do not fit: record it, do not kill the run
        a = b = ref = None
        torch.cuda.empty_cache()
        return {"rank_s": rank_s, "candidates": [], "skipped": f"oom_inputs: {str(e).splitlines()[0][:150]}"}
    backends = [("triton", key, TritonMatmul(ranker.cfgs[ranker.pool_keys.index(key)]), 0.0) for key in keys]
    lt_setup_s = 0.0
    if lt_n > 0:
        from baselines.cublaslt import candidates as lt_candidates
        t = time.perf_counter()
        lts = lt_candidates(M, N, K, dtype, lt_n)
        lt_setup_s = time.perf_counter() - t
        backends += [("cublaslt", c.key, c, lt_setup_s if i == 0 else 0.0) for i, c in enumerate(lts)]
    cands, bk = [], {}
    for family, key, backend, extra_s in backends:
        t = time.perf_counter()
        chk = validate(backend, M, N, K, dtype, ref=ref, inputs=(a, b))  # first call: JIT compile + correctness
        torch.cuda.synchronize()
        x = {"config_key": key, "valid": chk.valid, "failure_reason": chk.failure_reason,
             "compile_s": time.perf_counter() - t + extra_s}
        if lt_n > 0:
            x["family"] = family
        if chk.valid:
            t = time.perf_counter()
            x["cold_us"] = benchmark(backend.make_runner(a, b), cold_cfg).median_us
            x["time_s"] = time.perf_counter() - t
        cands.append(x)
        bk[key] = backend
    ok = [x for x in cands if x["valid"]]
    if ok and min(x["cold_us"] for x in ok) < SMALL_US:
        for x in ok:
            t = time.perf_counter()
            x["graph_us"] = benchmark(bk[x["config_key"]].make_runner(a, b), graph_cfg).median_us
            x["time_s"] += time.perf_counter() - t
    del a, b, ref
    torch.cuda.empty_cache()
    return {"rank_s": rank_s, "candidates": cands, **({"lt_setup_s": lt_setup_s} if lt_n > 0 else {})}


def run(shapes: Iterable[dict], ranker_stem: Path, out_dir: Path, *, k: int = 5, dtype: str = "fp16",
        order: str = "plain", out_name: str = "online_lgbm_rank.jsonl", session_id: str | None = None, lt_n: int = 0,
        log: Callable[[str], None] = print) -> Path:
    from analysis.learned import Ranker
    from tune.runner import DTYPES
    import os
    session_id = session_id or time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    env = write_env(out_dir, session_id=session_id, kind="online")
    ranker = Ranker.load(ranker_stem)
    path = out_dir / out_name
    done = set()
    if path.exists():
        rule = None if order == "plain" else ranker.rule(order)
        done = {(r["M"], r["N"], r["K"]) for r in map(json.loads, path.read_text().splitlines())
                if r.get("k") == k and r.get("order", "plain") == order and r.get("rule") == rule
                and r.get("lt_n", 0) == lt_n}
    shapes = [s for s in shapes if (s["M"], s["N"], s["K"]) not in done]
    t0 = time.perf_counter()
    with RecordWriter(path) as w:
        for i, s in enumerate(shapes):
            M, N, K = s["M"], s["N"], s["K"]
            t = time.perf_counter()
            r = search_shape(ranker, M, N, K, k, DTYPES[dtype], order, lt_n)
            row = {"M": M, "N": N, "K": K, "dtype": dtype, "k": k, "order": order, "lt_n": lt_n,
                   "rule": None if order == "plain" else ranker.rule(order),
                   "split": s.get("split"), "suite": s.get("suite"),
                   "session_id": session_id, "hw_id": env["hw_id"], "driver": env["driver"], **provenance(env),
                   "ranker": str(ranker_stem), "ranker_meta": ranker.meta,
                   "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"), "wall_s": time.perf_counter() - t,
                   **r, "choice": {str(j): choose(r["candidates"], j) for j in range(1, k + 1)},
                   **({"choice_lt": {f"{j}_lt{n}": choose(r["candidates"], j, n) for j in range(1, k + 1)
                                     for n in sorted({0, 1, 4, lt_n}) if n <= lt_n}} if lt_n else {}),
                   "timestamp": time.time()}
            w.write(row)
            el = time.perf_counter() - t0
            log(f"[{i + 1}/{len(shapes)}] {M}x{N}x{K} -> {row['choice'][str(k)]['key']} ({row['wall_s']:.1f}s) | "
                f"{el:.0f}s elapsed, ETA {el / (i + 1) * (len(shapes) - i - 1):.0f}s")
    return path
