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
from an earlier stage-2 session that timed the same configs. k is per shape when a KPolicy is given (`select_kernel` for one shape, `run(policy=...)` / `tune online --k 3 --k-frequent 5`
for a file of shapes): a shape that carries `calls` (expected launches, >= policy.frequent_calls) or `frequent: true` gets
k_frequent candidates, every other shape k. The signal is the caller's; the ranker cannot know how often a kernel runs.

With lt_n > 0 (`tune online --lt-n 16`) cuBLASLt's heuristic top lt_n joins the same pool (joint selection): they need
no compile, so they cost only validation + short timing; `choice_lt` stores the choice for every k' and lt_n' in
{0, 1, 4, lt_n} (lt_n' = 0 is Triton only, 1 is B2 + Triton). Picks are scored against a full-protocol stage-2
session in analysis/learned.py (`build_topk`).
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch

from tune.correctness import placed_inputs, reference, validate
from tune.env import provenance, write_env
from tune.records import RecordWriter

# How many Triton candidates to compile and time. The ranker cannot know how often a shape will be launched, so the
# call-count signal comes from the caller (KPolicy). k = 3 is the default (~0.5 s first use, ~0.6 s joint with cuBLASLt);
# k = 5 (~0.8-0.9 s) pays back only for shapes launched often: on the 86 held-out shapes, 3 -> 5 saved 0.2 us/call on
# average (10 of 86 shapes improved) for 0.3 s, a payback of ~1.4M calls overall, ~0.6M at >= 50 us and ~13M below 50 us.
# FREQUENT_CALLS = 1M is that overall break-even rounded; it is measured on one GPU, so treat it as a tunable.
K_DEFAULT = 3
K_FREQUENT = 5
FREQUENT_CALLS = 1_000_000

SHORT_REPS = 10
SHORT_WARMUP_MS = 2.0
GRAPH_BATCH = 20
GRAPH_REPS = 5
SMALL_US = 50.0


@dataclass(frozen=True)
class KPolicy:
    """Maps a caller-supplied launch-count signal to the number of Triton candidates to search.

    k_for(calls=None, frequent=None): an explicit `frequent` flag wins; otherwise the shape is frequent when it is
    expected to run at least `frequent_calls` times. No signal means the default k (never the larger one)."""
    k: int = K_DEFAULT
    k_frequent: int = K_FREQUENT
    frequent_calls: int = FREQUENT_CALLS

    def __post_init__(self):
        if self.k < 1 or self.k_frequent < self.k or self.frequent_calls < 1:
            raise ValueError(f"need 1 <= k <= k_frequent and frequent_calls >= 1, got {self}")

    def k_for(self, calls: int | None = None, frequent: bool | None = None) -> int:
        if calls is not None and calls < 0:
            raise ValueError(f"calls must be >= 0, got {calls}")
        if frequent is None:
            frequent = calls is not None and calls >= self.frequent_calls
        return self.k_frequent if frequent else self.k


def shape_k(shape: dict, k: int, policy: KPolicy | None) -> int:
    """k for one shape of a shapes file: the policy's (from the shape's `calls` / `frequent` fields) or the global k."""
    return policy.k_for(shape.get("calls"), shape.get("frequent")) if policy else k


PROPOSAL_FAMILIES = ("cublaslt_knn", "cublaslt_rank")   # cuBLASLt candidates that are not the heuristic's
ROUTER_CONFIRM = 3   # the router times cuBLASLt first; re-timing the top 3 in reverse order cancels the resulting position bias
RANK_DEFAULT = 5   # learned cuBLASLt ranker picks per search (held-out: top-5 = top-10 = top-16 in the online choice; reports/week1.md 2026-11-04)
KNN_DEFAULT = 8   # nearest tune shapes for the cuBLASLt proposals: chosen on a tune-only leave-one-out scan (reports/week1.md 2026-11-04)


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
    """The first k Triton candidates plus the first lt_n cuBLASLt heuristic ones (all of them for None); rows written
    before cuBLASLt candidates existed have no "family" and are all Triton, so this is cands[:k] for them. kNN proposals
    (family "cublaslt_knn", `--lt-knn`) are not heuristic candidates: they count toward no lt_n and join only when
    lt_n is None, so `choice_lt[j_lt<n>]` stays the heuristic-only search and `choice` is the full pool."""
    out, n_tri, n_lt = [], 0, 0
    for x in cands:
        if x.get("family") in PROPOSAL_FAMILIES:
            if lt_n is None:
                out.append(x)
        elif _is_lt(x):
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
    best = min(ok, key=lambda x: x.get("confirm_us", x["graph_us"] if small else x["cold_us"]))
    return {"key": best["config_key"], "rank": c.index(best), "mode": "graph" if small else "cold",
            "family": best.get("family", "triton"), **cost}


def all_choices(cands: list[dict], k: int, lt_n: int) -> dict:
    """The row fields for a search of k Triton candidates (+ lt_n cuBLASLt): `choice` for every k' <= k (all cuBLASLt
    candidates included) and, with lt_n > 0, `choice_lt` for k' x lt_n' in {0, 1, 4, lt_n}."""
    out = {"choice": {str(j): choose(cands, j) for j in range(1, k + 1)}}
    if lt_n:
        out["choice_lt"] = {f"{j}_lt{n}": choose(cands, j, n) for j in range(1, k + 1) for n in sorted({0, 1, 4, lt_n}) if n <= lt_n}
    return out


def search_shape(ranker, M: int, N: int, K: int, k: int, dtype=torch.float16, order: str = "plain",
                 lt_n: int = 0, lt_knn: int = 0, lt_rank: int = 0, router=None, confirm: int = 0) -> dict:
    """Rank (plain or diverse top-k, Ranker.rank), then compile + validate + short-time the top k on one placement.
    lt_n > 0 also times cuBLASLt's heuristic top lt_n (no compile; same validate + short timing) in the same pool, so
    the choice is joint over both families. lt_knn > 0 (needs lt_n > 0) adds the cuBLASLt configs that the deep searches
    of the lt_knn nearest tune shapes found (`baselines/cublaslt_knn.py`, top 2 each, minus those the heuristic already
    returned), validated and short-timed like the rest. lt_rank > 0 (needs lt_n > 0) adds the lt_rank configs a learned
    ranker over all valid cuBLASLt configs predicts fastest (`baselines/cublaslt_rank.py`; family "cublaslt_rank", not
    deduplicated against the k-NN proposals so subsets of the pool can be scored from one run). `router` (a
    `tune.router.Router`, needs lt_n > 0) times the cuBLASLt candidates FIRST and skips the Triton half (ranking, compile,
    validation, timing) when P(Triton still wins) is below its threshold; a shape that is not skipped runs the same
    candidates as without a router. `confirm` = n > 0 re-times the n fastest candidates once more in the REVERSE order of
    their first ranking, after everything else was timed, and ranks them by the mean of the two timings (`confirm_us`):
    a candidate timed first or last in the pass gets the opposite position the second time, which cancels the
    position / GPU-state bias that made the pick change with the timing order (reports/week1.md 2026-11-04 later).
    Returns the per-candidate record (`router`: p_win, t_lt_us, skipped)."""
    from kernels.triton_matmul import TritonMatmul
    from tune.timing import benchmark
    cold_cfg, graph_cfg = short_configs()
    try:
        a, b, _ = placed_inputs(M, N, K, dtype, seed=0)
        ref = reference(a, b)
    except torch.OutOfMemoryError as e:  # the shape's tensors + fp32 reference do not fit: record it, do not kill the run
        a = b = ref = None
        torch.cuda.empty_cache()
        return {"rank_s": 0.0, "candidates": [], "skipped": f"oom_inputs: {str(e).splitlines()[0][:150]}"}
    lt_backends = []
    lt_setup_s = 0.0
    if lt_n > 0:
        from baselines.cublaslt import candidates as lt_candidates
        t = time.perf_counter()
        lts = lt_candidates(M, N, K, dtype, lt_n)
        lt_setup_s = time.perf_counter() - t
        lt_backends += [("cublaslt", c.key, c, lt_setup_s if i == 0 else 0.0) for i, c in enumerate(lts)]
        if lt_knn > 0:
            from baselines.cublaslt_knn import candidates as knn_candidates, cfg_tuple
            t = time.perf_counter()
            known = {cfg_tuple(c.cfg) for c in lts}
            knn = knn_candidates(M, N, K, dtype, lt_knn, problem=lts[0].problem if lts else None, skip=known)
            knn_setup_s = time.perf_counter() - t
            lt_backends += [("cublaslt_knn", c.key, c, knn_setup_s if i == 0 else 0.0) for i, c in enumerate(knn)]
        if lt_rank > 0:
            from baselines.cublaslt_rank import candidates as rank_candidates
            from baselines.cublaslt_knn import cfg_tuple
            t = time.perf_counter()
            known = {cfg_tuple(c.cfg) for c in lts}
            ranked = rank_candidates(M, N, K, dtype, lt_rank, problem=lts[0].problem if lts else None, skip=known)
            rank_setup_s = time.perf_counter() - t
            lt_backends += [("cublaslt_rank", c.key, c, rank_setup_s if i == 0 else 0.0) for i, c in enumerate(ranked)]
    cands, bk = [], {}

    def measure(backends):
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

    def graph_pass(xs):
        """Graph-time the valid candidates `xs` (those not timed yet) when the fastest cold time is below SMALL_US."""
        ok = [x for x in cands if x["valid"]]
        if ok and min(x["cold_us"] for x in ok) < SMALL_US:
            for x in xs:
                if x["valid"] and "graph_us" not in x:
                    t = time.perf_counter()
                    x["graph_us"] = benchmark(bk[x["config_key"]].make_runner(a, b), graph_cfg).median_us
                    x["time_s"] += time.perf_counter() - t

    def triton_backends():
        t0 = time.perf_counter()
        keys = ranker.rank(M, N, K, k, order=order)
        return [("triton", key, TritonMatmul(ranker.cfgs[ranker.pool_keys.index(key)]), 0.0) for key in keys], time.perf_counter() - t0

    rank_s, info = 0.0, None
    if router is not None and lt_backends:
        measure(lt_backends)
        graph_pass(cands)
        ok = [x for x in cands if x["valid"] and x.get("cold_us") is not None]
        if ok:
            small = min(x["cold_us"] for x in ok) < SMALL_US and all(x.get("graph_us") is not None for x in ok)
            t_lt = min(x["graph_us"] if small else x["cold_us"] for x in ok)
            skip, p = router.skip_triton(M, N, K, t_lt)
        else:                                              # no valid cuBLASLt candidate: Triton is the only way out
            t_lt, skip, p = None, False, None
        info = {"p_win": p, "t_lt_us": t_lt, "skipped": bool(skip), "threshold": router.threshold}
        if not skip:
            tb, rank_s = triton_backends()
            measure(tb)
    else:
        tb, rank_s = triton_backends()
        measure(tb + lt_backends)
    graph_pass(cands)
    if confirm > 0:
        ok = [x for x in cands if x["valid"] and x.get("cold_us") is not None]
        small = bool(ok) and min(x["cold_us"] for x in ok) < SMALL_US and all(x.get("graph_us") is not None for x in ok)
        metric = "graph_us" if small else "cold_us"
        top = sorted(ok, key=lambda x: x[metric])[:confirm]
        for x in reversed(top):
            t = time.perf_counter()
            again = benchmark(bk[x["config_key"]].make_runner(a, b), graph_cfg if small else cold_cfg).median_us
            x["confirm_us"] = 0.5 * (x[metric] + again)
            x["time_s"] += time.perf_counter() - t
        for x in ok:                                   # unconfirmed candidates were all slower than the confirmed ones
            x.setdefault("confirm_us", x[metric])
    del a, b, ref
    torch.cuda.empty_cache()
    return {"rank_s": rank_s, "candidates": cands, **({"lt_setup_s": lt_setup_s} if lt_n > 0 else {}),
            **({"lt_knn": lt_knn} if lt_knn > 0 else {}), **({"lt_rank": lt_rank} if lt_rank > 0 else {}),
            **({"router": info} if info else {})}


_DTYPE_NAMES = {torch.float16: "fp16", torch.bfloat16: "bf16"}


def select_kernel(ranker, M: int, N: int, K: int, *, calls: int | None = None, frequent: bool | None = None,
                  policy: KPolicy = KPolicy(), lt_n: int = 16, order: str = "diverse", dtype=torch.float16,
                  cache=True, use_bundle: bool = True, lt_knn: int | None = None, lt_rank: int = RANK_DEFAULT,
                  router: bool | str = "auto") -> dict:
    """First-use kernel selection for one shape with the call-count-aware k policy. `calls` (expected launches) or
    `frequent` (explicit) come from the caller; the ranker cannot know them. Searches the policy's k Triton candidates
    plus cuBLASLt's heuristic top lt_n (0 = Triton only) plus, with lt_n > 0, the configs the deep searches of the lt_knn
    nearest tune shapes found (`baselines/cublaslt_knn.py`; 0 = off; no table for this GPU = none) and the lt_rank configs a
    learned ranker over all valid cuBLASLt configs predicts fastest (`baselines/cublaslt_rank.py`; 0 = off). lt_knn=None
    means automatic: KNN_DEFAULT neighbours only when the ranker is off or has no model for this GPU, else none (the
    ranker alone beat the proposals; the union adds little for +0.1 s). `router` (`tune/router.py`) times the cuBLASLt
    candidates first and skips the Triton half when P(Triton still wins) is under the router's threshold: "auto" (default)
    = on only when the Triton kernels will really compile (no bundle, or a bundle built for another Triton / GPU), off
    when the bundle serves them (the Triton half is then ~0.06 s of ~0.17 s); True / False force it; no model for this GPU
    = off. A skipped shape's cached winner is the cuBLASLt one. Returns {"k", "reason", "choice", "candidates", "rank_s",
    "cached"}. choice = {"key", "family", "mode", "rank", "compile_s", "time_s"} (key None if no candidate was valid); a shape
    whose tensors do not fit in memory returns {"skipped": ...} with choice None.

    Two things make a repeat or a fresh clone cheap (both default on, both opt-out because experiments measure the search
    itself and `demo.py` passes `cache=None, use_bundle=False`):
      * `use_bundle`: install the prebuilt Triton kernel bundle (`kernels/bundle.py`), so a candidate the rankers can
        propose is loaded instead of compiled. Any kernel not in it, or a bundle built for another Triton / GPU
        architecture, compiles as before (a mismatch is warned about once);
      * `cache`: the per-shape winner cache (`tune/winner_cache.py`). True = the default file for this GPU (None / False /
        `TUNE_WINNER_CACHE=off` = off; or pass a `WinnerCache`). A stored winner from a search at least as wide as this
        one is returned with no ranking, compiling or timing: {"reason": "cached", "cached": True, "candidates": []}. A
        miss searches and stores the winner. Cache problems are warned about and never fail the selection."""
    import warnings
    bundle_serves = False
    if use_bundle:
        from kernels import bundle
        info = bundle.install()
        bundle_serves = bool(info.get("installed") and info.get("matches"))
    wc = _winner_cache(cache)
    if lt_n and lt_rank:
        from baselines.cublaslt_rank import ranker as _lt_ranker
        if _lt_ranker() is None:
            lt_rank = 0
    if lt_knn is None:
        lt_knn = 0 if (lt_rank or not lt_n) else KNN_DEFAULT
    if not lt_n:
        lt_knn = lt_rank = 0
    rt = None
    if lt_n and (router is True or (router == "auto" and not bundle_serves)):
        from tune.router import router as _router
        rt = _router()
    k = policy.k_for(calls, frequent)
    dname = _DTYPE_NAMES.get(dtype, str(dtype))
    if wc is not None:
        try:
            from tune.winner_cache import backend_for
            hit = wc.get(M, N, K, dname, k=k, lt_n=lt_n, lt_knn=lt_knn, lt_rank=lt_rank)
            if hit is not None and backend_for(hit["choice"], M, N, K, dtype, hit["lt_n"]) is not None:
                return {"k": hit["k"], "reason": "cached", "rank_s": 0.0, "choice": hit["choice"], "candidates": [],
                        "cached": True}
        except Exception as e:                      # a broken cache must not break selection: search instead
            warnings.warn(f"winner cache lookup failed ({type(e).__name__}: {e}); searching")
    r = search_shape(ranker, M, N, K, k, dtype, order, lt_n,
                     **({"lt_knn": lt_knn} if lt_knn else {}), **({"lt_rank": lt_rank} if lt_rank else {}),
                     **({"router": rt, "confirm": ROUTER_CONFIRM} if rt is not None else {}))
    out = {"k": k, "reason": "frequent" if k == policy.k_frequent and k != policy.k else "default", "rank_s": r["rank_s"],
           "cached": False}
    if r.get("skipped"):
        return {**out, "skipped": r["skipped"], "choice": None, "candidates": []}
    choice = choose(r["candidates"], k)
    if wc is not None:
        try:
            wc.put(M, N, K, dname, k=k, lt_n=lt_n, lt_knn=lt_knn, lt_rank=lt_rank, choice=choice, candidates=r["candidates"], order=order,
                   ranker=f"lgbm_rank/{getattr(ranker, 'features', None) or 'v1'}")
        except Exception as e:
            warnings.warn(f"winner cache write failed ({type(e).__name__}: {e})")
    return {**out, "choice": choice, "candidates": r["candidates"]}


def _winner_cache(cache):
    """`cache` argument of `select_kernel` -> a WinnerCache or None."""
    if cache is None or cache is False:
        return None
    if cache is True:
        from tune.winner_cache import WinnerCache
        return WinnerCache.default()
    return cache


def run(shapes: Iterable[dict], ranker_stem: Path, out_dir: Path, *, k: int = 5, dtype: str = "fp16",
        order: str = "plain", out_name: str = "online_lgbm_rank.jsonl", session_id: str | None = None, lt_n: int = 0,
        lt_knn: int = 0, lt_rank: int = 0, use_router: bool = False, router_threshold: float | None = None, confirm: int = 0,
        policy: KPolicy | None = None,
        log: Callable[[str], None] = print) -> Path:
    from analysis.learned import Ranker
    from tune.runner import DTYPES
    import os
    session_id = session_id or time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    env = write_env(out_dir, session_id=session_id, kind="online")
    ranker = Ranker.load(ranker_stem)
    rt = None
    if use_router:
        from tune.router import router as _router
        rt = _router()
        if rt is None:
            raise RuntimeError("--router: no family router model for this GPU (python -m analysis.router fit --save data/<hw>/router)")
        if router_threshold is not None:                    # control runs: 0 = cuBLASLt timed first but Triton never skipped
            rt.threshold = float(router_threshold)
    path = out_dir / out_name
    done = set()
    if path.exists():
        rule = None if order == "plain" else ranker.rule(order)
        done = {(r["M"], r["N"], r["K"], r.get("k")) for r in map(json.loads, path.read_text().splitlines())
                if r.get("order", "plain") == order and r.get("rule") == rule and r.get("lt_n", 0) == lt_n and r.get("lt_knn", 0) == lt_knn and r.get("lt_rank", 0) == lt_rank
                and bool(r.get("router")) == use_router and r.get("confirm", 0) == confirm}
    # Per-shape k: the policy's, from the shape's own `calls` / `frequent` fields; without a policy, the global k.
    shapes = [dict(s, _k=shape_k(s, k, policy)) for s in shapes]
    shapes = [s for s in shapes if (s["M"], s["N"], s["K"], s["_k"]) not in done]
    t0 = time.perf_counter()
    with RecordWriter(path) as w:
        for i, s in enumerate(shapes):
            M, N, K, ks = s["M"], s["N"], s["K"], s["_k"]
            t = time.perf_counter()
            r = search_shape(ranker, M, N, K, ks, DTYPES[dtype], order, lt_n,
                             **({"lt_knn": lt_knn} if lt_knn else {}), **({"lt_rank": lt_rank} if lt_rank else {}),
                             **({"router": rt} if rt is not None else {}), **({"confirm": confirm} if confirm else {}))
            row = {"M": M, "N": N, "K": K, "dtype": dtype, "k": ks, "order": order, "lt_n": lt_n, "lt_knn": lt_knn, "lt_rank": lt_rank, "confirm": confirm,
                   **({"calls": s.get("calls"), "frequent": s.get("frequent"),
                       "k_reason": "frequent" if ks == policy.k_frequent and ks != policy.k else "default"} if policy else {}),
                   "rule": None if order == "plain" else ranker.rule(order),
                   "split": s.get("split"), "suite": s.get("suite"),
                   "session_id": session_id, "hw_id": env["hw_id"], "driver": env["driver"], **provenance(env),
                   "ranker": str(ranker_stem), "ranker_meta": ranker.meta,
                   "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"), "wall_s": time.perf_counter() - t,
                   **r, **all_choices(r["candidates"], ks, lt_n), "timestamp": time.time()}
            w.write(row)
            el = time.perf_counter() - t0
            log(f"[{i + 1}/{len(shapes)}] {M}x{N}x{K} k={ks} -> {row['choice'][str(ks)]['key']} ({row['wall_s']:.1f}s) | "
                f"{el:.0f}s elapsed, ETA {el / (i + 1) * (len(shapes) - i - 1):.0f}s")
    return path
