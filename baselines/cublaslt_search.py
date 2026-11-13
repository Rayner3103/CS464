"""Deeper B3: search cuBLASLt's configuration space beyond its heuristic's top-N.

The heuristic (baselines/cublaslt.py) returns only a few candidates per shape, often 1-4. This module:
  1. enumerates configurations: every algorithm id x tile x stages, then the valid bases expanded over
     split-K x reduction scheme x CTA swizzle x custom option, each filtered by cublasLtMatmulAlgoCheck
     and the workspace cap;
  2. seeds the screen with the best-fit heuristic top-N and a per-algorithm heuristic search
     (CUBLASLT_SEARCH_LIMITED_BY_ALGO_ID);
  3. screens configs with a quick event timer (hot L2, 1 warmup + 3 timed launches), seeds first, then a seeded
     random sample of the rest, until a per-shape time budget runs out;
  4. returns the fastest `top_k` distinct configs. The harness then validates and times them with the full protocol
     (family "cublaslt_deep").

Screening runs in a subprocess per shape (python -m baselines.cublaslt_search ...), so a faulting configuration
cannot take down the benchmark run. Results are cached as JSON per shape.
"""
from __future__ import annotations

import argparse
import ctypes as C
import json
import random
import time
from pathlib import Path

import torch

from baselines.cublaslt import (CUBLAS_COMPUTE_32F, CUDA_R_32F, WORKSPACE_BYTES, _CUDA_DTYPE, _Algo, _HeuristicResult,
                                _lib, _Problem, algo_config)

SEARCH_VERSION = 1
CAP_SPLITK_SUPPORT, CAP_REDUCTION_MASK, CAP_SWIZZLE_SUPPORT, CAP_TILE_IDS, CAP_CUSTOM_MAX, CAP_STAGES_IDS = 0, 1, 2, 6, 7, 13
CFG_TILE, CFG_SPLITK, CFG_REDUCTION, CFG_SWIZZLE, CFG_CUSTOM, CFG_STAGES = 1, 2, 3, 4, 5, 6
PREF_SEARCH_MODE, SEARCH_LIMITED_BY_ALGO_ID = 0, 1
SPLITK_VALUES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
MAX_CUSTOM_TILED = 7      # custom options explored per tiled algorithm (tile-less ones, e.g. GEMV, get all, <= 255)
CFG_KEYS = ("algo_id", "tile_id", "stages_id", "splitk", "reduction", "swizzle", "custom")
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


def _cap_array(lib, algo, attr) -> list[int]:
    written = C.c_size_t()
    lib.cublasLtMatmulAlgoCapGetAttribute(C.byref(algo), attr, None, 0, C.byref(written))
    n = written.value // 4
    if not n:
        return []
    buf = (C.c_uint32 * n)()
    if lib.cublasLtMatmulAlgoCapGetAttribute(C.byref(algo), attr, buf, C.sizeof(buf), C.byref(written)) != 0:
        return []
    return list(buf)


def _cap_int(lib, algo, attr) -> int:
    v, written = C.c_int32(), C.c_size_t()
    return v.value if lib.cublasLtMatmulAlgoCapGetAttribute(C.byref(algo), attr, C.byref(v), 4, C.byref(written)) == 0 else 0


def _set(lib, algo, attr, value: int):
    v = C.c_uint32(value)
    return lib.cublasLtMatmulAlgoConfigSetAttribute(C.byref(algo), attr, C.byref(v), C.sizeof(v))


def _init(lib, handle, ctype: int, algo_id: int) -> _Algo | None:
    algo = _Algo()
    st = lib.cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, ctype, ctype, ctype, ctype, algo_id, C.byref(algo))
    return algo if st == 0 else None


def build_algo(problem: _Problem, ctype: int, cfg: dict) -> _HeuristicResult | None:
    """Recreate a configuration from its attributes; returns a checked result (algo + workspace) or None."""
    lib, handle = problem.lib, problem.handle
    algo = _init(lib, handle, ctype, cfg["algo_id"])
    if algo is None:
        return None
    for attr, key in ((CFG_TILE, "tile_id"), (CFG_STAGES, "stages_id"), (CFG_SPLITK, "splitk"),
                      (CFG_REDUCTION, "reduction"), (CFG_SWIZZLE, "swizzle"), (CFG_CUSTOM, "custom")):
        if cfg.get(key) is not None and _set(lib, algo, attr, cfg[key]) != 0:
            return None
    res = _HeuristicResult()
    if lib.cublasLtMatmulAlgoCheck(handle, problem.desc, problem.a, problem.b, problem.c, problem.c,
                                   C.byref(algo), C.byref(res)) != 0:
        return None
    if res.workspaceSize > WORKSPACE_BYTES:
        return None
    res.algo = algo  # AlgoCheck never writes result.algo
    return res


def _key(cfg: dict) -> tuple:
    return tuple(cfg.get(k) or 0 for k in CFG_KEYS)  # unset tile/stages (None) == 0 (UNDEFINED)


def enumerate_configs(problem: _Problem, ctype: int, M: int, N: int, K: int) -> tuple[list[dict], int]:
    lib, handle = problem.lib, problem.handle
    ids = (C.c_int * 1024)()
    n = C.c_int()
    lib.cublasLtMatmulAlgoGetIds(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, ctype, ctype, ctype, ctype, 1024, ids, C.byref(n))
    out, checks = [], 0
    for aid in ids[:n.value]:
        algo = _init(lib, handle, ctype, aid)
        if algo is None:
            continue
        tiles = _cap_array(lib, algo, CAP_TILE_IDS) or [None]
        stages = _cap_array(lib, algo, CAP_STAGES_IDS) or [None]
        splitk_ok = _cap_int(lib, algo, CAP_SPLITK_SUPPORT) == 1
        red_mask = _cap_int(lib, algo, CAP_REDUCTION_MASK)
        swizzles = [0, 1] if _cap_int(lib, algo, CAP_SWIZZLE_SUPPORT) == 1 else [0]
        cmax = _cap_int(lib, algo, CAP_CUSTOM_MAX)
        customs = list(range(min(cmax, 255) + 1)) if tiles == [None] else list(range(min(cmax, MAX_CUSTOM_TILED) + 1))
        # Phase A: valid (tile, stages) bases at defaults.
        bases = []
        for t in tiles:
            for s in stages:
                cfg = {"algo_id": aid, "tile_id": t, "stages_id": s, "splitk": 1, "reduction": 0, "swizzle": 0, "custom": 0}
                checks += 1
                if build_algo(problem, ctype, cfg) is not None:
                    bases.append(cfg)
        # Phase B: expand the valid bases (split-K only where each split keeps a useful K slice).
        splitks = [s for s in SPLITK_VALUES if s == 1 or (splitk_ok and K // s >= 64)]
        reds = [r for r in (1, 2, 4) if red_mask & r]
        for base in bases:
            for c in customs:
                for sw in swizzles:
                    for sk in splitks:
                        for red in ([0] if sk == 1 else reds):
                            cfg = dict(base, custom=c, swizzle=sw, splitk=sk, reduction=red)
                            if cfg == base:
                                out.append(cfg)
                                continue
                            checks += 1
                            if build_algo(problem, ctype, cfg) is not None:
                                out.append(cfg)
    return out, checks


def heuristic_seeds(problem: _Problem, ctype: int, n_best_fit: int = 16, n_per_algo: int = 8) -> list[dict]:
    lib, handle = problem.lib, problem.handle
    seeds = [algo_config(lib, r.algo) for r in problem.heuristics(n_best_fit)]
    ids = (C.c_int * 1024)()
    n = C.c_int()
    lib.cublasLtMatmulAlgoGetIds(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, ctype, ctype, ctype, ctype, 1024, ids, C.byref(n))
    pref = C.c_void_p()
    lib.cublasLtMatmulPreferenceCreate(C.byref(pref))
    ws, mode = C.c_uint64(WORKSPACE_BYTES), C.c_uint32(SEARCH_LIMITED_BY_ALGO_ID)
    lib.cublasLtMatmulPreferenceSetAttribute(pref, 1, C.byref(ws), C.sizeof(ws))
    lib.cublasLtMatmulPreferenceSetAttribute(pref, PREF_SEARCH_MODE, C.byref(mode), C.sizeof(mode))
    for aid in ids[:n.value]:
        algo = _init(lib, handle, ctype, aid)
        if algo is None:
            continue
        results = (_HeuristicResult * n_per_algo)()
        for i in range(n_per_algo):  # LIMITED_BY_ALGO_ID needs every result slot pre-initialized (else status 7)
            results[i].algo = algo
        cnt = C.c_int(0)
        if lib.cublasLtMatmulAlgoGetHeuristic(handle, problem.desc, problem.a, problem.b, problem.c, problem.c, pref,
                                              n_per_algo, results, C.byref(cnt)) == 0:
            seeds += [algo_config(lib, results[i].algo) for i in range(cnt.value) if results[i].state == 0]
    lib.cublasLtMatmulPreferenceDestroy(pref)
    uniq, seen = [], set()
    for s in seeds:
        if _key(s) not in seen:
            seen.add(_key(s))
            uniq.append(s)
    return uniq


def _screen_time_us(lib, handle, problem, res, a, b, c, ws, reps: int = 3) -> float:
    alpha, beta = C.c_float(1.0), C.c_float(0.0)
    stream = C.c_void_p(torch.cuda.current_stream().cuda_stream)
    ws_ptr, ws_size = C.c_void_p(ws.data_ptr()), C.c_size_t(int(res.workspaceSize))
    pa, pb, pc = C.c_void_p(b.data_ptr()), C.c_void_p(a.data_ptr()), C.c_void_p(c.data_ptr())

    def launch():
        st = lib.cublasLtMatmul(handle, problem.desc, C.byref(alpha), pa, problem.a, pb, problem.b, C.byref(beta),
                                pc, problem.c, pc, problem.c, C.byref(res.algo), ws_ptr, ws_size, stream)
        if st != 0:
            raise RuntimeError(f"cublasLtMatmul status {st}")
    launch()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        launch()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) * 1e3 / reps


def search(M: int, N: int, K: int, dtype: str = "fp16", budget_s: float = 20.0, top_k: int = 8, seed: int = 0,
           keep_all: bool = False) -> dict:
    tdtype = DTYPES[dtype]
    ctype = _CUDA_DTYPE[tdtype]
    problem = _Problem(M, N, K, tdtype)
    lib, handle = problem.lib, problem.handle
    t0 = time.perf_counter()
    configs, n_checks = enumerate_configs(problem, ctype, M, N, K)
    t_enum = time.perf_counter() - t0
    seeds = heuristic_seeds(problem, ctype)
    seed_keys = {_key(s) for s in seeds}
    rest = [c for c in configs if _key(c) not in seed_keys]
    random.Random(seed).shuffle(rest)
    order = seeds + rest

    g = torch.Generator(device="cuda").manual_seed(0)
    a = torch.randn((M, K), generator=g, device="cuda").to(tdtype)
    b = torch.randn((K, N), generator=g, device="cuda").to(tdtype)
    c = torch.empty((M, N), device="cuda", dtype=tdtype)
    ws = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device="cuda")
    screened, failed = [], 0
    t_screen0 = time.perf_counter()
    for i, cfg in enumerate(order):
        if time.perf_counter() - t_screen0 > budget_s:
            break
        res = build_algo(problem, ctype, cfg)
        if res is None:
            failed += 1
            continue
        try:
            us = _screen_time_us(lib, handle, problem, res, a, b, c, ws)
        except RuntimeError:
            failed += 1
            continue
        screened.append({"cfg": cfg, "us": us, "seed": i < len(seeds)})
    screened.sort(key=lambda x: x["us"])
    top, seen = [], set()
    for s in screened:
        if _key(s["cfg"]) not in seen:
            seen.add(_key(s["cfg"]))
            top.append(s)
        if len(top) == top_k:
            break
    return {"search_version": SEARCH_VERSION, "M": M, "N": N, "K": K, "dtype": dtype, "budget_s": budget_s,
            "n_algo_checks": n_checks, "n_valid_configs": len(configs), "n_seeds": len(seeds),
            "n_screened": len(screened), "n_screen_failed": failed,
            "screened_all": len(screened) + failed >= len(order), "enumerate_s": round(t_enum, 3),
            "screen_s": round(time.perf_counter() - t_screen0, 3),
            "best_seed_us": min((s["us"] for s in screened if s["seed"]), default=None),
            "top": top, **({"screened": screened} if keep_all else {})}


def cache_path(cache_dir: Path, M: int, N: int, K: int, dtype: str) -> Path:
    return Path(cache_dir) / f"{M}x{N}x{K}_{dtype}.json"


def search_isolated(M: int, N: int, K: int, dtype: str, cache_dir: Path, budget_s: float = 20.0, top_k: int = 8,
                    timeout_s: float = 600.0) -> dict:
    """Run `search` in a subprocess (crash/hang isolation) and cache its JSON result per shape."""
    import subprocess
    import sys
    path = cache_path(cache_dir, M, N, K, dtype)
    if path.exists():
        d = json.loads(path.read_text())
        if d.get("search_version") == SEARCH_VERSION and d.get("budget_s") == budget_s:
            return d
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "baselines.cublaslt_search", str(M), str(N), str(K), "--dtype", dtype,
           "--budget-s", str(budget_s), "--top-k", str(top_k), "--out", str(path)]
    try:
        p = subprocess.run(cmd, cwd=Path(__file__).resolve().parent.parent, capture_output=True, text=True,
                           timeout=timeout_s)
        ok = p.returncode == 0 and path.exists()
        err = None if ok else (p.stderr.strip().splitlines() or ["?"])[-1][:300]
    except subprocess.TimeoutExpired:
        ok, err = False, f"timeout after {timeout_s}s"
    if not ok:
        d = {"search_version": SEARCH_VERSION, "M": M, "N": N, "K": K, "dtype": dtype, "budget_s": budget_s,
             "error": err, "top": []}
        path.write_text(json.dumps(d, indent=1))
        return d
    return json.loads(path.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("M", type=int)
    ap.add_argument("N", type=int)
    ap.add_argument("K", type=int)
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--budget-s", type=float, default=20.0)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-all", action="store_true",
                    help="also write every screened config's time (`screened`), not only the top-k: the training set of "
                         "a learned cuBLASLt ranker (baselines/cublaslt_screen.py)")
    a = ap.parse_args()
    d = search(a.M, a.N, a.K, a.dtype, a.budget_s, a.top_k, keep_all=a.keep_all)
    Path(a.out).write_text(json.dumps(d, indent=1))


if __name__ == "__main__":
    main()
