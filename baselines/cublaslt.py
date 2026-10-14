"""B2 / B3: cuBLASLt via ctypes (plan 4.1).

B2 = the heuristic's top-1 algorithm (cuBLAS's own selector).
B3 = best of the heuristic's top-N candidates after timing each one (the vendor-library oracle).

Each candidate is exposed as its own backend, so the harness validates and times it like any other kernel.
B2 is the row with heuristic_rank 0; B3 is the per-shape minimum over all cuBLASLt rows.

The library is the same libcublasLt that torch already loaded, so B1 (torch.matmul) and B2/B3 share one cuBLAS build.

Row-major trick: C[M,N] = A[M,K] @ B[K,N] in row-major is C^T = B^T A^T in column-major. Row-major B's memory read as
column-major is an N x K matrix with ld = N, i.e. B^T, and likewise for A^T and C^T. So we call cuBLASLt with
m=N, n=M, k=K, "A"=B, "B"=A, no transposes.
"""
from __future__ import annotations

import ctypes as C
import re
from functools import lru_cache

import torch

# cuBLAS / CUDA enums (cublas_api.h, library_types.h, cublasLt.h)
CUBLAS_COMPUTE_32F = 68
CUDA_R_32F, CUDA_R_16F, CUDA_R_16BF = 0, 2, 14
_CUDA_DTYPE = {torch.float16: CUDA_R_16F, torch.bfloat16: CUDA_R_16BF}
PREF_MAX_WORKSPACE_BYTES = 1
# cublasLtMatmulAlgoConfigAttributes_t: id -> (name, ctype)
ALGO_ATTRS = {0: ("algo_id", C.c_int32), 1: ("tile_id", C.c_uint32), 2: ("splitk", C.c_uint32),
              3: ("reduction", C.c_uint32), 4: ("swizzle", C.c_uint32), 5: ("custom", C.c_uint32),
              6: ("stages_id", C.c_uint32)}
WORKSPACE_BYTES = 32 * 2**20


class _Algo(C.Structure):
    _fields_ = [("data", C.c_uint64 * 8)]


class _HeuristicResult(C.Structure):
    _fields_ = [("algo", _Algo), ("workspaceSize", C.c_size_t), ("state", C.c_int), ("wavesCount", C.c_float),
                ("reserved", C.c_int * 4)]


@lru_cache(maxsize=1)
def _lib():
    torch.ones(1, device="cuda") @ torch.ones(1, device="cuda")  # make torch load its cuBLAS
    maps = open("/proc/self/maps").read()
    paths = sorted(set(re.findall(r"\S*libcublasLt\.so[\S]*", maps)))
    if not paths:
        raise RuntimeError("torch has not loaded libcublasLt; cannot bind B2/B3")
    lib = C.CDLL(paths[0])
    lib.cublasLtGetVersion.restype = C.c_size_t
    handle = C.c_void_p()
    _check(lib.cublasLtCreate(C.byref(handle)), "cublasLtCreate")
    return lib, handle, paths[0]


def _check(status: int, what: str):
    if status != 0:
        raise RuntimeError(f"{what} failed with cublasStatus {status}")


def library_info() -> dict:
    lib, _, path = _lib()
    return {"path": path, "version": int(lib.cublasLtGetVersion())}


class _Problem:
    """Matmul + layout descriptors for one (M, N, K, dtype); owns them and destroys them when garbage-collected."""

    def __init__(self, M: int, N: int, K: int, dtype):
        self.lib, self.handle, _ = _lib()
        lib = self.lib
        t = _CUDA_DTYPE[dtype]
        self.desc = C.c_void_p()
        _check(lib.cublasLtMatmulDescCreate(C.byref(self.desc), CUBLAS_COMPUTE_32F, CUDA_R_32F), "MatmulDescCreate")
        self.a, self.b, self.c = C.c_void_p(), C.c_void_p(), C.c_void_p()
        # Column-major view: "A" = B^T (N x K, ld N), "B" = A^T (K x M, ld K), C^T (N x M, ld N).
        _check(lib.cublasLtMatrixLayoutCreate(C.byref(self.a), t, C.c_uint64(N), C.c_uint64(K), C.c_int64(N)), "LayoutA")
        _check(lib.cublasLtMatrixLayoutCreate(C.byref(self.b), t, C.c_uint64(K), C.c_uint64(M), C.c_int64(K)), "LayoutB")
        _check(lib.cublasLtMatrixLayoutCreate(C.byref(self.c), t, C.c_uint64(N), C.c_uint64(M), C.c_int64(N)), "LayoutC")
        self.pref = C.c_void_p()
        _check(lib.cublasLtMatmulPreferenceCreate(C.byref(self.pref)), "PreferenceCreate")
        ws = C.c_uint64(WORKSPACE_BYTES)
        _check(lib.cublasLtMatmulPreferenceSetAttribute(self.pref, PREF_MAX_WORKSPACE_BYTES, C.byref(ws), C.sizeof(ws)),
               "PreferenceSetAttribute")

    def heuristics(self, n: int) -> list[_HeuristicResult]:
        results = (_HeuristicResult * n)()
        count = C.c_int(0)
        _check(self.lib.cublasLtMatmulAlgoGetHeuristic(self.handle, self.desc, self.a, self.b, self.c, self.c, self.pref,
                                                       n, results, C.byref(count)), "AlgoGetHeuristic")
        return [results[i] for i in range(count.value) if results[i].state == 0]

    def __del__(self):
        try:
            for fn, h in (("cublasLtMatmulPreferenceDestroy", self.pref), ("cublasLtMatrixLayoutDestroy", self.a),
                          ("cublasLtMatrixLayoutDestroy", self.b), ("cublasLtMatrixLayoutDestroy", self.c),
                          ("cublasLtMatmulDescDestroy", self.desc)):
                getattr(self.lib, fn)(h)
        except Exception:
            pass


def algo_config(lib, algo: _Algo) -> dict:
    out = {}
    for attr, (name, ctype) in ALGO_ATTRS.items():
        v, written = ctype(), C.c_size_t()
        if lib.cublasLtMatmulAlgoConfigGetAttribute(C.byref(algo), attr, C.byref(v), C.sizeof(v), C.byref(written)) == 0:
            out[name] = int(v.value)
    return out


class CublasLtMatmul:
    """One heuristic candidate for one problem shape."""

    family = "cublaslt"

    def __init__(self, problem: _Problem, result: _HeuristicResult, rank: int, family: str = "cublaslt",
                 prefix: str = "h", extra: dict | None = None):
        self.problem, self.result, self.rank, self.family = problem, result, rank, family
        rank_key = "heuristic_rank" if prefix == "h" else "search_rank"
        self.cfg = {rank_key: rank, "workspace_bytes": int(result.workspaceSize),
                    "waves": round(float(result.wavesCount), 3), **algo_config(problem.lib, result.algo), **(extra or {})}
        self.key = (f"{prefix}{rank}_a{self.cfg.get('algo_id')}_t{self.cfg.get('tile_id')}_s{self.cfg.get('stages_id')}"
                    f"_sk{self.cfg.get('splitk')}_r{self.cfg.get('reduction')}_sw{self.cfg.get('swizzle')}"
                    f"_c{self.cfg.get('custom')}")

    def make_runner(self, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor | None = None):
        assert a.is_contiguous() and b.is_contiguous(), "row-major contiguous inputs only (NN layout)"
        M, K = a.shape
        N = b.shape[1]
        c = out if out is not None else torch.empty((M, N), device=a.device, dtype=a.dtype)
        assert c.is_contiguous()
        p = self.problem
        lib, handle = p.lib, p.handle
        ws = torch.empty(max(int(self.result.workspaceSize), 1), dtype=torch.uint8, device=a.device)
        alpha, beta = C.c_float(1.0), C.c_float(0.0)
        algo = self.result.algo
        args_ptrs = (C.c_void_p(b.data_ptr()), C.c_void_p(a.data_ptr()), C.c_void_p(c.data_ptr()))
        ws_ptr, ws_size = C.c_void_p(ws.data_ptr()), C.c_size_t(int(self.result.workspaceSize))

        def run():
            stream = C.c_void_p(torch.cuda.current_stream(a.device).cuda_stream)
            st = lib.cublasLtMatmul(handle, p.desc, C.byref(alpha), args_ptrs[0], p.a, args_ptrs[1], p.b,
                                    C.byref(beta), args_ptrs[2], p.c, args_ptrs[2], p.c, C.byref(algo),
                                    ws_ptr, ws_size, stream)
            if st != 0:
                raise RuntimeError(f"cublasLtMatmul failed with cublasStatus {st}")
            return c
        run._keepalive = (ws, alpha, beta, algo, args_ptrs)  # keep ctypes buffers alive with the closure
        return run


def deep_candidates(M: int, N: int, K: int, dtype=torch.float16, budget_s: float = 20.0, top_k: int = 8,
                    cache_dir=None) -> list[CublasLtMatmul]:
    """Deeper B3: the fastest `top_k` configs from an enumerate-and-screen search (baselines/cublaslt_search.py)."""
    from pathlib import Path
    from baselines.cublaslt_search import build_algo, search_isolated
    if cache_dir is None:  # the runner points this at <out_dir>/lt_search for each run
        import os
        from tune.env import default_data_dir
        cache_dir = os.environ.get("TUNE_LT_SEARCH_DIR") or default_data_dir() / "lt_search"
    dname = {torch.float16: "fp16", torch.bfloat16: "bf16"}[dtype]
    d = search_isolated(M, N, K, dname, Path(cache_dir), budget_s=budget_s, top_k=top_k)
    stats = {k: d.get(k) for k in ("n_valid_configs", "n_screened", "screened_all", "best_seed_us", "error")}
    p = _Problem(M, N, K, dtype)
    out = []
    for rank, t in enumerate(d.get("top", [])):
        res = build_algo(p, _CUDA_DTYPE[dtype], t["cfg"])
        if res is not None:
            out.append(CublasLtMatmul(p, res, rank, family="cublaslt_deep", prefix="d",
                                      extra={"screen_us": round(t["us"], 3), **stats}))
    return out


def candidates(M: int, N: int, K: int, dtype=torch.float16, n: int = 16) -> list[CublasLtMatmul]:
    """Top-n heuristic candidates for this shape (rank 0 = B2). Fewer may come back."""
    p = _Problem(M, N, K, dtype)
    return [CublasLtMatmul(p, r, i) for i, r in enumerate(p.heuristics(n))]
