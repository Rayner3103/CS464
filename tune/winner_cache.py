"""Per-shape winner cache: remember which kernel the online search picked, so a shape is searched once per machine.

`tune.online.select_kernel` ranks, compiles, validates and short-times k Triton configs plus cuBLASLt's heuristic top-N
(~0.6 s). The winner is a measurement, so it stays valid for as long as the things that determine it stay the same; a
repeat call to the same shape can skip all of it. This module is that memory: an append-only JSONL file, one line per
searched shape, last line wins (the same crash-safe style as `tune/records.py`: one fsync'd line per write).

Key = machine + software + shape. The machine / software part (`environment()`) is the GPU id, the Triton version, the
cuBLASLt library version and the driver: a winner timed under another cuBLASLt build or driver may not be the winner
here, so such lines are ignored (kept in the file, never deleted). The ranker is NOT part of the key: a timed winner is
a winner whichever ranker proposed it. It is recorded for provenance.

An entry is reused when it came from a search at least as wide as the one asked for (stored k >= k and stored lt_n >=
lt_n): a shape first seen with k = 3 is searched again, once, when a caller later marks it frequent (k = 5), and the
wider result replaces it. `select_kernel(..., cache=None)` turns the cache off (experiments, `demo.py`: they measure the
search itself).

A cuBLASLt winner is stored as its heuristic key (`h<rank>_a<algo>_t<tile>_...`); `select_kernel` rebuilds the candidate list
and treats a key that no longer exists as a miss (`backend_for` returns None), so a stale entry costs one re-search, never a
wrong kernel. `backend_for(sel["choice"], M, N, K)` turns a choice (cached or fresh) into a runnable backend.

LIMITATIONS: entries are per machine and per software environment (nothing is shared across GPUs); the cache never
expires on its own (delete the file, or change environment, to force a re-search); a winner is the fastest of a short online
timing (10 cold samples, one placement), so its ranking against near-ties is only as good as `choose`'s (no tie margin).
Concurrent writers append whole lines under a file lock; a reader only sees entries from before it loaded the file
(`reload()` to refresh).
"""
from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path

ENV_PATH = "TUNE_WINNER_CACHE"     # file path, or "off" to disable the default cache
SCHEMA = 1


def environment() -> dict:
    """What a timed winner depends on besides the shape: GPU, Triton, cuBLASLt build, driver."""
    import triton
    from baselines.cublaslt import library_info
    from tune.env import _nvml_handle, _safe, hw_id
    import pynvml
    h = _nvml_handle()
    driver = _safe(pynvml.nvmlSystemGetDriverVersion) if h else None
    return {"hw_id": hw_id(), "triton": triton.__version__, "cublaslt": library_info()["version"], "driver": driver}


def default_path() -> Path:
    from tune.env import default_data_dir
    return Path(os.environ.get(ENV_PATH) or default_data_dir() / "winner_cache.jsonl")


class WinnerCache:
    def __init__(self, path: str | os.PathLike | None = None, env: dict | None = None):
        self.path = Path(path) if path else default_path()
        self.env = env if env is not None else environment()
        self._rows: dict[tuple, dict] = {}
        self.reload()

    @classmethod
    def default(cls) -> "WinnerCache | None":
        """The cache at its standard place, or None when `TUNE_WINNER_CACHE=off`."""
        if os.environ.get(ENV_PATH, "").lower() == "off":
            return None
        return cls()

    @staticmethod
    def _key(M: int, N: int, K: int, dtype: str) -> tuple:
        return (int(M), int(N), int(K), dtype)

    def reload(self) -> None:
        """(Re)read the file; lines from another environment or an unreadable line are skipped."""
        self._rows = {}
        try:
            lines = self.path.read_text().splitlines()
        except OSError:
            return
        for line in lines:
            try:
                r = json.loads(line)
            except ValueError:
                continue                      # a torn last line from a crash
            if r.get("schema") != SCHEMA or r.get("env") != self.env:
                continue
            self._rows[self._key(r["M"], r["N"], r["K"], r["dtype"])] = r

    def __len__(self) -> int:
        return len(self._rows)

    def get(self, M: int, N: int, K: int, dtype: str = "fp16", *, k: int = 1, lt_n: int = 0, lt_knn: int = 0,
            lt_rank: int = 0) -> dict | None:
        """The stored entry if its search was at least as wide as (k, lt_n, lt_knn), else None (entries written before the
        kNN proposals existed have lt_knn 0)."""
        r = self._rows.get(self._key(M, N, K, dtype))
        if r is None or r["k"] < k or r["lt_n"] < lt_n or r.get("lt_knn", 0) < lt_knn or r.get("lt_rank", 0) < lt_rank:
            return None
        return r

    def put(self, M: int, N: int, K: int, dtype: str, *, k: int, lt_n: int, choice: dict, candidates: list[dict] | None = None,
            order: str | None = None, ranker: str | None = None, lt_knn: int = 0, lt_rank: int = 0) -> dict | None:
        """Store a search result's winner. Searches that found no valid kernel are not stored. Returns the entry."""
        if not choice or choice.get("key") is None:
            return None
        win = next((c for c in (candidates or []) if c.get("config_key") == choice["key"]), {})
        row = {"schema": SCHEMA, "env": self.env, "M": int(M), "N": int(N), "K": int(K), "dtype": dtype, "k": int(k),
               "lt_n": int(lt_n), "lt_knn": int(lt_knn), "lt_rank": int(lt_rank), "order": order, "ranker": ranker,
               "choice": {x: choice.get(x) for x in ("key", "family", "mode", "rank", "compile_s", "time_s")},
               "winner_us": win.get("graph_us") if choice.get("mode") == "graph" else win.get("cold_us"),
               "timestamp": time.time()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps(row) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._rows[self._key(M, N, K, dtype)] = row
        return row


def backend_for(choice: dict, M: int, N: int, K: int, dtype=None, lt_n: int = 16):
    """The runnable backend a choice names (`.make_runner(a, b)`), or None when it no longer exists (a cuBLASLt key the
    library does not return any more): the caller then searches again. `choice` is `select_kernel(...)["choice"]` or a
    stored entry's `["choice"]`; `lt_n` bounds the cuBLASLt heuristic list rebuilt to find the named candidate."""
    import torch
    dtype = dtype or torch.float16
    if choice.get("key") is None:
        return None
    if choice.get("family", "triton") == "triton":
        from kernels.triton_matmul import TritonMatmul
        from tune.sweep import parse_key
        return TritonMatmul(parse_key(choice["key"]))
    if choice.get("family") in ("cublaslt_knn", "cublaslt_rank"):
        from baselines.cublaslt_knn import from_key
        return from_key(choice["key"], M, N, K, dtype)
    from baselines.cublaslt import candidates as lt_candidates
    return next((c for c in lt_candidates(M, N, K, dtype, max(lt_n, 1)) if c.key == choice["key"]), None)
