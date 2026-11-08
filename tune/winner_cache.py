"""Per-shape winner cache: remember which kernel the online search picked, so a shape is searched once per machine.

`tune.online.select_kernel` ranks, compiles, validates and short-times k Triton configs plus cuBLASLt's heuristic top-N
(~0.6 s). The winner is a measurement, so it stays valid for as long as the things that determine it stay the same; a
repeat call to the same shape can skip all of it. This module is that memory: an append-only JSONL file, one line per
searched shape (the same crash-safe style as `tune/records.py`: one fsync'd line per write).

Key = shape + dtype; each line also carries the environment it was timed in (`environment()`): GPU id, Triton, torch,
CUDA runtime, cuBLASLt build, driver, VBIOS, a hash of the Triton kernel's source, the declared clock-lock state, and,
when the caller passes them, the ids of the models that proposed the candidates. Comparing a line's environment with
the current one sorts it into one of three tiers (schema 2):

  * current: every field the current environment declares matches. Served.
  * soft-stale: only software / clock / model fields differ (driver, torch, Triton, CUDA, cuBLASLt, VBIOS, clock lock,
    models), or the line predates a field (schema 1 lines have only GPU, Triton, cuBLASLt and driver). The kernel it
    names still exists and is still a correct GEMM; it is just no longer known to be the fastest. `get` returns it only
    with `allow_stale=True` (the lscli daemon: keep serving it until an idle-time re-search replaces it); the caller then
    runs the correctness gate once in the new environment and records that with `mark_verified`.
  * hard-stale: another GPU (`hw_id`), or for a Triton winner a different kernel source (`kernel_src`: the same key now
    names different code). Never returned. A line that does not carry the field (older schema) is soft-stale for it.

A field the current environment leaves as None (e.g. no declared clock lock, no model ids) is not compared. Among the
lines of one tier the last one wins. An entry is reused when it came from a search at least as wide as the one asked for
(stored k >= k, lt_n >= lt_n, lt_knn >= lt_knn, lt_rank >= lt_rank): a shape first seen with k = 3 is searched again,
once, when a caller later marks it frequent (k = 5), and the wider result replaces it. `select_kernel(..., cache=None)`
turns the cache off (experiments: they measure the search itself).

A cuBLASLt winner is stored as its key (`h<rank>_a<algo>_t<tile>_...`); `select_kernel` rebuilds the candidate and treats a
key that no longer exists as a miss (`backend_for` returns None), so a stale entry costs one re-search, never a wrong
kernel. `backend_for(sel["choice"], M, N, K)` turns a choice (cached or fresh) into a runnable backend.

The file only grows until `compact()` rewrites it (atomically, under the same lock the writers take): for each shape and
GPU, a line that a later current-environment search re-searched is dropped, otherwise the last line per distinct
environment is kept, so soft-stale entries stay until they are re-searched. Lines of other GPUs are kept (the card may
come back).

LIMITATIONS: entries are per machine (nothing is shared across GPUs); the cache never expires by age; a winner is the
fastest of a short online timing (10 cold samples, one placement), so its ranking against near-ties is only as good as
`choose`'s (no tie margin). Concurrent writers append whole lines under a file lock; a reader only sees entries from
before it loaded the file (`reload()` to refresh).
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
import uuid
from pathlib import Path

ENV_PATH = "TUNE_WINNER_CACHE"     # file path, or "off" to disable the default cache
SCHEMA = 2
READS = (1, 2)                     # schema 1 lines (4-field environment) are read as soft-stale
HARD_FIELDS = ("hw_id",)           # another GPU: the measurement says nothing here
TRITON_HARD_FIELDS = ("kernel_src",)  # the Triton kernel's code changed: a Triton key now names a different kernel
_RESERVED = ("stale", "verified")  # annotations `get` adds to the rows it returns; never stored


def kernel_src_hash() -> str:
    """sha256 (16 hex) of `kernels/triton_matmul.py`: any edit changes what a Triton config key compiles to."""
    src = Path(__file__).resolve().parent.parent / "kernels" / "triton_matmul.py"
    return hashlib.sha256(src.read_bytes()).hexdigest()[:16]


def clock_lock_declared(value: str | None = None) -> str | None:
    """The declared SM clock policy (`TUNE_CLOCK_LOCK` / `--clock-lock`) normalized: "unlocked", a MHz string, or None
    when nothing was declared (then the clock is not part of the comparison)."""
    v = value if value is not None else os.environ.get("TUNE_CLOCK_LOCK")
    if v is None or not str(v).strip():
        return None
    v = str(v).strip().lower()
    return "unlocked" if v in ("none", "off", "0", "unlocked") else v


def environment(*, models: dict | None = None, clock_lock: str | None = None) -> dict:
    """What a timed winner depends on besides the shape. `models` (ids of the rankers / router that proposed the
    candidates) and `clock_lock` (a policy to declare instead of `TUNE_CLOCK_LOCK`) come from the caller."""
    import torch
    import triton
    from baselines.cublaslt import library_info
    from tune.env import _nvml_handle, _safe, hw_id
    import pynvml
    h = _nvml_handle()
    driver = _safe(pynvml.nvmlSystemGetDriverVersion) if h else None
    vbios = _safe(pynvml.nvmlDeviceGetVbiosVersion, h) if h else None
    env = {"hw_id": hw_id(), "triton": triton.__version__, "cublaslt": library_info()["version"], "driver": driver,
           "torch": torch.__version__, "cuda_runtime": torch.version.cuda, "vbios": vbios,
           "kernel_src": kernel_src_hash(), "clock_lock": clock_lock_declared(clock_lock)}
    if models is not None:
        env["models"] = models
    return env


def stale_fields(row_env: dict, env: dict) -> list[str]:
    """Fields the current environment declares (not None) that the stored line does not match."""
    return [f for f, v in env.items() if v is not None and row_env.get(f) != v]


def tier(row: dict, env: dict) -> tuple[str, list[str]]:
    """("current" | "soft" | "hard", differing fields) of a stored line against the environment `env`."""
    row_env = row.get("env") or {}
    fields = stale_fields(row_env, env)
    family = (row.get("choice") or {}).get("family", "triton")
    # A field the line does not carry (written before it existed) is unknown, not changed: soft, never hard.
    if any(f in row_env and (f in HARD_FIELDS or (f in TRITON_HARD_FIELDS and family == "triton")) for f in fields):
        return "hard", fields
    return ("soft" if fields else "current"), fields


def _row_id(r: dict, line: str) -> str:
    return r.get("id") or hashlib.sha1(line.encode()).hexdigest()[:16]


def default_path() -> Path:
    from tune.env import default_data_dir
    return Path(os.environ.get(ENV_PATH) or default_data_dir() / "winner_cache.jsonl")


class WinnerCache:
    def __init__(self, path: str | os.PathLike | None = None, env: dict | None = None):
        self.path = Path(path) if path else default_path()
        self.env = env if env is not None else environment()
        self._rows: dict[tuple, dict[str, dict]] = {}     # key -> {"current": row, "soft": row} (last line of each tier)
        self._verified: set[str] = set()                   # entry ids verified in exactly this environment
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

    @staticmethod
    def _parse(lines: list[str]):
        """(row, line) for every readable line of a known schema; a torn last line from a crash is skipped."""
        for line in lines:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if isinstance(r, dict) and r.get("schema") in READS:
                yield r, line

    def reload(self) -> None:
        """(Re)read the file; hard-stale lines (another GPU, changed kernel source) and unreadable lines are skipped."""
        self._rows, self._verified = {}, set()
        try:
            lines = self.path.read_text().splitlines()
        except OSError:
            return
        for r, line in self._parse(lines):
            if r.get("kind") == "verified":
                if r.get("env") == self.env:
                    self._verified.add(r.get("id"))
                continue
            t, fields = tier(r, self.env)
            if t == "hard":
                continue
            r = {k: v for k, v in r.items() if k not in _RESERVED}
            r["id"] = _row_id(r, line)
            r["_stale"] = fields
            self._rows.setdefault(self._key(r["M"], r["N"], r["K"], r["dtype"]), {})[t] = r

    def __len__(self) -> int:
        """Shapes with a current entry."""
        return sum(1 for v in self._rows.values() if "current" in v)

    @staticmethod
    def _wide_enough(r: dict | None, k: int, lt_n: int, lt_knn: int, lt_rank: int) -> bool:
        return (r is not None and r["k"] >= k and r["lt_n"] >= lt_n and r.get("lt_knn", 0) >= lt_knn
                and r.get("lt_rank", 0) >= lt_rank)

    def _out(self, r: dict) -> dict:
        out = {k: v for k, v in r.items() if k != "_stale"}
        out["stale"] = list(r["_stale"])
        out["verified"] = not r["_stale"] or r["id"] in self._verified
        return out

    def get(self, M: int, N: int, K: int, dtype: str = "fp16", *, k: int = 1, lt_n: int = 0, lt_knn: int = 0,
            lt_rank: int = 0, allow_stale: bool = False) -> dict | None:
        """The stored entry if its search was at least as wide as (k, lt_n, lt_knn, lt_rank), else None (entries written
        before the kNN proposals / the cuBLASLt ranker existed have lt_knn / lt_rank 0). A current entry is preferred; with
        allow_stale a soft-stale one is returned when no current one is wide enough. The returned copy carries `stale` (the
        differing fields, [] when current) and `verified` (True when current, or when `mark_verified` ran for it in this
        environment)."""
        tiers = self._rows.get(self._key(M, N, K, dtype), {})
        for t in ("current", "soft") if allow_stale else ("current",):
            r = tiers.get(t)
            if self._wide_enough(r, k, lt_n, lt_knn, lt_rank):
                return self._out(r)
        return None

    def stale_entries(self) -> list[dict]:
        """The soft-stale entries of shapes with no current entry: what an idle-time re-search should redo."""
        return [self._out(v["soft"]) for v in self._rows.values() if "soft" in v and "current" not in v]

    def _append(self, row: dict) -> None:
        """Append one fsync'd line under the file lock. If `compact` replaced the file while this writer waited for the
        lock, the handle points at the old (unlinked) file: reopen and write to the new one."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(row) + "\n"
        for _ in range(10):
            with open(self.path, "a") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    if os.fstat(f.fileno()).st_ino != os.stat(self.path).st_ino:
                        continue
                except FileNotFoundError:
                    continue
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
                return
        raise OSError(f"{self.path}: kept being replaced while appending")

    def put(self, M: int, N: int, K: int, dtype: str, *, k: int, lt_n: int, choice: dict, candidates: list[dict] | None = None,
            order: str | None = None, ranker: str | None = None, lt_knn: int = 0, lt_rank: int = 0) -> dict | None:
        """Store a search result's winner in the current environment. Searches that found no valid kernel are not stored.
        Returns the entry."""
        if not choice or choice.get("key") is None:
            return None
        win = next((c for c in (candidates or []) if c.get("config_key") == choice["key"]), {})
        row = {"schema": SCHEMA, "id": uuid.uuid4().hex[:16], "env": self.env, "M": int(M), "N": int(N), "K": int(K),
               "dtype": dtype, "k": int(k), "lt_n": int(lt_n), "lt_knn": int(lt_knn), "lt_rank": int(lt_rank),
               "order": order, "ranker": ranker,
               "choice": {x: choice.get(x) for x in ("key", "family", "mode", "rank", "compile_s", "time_s")},
               "winner_us": win.get("graph_us") if choice.get("mode") == "graph" else win.get("cold_us"),
               "timestamp": time.time()}
        self._append(row)
        self._rows.setdefault(self._key(M, N, K, dtype), {})["current"] = {**row, "_stale": []}
        return row

    def mark_verified(self, entry: dict) -> None:
        """Record that a (soft-stale) entry passed the correctness gate in the current environment, so it is not
        re-checked by this or a later process with the same environment."""
        if entry.get("verified") and not entry.get("stale"):
            return
        self._append({"schema": SCHEMA, "kind": "verified", "id": entry["id"], "env": self.env, "timestamp": time.time()})
        self._verified.add(entry["id"])

    def compact(self) -> dict:
        """Rewrite the file without the lines a re-search made obsolete; returns {"before", "after"} line counts.

        Per (shape, dtype, GPU): when a line is current in this environment, only the last current line stays (every
        older line of that shape and GPU, stale or not, was re-searched); otherwise the last line per distinct
        environment stays (soft-stale entries are kept until re-searched). Verification lines stay while their entry
        does. Unreadable lines are dropped. Atomic (temp file + rename) under the writers' lock."""
        if not self.path.exists():
            return {"before": 0, "after": 0}
        with open(self.path, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            lines = f.read().splitlines()
            entries, verified = [], []
            for r, line in self._parse(lines):
                (verified if r.get("kind") == "verified" else entries).append((r, line))
            groups: dict[tuple, list] = {}
            for i, (r, line) in enumerate(entries):
                groups.setdefault((r["M"], r["N"], r["K"], r["dtype"], (r.get("env") or {}).get("hw_id")), []).append(i)
            keep: set[int] = set()
            for idx in groups.values():
                cur = [i for i in idx if tier(entries[i][0], self.env)[0] == "current"]
                if cur:
                    keep.add(cur[-1])
                    continue
                last: dict[str, int] = {}
                for i in idx:
                    last[json.dumps(entries[i][0].get("env"), sort_keys=True)] = i
                keep.update(last.values())
            kept_ids = {_row_id(entries[i][0], entries[i][1]) for i in keep}
            out = [entries[i][1] for i in sorted(keep)] + [line for r, line in verified if r.get("id") in kept_ids]
            tmp = self.path.with_name(self.path.name + f".tmp{os.getpid()}")
            with open(tmp, "w") as g:
                g.write("".join(x + "\n" for x in out))
                g.flush()
                os.fsync(g.fileno())
            os.replace(tmp, self.path)
        self.reload()
        return {"before": len(lines), "after": len(out)}


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
