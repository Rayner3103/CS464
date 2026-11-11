"""Per-shape winner cache: remember which kernel the online search picked, so a shape is searched once per machine.

`tune.online.select_kernel` ranks, compiles, validates and short-times k Triton configs plus cuBLASLt's heuristic top-N
(~0.6 s). The winner is a measurement, so it stays valid for as long as the things that determine it stay the same; a
repeat call to the same shape can skip all of it. This module is that memory: an append-only JSONL file, one line per
searched shape (the same crash-safe style as `tune/records.py`: one fsync'd line per write).

Key = shape + dtype; each line also carries the environment it was timed in (`environment()`): GPU id, Triton, torch,
CUDA runtime, cuBLASLt build, driver, VBIOS, a hash of the Triton kernel's source, the declared clock-lock state, and,
when the caller passes them, the ids of the models that proposed the candidates (`model_id`; `select_kernel`'s default
cache passes the ids of the Triton ranker it was given and of this GPU's cuBLASLt ranker and router files, so a
retrained model makes the old winners soft-stale). Comparing a line's environment with
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

The file only grows until `compact()` rewrites it (atomically, under the same lock the writers take). What it may drop
is decided by who else can still be reading the line:

  * always: a line that a later line of the SAME environment (every field equal) replaced. That search superseded it
    for every reader, whatever its own environment: a reader keeps only the last line per tier anyway.
  * only inside the compactor's own `env_id` scope: a line that is stale here, was written before this environment's
    current line for the shape, and whose search was not wider than it. `env_id` is an optional field of the
    environment naming the one interpreter the entry belongs to (lscli sets it per registered Python environment;
    `environment(env_id=...)`). Inside a scope the environment only changes over time (an upgrade), so a re-searched
    line has no reader left. Without an `env_id`, several interpreters may share the file (two virtualenvs with
    different torch builds on one GPU): a line that is stale for the compactor may be another one's current entry, so
    nothing but exact duplicates is dropped and the file keeps one line per shape per environment it has seen.

Lines of other GPUs and of other `env_id`s are never dropped (the card may come back; the other interpreter compacts its
own scope).

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
PROPOSAL_FAMILIES = ("cublaslt_knn", "cublaslt_rank")   # cuBLASLt candidates rebuilt from their key alone (`backend_for`)


def kernel_src_hash() -> str:
    """sha256 (16 hex) of `kernels/triton_matmul.py`: any edit changes what a Triton config key compiles to."""
    src = Path(__file__).resolve().parent.parent / "kernels" / "triton_matmul.py"
    return hashlib.sha256(src.read_bytes()).hexdigest()[:16]


_MODEL_IDS: dict[tuple, str] = {}


def model_id(stem: str | os.PathLike | None) -> str | None:
    """Identity of a saved model (`<stem>.txt` + `<stem>.json`: the Triton ranker, the cuBLASLt ranker, the router):
    `<stem name>:<12 hex of the sha256 of both files>`, the form lscli uses for its model sets. None without a stem or
    when a file is missing. A retrained model gets another id under the same file name. Hashed once per process and file
    state (path, size, mtime): the Triton ranker is ~36 MB."""
    if not stem:
        return None
    files = [Path(str(stem) + ext) for ext in (".txt", ".json")]
    try:
        stats = [f.stat() for f in files]
    except OSError:
        return None
    key = tuple((str(f), st.st_size, st.st_mtime_ns) for f, st in zip(files, stats))
    if key not in _MODEL_IDS:
        h = hashlib.sha256()
        for f in files:
            h.update(f.read_bytes())
        _MODEL_IDS[key] = f"{Path(stem).name}:{h.hexdigest()[:12]}"
    return _MODEL_IDS[key]


def clock_lock_declared(value: str | None = None) -> str | None:
    """The declared SM clock policy (`TUNE_CLOCK_LOCK` / `--clock-lock`) normalized: "unlocked", a MHz string, or None
    when nothing was declared (then the clock is not part of the comparison)."""
    v = value if value is not None else os.environ.get("TUNE_CLOCK_LOCK")
    if v is None or not str(v).strip():
        return None
    v = str(v).strip().lower()
    return "unlocked" if v in ("none", "off", "0", "unlocked") else v


def environment(*, models: dict | None = None, clock_lock: str | None = None, env_id: str | None = None) -> dict:
    """What a timed winner depends on besides the shape. `models` (ids of the rankers / router that proposed the
    candidates), `clock_lock` (a policy to declare instead of `TUNE_CLOCK_LOCK`) and `env_id` (a name for this
    interpreter when several share the file; it scopes `compact`) come from the caller."""
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
    if env_id is not None:
        env["env_id"] = env_id
    return env


def stale_fields(row_env: dict, env: dict) -> list[str]:
    """Fields the current environment declares (not None) that the stored line does not match."""
    return [f for f, v in env.items() if v is not None and row_env.get(f) != v]


def tier(row: dict, env: dict) -> tuple[str, list[str]]:
    """("current" | "soft" | "hard", differing fields) of a stored line against the environment `env`."""
    row_env = row.get("env") or {}
    fields = stale_fields(row_env, env)
    family = (row.get("choice") or {}).get("family") or "triton"
    # A field the line does not carry (written before it existed) is unknown, not changed: soft, never hard.
    if any(f in row_env and (f in HARD_FIELDS or (f in TRITON_HARD_FIELDS and family == "triton")) for f in fields):
        return "hard", fields
    return ("soft" if fields else "current"), fields


WIDTH = ("k", "lt_n", "lt_knn", "lt_rank")


def _wider(a: dict, b: dict) -> bool:
    """Entry `a` came from a search wider than `b`'s in some dimension: `get` can serve `a` where `b` is a miss."""
    return any(a.get(w, 0) > b.get(w, 0) for w in WIDTH)


def _row_id(r: dict, line: str) -> str:
    return r.get("id") or hashlib.sha1(line.encode()).hexdigest()[:16]


def default_enabled() -> bool:
    """False when `TUNE_WINNER_CACHE=off`."""
    return os.environ.get(ENV_PATH, "").lower() != "off"


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
    def default(cls, models: dict | None = None) -> "WinnerCache | None":
        """The cache at its standard place, or None when `TUNE_WINNER_CACHE=off`. `models` = the model ids to declare
        in its environment (`environment(models=...)`)."""
        if not default_enabled():
            return None
        return cls(env=environment(models=models))

    @staticmethod
    def _key(M: int, N: int, K: int, dtype: str) -> tuple:
        return (int(M), int(N), int(K), dtype)

    @staticmethod
    def _parse(lines: list[str]):
        """(row, line) for every readable line of a known schema; a torn last line from a crash, or a line whose key
        fields are missing or of the wrong type (a damaged file), is skipped."""
        for line in lines:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if not isinstance(r, dict) or r.get("schema") not in READS:
                continue
            if r.get("kind") == "verified" or (all(isinstance(r.get(x), int) for x in ("M", "N", "K", "k", "lt_n"))
                                               and isinstance(r.get("dtype"), str)):
                yield r, line                 # an entry without usable key fields is skipped like a torn line

    def reload(self) -> None:
        """(Re)read the file; hard-stale lines (another GPU, changed kernel source) and unreadable lines are skipped."""
        self._rows, self._verified = {}, set()
        try:
            lines = self.path.read_text(errors="replace").splitlines()   # damaged bytes spoil their line, not the file
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
        allow_stale a soft-stale one is returned when no current one is wide enough (so a wider stale search can be served
        where `allow_stale=False` would re-search; the caller queues the re-search). The returned copy carries `stale` (the
        differing fields, [] when current) and `verified` (True when current, or when `mark_verified` ran for it in this
        environment)."""
        tiers = self._rows.get(self._key(M, N, K, dtype), {})
        for t in ("current", "soft") if allow_stale else ("current",):
            r = tiers.get(t)
            if self._wide_enough(r, k, lt_n, lt_knn, lt_rank):
                return self._out(r)
        return None

    def stale_entries(self) -> list[dict]:
        """The soft-stale entries `get(allow_stale=True)` can still return: those of shapes with no current entry, and
        those whose search was wider than the shape's current entry (served for the wider request). What an idle-time
        re-search should redo; a list that left the second kind out would let them be served for ever."""
        return [self._out(v["soft"]) for v in self._rows.values()
                if "soft" in v and ("current" not in v or _wider(v["soft"], v["current"]))]

    def _locked(self, mode: str):
        """Open the file and take the exclusive lock, re-opening when `compact` replaced it while this process waited
        (the handle would point at the old, unlinked file). Returns the open, locked file; closing it releases the lock."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(20):
            f = open(self.path, mode, **({} if "b" in mode else {"errors": "replace"}))
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                if os.fstat(f.fileno()).st_ino == os.stat(self.path).st_ino:
                    return f
            except FileNotFoundError:
                pass
            f.close()
        raise OSError(f"{self.path}: kept being replaced while waiting for its lock")

    def _append(self, row: dict) -> None:
        """Append one fsync'd line under the file lock. A torn last line (a crash mid-write) is terminated first, so the
        new line is not glued onto it."""
        with self._locked("a+b") as f:
            f.seek(0, os.SEEK_END)
            torn = False
            if f.tell() > 0:
                f.seek(-1, os.SEEK_END)
                torn = f.read(1) != b"\n"
            f.write((b"\n" if torn else b"") + (json.dumps(row) + "\n").encode())
            f.flush()
            os.fsync(f.fileno())

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
        if not entry.get("stale") or entry["id"] in self._verified:
            return
        self._append({"schema": SCHEMA, "kind": "verified", "id": entry["id"], "env": self.env, "timestamp": time.time()})
        self._verified.add(entry["id"])

    def compact(self) -> dict:
        """Rewrite the file without the lines no reader can be served any more; returns {"before", "after"} line counts.

        Per (shape, dtype, GPU, env_id): the last line per distinct environment is a candidate, every earlier line of the
        same environment is dropped. The candidates all stay, except in this cache's own scope (the group's `env_id` is
        this environment's, and not None): there a candidate that is not current here is dropped when a current line was
        written after it and its search was not wider in some dimension (k, lt_n, lt_knn, lt_rank; `get(allow_stale=True)`
        can still serve a wider one). A line written after the current one stays even when it is stale here: it comes
        from a newer state of the interpreter than the environment this compactor was built with. Verification lines stay
        while their entry does. Unreadable lines are dropped. Atomic (temp file + rename) under the writers' lock; a
        compactor that waited for the lock while another one replaced the file re-opens the new file."""
        if not self.path.exists():
            return {"before": 0, "after": 0}
        with self._locked("a+") as f:
            f.seek(0)
            lines = f.read().splitlines()
            entries, verified = [], []
            for r, line in self._parse(lines):
                (verified if r.get("kind") == "verified" else entries).append((r, line))
            groups: dict[tuple, list] = {}
            for i, (r, line) in enumerate(entries):
                e = r.get("env") or {}
                groups.setdefault((r["M"], r["N"], r["K"], r["dtype"], e.get("hw_id"), e.get("env_id")), []).append(i)
            scope = self.env.get("env_id")
            keep: set[int] = set()
            for gkey, idx in groups.items():
                last: dict[str, int] = {}
                for i in idx:
                    last[json.dumps(entries[i][0].get("env"), sort_keys=True)] = i
                cand = sorted(last.values())
                current = {i for i in cand if tier(entries[i][0], self.env)[0] == "current"}
                if scope is None or gkey[-1] != scope or not current:
                    keep.update(cand)
                    continue
                newest = max(current)                          # the line `get` serves here
                keep.update(i for i in cand if i in current or i > newest or _wider(entries[i][0], entries[newest][0]))
            kept_ids = {_row_id(entries[i][0], entries[i][1]) for i in keep}
            out = [entries[i][1] for i in sorted(keep)] + [line for r, line in verified if r.get("id") in kept_ids]
            tmp = self.path.with_name(f"{self.path.name}.tmp-{uuid.uuid4().hex[:12]}")
            with open(tmp, "w") as g:
                g.write("".join(x + "\n" for x in out))
                g.flush()
                os.fsync(g.fileno())
            os.replace(tmp, self.path)
            d = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(d)                          # make the rename itself durable
            finally:
                os.close(d)
        self.reload()
        return {"before": len(lines), "after": len(out)}


def backend_for(choice: dict, M: int, N: int, K: int, dtype=None, lt_n: int = 16, heuristics: list | None = None):
    """The runnable backend a choice names (`.make_runner(a, b)`), or None when it no longer exists (a cuBLASLt key the
    library does not return any more): the caller then searches again. `choice` is `select_kernel(...)["choice"]` or a
    stored entry's `["choice"]`; `lt_n` bounds the cuBLASLt heuristic list rebuilt to find the named candidate, unless the
    caller passes that list (`heuristics`, e.g. `tune.online.backends_from_keys`, which rebuilds many keys of one shape).
    A malformed Triton key raises ValueError (`tune.sweep.parse_key`)."""
    import torch
    dtype = dtype or torch.float16
    if choice.get("key") is None:
        return None
    if (choice.get("family") or "triton") == "triton":
        from kernels.triton_matmul import TritonMatmul
        from tune.sweep import parse_key
        return TritonMatmul(parse_key(choice["key"]))
    if choice.get("family") in PROPOSAL_FAMILIES:
        from baselines.cublaslt_knn import from_key
        return from_key(choice["key"], M, N, K, dtype)
    if heuristics is None:
        from baselines.cublaslt import candidates as lt_candidates
        heuristics = lt_candidates(M, N, K, dtype, max(lt_n, 1))
    return next((c for c in heuristics if c.key == choice["key"]), None)
