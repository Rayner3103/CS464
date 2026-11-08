"""The usage record: what the user's programs run, and how often (spec 3.2-3.3).

Clients count calls in memory and flush increments to their own `profile/<pid>-<t0>.jsonl` (no lock contention between
processes). The daemon merges them into `profile.json`, where each shape's count decays with a half-life
(`search.half_life_days`), so priorities follow current habits: a shape run 2M times last month and not since weighs as
much as one run 1M times today (half-life 30 days). Merging remembers how far it read each log, so a log that is still
being written is read incrementally, and deletes the logs of processes that have exited once read to the end.
"""
from __future__ import annotations

import os
import time
from collections import Counter
from pathlib import Path

from lscli.files import append_jsonl, read_json, read_jsonl, write_json

LAYOUTS = ("NN", "NT")
DAY = 86400.0


def _key(r: dict) -> tuple:
    return (int(r["M"]), int(r["N"]), int(r["K"]), r["dtype"], r.get("layout", "NN"))


class UsageLog:
    """Per-process counter (client side). `add` is the hot path: one Counter update, no I/O."""

    def __init__(self, paths, env_id: str):
        self.path = Path(paths.profile_dir) / f"{os.getpid()}-{int(time.time())}.jsonl"
        self.env_id = env_id
        self.calls: Counter = Counter()
        self.hits: Counter = Counter()
        self.fallbacks: Counter = Counter()

    def add(self, M: int, N: int, K: int, dtype: str, layout: str = "NN", hit: bool | None = None) -> None:
        k = (M, N, K, dtype, layout)
        self.calls[k] += 1
        if hit is True:
            self.hits[k] += 1
        elif hit is False:
            self.fallbacks[k] += 1

    def flush(self) -> int:
        """Write the increments since the last flush; returns the number of shapes written."""
        if not self.calls:
            return 0
        t = time.time()
        rows = [{"t": t, "env_id": self.env_id, "M": k[0], "N": k[1], "K": k[2], "dtype": k[3], "layout": k[4],
                 "calls": n, "hits": self.hits.get(k, 0), "fallbacks": self.fallbacks.get(k, 0)}
                for k, n in self.calls.items()]
        append_jsonl(self.path, rows)
        self.calls.clear()
        self.hits.clear()
        self.fallbacks.clear()
        return len(rows)


def _pid_alive(name: str) -> bool:
    try:
        pid = int(name.split("-", 1)[0])
        os.kill(pid, 0)
        return True
    except (ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True


def decay(calls: float, since: float, now: float, half_life_days: float) -> float:
    return calls * 0.5 ** (max(0.0, now - since) / (half_life_days * DAY)) if half_life_days > 0 else calls


def merge(paths, half_life_days: float, now: float | None = None, write: bool = True) -> dict:
    """Fold every unread log line into the profile and return it. write=False computes it without touching any file
    (`lscli profile` while the daemon is not running)."""
    now = time.time() if now is None else now
    prof = read_json(paths.profile, None) or {"updated": now, "half_life_days": half_life_days, "shapes": [], "offsets": {}}
    shapes = {_key(s): dict(s) for s in prof["shapes"]}
    for s in shapes.values():
        s["calls"] = decay(s["calls"], prof["updated"], now, half_life_days)
    offsets = dict(prof.get("offsets", {}))
    done = []
    for log in sorted(Path(paths.profile_dir).glob("*.jsonl")):
        rows, end = read_jsonl(log, offsets.get(log.name, 0))
        for r in rows:
            try:
                k = _key(r)
            except (KeyError, ValueError, TypeError):
                continue
            s = shapes.setdefault(k, {"M": k[0], "N": k[1], "K": k[2], "dtype": k[3], "layout": k[4], "calls": 0.0,
                                      "calls_total": 0, "hits": 0, "fallbacks": 0, "first_seen": r.get("t", now),
                                      "envs": []})
            n = int(r.get("calls", 0))
            s["calls"] += decay(n, r.get("t", now), now, half_life_days)
            s["calls_total"] += n
            s["hits"] = s.get("hits", 0) + int(r.get("hits", 0))
            s["fallbacks"] = s.get("fallbacks", 0) + int(r.get("fallbacks", 0))
            s["last_seen"] = max(s.get("last_seen", 0), r.get("t", now))
            if r.get("env_id") and r["env_id"] not in s.setdefault("envs", []):
                s["envs"].append(r["env_id"])
        offsets[log.name] = end
        if not _pid_alive(log.name) and end >= log.stat().st_size:
            done.append(log)
    out = {"updated": now, "half_life_days": half_life_days, "offsets": offsets,
           "shapes": sorted(shapes.values(), key=lambda s: -s["calls"])}
    if write:
        write_json(paths.profile, {**out, "offsets": {k: v for k, v in offsets.items() if k not in {d.name for d in done}}})
        for d in done:
            d.unlink(missing_ok=True)
    return out


def load(paths) -> dict:
    return read_json(paths.profile, None) or {"updated": time.time(), "half_life_days": 30, "shapes": [], "offsets": {}}


def calls_now(prof: dict, shape: dict, now: float | None = None) -> float:
    """A profile entry's decayed count as of now (the stored count is as of `updated`)."""
    return decay(shape["calls"], prof["updated"], time.time() if now is None else now, prof.get("half_life_days", 30))


def lookup(prof: dict) -> dict[tuple, dict]:
    return {_key(s): s for s in prof.get("shapes", [])}
