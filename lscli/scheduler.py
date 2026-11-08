"""What to do next with an open gate (spec 4.3): one ordered list of tasks, rebuilt every tick from the usage record,
the queue and each registered env's view of the winner cache.

Classes, most urgent first: 1 search (no usable entry: queued misses / manual requests, and profile shapes), 2 widen
(a shape became frequent but its entry was searched with k < k_frequent), 3 re-search a soft-stale entry, 4 train
(added by the training orchestration), 5 re-search an entry superseded by new models (its only stale field is
`models`). Within a class: decayed calls x a roofline estimate of one call (time the user would save, roughly).
A (shape, env) that failed twice is skipped until the daemon restarts.
"""
from __future__ import annotations

from dataclasses import dataclass, field

PEAK_FLOPS = 60e12          # ordering only: absolute values do not matter
PEAK_BYTES = 600e9
CLASSES = {"search": 1, "widen": 2, "stale": 3, "train": 4, "superseded": 5}
MAX_FAILURES = 2


def t_est_s(M: int, N: int, K: int, elem: int = 2) -> float:
    return max(2.0 * M * N * K / PEAK_FLOPS, elem * (M * K + K * N + M * N) / PEAK_BYTES)


@dataclass(order=True)
class Task:
    sort_key: tuple = field(init=False, repr=False)
    kind: str
    M: int = 0
    N: int = 0
    K: int = 0
    dtype: str = "fp16"
    env_id: str = ""
    k: int = 3
    calls: float = 0.0
    reason: str = ""
    stage: str | None = None
    t: float = 0.0                       # queue time of a request: equal-score requests go oldest first

    def __post_init__(self):
        self.sort_key = (CLASSES[self.kind], -self.calls * t_est_s(self.M or 1, self.N or 1, self.K or 1), self.t)

    @property
    def key(self) -> tuple:
        return (self.M, self.N, self.K, self.dtype, self.env_id)


def classify(entry: dict | None, need_k: int) -> str | None:
    """Task kind for one (shape, env) given its cache entry (get(k=1, lt_n=0, allow_stale=True)), or None if satisfied."""
    if entry is None:
        return "search"
    if entry["stale"]:
        return "superseded" if set(entry["stale"]) <= {"models"} else "stale"
    return "widen" if entry["k"] < need_k else None


def plan(cfg: dict, prof: dict, pending: list[dict], caches: dict, failures: dict, now: float | None = None) -> list[Task]:
    """`caches` = {env_id: WinnerCache in that env's environment} (registered envs only); `failures` = {Task.key: n}."""
    from lscli.usage import calls_now
    s = cfg["search"]
    want: dict[tuple, dict] = {}
    for sh in prof.get("shapes", []):
        if sh.get("layout", "NN") != "NN":
            continue
        c = calls_now(prof, sh, now)
        for eid in sh.get("envs", []):
            if eid in caches:
                want[(sh["M"], sh["N"], sh["K"], sh["dtype"], eid)] = {"calls": c, "reason": "profile"}
    for r in pending:
        key = (int(r["M"]), int(r["N"]), int(r["K"]), r["dtype"], r["env_id"])
        if key[4] in caches:
            w = want.setdefault(key, {"calls": 0.0})
            w["reason"], w["t"] = r.get("reason", "miss"), r.get("t") or 0.0
    tasks = []
    for key, w in want.items():
        if failures.get(key, 0) >= MAX_FAILURES:
            continue
        need_k = s["k_frequent"] if w["calls"] >= s["frequent_calls"] else s["k"]
        e = caches[key[4]].get(*key[:4], k=1, lt_n=0, allow_stale=True)
        kind = classify(e, need_k)
        if kind is not None:
            tasks.append(Task(kind, *key[:4], env_id=key[4], k=need_k, calls=w["calls"], reason=w["reason"],
                              t=w.get("t", 0.0)))
    return sorted(tasks)


def satisfied(request: dict, caches: dict, failures: dict, unknown: set = frozenset()) -> bool:
    """A queued request needs no more work: its env is no longer registered, it failed out, or a current entry serves
    it. An env that is registered but whose environment could not be probed yet (`unknown`) keeps its requests."""
    key = (int(request["M"]), int(request["N"]), int(request["K"]), request["dtype"], request["env_id"])
    if key[4] in unknown:
        return False
    if key[4] not in caches or failures.get(key, 0) >= MAX_FAILURES:
        return True
    e = caches[key[4]].get(*key[:4], k=1, lt_n=0, allow_stale=True)
    return e is not None and not e["stale"]
