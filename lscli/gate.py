"""The idle gate (spec 4.1) and the process bookkeeping it needs.

Two rules, because the daemon's own job drives the GPU:
  * opening: no compute process on the GPU and utilization below the threshold for `gpu_idle_minutes`, inside the
    window, on AC power, not paused, no backoff, no foreign pipeline run;
  * staying open (a job runs): the same minus utilization, and "no compute process" means none outside the job's process
    tree (the job's root and every descendant, walked from /proc each poll: sweep compile workers are grandchildren).

Everything that reads the machine (NVML, /proc, /sys, flock) is a small function taking its inputs explicitly, so the
rules are testable without a GPU. The daemon process itself never creates a CUDA context: NVML only.
"""
from __future__ import annotations

import datetime as _dt
import fcntl
import os
from dataclasses import dataclass, field
from pathlib import Path


# ------------------------------------------------------------------------------------------------------------ inputs
def in_window(window: str, now: _dt.datetime | None = None) -> bool:
    """`HH:MM-HH:MM` local time (may wrap midnight); "" = always. The end minute is excluded."""
    if not window:
        return True
    now = now or _dt.datetime.now()
    a, b = window.split("-")
    start = int(a[:2]) * 60 + int(a[3:])
    end = int(b[:2]) * 60 + int(b[3:])
    t = now.hour * 60 + now.minute
    if start == end:
        return True
    return start <= t < end if start < end else (t >= start or t < end)


def on_battery(sys_root: str | os.PathLike = "/sys/class/power_supply") -> bool:
    """True when any battery reports Discharging (desktops have none: False)."""
    root = Path(sys_root)
    try:
        supplies = list(root.iterdir())
    except OSError:
        return False
    for s in supplies:
        try:
            if (s / "type").read_text().strip() == "Battery" and (s / "status").read_text().strip() == "Discharging":
                return True
        except OSError:
            continue
    return False


def lock_held(path: str | os.PathLike) -> bool:
    """True when another process holds an exclusive flock on `path` (train_best.sh's pipeline/.lock)."""
    try:
        with open(path, "a") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(f, fcntl.LOCK_UN)
            return False
    except OSError:
        return False


def descendants(pid: int, proc: str = "/proc") -> set[int]:
    """`pid` and every live descendant (from /proc/<pid>/task/*/children)."""
    out, todo = set(), [pid]
    while todo:
        p = todo.pop()
        if p in out:
            continue
        out.add(p)
        try:
            tasks = os.listdir(f"{proc}/{p}/task")
        except OSError:
            continue
        for t in tasks:
            try:
                kids = Path(f"{proc}/{p}/task/{t}/children").read_text().split()
            except OSError:
                continue
            todo += [int(k) for k in kids]
    return out


@dataclass
class GpuSnapshot:
    pids: list[int]                 # compute processes on the GPU (NVML, host PIDs)
    util: int | None                # percent, None when NVML cannot tell
    ok: bool = True                 # False: NVML unavailable (the gate stays closed)
    names: dict = field(default_factory=dict)


def gpu_snapshot(index: int = 0) -> GpuSnapshot:
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(index)
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
        util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
    except Exception:
        return GpuSnapshot([], None, ok=False)
    pids = [int(p.pid) for p in procs]
    names = {}
    for p in pids:
        try:
            names[p] = Path(f"/proc/{p}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()[:120]
        except OSError:
            names[p] = "?"
    return GpuSnapshot(pids, util, True, names)


def gpu_identity(index: int = 0) -> dict:
    """NVML UUID / name of the daemon's GPU (no CUDA context)."""
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(index)
        uuid, name = pynvml.nvmlDeviceGetUUID(h), pynvml.nvmlDeviceGetName(h)
        n = pynvml.nvmlDeviceGetCount()
        return {"uuid": uuid.decode() if isinstance(uuid, bytes) else uuid,
                "name": name.decode() if isinstance(name, bytes) else name, "count": n}
    except Exception as e:
        return {"uuid": None, "name": None, "count": 0, "error": f"{type(e).__name__}: {e}"}


# ------------------------------------------------------------------------------------------------------------ the rules
class Gate:
    """Keeps the 'GPU idle since' clock between polls and applies the opening / staying-open rules."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.busy_until: float | None = None        # last time the GPU was seen busy (None: not yet observed)
        self.grace_until = 0.0                       # utilization right after our own job is still that job's

    def job_ended(self, now: float, grace_s: float = 5.0) -> None:
        self.grace_until = now + grace_s

    def observe(self, snap: GpuSnapshot, now: float, own: set[int] = frozenset()) -> list[int]:
        """Update the idle clock; returns the foreign compute PIDs."""
        foreign = [p for p in snap.pids if p not in own]
        util_busy = (not own and now >= self.grace_until and snap.util is not None
                     and snap.util >= self.cfg["idle"]["gpu_util_max"])
        busy = not snap.ok or bool(foreign) or util_busy
        if busy or self.busy_until is None:
            self.busy_until = now
        return foreign

    def reasons(self, *, snap: GpuSnapshot, now: float, local: _dt.datetime, battery: bool, paused: bool,
                backoff_until: float | None, pipeline_locked: bool, own: set[int] | None = None) -> list[str]:
        """Why the gate is closed ([] = open). `own` = the running job's process tree (None: no job runs)."""
        idle = self.cfg["idle"]
        foreign = self.observe(snap, now, own or set())
        out = []
        if not snap.ok:
            out.append("NVML unavailable")
        if foreign:
            out.append("compute process " + ", ".join(f"{p} ({snap.names.get(p, '?')})" for p in foreign[:3]))
        if own is None:
            need = idle["gpu_idle_minutes"] * 60
            if not foreign and snap.ok and now - self.busy_until < need:
                out.append(f"GPU idle {int(now - self.busy_until)}s of {int(need)}s")
        if not in_window(idle["window"], local):
            out.append(f"outside window {idle['window']}")
        if idle["require_ac"] and battery:
            out.append("on battery")
        if paused:
            out.append("paused")
        if backoff_until and now < backoff_until:
            out.append(f"backoff {int(backoff_until - now)}s after a preemption")
        if pipeline_locked:
            out.append("another train_best.sh run holds pipeline/.lock")
        return out
