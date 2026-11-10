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


def lock_held(path: str | os.PathLike, locks: str = "/proc/locks", exclude=frozenset(), proc: str = "/proc") -> bool:
    """True when a process outside `exclude` holds a flock on `path` (train_best.sh's pipeline/.lock). Read from
    /proc/locks, never by trying the lock: a probe that took it, even for a microsecond, could make a concurrent
    `flock -n` fail. `exclude` = the running job's tree: the daemon's own train_best run holds that lock.

    The pid /proc/locks gives is the process that TOOK the lock, not the one holding it: train_best.sh does
    `exec 9>.lock; flock -n 9`, so the pid is the short-lived `flock` child, already gone, never in the job's tree. A
    lock is therefore also ours when a process in `exclude` holds it through one of its own fds (/proc/<pid>/fdinfo lists
    a lock on the open file description that owns it). /proc/locks and the fds are read one after the other, so a job
    that releases the lock and exits in between would look foreign: a foreign reading counts only when a second one
    agrees."""
    return _foreign_lock(path, locks, exclude, proc) and _foreign_lock(path, locks, exclude, proc)


def _foreign_lock(path, locks: str, exclude, proc: str) -> bool:
    try:
        st = os.stat(path)
        text = Path(locks).read_text()
    except OSError:
        return False
    want = (os.major(st.st_dev), os.minor(st.st_dev), st.st_ino)
    ours = None
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 6 or parts[1] == "->":                 # "->" lines are waiters, not holders
            continue
        try:
            maj, mnr, ino = parts[5].split(":")
            if (int(maj, 16), int(mnr, 16), int(ino)) != want or int(parts[4]) in exclude:
                continue
        except ValueError:
            continue
        if ours is None:
            ours = _fd_locks(exclude, want, proc)
        if tuple(parts[1:]) not in ours:
            return True
    return False


def _fd_locks(pids, want: tuple[int, int, int], proc: str = "/proc") -> set[tuple[str, ...]]:
    """The locks on file `want` (major, minor, inode) that the processes `pids` hold through their own fds, as the
    fields of their /proc/locks lines after the ordinal (/proc/<pid>/fdinfo/<fd>: `lock:\t1: FLOCK ADVISORY WRITE ...`)."""
    out: set[tuple[str, ...]] = set()
    for pid in pids:
        fd_dir = Path(proc) / str(pid) / "fd"
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue
        for fd in fds:
            try:
                st = os.stat(fd_dir / fd)
                if (os.major(st.st_dev), os.minor(st.st_dev), st.st_ino) != want:
                    continue
                info = (Path(proc) / str(pid) / "fdinfo" / fd).read_text()
            except OSError:
                continue
            for ln in info.splitlines():
                if ln.startswith("lock:"):
                    out.add(tuple(ln.split()[2:]))
    return out


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


def _always_alive(pid: int) -> bool:          # a snapshot built by hand (tests): every listed pid runs
    return True


@dataclass
class GpuSnapshot:
    pids: list[int]                 # compute processes on the GPU (NVML, host PIDs)
    util: int | None                # percent, None when NVML cannot tell
    ok: bool = True                 # False: NVML unavailable (the gate stays closed)
    names: dict = field(default_factory=dict)
    # Asked again when the gate decides, i.e. AFTER the job's tree was walked: a job child that NVML listed and that
    # exited between the snapshot and the walk is in neither, and must not look foreign. The sweep's compile pool
    # (max_tasks_per_child=1) replaces a worker every chunk, so this race preempted tune_sweep within seconds, every time.
    alive: object = field(default=_always_alive, repr=False, compare=False)


def alive(pid: int, proc: str | os.PathLike = "/proc") -> bool:
    """False once `pid` is exiting or has exited: no /proc entry, a zombie / dead state, or no memory left (empty cmdline
    and an all-zero statm: the kernel drops a process's memory as it starts to exit, before it is a zombie; a live
    process that blanked its argv still has memory). NVML still lists a process while its CUDA context is torn down: a
    job's short-lived child (a sweep compile worker) seen there and reaped before the job's tree was walked looked
    foreign and preempted the job. A process on its way out uses no GPU, whoever it belonged to. Assumes the daemon sees
    the host's pids (a per-user service, not a container): a pid it cannot see counts as exited."""
    d = Path(proc, str(pid))
    try:
        stat = (d / "stat").read_text()
        if stat[stat.rindex(")") + 2:][:1] in ("Z", "X", "x"):
            return False
        if (d / "cmdline").read_bytes().strip(b"\0"):
            return True
        return any(v != "0" for v in (d / "statm").read_text().split())
    except (OSError, ValueError):
        return False


def gpu_snapshot(index: int = 0) -> GpuSnapshot:
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(index)
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
        util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
    except Exception:
        return GpuSnapshot([], None, ok=False)
    pids = [pid for pid in (int(p.pid) for p in procs) if alive(pid)]
    names = {}
    for p in pids:
        try:
            names[p] = Path(f"/proc/{p}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()[:120]
        except OSError:
            names[p] = "?"
    return GpuSnapshot(pids, util, True, names, alive=alive)


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
        foreign = [p for p in snap.pids if p not in own and snap.alive(p)]   # after `own` was walked
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
