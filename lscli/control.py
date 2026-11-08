"""Daemon status and control documents (spec 3.8): `state.json` is written by the daemon only, `control.json` by the
CLI only, so neither needs a lock (both are replaced atomically)."""
from __future__ import annotations

import os
import time

from lscli.files import read_json, write_json


def read_state(paths) -> dict | None:
    return read_json(paths.state, None)


def daemon_alive(state: dict | None) -> bool:
    """The pid in state.json is alive AND is an lscli process (a reused pid of another program is not the daemon)."""
    if not state or not state.get("pid"):
        return False
    pid = int(state["pid"])
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            argv = f.read().split(b"\0")
    except OSError:
        return True
    for i, a in enumerate(argv):                       # `python -m lscli daemon run` or `.../bin/lscli daemon run`
        if (a == b"lscli" or a.endswith(b"/lscli")) and b"daemon" in argv[i + 1:]:
            return True
    return False


def read_control(paths) -> dict:
    return read_json(paths.control, {}) or {}


def pause(paths, seconds: float | None = None) -> dict:
    """Pause background work: for `seconds`, or until `resume` (None)."""
    doc = {**read_control(paths), "paused": True, "paused_until": time.time() + seconds if seconds is not None else None,
           "updated": time.time()}
    write_json(paths.control, doc)
    return doc


def resume(paths) -> dict:
    doc = {**read_control(paths), "paused": False, "paused_until": None, "updated": time.time()}
    write_json(paths.control, doc)
    return doc


def paused(control: dict, now: float | None = None) -> bool:
    now = time.time() if now is None else now
    return bool(control.get("paused")) and (control.get("paused_until") is None or control["paused_until"] > now)


def request_reload(paths) -> None:
    """Ask the daemon to re-read config.toml (it compares this stamp each loop)."""
    write_json(paths.control, {**read_control(paths), "reload": time.time()})
