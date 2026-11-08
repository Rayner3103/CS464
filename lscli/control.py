"""Daemon status and control documents (spec 3.8): `state.json` is written by the daemon only, `control.json` by the
CLI only, so neither needs a lock (both are replaced atomically)."""
from __future__ import annotations

import os
import time

from lscli.files import read_json, write_json


def read_state(paths) -> dict | None:
    return read_json(paths.state, None)


def daemon_alive(state: dict | None) -> bool:
    if not state or not state.get("pid"):
        return False
    try:
        os.kill(int(state["pid"]), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def read_control(paths) -> dict:
    return read_json(paths.control, {}) or {}


def pause(paths, seconds: float | None = None) -> dict:
    """Pause background work: for `seconds`, or until `resume` (None)."""
    doc = {**read_control(paths), "paused": True, "paused_until": time.time() + seconds if seconds else None,
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
