"""Registered client interpreters (`envs.json`, spec 3.5). An env is identified by the ABSOLUTE, UNRESOLVED path of its
interpreter: a venv's `bin/python` is a symlink to the base interpreter, so resolving it would run the base Python
without the venv's packages and give every venv on that Python the same id. Versions are probed by running the
interpreter (the daemon never imports the user's packages into its own process)."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from lscli.files import read_json, write_json

_PROBE = r"""
import json, sys
out = {"python": sys.executable, "prefix": sys.prefix, "py": "%d.%d.%d" % sys.version_info[:3]}
try:
    import torch
    out.update(torch=torch.__version__, cuda_runtime=torch.version.cuda, cuda=torch.cuda.is_available())
except Exception as e:
    out["torch_error"] = f"{type(e).__name__}: {e}"
try:
    import triton
    out["triton"] = triton.__version__
except Exception as e:
    out["triton_error"] = f"{type(e).__name__}: {e}"
print(json.dumps(out))
"""


def interpreter_path(python: str | os.PathLike) -> str:
    return os.path.abspath(os.path.expanduser(str(python)))


def env_id(python: str | os.PathLike) -> str:
    return hashlib.sha256(interpreter_path(python).encode()).hexdigest()[:12]


def probe(python: str | os.PathLike, timeout: float = 120.0) -> dict:
    """Versions of an interpreter's torch / triton / CUDA runtime, by running it. Raises RuntimeError if it cannot run."""
    try:
        r = subprocess.run([str(python), "-c", _PROBE], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise RuntimeError(f"cannot run {python}: {e}") from None
    if r.returncode != 0:
        raise RuntimeError(f"{python} failed: {r.stderr.strip()[-400:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def load(paths) -> dict:
    return read_json(paths.envs, {}) or {}


def register(paths, python: str | os.PathLike, info: dict | None = None) -> tuple[str, dict]:
    info = dict(info if info is not None else probe(python))
    eid = env_id(python)
    envs = load(paths)
    info["python"] = interpreter_path(python)
    info["registered"] = envs.get(eid, {}).get("registered", time.time())
    info["updated"] = time.time()
    envs[eid] = info
    write_json(paths.envs, envs)
    return eid, info


def unregister(paths, python_or_id: str) -> bool:
    envs = load(paths)
    eid = python_or_id if python_or_id in envs else env_id(python_or_id)
    if eid not in envs:
        return False
    del envs[eid]
    write_json(paths.envs, envs)
    return True


def resolve(paths, ident: str | None) -> dict[str, dict]:
    """{env_id: record} for an id, an interpreter path, or None (= every registered env)."""
    envs = load(paths)
    if ident is None:
        return envs
    eid = ident if ident in envs else env_id(ident) if Path(ident).exists() else ident
    if eid not in envs:
        raise KeyError(f"no registered env {ident!r} (lscli envs lists them)")
    return {eid: envs[eid]}
