"""Where lscli keeps its state (spec section 3). `LSCLI_HOME` / `LSCLI_CONFIG` move it; `--home` on the CLI sets
`LSCLI_HOME` for the process. Every path is derived here so the client, the worker and the daemon agree."""
from __future__ import annotations

import os
from pathlib import Path


def home() -> Path:
    return Path(os.environ.get("LSCLI_HOME") or Path.home() / ".local" / "share" / "lscli")


def config_path() -> Path:
    return Path(os.environ.get("LSCLI_CONFIG") or Path.home() / ".config" / "lscli" / "config.toml")


class Paths:
    """The files under one LSCLI_HOME."""

    def __init__(self, root: str | os.PathLike | None = None):
        self.root = Path(root) if root else home()

    def ensure(self) -> "Paths":
        """Create the home with mode 0700 (it holds the usage record: shapes and interpreter paths)."""
        self.root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)
        return self

    winners = property(lambda s: s.root / "winners.jsonl")
    profile_dir = property(lambda s: s.root / "profile")
    profile = property(lambda s: s.root / "profile.json")
    queue = property(lambda s: s.root / "queue.jsonl")
    queue_offset = property(lambda s: s.root / "queue.offset")
    gpu_lock = property(lambda s: s.root / "gpu.lock")
    envs = property(lambda s: s.root / "envs.json")
    models = property(lambda s: s.root / "models")
    train = property(lambda s: s.root / "train")
    state = property(lambda s: s.root / "state.json")
    control = property(lambda s: s.root / "control.json")
    client = property(lambda s: s.root / "client.json")
    logs = property(lambda s: s.root / "logs")
    src = property(lambda s: s.root / "src")
    venv = property(lambda s: s.root / "venv")
