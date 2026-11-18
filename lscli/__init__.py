"""lscli: the learned-selector CLI and daemon (docs/lscli/spec.md).

The package is split by where it runs: `paths`, `files`, `config`, `envs`, `usage`, `queue`, `control` and `worker` are
light (stdlib + torch / triton, imported lazily) because the client and the measure worker run inside the user's own
Python env; ranking, training and scheduling live in the daemon's venv and import the research harness (`analysis`,
`pipeline/`) only there.
"""
from __future__ import annotations

__version__ = "0.1.0"


def enable(*args, **kwargs) -> bool:
    """Patch torch's NN matmul for the rest of the process (lscli.client.enable)."""
    from lscli.client import enable as _enable
    return _enable(*args, **kwargs)


def disable() -> None:
    """Restore torch (lscli.client.disable)."""
    from lscli.client import disable as _disable
    _disable()
