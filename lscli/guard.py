"""Runtime guard (spec 5.1): the harness packages lscli imports must come from lscli's own source directory.

On a machine that develops this repo, a CS464 checkout reached through the cwd or PYTHONPATH can shadow lscli's copy
of `tune` / `kernels` / `baselines` / `analysis`, and which one a process gets depends on where it started. Mixing an
lscli from one tree with a harness from another can select or time the wrong kernels silently, so the client refuses
to patch and the worker refuses to run (before touching CUDA). Stdlib only: the client calls this before importing
anything of the harness, and finding a package's location does not import it.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HARNESS = ("tune", "kernels", "baselines", "analysis")
PREFIX = "harness mismatch"          # the worker's error text starts with it; the daemon keys on it


def source_root() -> Path:
    """The directory lscli itself was imported from (src/ of an install, or a checkout's root)."""
    return Path(__file__).resolve().parent.parent


def _package_dir(name: str) -> Path | None:
    """Where `import name` resolves (or already resolved) in this process; None when it is not importable."""
    mod = sys.modules.get(name)
    where = getattr(mod, "__file__", None) if mod is not None else None
    if where is None and mod is not None:
        where = next(iter(getattr(mod, "__path__", []) or []), None)
    if where is None:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            return None
        if spec is None:
            return None
        where = spec.origin if spec.origin not in (None, "namespace") else next(iter(spec.submodule_search_locations or []), None)
        if where is None:
            return None
    p = Path(where).resolve()
    return p.parent if p.name == "__init__.py" else p      # a package's directory, or a module file / namespace dir


def mismatches() -> list[dict]:
    """[{package, found, expected}] for every harness package that resolves outside lscli's source directory."""
    root = source_root()
    out = []
    for name in HARNESS:
        d = _package_dir(name)
        if d is not None and d.parent != root:
            out.append({"package": name, "found": str(d), "expected": str(root / name)})
    return out


def describe(found: list[dict]) -> str:
    return f"{PREFIX}: " + "; ".join(f"{m['package']} resolves to {m['found']}, not lscli's copy {m['expected']}"
                                     for m in found)
