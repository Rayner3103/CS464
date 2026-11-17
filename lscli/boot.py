"""Start-up hook: `lscli.pth` imports this module only when LSCLI_PATCH=1 (and LSCLI_DISABLE is not 1).

It must cost nothing for programs that never use torch, and must not initialize anything early: importing torch from a
.pth would run before later .pth files and before the program sets CUDA_VISIBLE_DEVICES. So this module imports only
the stdlib and installs a meta-path finder that enables the client right after the program's own `import torch`
finishes. If torch is already imported (unusual for a .pth), it enables now. Any failure is one line on stderr.
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import sys


def _enable() -> None:
    try:
        from lscli.client import enable
        enable()
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"lscli: not enabled ({type(e).__name__}: {e})\n")


class _TorchHook(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name != "torch":
            return None
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass
        spec = importlib.util.find_spec("torch")
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return spec
        exec_module = spec.loader.exec_module

        def run(module):
            exec_module(module)
            _enable()
        spec.loader.exec_module = run
        return spec


def install() -> None:
    if "torch" in sys.modules:
        _enable()
    elif not any(isinstance(f, _TorchHook) for f in sys.meta_path):
        sys.meta_path.insert(0, _TorchHook())


install()
