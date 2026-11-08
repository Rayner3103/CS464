"""Start-up hook: `lscli.pth` imports this module only when LSCLI_PATCH=1 (and LSCLI_DISABLE is not 1). It must never
break the interpreter it runs in, so any failure is one line on stderr."""
from __future__ import annotations

import sys

try:
    from lscli.client import enable

    enable()
except Exception as e:  # noqa: BLE001
    sys.stderr.write(f"lscli: not enabled ({type(e).__name__}: {e})\n")
