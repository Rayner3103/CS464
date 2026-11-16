"""hw_id per GPU UUID, so environment probes need no CUDA context (spec section 4.1, environment probes).

An environment doc (`tune.winner_cache.environment`) needs a CUDA context for two fields only: `hw_id`
(`torch.cuda.get_device_properties`: SM count and memory, which NVML cannot give exactly; NVML's SM count is not
supported on GeForce cards, and its memory total differs from torch's by more than the rounding on some GPUs) and the
cuBLASLt version, which the worker reads CUDA-free from the library `import torch` maps (`lscli.worker
static_environment`). So `hw_id` is computed once per GPU with CUDA, in a probe the daemon runs only with the gate
open and `gpu.lock` held, and stored here keyed by the NVML UUID; every later probe reuses it and opens no context.

The cached value is used only when it names the device the probe would see: NVML lists exactly one GPU and
`CUDA_VISIBLE_DEVICES` is unset or "0". Otherwise `known` returns None and the caller probes with CUDA (gated).
"""
from __future__ import annotations

import os

from lscli.files import locked, read_json, write_json

NEEDS_CUDA = "needs CUDA"     # error prefix of a CUDA-free probe that cannot answer without a context


def _single_device(identity: dict) -> bool:
    return (identity.get("count") == 1 and bool(identity.get("uuid"))
            and os.environ.get("CUDA_VISIBLE_DEVICES") in (None, "0"))


def known(paths, identity: dict) -> str | None:
    """The cached hw_id of the GPU `identity` (gate.gpu_identity()) names, or None (unknown, or not usable here)."""
    if not _single_device(identity):
        return None
    hw = (read_json(paths.gpu_hw, {}) or {}).get(identity["uuid"])
    return hw if isinstance(hw, str) and hw else None


def remember(paths, identity: dict, hw_id: str | None) -> bool:
    """Store `hw_id` for this GPU (from a probe that ran with CUDA). Returns True when the stored value changed."""
    if not hw_id or not _single_device(identity):
        return False
    with locked(paths.gpu_hw):
        doc = read_json(paths.gpu_hw, {}) or {}
        if doc.get(identity["uuid"]) == hw_id:
            return False
        doc[identity["uuid"]] = hw_id
        write_json(paths.gpu_hw, doc)
    return True
