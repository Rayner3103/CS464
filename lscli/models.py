"""Which models the daemon ranks with (spec 3.7, 4.5): this GPU's own set once trained, else the zero-shot set.

A model set is a directory with MODELS.json:
  {"hw_id": ..., "source": "native" | "zero_shot", "models": {"triton": {"stem": "lgbm_rank_v2"}, "lt_rank": {...},
   "router": {...}}}
Stems are relative to the directory. A model's id is `<stem>:<first 12 hex of sha256 of its .txt and .json>`; the
ids go into every winner's environment (`env_extra["models"]`), so a new set makes old winners soft-stale
("superseded"). Loading never touches CUDA: the Triton ranker and the router are LightGBM boosters + JSON. The cuBLASLt
ranker needs descriptors (GPU), so the daemon passes only its stem to `lscli.propose`, which runs as a job.

Foreign-GPU rule (as in select_kernel): a cuBLASLt ranker or router trained on another architecture is not used; the
Triton ranker is used on any GPU (zero-shot), because its picks are validated and timed like any candidate.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from lscli.files import read_json

KINDS = ("triton", "lt_rank", "router")


def _arch(hw_id: str | None) -> str | None:
    """'rtx3080_sm86_68sms_10gb' -> 'sm86'."""
    if not hw_id:
        return None
    return next((p for p in hw_id.split("_") if p.startswith("sm") and p[2:].isdigit()), None)


def model_id(stem: Path) -> str | None:
    h = hashlib.sha256()
    for ext in (".txt", ".json"):
        f = Path(str(stem) + ext)
        if not f.exists():
            return None
        h.update(f.read_bytes())
    return f"{stem.name}:{h.hexdigest()[:12]}"


class ModelSet:
    def __init__(self, root: Path, doc: dict, hw_id: str | None):
        self.root, self.doc, self.hw_id = Path(root), doc, hw_id
        self.source = doc.get("source", "native")
        self.stems: dict[str, Path | None] = {}
        for kind in KINDS:
            m = (doc.get("models") or {}).get(kind)
            stem = self.root / m["stem"] if m and m.get("stem") else None
            if stem is not None and model_id(stem) is None:
                stem = None
            if kind != "triton" and stem is not None and _arch(doc.get("hw_id")) != _arch(hw_id):
                stem = None                          # cuBLASLt ids / router calibration are per architecture
            self.stems[kind] = stem
        self._triton = self._router = None

    @property
    def complete(self) -> bool:
        return all(self.stems[k] is not None for k in KINDS)

    def ids(self) -> dict:
        return {"source": self.source, **{k: (model_id(s) if s else None) for k, s in self.stems.items()}}

    def stem_paths(self) -> dict:
        return {k: (str(s) if s else None) for k, s in self.stems.items()}

    def triton(self):
        if self._triton is None and self.stems["triton"] is not None:
            from analysis.learned import Ranker
            self._triton = Ranker.load(self.stems["triton"])
        return self._triton

    def router(self):
        if self._router is None and self.stems["router"] is not None:
            from tune.router import Router
            self._router = Router.load(self.stems["router"])
        return self._router


def active(paths, hw_id: str | None) -> ModelSet | None:
    """This GPU's set when it has a Triton ranker (spec: native once trained), else the zero-shot set, else None."""
    for d in (paths.models / hw_id if hw_id else None, paths.models / "zero_shot"):
        if d is None:
            continue
        doc = read_json(d / "MODELS.json", None)
        if doc:
            ms = ModelSet(d, doc, hw_id)
            if ms.stems["triton"] is not None:
                return ms
    return None
