"""Shape-conditioned cuBLASLt proposals: the deep-search winners of the nearest tune shapes.

cuBLASLt's heuristic returns a handful of configs per shape, and the 20 s enumerate-and-screen (cublaslt_search.py) finds
split-K / tile configs it never offers (B3+ beats the heuristic top-16 by > the tie margin on 30 of 86 held-out shapes).
Those winners are not a small set of recurring configs (116 distinct over 213 deep-needed shapes) but they cluster by
regime, so this proposes, for a new shape, the top `per_neighbor` configs of its `k` nearest tune shapes in a small
feature space (log2 M, N, K, 128x128 CTA fill on 68 SMs, 8-alignment; each standardized on the tune shapes). A proposal
needs no compile: it is rebuilt with `build_algo` (cublasLtMatmulAlgoCheck rejects a config invalid for the shape) and
short-timed like any other cuBLASLt candidate. Family "cublaslt_knn", key prefix "n".

Shipping: the table is also kept as a small committed snapshot (`reports/snapshot/lock1755/lt_knn_table.json`, tune
shapes' top-8 configs, `python -m baselines.cublaslt_knn --snapshot` regenerates it from `lt_search/`). `table()` uses the
local caches when they hold tune searches and falls back to the snapshot only on the GPU it was built for (hw_id); on any
other GPU it proposes nothing and warns once (tile / algorithm ids and their validity are per architecture).

Held-out discipline: the table is built from the `lt_search/` caches of TUNE-split shapes only (suites/v1 `split`),
never a held-out or aux shape, and a shape never proposes from itself. The caches live in `data/<hw>/lt_search/`
(`TUNE_LT_SEARCH_DIR` moves them); data/ is gitignored, so a fresh clone has no table until it runs the searches
(`cublaslt_deep:20` over the tune shapes) or ships a snapshot. With no table `propose` returns [].
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import warnings
from pathlib import Path

import numpy as np
import torch

from baselines.cublaslt import CublasLtMatmul, _CUDA_DTYPE, _Problem
from baselines.cublaslt_search import CFG_KEYS, build_algo

SM_COUNT = 68
FAMILY = "cublaslt_knn"
SNAPSHOT = Path(__file__).resolve().parent.parent / "reports" / "snapshot" / "lock1755" / "lt_knn_table.json"
TOP_STORED = 8
_KEY_RE = re.compile(r"^([nr])(\d+)_a(\w+)_t(\w+)_s(\w+)_sk(\w+)_r(\w+)_sw(\w+)_c(\w+)$")  # n = k-NN proposal, r = ranker pick


def features(M: int, N: int, K: int) -> np.ndarray:
    """log2 M, N, K, log2 of 128x128-tile CTAs per SM, and whether all three dims are multiples of 8."""
    ctas = math.ceil(M / 128) * math.ceil(N / 128)
    return np.array([math.log2(M), math.log2(N), math.log2(K), math.log2(ctas / SM_COUNT + 1e-9),
                     float(M % 8 == 0 and N % 8 == 0 and K % 8 == 0)])


def lt_search_dir() -> Path:
    from tune.env import default_data_dir
    return Path(os.environ.get("TUNE_LT_SEARCH_DIR") or default_data_dir() / "lt_search")


def tune_shapes() -> set[tuple[int, int, int]]:
    """(M, N, K) of every suite shape whose split is "tune" (suites/v1; the split was fixed before any result)."""
    root = Path(__file__).resolve().parent.parent / "suites" / "v1"
    out = set()
    for f in root.glob("*.json"):
        doc = json.loads(f.read_text())
        out |= {(s["M"], s["N"], s["K"]) for s in doc["shapes"] if s.get("split") == "tune"}
    return out


class ProposalTable:
    """Winner configs of the tune shapes' deep searches, with their standardized shape features."""

    def __init__(self, shapes: list[tuple[int, int, int]], tops: list[list[dict]]):
        self.shapes, self.tops = shapes, tops
        X = np.array([features(*s) for s in shapes]) if shapes else np.zeros((0, 5))
        self.sd = (X.std(0) if len(X) else np.zeros(5)) + 1e-9
        self.X = X / self.sd

    @classmethod
    def load(cls, search_dir: Path | None = None, dtype: str = "fp16") -> "ProposalTable":
        search_dir = Path(search_dir) if search_dir else lt_search_dir()
        tune = tune_shapes()
        shapes, tops = [], []
        for f in sorted(glob.glob(str(search_dir / f"*_{dtype}.json"))):
            d = json.loads(Path(f).read_text())
            s = (d["M"], d["N"], d["K"])
            if s in tune and d.get("top"):
                shapes.append(s)
                tops.append([t["cfg"] for t in d["top"]])
        return cls(shapes, tops)

    def __len__(self) -> int:
        return len(self.shapes)

    def propose(self, M: int, N: int, K: int, k: int = 5, per_neighbor: int = 2) -> list[dict]:
        """Distinct configs, nearest neighbour first: the top `per_neighbor` of each of the `k` nearest tune shapes
        (never the shape itself)."""
        if not len(self):
            return []
        d = np.sqrt(((self.X - features(M, N, K) / self.sd) ** 2).sum(1))
        out, seen = [], set()
        for i in np.argsort(d):
            if self.shapes[i] == (M, N, K):
                continue
            if k <= 0:
                break
            k -= 1
            for cfg in self.tops[i][:per_neighbor]:
                key = tuple(cfg.get(c) for c in CFG_KEYS)
                if key not in seen:
                    seen.add(key)
                    out.append(cfg)
        return out


    @classmethod
    def from_snapshot(cls, path: Path = SNAPSHOT) -> "ProposalTable":
        doc = json.loads(Path(path).read_text())
        return cls([tuple(e["shape"]) for e in doc["shapes"]],
                   [[dict(zip(CFG_KEYS, c)) for c in e["top"]] for e in doc["shapes"]])


def snapshot(dest: Path = SNAPSHOT, search_dir: Path | None = None, dtype: str = "fp16") -> Path:
    """Write the tune shapes' stored top configs (from `lt_search/`) as the committed, GPU-tagged snapshot."""
    from baselines.cublaslt import library_info
    from tune.env import default_data_dir
    search_dir = Path(search_dir) if search_dir else lt_search_dir()
    tune = tune_shapes()
    shapes = []
    for f in sorted(glob.glob(str(search_dir / f"*_{dtype}.json"))):
        d = json.loads(Path(f).read_text())
        s = (d["M"], d["N"], d["K"])
        if s in tune and d.get("top"):
            shapes.append({"shape": list(s), "top": [[t["cfg"].get(c) for c in CFG_KEYS] for t in d["top"][:TOP_STORED]]})
    doc = {"hw_id": default_data_dir().name, "cublaslt_version": library_info()["version"], "dtype": dtype,
           "cfg_keys": list(CFG_KEYS), "note": "top configs of the 20 s cuBLASLt enumerate-and-screen on TUNE-split shapes only "
           "(baselines/cublaslt_search.py); input of baselines/cublaslt_knn.py", "n_shapes": len(shapes), "shapes": shapes}
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(doc, separators=(",", ":")))
    return dest


_TABLE: dict[tuple[str, str], ProposalTable] = {}


def table(dtype: str = "fp16") -> ProposalTable:
    """The proposal table: local tune searches if there are any, else the committed snapshot on the GPU it was built for."""
    key = (str(lt_search_dir()), dtype)
    if key not in _TABLE:
        t = ProposalTable.load(dtype=dtype)
        if not len(t) and dtype == "fp16" and SNAPSHOT.exists():
            from tune.env import default_data_dir
            doc = json.loads(SNAPSHOT.read_text())
            if doc.get("hw_id") == default_data_dir().name:
                t = ProposalTable.from_snapshot()
            else:
                warnings.warn(f"cuBLASLt kNN proposals disabled: the snapshot was built on {doc.get('hw_id')}, "
                              f"this GPU is {default_data_dir().name} (tile / algorithm ids are per architecture)")
        _TABLE[key] = t
    return _TABLE[key]


def cfg_tuple(cfg: dict) -> tuple:
    return tuple(cfg.get(c) for c in CFG_KEYS)


def candidates(M: int, N: int, K: int, dtype=torch.float16, k: int = 5, per_neighbor: int = 2,
               problem: _Problem | None = None, skip: set | None = None) -> list[CublasLtMatmul]:
    """The runnable proposals for this shape (those cublasLtMatmulAlgoCheck accepts), not in `skip` (config tuples, e.g.
    the heuristic candidates' already in the pool)."""
    dname = {torch.float16: "fp16", torch.bfloat16: "bf16"}[dtype]
    p = problem or _Problem(M, N, K, dtype)
    out = []
    for cfg in table(dname).propose(M, N, K, k, per_neighbor):
        res = build_algo(p, _CUDA_DTYPE[dtype], cfg)
        if res is None:
            continue
        c = CublasLtMatmul(p, res, len(out), family=FAMILY, prefix="n")
        if skip and cfg_tuple(c.cfg) in skip:
            continue
        out.append(c)
    return out


def from_key(key: str, M: int, N: int, K: int, dtype=torch.float16) -> CublasLtMatmul | None:
    """Rebuild the candidate a stored key names (the winner cache); None if the library no longer accepts it."""
    m = _KEY_RE.match(key)
    if m is None:
        return None
    vals = [None if v == "None" else int(v) for v in m.groups()[2:]]
    cfg = dict(zip(("algo_id", "tile_id", "stages_id", "splitk", "reduction", "swizzle", "custom"), vals))
    p = _Problem(M, N, K, dtype)
    res = build_algo(p, _CUDA_DTYPE[dtype], cfg)
    if res is None:
        return None
    fam = FAMILY if m.group(1) == "n" else "cublaslt_rank"
    return CublasLtMatmul(p, res, int(m.group(2)), family=fam, prefix=m.group(1))


if __name__ == "__main__":
    import sys
    if "--snapshot" in sys.argv:
        out = snapshot()
        print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB, {json.loads(out.read_text())['n_shapes']} shapes)")
