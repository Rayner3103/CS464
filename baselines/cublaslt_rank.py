"""Learned ranker over cuBLASLt configurations (runtime side): enumerate the configs valid for a shape, score them, return the best m.

Model: a LightGBM regressor of log(t / t_best_of_shape) per (shape, config) trained by `analysis/lt_rank.py` on the full per-config
screen of the TUNE shapes (`baselines/cublaslt_screen.py`), saved as `data/<hw>/lt_rank.txt/.json`. At run time the shape's valid
configs come from `cublaslt_search.enumerate_configs` (the same enumeration the screen used; cublasLtMatmulAlgoCheck-filtered, ~0.1-0.3 s
because of thousands of algo checks), the features below are computed for all of them, and the m predicted-fastest are rebuilt with
`build_algo` and short-timed like any cuBLASLt candidate (family "cublaslt_rank", key prefix "r"). Features include fill terms from
the tile dimensions (cublasLtMatmulTile_t ids 1-33, checked against cublasLt.h of CUDA 13; unknown ids give NaN fill features), so the model is
valid for the architecture and cuBLASLt version it was trained on (`meta["hw_id"]`); another GPU gets no proposals and a warning.

Other GPUs (RTX 3090, RTX 2060), two routes, both explicit:
  * retrain on that GPU (the default for anything but the same architecture): `python -m baselines.cublaslt_screen`, then
    `python -m analysis.lt_rank --save data/<hw>/lt_rank`. The fit reads this device's SM count and stores it as `meta["sms"]`, so
    the CTA / wave features describe the GPU the screen was measured on; `ranker()` then finds `data/<hw>/lt_rank` as before;
  * zero-shot transfer of a model fit elsewhere (`pipeline/zero_shot.sh --lt-rank 5`, `analysis.zero_shot run --lt-rank 5`): the
    guard is lifted per process with `TUNE_LT_RANK_FOREIGN=1` (the model is read from `TUNE_LT_RANK=<stem>`), and the SM count
    in the fill features is the variant of the Triton zero-shot experiment: `as_is` keeps the model's (3080: 68), `native` replaces
    it with this device's (`TUNE_LT_RANK_SMS=native`, or an integer). Tile and algorithm ids are per architecture, so this only
    means something on the same one (the 3090 is sm_86 like the 3080; the 2060 is sm_75: expect it to fail, and measure it).
Every pick is validated and short-timed whatever the model, so a bad transfer costs time, never correctness.
"""
from __future__ import annotations

import json
import math
import os
import warnings
from pathlib import Path

import numpy as np
import torch

from baselines.cublaslt import CublasLtMatmul, _CUDA_DTYPE, _Problem
from baselines.cublaslt_search import CFG_KEYS, build_algo, enumerate_configs

FAMILY = "cublaslt_rank"

# cublasLtMatmulTile_t (m x n) for the ids 1..37 (cublasLt.h); unknown ids -> 0 x 0. Both orientations are fed to the model.
TILE_MN = {1: (8, 8), 2: (8, 16), 3: (16, 8), 4: (8, 32), 5: (16, 16), 6: (32, 8), 7: (8, 64), 8: (16, 32), 9: (32, 16), 10: (64, 8),
           11: (32, 32), 12: (32, 64), 13: (64, 32), 14: (32, 128), 15: (64, 64), 16: (128, 32), 17: (64, 128), 18: (128, 64),
           19: (64, 256), 20: (128, 128), 21: (256, 64), 22: (64, 512), 23: (128, 256), 24: (256, 128), 25: (512, 64), 26: (64, 96),
           27: (96, 64), 28: (96, 128), 29: (128, 160), 30: (160, 128), 31: (192, 128), 32: (128, 192), 33: (128, 96)}
SMS = 68
Y_CLIP = math.log(8.0)
FEATURES = ["lM", "lN", "lK", "al_m", "al_n", "al_k", "algo", "tile", "stages", "splitk", "lsplit", "reduction", "swizzle", "custom",
            "tm", "tn", "ctas_a", "ctas_b", "waves_a", "waves_b", "pad_a", "pad_b", "k_per_split", "lk_per_split"]
CATS = ["algo", "tile", "stages"]


def device_sms() -> int:
    """SM count of the device the process runs on (SMS, the 3080's, when torch cannot say)."""
    try:
        return int(torch.cuda.get_device_properties(0).multi_processor_count)
    except Exception:
        return SMS


def features(shape: tuple, cfgs: np.ndarray, sms: int = SMS) -> np.ndarray:
    """(n, len(FEATURES)) float matrix for one shape's configs. `sms` is the SM count of the GPU the model describes."""
    M, N, K = shape
    n = len(cfgs)
    tm = np.array([TILE_MN.get(int(t), (0, 0))[0] for t in cfgs[:, 1]], float)
    tn = np.array([TILE_MN.get(int(t), (0, 0))[1] for t in cfgs[:, 1]], float)
    with np.errstate(divide="ignore", invalid="ignore"):
        # cuBLASLt sees the column-major problem (N x M): tile m covers N, tile n covers M; "b" is the other orientation
        ca = np.ceil(N / tm) * np.ceil(M / tn)
        cb = np.ceil(M / tm) * np.ceil(N / tn)
        pa = (np.ceil(N / tm) * tm * np.ceil(M / tn) * tn) / (M * N)
        pb = (np.ceil(M / tm) * tm * np.ceil(N / tn) * tn) / (M * N)
    nofit = tm == 0
    for a in (ca, cb, pa, pb):
        a[nofit] = np.nan
    sk = np.maximum(cfgs[:, 3], 1).astype(float)
    cols = [np.full(n, math.log2(M)), np.full(n, math.log2(N)), np.full(n, math.log2(K)),
            np.full(n, M % 8 == 0), np.full(n, N % 8 == 0), np.full(n, K % 8 == 0),
            cfgs[:, 0], cfgs[:, 1], cfgs[:, 2], sk, np.log2(sk), cfgs[:, 4], cfgs[:, 5], cfgs[:, 6], tm, tn,
            ca * sk, cb * sk, ca * sk / sms, cb * sk / sms, pa, pb, K / sk, np.log2(K / sk)]
    return np.column_stack(cols).astype(np.float32)


class LtRanker:
    def __init__(self, booster, meta: dict, sms: int | None = None):
        self.booster, self.meta = booster, meta
        self.sms = int(sms if sms is not None else meta.get("sms", SMS))   # models saved before `sms` existed are the 3080's

    @classmethod
    def load(cls, stem: str | Path, sms: int | str | None = None) -> "LtRanker":
        """sms: None = the model's own (`as_is`), "native" = this device's, or an integer."""
        import lightgbm as lgb
        stem = Path(stem)
        return cls(lgb.Booster(model_file=str(stem) + ".txt"), json.loads(Path(str(stem) + ".json").read_text()),
                   device_sms() if sms == "native" else (int(sms) if sms not in (None, "") else None))

    def rank(self, M: int, N: int, K: int, problem: _Problem, ctype: int, m: int) -> tuple[list[dict], dict]:
        """The m predicted-fastest valid configs (best first) and {"enumerate_s", "n_valid"}."""
        import time
        t = time.perf_counter()
        configs, _ = enumerate_configs(problem, ctype, M, N, K)
        t_enum = time.perf_counter() - t
        cfgs = np.array([[-1 if c.get(k) is None else c[k] for k in CFG_KEYS] for c in configs], dtype=np.int64)
        if not len(cfgs):
            return [], {"enumerate_s": t_enum, "n_valid": 0}
        pred = self.booster.predict(features((M, N, K), cfgs, self.sms))
        order = np.argsort(pred, kind="stable")[:m]
        return [configs[i] for i in order], {"enumerate_s": t_enum, "n_valid": len(configs)}


_RANKERS: dict[tuple, LtRanker | None] = {}


SNAPSHOT_STEM = Path(__file__).resolve().parent.parent / "reports" / "snapshot" / "lock1755" / "lt_rank"


def default_stem() -> Path:
    """`TUNE_LT_RANK` if set, else data/<hw>/lt_rank if it exists, else the committed snapshot (usable only on the GPU it was trained on)."""
    if os.environ.get("TUNE_LT_RANK"):
        return Path(os.environ["TUNE_LT_RANK"])
    from tune.env import default_data_dir
    local = default_data_dir() / "lt_rank"
    return local if Path(str(local) + ".txt").exists() else SNAPSHOT_STEM


def ranker(stem: str | Path | None = None, *, foreign: bool | None = None, sms: int | str | None = None) -> LtRanker | None:
    """The ranker for this GPU, or None (with one warning) when there is no model or it was trained for another GPU.
    `foreign` (default: `TUNE_LT_RANK_FOREIGN=1`) lifts that refusal for a zero-shot run; `sms` (default `TUNE_LT_RANK_SMS`) is
    None / "native" / an integer, see `LtRanker.load`. Cached per (stem, foreign, sms)."""
    stem = str(stem or default_stem())
    if foreign is None:
        foreign = os.environ.get("TUNE_LT_RANK_FOREIGN") == "1"
    if sms is None:
        sms = os.environ.get("TUNE_LT_RANK_SMS") or None
    key = (stem, bool(foreign), None if sms is None else str(sms))
    if key not in _RANKERS:
        try:
            r = LtRanker.load(stem, sms)
            from tune.env import default_data_dir
            if r.meta.get("hw_id") not in (None, default_data_dir().name):
                if foreign:
                    warnings.warn(f"cuBLASLt ranker: zero-shot use of a model trained on {r.meta['hw_id']} on {default_data_dir().name} "
                                  f"(SM count {r.sms}); tile / algorithm ids are per architecture")
                else:
                    warnings.warn(f"cuBLASLt ranker disabled: trained on {r.meta['hw_id']}, this GPU is {default_data_dir().name}")
                    r = None
        except Exception as e:
            warnings.warn(f"cuBLASLt ranker unavailable ({type(e).__name__}: {e})")
            r = None
        _RANKERS[key] = r
    return _RANKERS[key]


def candidates(M: int, N: int, K: int, dtype=torch.float16, m: int = 10, problem: _Problem | None = None,
               skip: set | None = None, stem: str | Path | None = None) -> list[CublasLtMatmul]:
    """The ranker's top-m runnable configs for this shape, not in `skip` (config tuples already in the pool)."""
    r = ranker(stem)
    if r is None:
        return []
    p = problem or _Problem(M, N, K, dtype)
    picks, _ = r.rank(M, N, K, p, _CUDA_DTYPE[dtype], m + (len(skip) if skip else 0))
    out = []
    for cfg in picks:
        if skip and tuple(cfg.get(c) for c in CFG_KEYS) in skip:
            continue
        res = build_algo(p, _CUDA_DTYPE[dtype], cfg)
        if res is None:
            continue
        out.append(CublasLtMatmul(p, res, len(out), family=FAMILY, prefix="r"))
        if len(out) == m:
            break
    return out
