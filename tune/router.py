"""Cost-only family router: after the cuBLASLt pool is timed, decide whether the Triton search is worth running.

The online search spends most of its first-use cost on the Triton side (rank, JIT compile or bundle load, validate, short-time) and, since the
Triton family only wins on ~20% of shapes (tune: its screen-oracle beats B3+ by > the tie margin on 86 of 477), most of that work is wasted on
shapes where cuBLASLt is clearly ahead. cuBLASLt candidates need no compile, so they are timed FIRST; the router then sees how close the best
of them is to the roofline and decides from the shape alone whether Triton could still beat it:

    P(win) = P(Triton's best config beats the cuBLASLt value by more than the tie margin | shape, t_cublaslt / roofline)

A LightGBM classifier trained on the 477 tune shapes' full-protocol stage 2 (the screen's top-8 Triton configs vs B3+; one session, graph-amortized
below 50 us), threshold chosen on grouped out-of-fold tune scores only (`analysis/router.py`). The Triton search is skipped when P(win) < threshold.
A skipped shape keeps its best cuBLASLt candidate; a shape that is not skipped pays exactly what it paid before (the order of the two halves is
the only difference). Not a speedup when the prebuilt kernel bundle serves the Triton kernels (the whole search is then ~0.17 s and the Triton
half ~0.06 s): `select_kernel(router="auto")` turns it on only when the bundle does not cover this Triton / GPU.

Model: `data/<hw>/router.txt/.json` or the committed snapshot `reports/snapshot/lock1755/router.*`, tagged with the hw_id it was trained for.

Other GPUs: the model is refused (the search then runs both halves) unless a zero-shot run lifts the guard with `TUNE_ROUTER_FOREIGN=1`
(`TUNE_ROUTER=<stem>` picks the model; `TUNE_ROUTER_SMS=native|<int>` sets the SM count of the `l_ctas_per_sm` feature: unset keeps
the model's, `native` uses this device's, the same as_is / native split as the rankers). Retraining it needs that GPU's whole tune
stage 2 (`analysis/router.py fit`), so there is no pipeline stage for it: a router on another GPU is a transfer test, scored with
`analysis/confirm_ab.py` from `tune online --router --confirm 3`. On a GPU where Triton is far behind cuBLASLt (the 2060) it should
skip almost everything; whether that is right is what the run measures.
"""
from __future__ import annotations

import json
import math
import os
import warnings
from pathlib import Path

import numpy as np

FEATURES = ["lM", "lN", "lK", "l_roofline_us", "l_t_lt_over_roofline", "l_ctas_per_sm", "al_m", "al_n", "al_k", "l_arith_intensity"]
SNAPSHOT_STEM = Path(__file__).resolve().parent.parent / "reports" / "snapshot" / "lock1755" / "router"
SMS = 68


def features(M: int, N: int, K: int, t_lt_us: float, sms: int = SMS) -> list[float]:
    from suites.generate import roofline_us
    roof = roofline_us(M, N, K)
    ctas = math.ceil(M / 128) * math.ceil(N / 128)
    return [math.log2(M), math.log2(N), math.log2(K), math.log(roof), math.log(max(t_lt_us, 1e-3) / roof), math.log(ctas / sms + 1e-9),
            float(M % 8 == 0), float(N % 8 == 0), float(K % 8 == 0), math.log(2 * M * N * K / (2 * (M * K + K * N + M * N)))]


class Router:
    def __init__(self, booster, meta: dict, sms: int | None = None):
        self.booster, self.meta = booster, meta
        self.threshold = float(meta["threshold"])
        self.sms = int(sms if sms is not None else meta.get("sms", SMS))   # models saved before `sms` existed are the 3080's

    @classmethod
    def load(cls, stem: str | Path, sms: int | str | None = None) -> "Router":
        """sms: None = the model's own (`as_is`), "native" = this device's, or an integer."""
        import lightgbm as lgb
        stem = Path(stem)
        if sms == "native":
            from baselines.cublaslt_rank import device_sms
            sms = device_sms()
        return cls(lgb.Booster(model_file=str(stem) + ".txt"), json.loads(Path(str(stem) + ".json").read_text()),
                   int(sms) if sms not in (None, "") else None)

    def p_win(self, M: int, N: int, K: int, t_lt_us: float) -> float:
        return float(self.booster.predict(np.array([features(M, N, K, t_lt_us, self.sms)]))[0])

    def skip_triton(self, M: int, N: int, K: int, t_lt_us: float) -> tuple[bool, float]:
        p = self.p_win(M, N, K, t_lt_us)
        return p < self.threshold, p


_ROUTERS: dict[tuple, Router | None] = {}


def default_stem() -> Path:
    """`TUNE_ROUTER` if set, else data/<hw>/router if it exists, else the committed snapshot (usable only on the GPU it was trained on)."""
    if os.environ.get("TUNE_ROUTER"):
        return Path(os.environ["TUNE_ROUTER"])
    from tune.env import default_data_dir
    local = default_data_dir() / "router"
    return local if Path(str(local) + ".txt").exists() else SNAPSHOT_STEM


def router(stem: str | Path | None = None, *, foreign: bool | None = None, sms: int | str | None = None) -> Router | None:
    """The router for this GPU, or None (with one warning) when there is no model or it was trained for another GPU.
    `foreign` (default `TUNE_ROUTER_FOREIGN=1`) lifts that refusal for a zero-shot run; `sms` (default `TUNE_ROUTER_SMS`) is
    None / "native" / an integer, see `Router.load`. Cached per (stem, foreign, sms)."""
    stem = str(stem or default_stem())
    if foreign is None:
        foreign = os.environ.get("TUNE_ROUTER_FOREIGN") == "1"
    if sms is None:
        sms = os.environ.get("TUNE_ROUTER_SMS") or None
    key = (stem, bool(foreign), None if sms is None else str(sms))
    if key not in _ROUTERS:
        try:
            r = Router.load(stem, sms)
            from tune.env import default_data_dir
            if r.meta.get("hw_id") not in (None, default_data_dir().name):
                if foreign:
                    warnings.warn(f"family router: zero-shot use of a model trained on {r.meta['hw_id']} on {default_data_dir().name} "
                                  f"(SM count {r.sms})")
                else:
                    warnings.warn(f"family router disabled: trained on {r.meta['hw_id']}, this GPU is {default_data_dir().name}")
                    r = None
        except Exception as e:
            warnings.warn(f"family router unavailable ({type(e).__name__}: {e})")
            r = None
        _ROUTERS[key] = r
    return _ROUTERS[key]
