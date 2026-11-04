"""kNN cuBLASLt proposals (`baselines/cublaslt_knn.py`) and their place in the online pool (`tune.online._subset`).
CPU only except the key round-trip, which needs the GPU for a cuBLASLt handle."""
import json

import pytest
import torch

from baselines import cublaslt_knn as knn
from tune import online


def _cfg(algo, tile=11, sk=1):
    return {"algo_id": algo, "tile_id": tile, "stages_id": 0, "splitk": sk, "reduction": 0, "swizzle": 0, "custom": 0}


def _write(d, M, N, K, cfgs):
    (d / f"{M}x{N}x{K}_fp16.json").write_text(json.dumps(
        {"M": M, "N": N, "K": K, "dtype": "fp16", "top": [{"cfg": c, "us": 1.0 + i, "seed": False} for i, c in enumerate(cfgs)]}))


def test_propose_nearest_first_dedup_and_never_self():
    t = knn.ProposalTable([(1000, 1000, 1000), (1024, 1024, 1024), (8, 8, 16384)],
                          [[_cfg(1), _cfg(2)], [_cfg(2), _cfg(3)], [_cfg(9), _cfg(10)]])
    got = [c["algo_id"] for c in t.propose(1010, 1010, 1010, k=2, per_neighbor=2)]
    assert sorted(got) == [1, 2, 3] and len(got) == len(set(got))   # algo 2 is shared by both neighbours: kept once
    assert 9 not in got and 10 not in got                           # the far shape is not among the 2 nearest
    # a shape never proposes from itself: the nearest *other* shape (1024^3) is the only neighbour at k = 1
    assert [c["algo_id"] for c in t.propose(1000, 1000, 1000, k=1, per_neighbor=1)] == [2]


def test_load_keeps_only_tune_shapes(tmp_path, monkeypatch):
    monkeypatch.setattr(knn, "tune_shapes", lambda: {(1000, 1000, 1000)})
    _write(tmp_path, 1000, 1000, 1000, [_cfg(1)])
    _write(tmp_path, 9500, 9500, 9500, [_cfg(2)])             # not a tune shape (held-out region): must never be proposed
    t = knn.ProposalTable.load(tmp_path)
    assert t.shapes == [(1000, 1000, 1000)]
    assert t.propose(2000, 2000, 2000) == [_cfg(1)]


def test_tune_split_excludes_every_heldout_shape():
    import glob
    held, tune = set(), knn.tune_shapes()
    for f in glob.glob("suites/v1/*.json"):
        held |= {(s["M"], s["N"], s["K"]) for s in json.load(open(f))["shapes"] if s["split"].startswith("heldout")}
    assert tune and held and not (tune & held)


def test_empty_table_proposes_nothing():
    assert knn.ProposalTable([], []).propose(64, 64, 64) == []


def test_knn_candidates_are_not_heuristic_candidates():
    def c(key, fam, us):
        return {"config_key": key, "valid": True, "cold_us": us, "compile_s": 0.0, "time_s": 0.0, "family": fam}
    cands = [dict(c("bm32", "triton", 30.0)), c("h0", "cublaslt", 20.0), c("h1", "cublaslt", 21.0), c("n0", "cublaslt_knn", 10.0)]
    assert online.choose(cands, 1, 16)["key"] == "h0"         # heuristic-only search ignores the proposals
    assert online.choose(cands, 1, None)["key"] == "n0"       # the full pool includes them
    assert online.choose(cands, 1, 1)["key"] == "h0"
    out = online.all_choices(cands, 1, 16)
    assert out["choice"]["1"]["key"] == "n0" and out["choice_lt"]["1_lt16"]["key"] == "h0"
    assert out["choice"]["1"]["family"] == "cublaslt_knn"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_key_round_trip():
    from baselines.cublaslt import candidates as heuristic
    h = heuristic(1024, 1024, 1024, torch.float16, 4)[0]
    cfg = {k: h.cfg.get(k) for k in ("algo_id", "tile_id", "stages_id", "splitk", "reduction", "swizzle", "custom")}
    from baselines.cublaslt import _CUDA_DTYPE
    from baselines.cublaslt_search import build_algo
    res = build_algo(h.problem, _CUDA_DTYPE[torch.float16], cfg)
    c = knn.CublasLtMatmul(h.problem, res, 3, family=knn.FAMILY, prefix="n")
    again = knn.from_key(c.key, 1024, 1024, 1024)
    assert again is not None and again.key == c.key and again.family == "cublaslt_knn"


def test_tile_table_matches_the_cublaslt_header():
    import os
    import re
    from baselines.cublaslt_rank import TILE_MN
    if not os.path.exists("/usr/include/cublasLt.h"):
        pytest.skip("no cublasLt.h")
    ref = {int(i): (int(a), int(b)) for a, b, i in re.findall(r"CUBLASLT_MATMUL_TILE_(\d+)x(\d+)\s*=\s*(\d+)", open("/usr/include/cublasLt.h").read())}
    assert all(TILE_MN[i] == ref[i] for i in TILE_MN) and set(ref) <= set(TILE_MN) | {0}


def test_ranker_features_shape_and_split_k_handling():
    import numpy as np
    from baselines.cublaslt_rank import FEATURES, features
    cfgs = np.array([[24, 11, 13, 1, 0, 0, 0], [24, 18, 0, 8, 2, 0, 0], [13, -1, -1, 1, 0, 0, 42]])   # last: tile-less (GEMV-like)
    f = features((68, 6420, 10359), cfgs)
    assert f.shape == (3, len(FEATURES))
    i = FEATURES.index("k_per_split")
    assert f[0, i] == 10359 and f[1, i] == pytest.approx(10359 / 8)
    assert np.isnan(f[2, FEATURES.index("ctas_a")])                 # no tile dimensions -> unknown fill, not zero
