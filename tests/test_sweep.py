import json

from kernels.config_space import HANDPICKED
from tune.sweep import coarse_subset, parse_key, read_partial, screen_path, top_configs


def _shape(M, N, K, suite="S3", split="tune", **kw):
    return {"M": M, "N": N, "K": K, "suite": suite, "split": split, **kw}


def test_parse_key_roundtrip():
    for c in HANDPICKED.values():
        assert parse_key(c.key()) == c


def test_coarse_subset_is_deterministic_and_respects_splits():
    shapes = [_shape(2 ** i, 64, 64) for i in range(12)]
    shapes += [_shape(8, 8, 8, split="heldout_region"), _shape(16, 16, 16, out_of_range=True),
               _shape(16384, 16384, 16384, suite="S1")]
    sub = coarse_subset(shapes, per_suite=4, max_gflop=1000)
    assert sub == coarse_subset(list(reversed(shapes)), per_suite=4, max_gflop=1000)
    ms = [s["M"] for s in sub]
    assert len(sub) == 4 and ms[0] == 1 and ms[-1] == 2048  # spans the FLOP range, both ends included
    assert all(s["split"] == "tune" and not s.get("out_of_range") for s in sub)
    assert all(s["suite"] != "S1" for s in sub)  # 16384^3 is 8796 GFLOP, above the cap


def test_coarse_subset_dedupes_across_suites():
    sub = coarse_subset([_shape(64, 64, 64, suite="S1"), _shape(64, 64, 64, suite="S2")], 5, 1000)
    assert len(sub) == 1


def test_top_configs_and_torn_partial(tmp_path):
    keys = [c.key() for c in HANDPICKED.values()]
    rows = [{"key": keys[0], "us": 30.0, "status": "ok"}, {"key": keys[1], "us": 10.0, "status": "ok"},
            {"key": keys[2], "us": 5.0, "status": "slow"}, {"key": keys[3], "us": None, "status": "invalid: x"},
            {"key": keys[4], "us": 20.0, "status": "ok"}]
    screen_path(tmp_path, 4, 5, 6).write_text(json.dumps({"rows": rows}))
    top = top_configs(tmp_path, 4, 5, 6, k=2)
    assert [t["cfg"].key() for t in top] == [keys[1], keys[4]]
    assert [t["screen_rank"] for t in top] == [0, 1]
    assert top_configs(tmp_path, 1, 2, 3, k=2) == []

    # Variants of one tile (here GROUP_M only) collapse to the fastest one.
    same_tile = [{"key": "bm64_bn64_bk32_w4_s4_g1_sk1", "us": 1.0, "status": "ok"},
                 {"key": "bm64_bn64_bk32_w4_s4_g8_sk1", "us": 1.1, "status": "ok"}]
    screen_path(tmp_path, 7, 8, 9).write_text(json.dumps({"rows": same_tile + rows}))
    assert [t["cfg"].key() for t in top_configs(tmp_path, 7, 8, 9, k=3)] == [same_tile[0]["key"], keys[1], keys[4]]

    p = tmp_path / "x.partial.jsonl"
    p.write_text(json.dumps(rows[0]) + "\n" + '{"key": "bm1')
    assert [r["key"] for r in read_partial(p)] == [keys[0]]


def test_workers_for_is_capped_by_the_scarcest_resource():
    from tune.sweep import workers_for
    assert workers_for(10, 12, 60.0, 10000.0) == (10, "")
    n, why = workers_for(10, 12, 16.0, 5000.0)             # 16 GB machine: (16 - 3) / 1 GiB = 13, VRAM (5000 - 768) / 450 = 9
    assert n == 9 and "VRAM" in why
    n, why = workers_for(0, 12, 8.0, 5000.0)               # auto = 10, but 8 GiB free leaves room for 5
    assert n == 5 and "host RAM" in why
    assert workers_for(10, 4, None, None)[0] == 4          # unknown memory is not a limit; CPUs still are
    assert workers_for(10, 12, 2.0, 100.0)[0] == 1         # never below one worker


def test_shape_vram_bytes_counts_fp32_reference():
    from tune.sweep import shape_vram_bytes
    M = N = K = 1024
    assert shape_vram_bytes(M, N, K) == 2 * 3 * 2**20 + 4 * 2 * 2**20 + 4 * 2**20 + 576 * 2**20


def test_prewarm_halves_the_pool_when_workers_die_and_keeps_finished_chunks():
    from kernels.triton_matmul import MatmulConfig
    from tune.sweep import prewarm
    cfgs = [MatmulConfig(16, 16, 16, num_warps=2, num_stages=2, GROUP_M=1, SPLIT_K=1 + i % 2) for i in range(8)]
    calls = []

    def pool(chunks, workers):
        calls.append((workers, sorted(chunks)))
        if workers > 2:                                    # pool broken after finishing the first two chunks only
            return {i: {f"c{i}": (0.0, None)} for i in sorted(chunks)[:2]}
        return {i: {f"c{i}": (0.0, None)} for i in chunks}

    res = prewarm(cfgs, 64, 64, 64, workers=8, pool=pool, log=lambda *_: None)
    assert [w for w, _ in calls] == [8, 4, 2]
    assert len(set(calls[1][1]) & set(calls[0][1][:2])) == 0   # finished chunks are not redone
    assert len(res) == 32                                      # 8 workers x 4 chunks, each reported once


def test_prewarm_gives_up_at_one_worker_instead_of_raising():
    from kernels.triton_matmul import MatmulConfig
    from tune.sweep import prewarm
    cfgs = [MatmulConfig(16, 16, 16, num_warps=2, num_stages=2, GROUP_M=1, SPLIT_K=1)]
    logs = []
    assert prewarm(cfgs, 64, 64, 64, workers=2, pool=lambda c, w: {}, log=logs.append) == {}
    assert any("lazily" in m for m in logs)
