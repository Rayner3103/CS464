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
