"""The fresh evaluation shapes (`suites/fresh.py`) are disjoint from everything a model was fit on. CPU only."""
import json
from pathlib import Path

from suites import fresh


def test_fresh_shapes_are_disjoint_from_suites_and_tune_neighbours():
    doc = json.loads((Path(fresh.ROOT) / "fresh" / "fresh465.json").read_text())
    allp, tune = fresh.suite_shapes()
    shapes = [(s["M"], s["N"], s["K"]) for s in doc["shapes"]]
    assert len(shapes) == len(set(shapes)) == 60
    assert not (set(shapes) & allp)                                   # in no suite at all
    assert not any(fresh.shares_two(t, tune) for t in shapes)         # no near-duplicate of a tune shape
    assert all(2e-9 * M * N * K <= fresh.MAX_GFLOP and max(M, N, K) <= fresh.MAX_DIM for M, N, K in shapes)
    assert {s["split"] for s in doc["shapes"]} == {"fresh"}


def test_generator_is_deterministic_and_respects_its_rules():
    a, b = fresh.build(465, 5, 5), fresh.build(465, 5, 5)
    assert a == b and len(a) == 10
    allp, tune = fresh.suite_shapes()
    assert all(fresh.ok((s["M"], s["N"], s["K"]), allp, tune) for s in a)
    assert fresh.build(466, 5, 5) != a
