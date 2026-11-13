"""Fresh evaluation shapes: new shapes that no fit, screen, selection or report has seen. CPU only.

`python -m suites.fresh [--seed 465] [--n-random 30] [--n-llm 30]` writes `suites/fresh/fresh<seed>.json` (same format as `suites/v1/*.json`, split
"fresh"). Two groups, so both regimes of the study are covered: (1) `fresh_random`, log-uniform M, N, K in [1, 16384] like S7; (2) `fresh_llm`, linear-layer
dimensions of models that are NOT in S2 (Qwen2-7B, Gemma-2-9B, Phi-3-mini, Mistral-Nemo-12B, Yi-6B) with token counts that are not in S2's list.

Disjointness (the point of the set): a shape is rejected if it is in any `suites/v1` suite, or shares two of M, N, K with any TUNE-split shape (the grouping the
cross-validation uses, so no near-duplicate of anything a ranker, router or table was fit on), or exceeds 1200 GFLOP (the sweep's cap, so a Triton screen oracle
exists as for the held-out shapes). The generator is deterministic in the seed; the shapes are chosen before any result on them exists.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MAX_DIM = 16384
MAX_GFLOP = 1200.0
# (name, K = in_features, N = out_features): C[M, N] = X[M, K] @ W[K, N]
LLM_LAYERS = [
    ("qwen2_7b.attn_o", 3584, 3584), ("qwen2_7b.attn_kv", 3584, 512), ("qwen2_7b.mlp_up", 3584, 18944), ("qwen2_7b.mlp_down", 18944, 3584),
    ("gemma2_9b.mlp_up", 3584, 14336), ("gemma2_9b.mlp_down", 14336, 3584), ("gemma2_9b.attn_o", 4096, 3584),
    ("phi3_mini.attn_qkv", 3072, 9216), ("phi3_mini.mlp_up", 3072, 16384), ("phi3_mini.mlp_down", 8192, 3072),
    ("mistral_nemo_12b.mlp_up", 5120, 14336), ("mistral_nemo_12b.mlp_down", 14336, 5120), ("mistral_nemo_12b.attn_kv", 5120, 1024),
    ("yi_6b.mlp_up", 4096, 11008), ("yi_6b.attn_kv", 4096, 512),
]
TOKENS = [3, 24, 96, 300, 1000, 3000, 6000]       # none of S2's {1, 2, 4, 8, 16, 32, 64, 128, 512, 2048, 8192}


def suite_shapes() -> tuple[set, set]:
    """(all shapes of suites/v1, tune-split shapes)."""
    allp, tune = set(), set()
    for f in (ROOT / "v1").glob("*.json"):
        for s in json.loads(f.read_text())["shapes"]:
            t = (s["M"], s["N"], s["K"])
            allp.add(t)
            if s.get("split") == "tune":
                tune.add(t)
    return allp, tune


def shares_two(t, others) -> bool:
    return any(sum(a == b for a, b in zip(t, o)) >= 2 for o in others)


def ok(t, allp, tune) -> bool:
    M, N, K = t
    return (1 <= min(t) and max(t) <= MAX_DIM and 2e-9 * M * N * K <= MAX_GFLOP and t not in allp and not shares_two(t, tune))


def build(seed: int = 465, n_random: int = 30, n_llm: int = 30) -> list[dict]:
    allp, tune = suite_shapes()
    rng = random.Random(seed)
    out, seen = [], set()
    while sum(s["suite"] == "fresh_random" for s in out) < n_random:
        t = tuple(max(1, int(round(math.exp(rng.uniform(0, math.log(MAX_DIM)))))) for _ in range(3))
        if ok(t, allp, tune) and t not in seen:
            seen.add(t)
            out.append({"M": t[0], "N": t[1], "K": t[2], "suite": "fresh_random", "split": "fresh"})
    cands = [(M, N, K, name) for name, K, N in LLM_LAYERS for M in TOKENS if ok((M, N, K), allp, tune)]
    rng.shuffle(cands)
    for M, N, K, name in cands[:n_llm]:
        if (M, N, K) not in seen:
            seen.add((M, N, K))
            out.append({"M": M, "N": N, "K": K, "suite": "fresh_llm", "split": "fresh", "layer": name})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=465)
    ap.add_argument("--n-random", type=int, default=30)
    ap.add_argument("--n-llm", type=int, default=30)
    a = ap.parse_args(argv)
    shapes = build(a.seed, a.n_random, a.n_llm)
    (ROOT / "fresh").mkdir(exist_ok=True)
    path = ROOT / "fresh" / f"fresh{a.seed}.json"
    path.write_text(json.dumps({"suite": "fresh", "seed": a.seed, "max_gflop": MAX_GFLOP, "n": len(shapes), "shapes": shapes}, indent=1))
    by = {}
    for s in shapes:
        by[s["suite"]] = by.get(s["suite"], 0) + 1
    print(f"wrote {path}: {len(shapes)} shapes {by}")


if __name__ == "__main__":
    main()
