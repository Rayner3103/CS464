"""Generate the versioned shape suites (plan 5.2) and the train/held-out split (plan 5.1).

    python -m suites.generate            # writes suites/v1/*.json

Deterministic: fixed seed, no dependence on the GPU except S6, which needs the SM count. S6 records the
SM count it was built for, and the suite is regenerated per GPU.

Held-out policy (fixed before any results exist):
  * Region holdout: every shape with 9000 <= K <= 12000 (contains LLaMA-7B's 11008 FFN dim) or
    300 <= M <= 450. This tests extrapolation into unseen K and M ranges.
  * Point holdout: 30% of S7 (random log-uniform), chosen by seeded RNG.
  * Everything else is "tune".
The edge and noise suites are for validation and methodology, not training, so their split is "aux".
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

VERSION = "v1"
SEED = 464
MAX_DIM = 16384
OUT_DIR = Path(__file__).resolve().parent / VERSION

# (name, K=in_features, N=out_features) for linear layers of common LLMs. C[M,N] = X[M,K] @ W[K,N].
LLM_LAYERS = [
    ("llama2_7b.attn_qkv", 4096, 12288), ("llama2_7b.attn_o", 4096, 4096),
    ("llama2_7b.mlp_gate_up", 4096, 11008), ("llama2_7b.mlp_down", 11008, 4096),
    ("llama2_13b.attn_o", 5120, 5120), ("llama2_13b.mlp_up", 5120, 13824), ("llama2_13b.mlp_down", 13824, 5120),
    ("llama3_8b.attn_kv", 4096, 1024), ("llama3_8b.mlp_up", 4096, 14336), ("llama3_8b.mlp_down", 14336, 4096),
    ("llama2_70b.attn_o", 8192, 8192), ("llama2_70b.mlp_up", 8192, 28672), ("llama2_70b.mlp_down", 28672, 8192),
]
TOKEN_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128, 512, 2048, 8192]


def _shape(M, N, K, suite, **meta):
    return {"M": int(M), "N": int(N), "K": int(K), "suite": suite, **meta}


def s1_square():
    pow2 = [2**i for i in range(6, 15)]  # 64 .. 16384
    off = [1000, 1023, 1025, 2000, 2047, 2049, 3000, 4095, 4097, 6000, 8191, 8193, 12000]
    return [_shape(n, n, n, "S1") for n in sorted(set(pow2 + off))]


def s2_llm():
    out = []
    for name, K, N in LLM_LAYERS:
        for M in TOKEN_COUNTS:
            # 28672 exceeds MAX_DIM: these layers are kept (they are real) but tagged out_of_range.
            out.append(_shape(M, N, K, "S2", layer=name, out_of_range=max(M, N, K) > MAX_DIM))
    return out


def s3_skinny():
    out = []
    for small in [1, 2, 4, 8, 16, 32, 64]:
        for big1, big2 in [(1024, 1024), (4096, 4096), (4096, 16384), (16384, 4096), (8192, 8192)]:
            out.append(_shape(small, big1, big2, "S3", orientation="small_M"))   # GEMV-like (decode)
            out.append(_shape(big1, small, big2, "S3", orientation="small_N"))
    return out


def s4_large_k():
    return [_shape(m, n, k, "S4") for m in [16, 64, 128, 256] for n in [16, 64, 128, 256]
            for k in [4096, 8192, 16384] if m <= n]


def s5_unaligned(rng: random.Random):
    primes = [7, 31, 127, 251, 509, 1021, 2039, 4093, 8191]
    odd = [33, 100, 257, 513, 769, 1000, 1537, 3001, 5003]
    out = []
    for _ in range(40):
        M, N, K = (rng.choice(primes + odd) for _ in range(3))
        out.append(_shape(M, N, K, "S5"))
    # Alignment ladder: same size, decreasing power-of-two divisibility.
    for d in [4096, 4095, 4094, 4092, 4088, 4080, 4064]:
        out.append(_shape(d, d, d, "S5", ladder=True))
    return out


def s6_wave_boundary(sm_count: int):
    """Shapes whose 128x128 tile count lands at w*SMs - 1, w*SMs, w*SMs + 1 (one-tile-per-SM occupancy).

    N is fixed so the tile count changes one tile-row at a time; the boundary is placed by tile rows
    and then fine-tuned with N. Also a 64x64-tile variant at 2 CTAs/SM.
    """
    out = []
    for tile, ctas_per_sm in [(128, 1), (64, 2)]:
        slots = sm_count * ctas_per_sm
        for tn in [8, 17]:  # tiles along N
            N = tn * tile
            for waves in [1, 2, 3, 4]:
                target = waves * slots
                for delta in (-tn, 0, tn):  # one tile-row below / at / above
                    tiles = max(tn, target + delta)
                    tm = math.ceil(tiles / tn)
                    M = tm * tile
                    if M <= MAX_DIM:
                        for K in [1024, 4096]:
                            out.append(_shape(M, N, K, "S6", tile=tile, waves_target=waves, tiles=tm * tn,
                                              wave_slots=slots))
    return out


def s7_random(rng: random.Random, n: int = 200):
    out = []
    for _ in range(n):
        M, N, K = (int(round(math.exp(rng.uniform(0, math.log(MAX_DIM))))) for _ in range(3))
        out.append(_shape(max(M, 1), max(N, 1), max(K, 1), "S7"))
    return out


def edge_suite():
    """Correctness-only suite (plan 3.4): degenerate dims, primes, non-multiples, K < BLOCK_K."""
    dims = [1, 2, 3, 7, 15, 16, 17, 31, 33, 63, 65, 127, 129, 255, 257]
    shapes = []
    for M in [1, 17, 64, 129, 257]:
        for N in [1, 15, 64, 127, 257]:
            for K in [1, 3, 16, 33, 255, 1031]:
                shapes.append(_shape(M, N, K, "edge"))
    shapes += [_shape(d, d, d, "edge") for d in dims]
    shapes += [_shape(1, 1, 16384, "edge"), _shape(1, 4096, 4096, "edge"), _shape(4093, 1, 4093, "edge"),
               _shape(2048, 2048, 8, "edge")]
    return shapes


def noise_set(all_shapes, rng: random.Random, n: int = 30):
    """~30 representative shapes for the noise-floor study, stratified across suites and runtime scales."""
    pick = []
    by_suite = {}
    for s in all_shapes:
        if not s.get("out_of_range"):
            by_suite.setdefault(s["suite"], []).append(s)
    per = {"S1": 6, "S2": 8, "S3": 4, "S4": 3, "S5": 3, "S6": 2, "S7": 4}
    for suite, k in per.items():
        pool = sorted(by_suite[suite], key=lambda s: s["M"] * s["N"] * s["K"])
        # Evenly spaced through the FLOP-sorted pool, so tiny and huge kernels are both represented.
        idx = [round(i * (len(pool) - 1) / max(k - 1, 1)) for i in range(k)]
        pick += [dict(pool[i], suite_origin=suite, suite="noise") for i in sorted(set(idx))]
    return pick[:n]


def assign_split(s, rng: random.Random):
    if s["suite"] in ("edge", "noise"):
        return "aux"
    if 9000 <= s["K"] <= 12000 or 300 <= s["M"] <= 450:
        return "heldout_region"
    if s["suite"] == "S7" and rng.random() < 0.3:
        return "heldout_random"
    return "tune"


def build(sm_count: int) -> dict[str, list[dict]]:
    rng = random.Random(SEED)
    suites = {
        "S1": s1_square(), "S2": s2_llm(), "S3": s3_skinny(), "S4": s4_large_k(),
        "S5": s5_unaligned(rng), "S6": s6_wave_boundary(sm_count), "S7": s7_random(rng),
    }
    for name, shapes in suites.items():  # dedupe within a suite, keep first occurrence
        seen, uniq = set(), []
        for s in shapes:
            if (s["M"], s["N"], s["K"]) not in seen:
                seen.add((s["M"], s["N"], s["K"]))
                uniq.append(s)
        suites[name] = uniq
    split_rng = random.Random(SEED + 1)
    for shapes in suites.values():
        for s in shapes:
            s["split"] = assign_split(s, split_rng)
    all_main = [s for v in suites.values() for s in v]
    suites["edge"] = [dict(s, split="aux") for s in edge_suite()]
    suites["noise"] = [dict(s, split="aux") for s in noise_set(all_main, random.Random(SEED + 2))]
    return suites


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sm-count", type=int, default=None, help="default: query GPU 0")
    args = ap.parse_args()
    sm = args.sm_count
    if sm is None:
        import torch
        sm = torch.cuda.get_device_properties(0).multi_processor_count
    suites = build(sm)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, shapes in suites.items():
        doc = {"suite": name, "version": VERSION, "seed": SEED, "max_dim": MAX_DIM,
               "built_for_sm_count": sm if name == "S6" else None, "n": len(shapes), "shapes": shapes}
        (OUT_DIR / f"{name}.json").write_text(json.dumps(doc, indent=1))
        splits = {}
        for s in shapes:
            splits[s["split"]] = splits.get(s["split"], 0) + 1
        print(f"{name:6s} n={len(shapes):4d} {splits}")


if __name__ == "__main__":
    main()
