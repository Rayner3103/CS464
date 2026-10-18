# Trivial selector baselines on held-out shapes (auto-generated)

Selectors are fit on the tune-split screen (477 shapes x 10840 configs) and scored on 86 held-out shapes (>= 50 us 30, < 50 us 56, heldout_random 55, heldout_region 31). Features: (log2 M, log2 N, log2 K), Euclidean, for nn1 / knn*; the same plus SM-fill and alignment groups (`shape_features`) for nn1f / knnf. Every pick passes the pruning rules for its target shape. `b6` = best single config on the tune screen (`bm32_bn32_bk32_w4_s5_g1_sk1`); `rule` = `analysis.selectors.rule_config`. Regret = t / oracle - 1.

## Screen: regret vs the held-out per-shape best (full pruned space)

| selector | in-sample (tune, grouped LOO) median / p95 | held-out all median / p95 | held-out >= 50 us median / p95 | held-out < 50 us median / p95 | held-out heldout_random median / p95 | held-out heldout_region median / p95 | held-out mean | picks the oracle |
|---|---|---|---|---|---|---|---|---|
| b6 | 68.4% / 343.1% | 48.1% / 487.0% | 85.0% / 395.7% | 28.9% / 524.6% | 40.0% / 345.5% | 64.7% / 711.6% | 113.6% | 0.0% |
| rule | 7.5% / 44.3% | 10.3% / 40.5% | 5.3% / 38.6% | 12.0% / 42.5% | 8.9% / 39.5% | 11.7% / 46.9% | 14.8% | 0.0% |
| nn1 | 4.9% / 52.8% | 5.4% / 131.8% | 2.2% / 195.4% | 8.3% / 107.9% | 7.8% / 151.4% | 4.0% / 116.4% | 30.7% | 3.5% |
| knn5 | 2.8% / 29.2% | 4.7% / 25.4% | 3.6% / 65.4% | 4.8% / 21.5% | 5.0% / 24.9% | 2.1% / 27.5% | 10.8% | 2.3% |
| knnf | 2.1% / 22.5% | 4.1% / 27.3% | 2.2% / 37.4% | 4.7% / 23.2% | 5.0% / 22.4% | 1.3% / 32.3% | 7.8% | 3.5% |
| nn1f | 2.9% / 29.7% | 4.5% / 39.8% | 1.7% / 41.9% | 5.3% / 39.6% | 4.6% / 42.6% | 3.5% / 30.0% | 10.2% | 7.0% |
| knn3 | 3.1% / 30.4% | 5.4% / 43.3% | 4.2% / 194.6% | 6.3% / 39.2% | 8.4% / 53.1% | 3.2% / 33.2% | 18.6% | 0.0% |
| knn10 | 3.0% / 26.5% | 5.0% / 24.6% | 3.7% / 30.3% | 5.6% / 22.8% | 5.8% / 23.0% | 4.6% / 25.7% | 8.5% | 2.3% |

In-sample uses grouped leave-one-out for the neighbour selectors (no tune shape sharing two of M, N, K with the target is a neighbour); the rule was written looking at these tune shapes, so its in-sample number is optimistic. For reference, the tune-fitted 48-config greedy portfolio's held-out screen p95 is in `sweep_portfolio_heldout.md` (an oracle over 48 configs, not a selector).

## Feature fit (tune shapes only)

Grid: fill weight (0, 0.5, 1, 1.5, 2, 3), align weight (0, 0.25, 0.5, 1), k (1, 3, 5, 7, 10) (dims weight 1). Chosen by the lowest capped mean regret under grouped leave-one-out on the tune shapes, before scoring any held-out shape. **knnf: fill weight 1.5, align weight 0.5, k = 3. nn1f: fill weight 0.5, align weight 0.5, k = 1.**

| fill | align | k | in-sample mean | median | p95 |
|---|---|---|---|---|---|
| 1.5 | 0.5 | 3 | 5.9% | 2.1% | 22.5% |
| 1 | 0.25 | 5 | 6.0% | 2.3% | 23.6% |
| 1 | 0.5 | 3 | 6.1% | 2.3% | 23.7% |
| 1 | 0.25 | 7 | 6.2% | 2.3% | 24.1% |
| 0.5 | 0.5 | 5 | 6.3% | 2.5% | 25.8% |
| 0.5 | 0.25 | 7 | 6.3% | 2.5% | 22.7% |
| 0.5 | 0.5 | 3 | 6.4% | 2.3% | 26.3% |
| 1 | 0.5 | 5 | 6.4% | 2.4% | 24.9% |
| 0 | 0 | 1 | 14.4% | 4.6% | 53.5% |
| 0 | 0 | 5 | 7.4% | 2.8% | 29.2% |

The last rows are the dims-only baselines (nn1, knn5) for comparison.

Worst held-out shapes (screen):

| selector | shape | split | oracle us | pick | regret |
|---|---|---|---|---|---|
| nn1 | 14244x16x1257 | heldout_random | 70.7 | `bm128_bn32_bk128_w4_s2_g1_sk1` | 484.1% |
| nn1 | 4x889x14165 | heldout_random | 50.2 | `bm16_bn64_bk128_w4_s4_g1_sk1` | 243.0% |
| nn1 | 15x3363x2632 | heldout_random | 38.8 | `bm16_bn128_bk32_w4_s4_g1_sk1` | 235.2% |
| nn1 | 384x2x10369 | heldout_region | 26.6 | `bm32_bn32_bk128_w4_s2_g1_sk2` | 174.1% |
| nn1 | 37x922x10110 | heldout_region | 52.2 | `bm32_bn32_bk128_w4_s5_g1_sk1` | 137.3% |
| knn5 | 28x3030x4012 | heldout_random | 50.2 | `bm32_bn128_bk32_w4_s5_g1_sk1` | 277.7% |
| knn5 | 2x8901x9643 | heldout_region | 255.0 | `bm16_bn128_bk32_w4_s4_g1_sk1` | 94.8% |
| knn5 | 68x6420x10359 | heldout_region | 412.7 | `bm32_bn128_bk32_w4_s3_g8_sk2` | 29.5% |
| knn5 | 49x59x920 | heldout_random | 9.2 | `bm32_bn32_bk16_w4_s5_g1_sk8` | 27.0% |
| knn5 | 37x922x10110 | heldout_region | 52.2 | `bm32_bn64_bk32_w4_s4_g1_sk8` | 25.4% |
| rule | 2x8901x9643 | heldout_region | 255.0 | `bm16_bn128_bk32_w4_s4_g1_sk1` | 94.8% |
| rule | 49x59x920 | heldout_random | 9.2 | `bm32_bn64_bk128_w4_s3_g8_sk2` | 77.5% |
| rule | 378x147x306 | heldout_region | 9.2 | `bm64_bn64_bk64_w4_s3_g8_sk2` | 55.6% |
| rule | 12x56x836 | heldout_random | 8.2 | `bm16_bn64_bk128_w4_s3_g1_sk2` | 47.0% |
| rule | 226x2141x1139 | heldout_random | 41.4 | `bm64_bn128_bk32_w4_s4_g8_sk1` | 41.0% |
| knnf | 68x6420x10359 | heldout_region | 412.7 | `bm32_bn64_bk16_w4_s2_g1_sk2` | 45.4% |
| knnf | 79x15624x2135 | heldout_random | 222.2 | `bm32_bn64_bk16_w4_s3_g1_sk2` | 41.0% |
| knnf | 443x4723x393 | heldout_region | 52.2 | `bm128_bn64_bk32_w4_s3_g4_sk1` | 33.0% |
| knnf | 384x2x10369 | heldout_region | 26.6 | `bm32_bn32_bk64_w4_s3_g16_sk8` | 31.5% |
| knnf | 3522x1x647 | heldout_random | 16.4 | `bm64_bn32_bk64_w8_s3_g4_sk4` | 27.6% |

## Stage 2 (cold): re-measured, `results_lock1755_selectors2.jsonl`

cold L2, per-launch, adaptive repeats, 3 input placements; all held-out shapes. Per shape: the screen's top-8 tiles, every selector pick, B1, B3 (cuBLASLt heuristic top-16) and B3+ (B3 plus the deep cuBLASLt search), all in one session. 86 shapes, 2459 rows, 0 invalid. Oracle = fastest Triton row.

Regret vs the per-shape Triton oracle, median / p95:

| selector | all (n=86) | >= 50 us (n=30) | < 50 us (n=56) |
|---|---|---|---|
| b6 | 46.9% / 481.7% | 78.0% / 391.8% | 26.8% / 517.7% |
| nn1 | 3.8% / 128.5% | 1.5% / 193.9% | 4.7% / 105.9% |
| knn5 | 2.9% / 23.7% | 3.7% / 65.2% | 2.4% / 18.8% |
| rule | 7.4% / 37.1% | 4.7% / 37.0% | 7.9% / 37.1% |
| nn1f | 3.5% / 40.0% | 1.5% / 39.6% | 3.8% / 38.1% |
| knnf | 2.7% / 22.9% | 1.9% / 36.6% | 2.9% / 17.9% |

Speedup vs B3+ (t_B3+ / t_pick; > 1 = the pick is faster), median [p10, p90], and shapes ahead by > 5% / within 2% / behind by > 5%:

| selector | all | >= 50 us | < 50 us |
|---|---|---|---|
| Triton oracle | 1.005 [0.83, 1.14] · 27/13/31 | 0.964 [0.83, 1.02] · 2/11/14 | 1.042 [0.83, 1.15] · 25/2/17 |
| b6 | 0.656 [0.24, 1.02] · 2/4/65 | 0.563 [0.23, 0.80] · 0/0/29 | 0.828 [0.25, 1.03] · 2/4/36 |
| nn1 | 0.942 [0.51, 1.11] · 18/14/46 | 0.904 [0.46, 1.01] · 2/7/19 | 0.962 [0.52, 1.11] · 16/7/27 |
| knn5 | 0.967 [0.70, 1.13] · 21/16/41 | 0.896 [0.64, 1.01] · 2/7/19 | 1.014 [0.74, 1.14] · 19/9/22 |
| rule | 0.918 [0.67, 1.09] · 15/8/49 | 0.879 [0.67, 0.98] · 2/4/19 | 0.930 [0.67, 1.10] · 13/4/30 |
| nn1f | 0.951 [0.69, 1.10] · 15/14/44 | 0.927 [0.68, 1.01] · 1/8/18 | 0.970 [0.70, 1.11] · 14/6/26 |
| knnf | 0.958 [0.70, 1.12] · 19/13/42 | 0.922 [0.68, 1.01] · 2/7/20 | 0.994 [0.72, 1.13] · 17/6/22 |

B3+ vs B1 on these shapes: median 1.047 [1.00, 1.35].

## Stage 2 (graph): re-measured, `results_lock1755_selectors2_graph.jsonl`

hot L2, CUDA-graph amortized (50 launches); held-out shapes with screen oracle < 50 us. Per shape: the screen's top-8 tiles, every selector pick, B1, B3 (cuBLASLt heuristic top-16) and B3+ (B3 plus the deep cuBLASLt search), all in one session. 56 shapes, 1625 rows, 0 invalid. Oracle = fastest Triton row.

Regret vs the per-shape Triton oracle, median / p95:

| selector | all (n=56) | B3+ < 5 us (n=33) | B3+ 5-15 us (n=10) | B3+ >= 15 us (n=13) |
|---|---|---|---|---|
| b6 | 24.6% / 459.3% | 14.2% / 226.2% | 58.4% / 337.0% | 89.8% / 791.6% |
| nn1 | 9.3% / 113.2% | 5.8% / 29.0% | 17.9% / 90.8% | 33.8% / 242.6% |
| knn5 | 3.5% / 20.9% | 1.8% / 14.3% | 4.1% / 15.9% | 5.2% / 31.2% |
| rule | 12.9% / 58.7% | 13.3% / 63.2% | 13.8% / 37.7% | 9.1% / 36.1% |
| nn1f | 8.2% / 41.2% | 3.1% / 23.1% | 20.2% / 55.8% | 12.3% / 61.9% |
| knnf | 3.8% / 26.7% | 2.8% / 15.4% | 6.3% / 23.4% | 7.0% / 30.2% |

Speedup vs B3+ (t_B3+ / t_pick; > 1 = the pick is faster), median [p10, p90], and shapes ahead by > 5% / within 2% / behind by > 5%:

| selector | all | B3+ < 5 us | B3+ 5-15 us | B3+ >= 15 us |
|---|---|---|---|---|
| Triton oracle | 1.018 [0.77, 1.18] · 24/5/20 | 1.027 [0.83, 1.17] · 15/2/11 | 1.075 [0.73, 1.28] · 6/1/3 | 0.962 [0.81, 1.16] · 3/2/6 |
| b6 | 0.800 [0.22, 1.02] · 0/6/40 | 0.947 [0.29, 1.03] · 0/5/18 | 0.710 [0.23, 0.92] · 0/1/9 | 0.502 [0.13, 0.64] · 0/0/13 |
| nn1 | 0.898 [0.46, 1.13] · 17/4/33 | 0.942 [0.75, 1.12] · 11/3/17 | 0.970 [0.50, 1.19] · 4/1/5 | 0.757 [0.31, 1.09] · 2/0/11 |
| knn5 | 0.975 [0.65, 1.17] · 20/5/25 | 0.984 [0.75, 1.14] · 13/3/12 | 1.015 [0.65, 1.22] · 5/0/4 | 0.904 [0.66, 1.13] · 2/2/9 |
| rule | 0.879 [0.53, 1.12] · 13/0/35 | 0.876 [0.53, 1.11] · 8/0/21 | 0.916 [0.65, 1.13] · 3/0/6 | 0.802 [0.66, 1.07] · 2/0/8 |
| nn1f | 0.905 [0.58, 1.11] · 14/7/32 | 0.942 [0.76, 1.11] · 9/4/17 | 0.967 [0.55, 1.12] · 3/2/5 | 0.818 [0.59, 1.07] · 2/1/10 |
| knnf | 0.952 [0.68, 1.15] · 18/4/28 | 0.979 [0.75, 1.14] · 12/2/16 | 1.029 [0.62, 1.20] · 4/0/4 | 0.924 [0.61, 1.05] · 2/2/8 |

B3+ vs B1 on these shapes: median 1.077 [1.00, 1.43].
