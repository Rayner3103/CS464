# Trivial selector baselines on held-out shapes (auto-generated)

Selectors are fit on the tune-split screen (477 shapes x 10840 configs) and scored on 86 held-out shapes (>= 50 us 30, < 50 us 56, heldout_random 55, heldout_region 31). Features: (log2 M, log2 N, log2 K), Euclidean, for nn1 / knn*; the same plus SM-fill and alignment groups (`shape_features`) for nn1f / knnf. Every pick passes the pruning rules for its target shape. `b6` = best single config on the tune screen (`bm32_bn32_bk32_w4_s5_g1_sk1`); `rule` = `analysis.selectors.rule_config`. Regret = t / oracle - 1.

## Screen: regret vs the held-out per-shape best (full pruned space)

| selector | in-sample (tune, grouped LOO) median / p95 / p99 | held-out all median / p95 / p99 | held-out >= 50 us median / p95 / p99 | held-out < 50 us median / p95 / p99 | held-out heldout_random median / p95 / p99 | held-out heldout_region median / p95 / p99 | held-out mean | picks the oracle |
|---|---|---|---|---|---|---|---|---|
| b6 | 68.4% / 343.1% / 1001.6% | 48.1% / 487.0% / 973.1% | 85.0% / 395.7% / 589.9% | 28.9% / 524.6% / 1081.4% | 40.0% / 345.5% / 616.8% | 64.7% / 711.6% / 1171.7% | 113.6% | 0.0% |
| rule | 7.5% / 44.3% / 76.0% | 10.3% / 40.5% / 80.1% | 5.3% / 38.6% / 78.6% | 12.0% / 42.5% / 65.4% | 8.9% / 39.5% / 61.0% | 11.7% / 46.9% / 83.0% | 14.8% | 0.0% |
| nn1 | 4.9% / 52.8% / 118.6% | 5.4% / 131.8% / 279.1% | 2.2% / 195.4% / 414.1% | 8.3% / 107.9% / 201.6% | 7.8% / 151.4% / 353.9% | 4.0% / 116.4% / 163.0% | 30.7% | 3.5% |
| knn5 | 2.8% / 29.2% / 42.6% | 4.7% / 25.4% / 122.2% | 3.6% / 65.4% / 224.7% | 4.8% / 21.5% / 26.1% | 5.0% / 24.9% / 142.3% | 2.1% / 27.5% / 75.2% | 10.8% | 2.3% |
| knnc | 2.5% / 25.6% / 40.5% | 5.3% / 29.3% / 62.6% | 3.6% / 34.0% / 96.7% | 6.3% / 28.2% / 43.5% | 7.1% / 29.8% / 46.1% | 3.5% / 28.8% / 92.7% | 10.7% | 4.7% |
| knnf | 2.1% / 22.5% / 38.6% | 4.1% / 27.3% / 41.7% | 2.2% / 37.4% / 44.1% | 4.7% / 23.2% / 29.4% | 5.0% / 22.4% / 33.8% | 1.3% / 32.3% / 41.7% | 7.8% | 3.5% |
| nn1f | 2.9% / 29.7% / 58.4% | 4.5% / 39.8% / 66.7% | 1.7% / 41.9% / 61.3% | 5.3% / 39.6% / 65.3% | 4.6% / 42.6% / 65.4% | 3.5% / 30.0% / 56.4% | 10.2% | 7.0% |
| knn3 | 3.1% / 30.4% / 73.4% | 5.4% / 43.3% / 306.9% | 4.2% / 194.6% / 423.6% | 6.3% / 39.2% / 57.6% | 8.4% / 53.1% / 371.5% | 3.2% / 33.2% / 79.2% | 18.6% | 0.0% |
| knn10 | 3.0% / 26.5% / 40.6% | 5.0% / 24.6% / 43.8% | 3.7% / 30.3% / 77.4% | 5.6% / 22.8% / 26.0% | 5.8% / 23.0% / 29.7% | 4.6% / 25.7% / 74.4% | 8.5% | 2.3% |

In-sample uses grouped leave-one-out for the neighbour selectors (no tune shape sharing two of M, N, K with the target is a neighbour); the rule was written looking at these tune shapes, so its in-sample number is optimistic. For reference, the tune-fitted 48-config greedy portfolio's held-out screen p95 is in `sweep_portfolio_heldout.md` (an oracle over 48 configs, not a selector).

## Feature fit (tune shapes only)

Grid: fill weight (0, 0.5, 1, 1.5, 2, 3), align weight (0, 0.25, 0.5, 1), k (1, 3, 5, 7, 10) (dims weight 1). Chosen by the lowest capped mean regret under grouped leave-one-out on the tune shapes, before scoring any held-out shape. **knnf: fill weight 1.5, align weight 0.5, k = 3. nn1f: fill weight 0.5, align weight 0.5, k = 1.**

| fill | align | k | in-sample mean | median | p95 | p99 |
|---|---|---|---|---|---|---|
| 1.5 | 0.5 | 3 | 5.9% | 2.1% | 22.5% | 38.6% |
| 1 | 0.25 | 5 | 6.0% | 2.3% | 23.6% | 46.4% |
| 1 | 0.5 | 3 | 6.1% | 2.3% | 23.7% | 39.1% |
| 1 | 0.25 | 7 | 6.2% | 2.3% | 24.1% | 50.2% |
| 0.5 | 0.5 | 5 | 6.3% | 2.5% | 25.8% | 35.9% |
| 0.5 | 0.25 | 7 | 6.3% | 2.5% | 22.7% | 50.6% |
| 0.5 | 0.5 | 3 | 6.4% | 2.3% | 26.3% | 40.3% |
| 1 | 0.5 | 5 | 6.4% | 2.4% | 24.9% | 46.4% |
| 0 | 0 | 1 | 14.4% | 4.6% | 53.5% | 111.5% |
| 0 | 0 | 5 | 7.4% | 2.8% | 29.2% | 46.4% |

The last rows are the dims-only baselines (nn1, knn5) for comparison.

Per-candidate k-NN (knnc), grid dims weight (0, 0.25, 0.5, 1) x k (3, 5, 10, 20) (candidate features log2_waves, wave_eff, log2_ksteps, log2_pad_mn, log2_pad_k, even_k share weight 1), same selection rule. **knnc: dims weight 0.5, k = 10.**

| dims weight | k | in-sample mean | median | p95 | p99 |
|---|---|---|---|---|---|
| 0.5 | 10 | 6.6% | 2.5% | 25.6% | 40.5% |
| 1 | 5 | 6.7% | 2.1% | 27.3% | 39.9% |
| 1 | 10 | 6.7% | 2.7% | 25.4% | 40.0% |
| 1 | 20 | 7.0% | 3.1% | 25.8% | 38.8% |
| 0.5 | 5 | 7.1% | 2.1% | 28.9% | 63.9% |
| 0.25 | 10 | 7.1% | 2.6% | 28.5% | 48.6% |
| 0.5 | 20 | 7.3% | 3.6% | 27.2% | 49.2% |
| 0.25 | 5 | 7.5% | 2.1% | 33.6% | 67.0% |
| 0.25 | 3 | 8.4% | 1.9% | 31.1% | 74.9% |
| 0.25 | 20 | 8.5% | 3.7% | 32.1% | 63.0% |
| 1 | 3 | 9.2% | 2.3% | 33.1% | 68.1% |
| 0.5 | 3 | 9.2% | 2.3% | 30.3% | 70.3% |
| 0 | 20 | 10.8% | 4.1% | 37.9% | 98.9% |
| 0 | 10 | 10.8% | 2.7% | 40.9% | 176.3% |
| 0 | 5 | 12.6% | 2.1% | 52.8% | 192.8% |
| 0 | 3 | 13.1% | 1.9% | 53.0% | 208.7% |

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
| knnc | 2x8901x9643 | heldout_region | 255.0 | `bm16_bn256_bk32_w8_s4_g1_sk1` | 119.7% |
| knnc | 3522x1x647 | heldout_random | 16.4 | `bm128_bn32_bk64_w4_s2_g1_sk8` | 52.6% |
| knnc | 79x15624x2135 | heldout_random | 222.2 | `bm64_bn128_bk16_w4_s4_g8_sk1` | 40.6% |
| knnc | 226x2141x1139 | heldout_random | 41.4 | `bm128_bn64_bk32_w4_s5_g1_sk1` | 36.1% |
| knnc | 342x88x6772 | heldout_region | 31.5 | `bm64_bn64_bk32_w4_s5_g1_sk8` | 29.8% |

## Stage 2 (cold): re-measured, `results_lock1755_selectors3.jsonl`

cold L2, per-launch, adaptive repeats, 3 input placements; all held-out shapes. Per shape: the screen's top-8 tiles, every selector pick, B1, B3 (cuBLASLt heuristic top-16) and B3+ (B3 plus the deep cuBLASLt search), all in one session. 86 shapes, 2526 rows, 0 invalid. Oracle = fastest Triton row.

Regret vs the per-shape Triton oracle, median / p95 / p99:

| selector | all (n=86) | >= 50 us (n=30) | < 50 us (n=56) |
|---|---|---|---|
| b6 | 48.7% / 485.8% / 947.5% | 78.5% / 393.8% / 586.0% | 24.5% / 522.2% / 1047.2% |
| nn1 | 4.1% / 129.4% / 276.0% | 1.7% / 194.2% / 401.9% | 5.5% / 105.8% / 197.9% |
| knn5 | 2.7% / 22.6% / 122.2% | 3.9% / 65.4% / 222.7% | 2.2% / 18.8% / 22.4% |
| rule | 7.9% / 37.0% / 67.9% | 4.6% / 37.4% / 78.6% | 8.7% / 36.1% / 50.5% |
| nn1f | 2.5% / 38.9% / 61.9% | 1.3% / 40.4% / 57.3% | 3.8% / 37.3% / 60.3% |
| knnf | 1.8% / 22.9% / 41.3% | 1.9% / 36.6% / 45.1% | 1.7% / 19.0% / 25.1% |
| knnc | 4.4% / 26.7% / 59.2% | 3.8% / 31.9% / 97.6% | 4.9% / 25.6% / 41.0% |

Speedup vs B3+ (t_B3+ / t_pick; > 1 = the pick is faster), median [p10, p90], and shapes ahead by > 5% / within 2% / behind by > 5%:

| selector | all | >= 50 us | < 50 us |
|---|---|---|---|
| Triton oracle | 1.004 [0.83, 1.14] · 30/15/30 | 0.964 [0.83, 1.01] · 2/12/14 | 1.050 [0.83, 1.15] · 28/3/16 |
| b6 | 0.658 [0.24, 1.02] · 5/10/64 | 0.562 [0.24, 0.80] · 0/1/29 | 0.842 [0.25, 1.03] · 5/9/35 |
| nn1 | 0.953 [0.51, 1.10] · 22/14/43 | 0.903 [0.46, 1.01] · 2/8/19 | 0.993 [0.52, 1.11] · 20/6/24 |
| knn5 | 0.968 [0.71, 1.11] · 26/12/41 | 0.894 [0.64, 1.01] · 2/7/19 | 1.015 [0.74, 1.12] · 24/5/22 |
| rule | 0.929 [0.67, 1.09] · 16/9/50 | 0.880 [0.67, 0.98] · 2/5/19 | 0.936 [0.67, 1.09] · 14/4/31 |
| nn1f | 0.957 [0.68, 1.09] · 19/15/42 | 0.926 [0.68, 1.01] · 1/9/18 | 0.989 [0.69, 1.11] · 18/6/24 |
| knnf | 0.967 [0.70, 1.11] · 21/14/40 | 0.916 [0.68, 1.01] · 2/7/19 | 0.985 [0.72, 1.12] · 19/7/21 |
| knnc | 0.952 [0.71, 1.10] · 18/9/43 | 0.899 [0.72, 1.01] · 2/6/20 | 0.994 [0.71, 1.11] · 16/3/23 |

B3+ vs B1 on these shapes: median 1.045 [1.00, 1.35].

## Stage 2 (graph): re-measured, `results_lock1755_selectors3_graph.jsonl`

hot L2, CUDA-graph amortized (50 launches); held-out shapes with screen oracle < 50 us. Per shape: the screen's top-8 tiles, every selector pick, B1, B3 (cuBLASLt heuristic top-16) and B3+ (B3 plus the deep cuBLASLt search), all in one session. 56 shapes, 1670 rows, 0 invalid. Oracle = fastest Triton row.

Regret vs the per-shape Triton oracle, median / p95 / p99:

| selector | all (n=56) | B3+ < 5 us (n=33) | B3+ 5-15 us (n=10) | B3+ >= 15 us (n=13) |
|---|---|---|---|---|
| b6 | 24.5% / 456.0% / 1235.5% | 14.1% / 227.8% / 980.3% | 58.5% / 338.5% / 378.0% | 88.4% / 796.7% / 1083.2% |
| nn1 | 9.2% / 115.9% / 246.9% | 5.8% / 29.3% / 56.1% | 17.9% / 92.7% / 105.7% | 36.0% / 244.0% / 271.3% |
| knn5 | 3.7% / 22.3% / 36.4% | 1.6% / 14.3% / 29.5% | 4.0% / 16.0% / 17.2% | 4.8% / 33.8% / 36.8% |
| rule | 12.7% / 58.5% / 77.1% | 12.9% / 63.7% / 83.0% | 13.5% / 37.8% / 44.2% | 9.3% / 35.9% / 38.8% |
| nn1f | 8.4% / 39.9% / 75.2% | 3.0% / 23.0% / 29.7% | 21.4% / 55.2% / 69.3% | 11.9% / 61.3% / 74.7% |
| knnf | 4.7% / 26.2% / 36.7% | 2.9% / 15.5% / 31.2% | 6.7% / 23.0% / 26.6% | 6.6% / 29.9% / 34.9% |
| knnc | 7.1% / 36.0% / 46.8% | 8.9% / 24.2% / 34.0% | 6.8% / 38.4% / 54.1% | 5.0% / 36.4% / 37.3% |

Speedup vs B3+ (t_B3+ / t_pick; > 1 = the pick is faster), median [p10, p90], and shapes ahead by > 5% / within 2% / behind by > 5%:

| selector | all | B3+ < 5 us | B3+ 5-15 us | B3+ >= 15 us |
|---|---|---|---|---|
| Triton oracle | 1.016 [0.77, 1.18] · 24/5/19 | 1.027 [0.83, 1.18] · 15/2/10 | 1.075 [0.73, 1.28] · 6/1/3 | 0.960 [0.81, 1.15] · 3/2/6 |
| b6 | 0.801 [0.22, 1.02] · 0/7/40 | 0.948 [0.29, 1.03] · 0/6/18 | 0.707 [0.23, 0.92] · 0/1/9 | 0.504 [0.13, 0.64] · 0/0/13 |
| nn1 | 0.899 [0.47, 1.13] · 17/4/33 | 0.948 [0.75, 1.12] · 11/3/17 | 0.970 [0.50, 1.19] · 4/1/5 | 0.732 [0.31, 1.09] · 2/0/11 |
| knn5 | 0.972 [0.65, 1.17] · 20/3/25 | 0.981 [0.74, 1.14] · 13/2/12 | 1.013 [0.64, 1.22] · 5/0/4 | 0.883 [0.67, 1.13] · 2/1/9 |
| rule | 0.881 [0.53, 1.11] · 13/2/36 | 0.879 [0.53, 1.12] · 8/0/22 | 0.916 [0.65, 1.14] · 3/0/6 | 0.814 [0.66, 1.06] · 2/2/8 |
| nn1f | 0.904 [0.59, 1.11] · 14/7/32 | 0.948 [0.76, 1.11] · 9/4/17 | 0.968 [0.55, 1.11] · 3/2/5 | 0.823 [0.59, 1.05] · 2/1/10 |
| knnf | 0.955 [0.69, 1.15] · 16/3/28 | 0.978 [0.75, 1.14] · 11/2/16 | 1.027 [0.62, 1.20] · 4/0/4 | 0.925 [0.62, 1.04] · 1/1/8 |
| knnc | 0.940 [0.66, 1.14] · 15/3/29 | 0.963 [0.75, 1.12] · 9/1/16 | 0.946 [0.64, 1.22] · 4/0/5 | 0.931 [0.61, 1.06] · 2/2/8 |

B3+ vs B1 on these shapes: median 1.075 [1.00, 1.42].
