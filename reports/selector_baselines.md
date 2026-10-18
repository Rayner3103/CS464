# Trivial selector baselines on held-out shapes (auto-generated)

Selectors are fit on the tune-split screen (477 shapes x 10840 configs) and scored on 86 held-out shapes (>= 50 us 30, < 50 us 56, heldout_random 55, heldout_region 31). Features for the neighbour selectors: (log2 M, log2 N, log2 K), Euclidean. Every pick passes the pruning rules for its target shape. `b6` = best single config on the tune screen (`bm32_bn32_bk32_w4_s5_g1_sk1`); `rule` = `analysis.selectors.rule_config`. Regret = t / oracle - 1.

## Screen: regret vs the held-out per-shape best (full pruned space)

| selector | in-sample (tune, LOO) median / p95 | held-out all median / p95 | held-out >= 50 us median / p95 | held-out < 50 us median / p95 | held-out heldout_random median / p95 | held-out heldout_region median / p95 | held-out mean | picks the oracle |
|---|---|---|---|---|---|---|---|---|
| b6 | 68.4% / 343.1% | 48.1% / 487.0% | 85.0% / 395.7% | 28.9% / 524.6% | 40.0% / 345.5% | 64.7% / 711.6% | 113.6% | 0.0% |
| rule | 7.5% / 44.3% | 10.3% / 40.5% | 5.3% / 38.6% | 12.0% / 42.5% | 8.9% / 39.5% | 11.7% / 46.9% | 14.8% | 0.0% |
| nn1 | 4.1% / 59.8% | 5.4% / 131.8% | 2.2% / 195.4% | 8.3% / 107.9% | 7.8% / 151.4% | 4.0% / 116.4% | 30.7% | 3.5% |
| knn5 | 2.3% / 23.2% | 4.7% / 25.4% | 3.6% / 65.4% | 4.8% / 21.5% | 5.0% / 24.9% | 2.1% / 27.5% | 10.8% | 2.3% |
| knn3 | 2.4% / 26.7% | 5.4% / 43.3% | 4.2% / 194.6% | 6.3% / 39.2% | 8.4% / 53.1% | 3.2% / 33.2% | 18.6% | 0.0% |
| knn10 | 2.5% / 24.3% | 5.0% / 24.6% | 3.7% / 30.3% | 5.6% / 22.8% | 5.8% / 23.0% | 4.6% / 25.7% | 8.5% | 2.3% |

In-sample uses leave-one-out for the neighbour selectors; the rule was written looking at these tune shapes, so its in-sample number is optimistic. For reference, the tune-fitted 48-config greedy portfolio's held-out screen p95 is in `sweep_portfolio_heldout.md` (an oracle over 48 configs, not a selector).

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

## Stage 2 (cold): re-measured, `results_lock1755_selectors.jsonl`

cold L2, per-launch, adaptive repeats, 3 input placements; all held-out shapes. Per shape: the screen's top-8 tiles, every selector pick, B1, B3 (cuBLASLt heuristic top-16) and B3+ (B3 plus the deep cuBLASLt search), all in one session. 86 shapes, 2375 rows, 0 invalid. Oracle = fastest Triton row.

Regret vs the per-shape Triton oracle, median / p95:

| selector | all (n=86) | >= 50 us (n=29) | < 50 us (n=57) |
|---|---|---|---|
| b6 | 48.8% / 483.9% | 74.8% / 327.3% | 26.5% / 590.4% |
| nn1 | 3.6% / 129.2% | 1.1% / 123.0% | 4.2% / 122.4% |
| knn5 | 2.1% / 23.7% | 3.1% / 69.0% | 1.9% / 16.7% |
| rule | 7.4% / 37.4% | 4.7% / 37.9% | 8.2% / 36.5% |

Speedup vs B3+ (t_B3+ / t_pick; > 1 = the pick is faster), median [p10, p90], and shapes ahead by > 5% / within 2% / behind by > 5%:

| selector | all | >= 50 us | < 50 us |
|---|---|---|---|
| Triton oracle | 1.001 [0.83, 1.14] · 26/18/33 | 0.954 [0.83, 1.02] · 2/11/14 | 1.033 [0.83, 1.16] · 24/7/19 |
| b6 | 0.648 [0.24, 1.02] · 3/7/67 | 0.566 [0.27, 0.77] · 0/1/28 | 0.821 [0.22, 1.03] · 3/6/39 |
| nn1 | 0.941 [0.52, 1.11] · 17/15/47 | 0.920 [0.51, 1.01] · 2/8/18 | 0.952 [0.52, 1.12] · 15/7/29 |
| knn5 | 0.962 [0.71, 1.12] · 22/15/42 | 0.897 [0.64, 1.01] · 2/7/18 | 1.000 [0.74, 1.14] · 20/8/24 |
| rule | 0.934 [0.68, 1.10] · 14/8/50 | 0.877 [0.67, 0.98] · 2/4/18 | 0.940 [0.68, 1.11] · 12/4/32 |

B3+ vs B1 on these shapes: median 1.042 [1.00, 1.34].

## Stage 2 (graph): re-measured, `results_lock1755_selectors_graph.jsonl`

hot L2, CUDA-graph amortized (50 launches); held-out shapes with screen oracle < 50 us. Per shape: the screen's top-8 tiles, every selector pick, B1, B3 (cuBLASLt heuristic top-16) and B3+ (B3 plus the deep cuBLASLt search), all in one session. 56 shapes, 1567 rows, 0 invalid. Oracle = fastest Triton row.

Regret vs the per-shape Triton oracle, median / p95:

| selector | all (n=56) | B3+ < 5 us (n=33) | B3+ 5-15 us (n=10) | B3+ >= 15 us (n=13) |
|---|---|---|---|---|
| b6 | 24.7% / 455.0% | 14.1% / 224.8% | 58.3% / 339.9% | 89.4% / 794.4% |
| nn1 | 9.2% / 115.7% | 5.8% / 29.1% | 17.8% / 92.8% | 34.7% / 244.5% |
| knn5 | 3.6% / 21.7% | 2.2% / 14.3% | 4.3% / 15.9% | 5.3% / 33.8% |
| rule | 13.0% / 59.2% | 13.5% / 62.8% | 13.5% / 37.6% | 8.9% / 35.6% |

Speedup vs B3+ (t_B3+ / t_pick; > 1 = the pick is faster), median [p10, p90], and shapes ahead by > 5% / within 2% / behind by > 5%:

| selector | all | B3+ < 5 us | B3+ 5-15 us | B3+ >= 15 us |
|---|---|---|---|---|
| Triton oracle | 1.017 [0.77, 1.18] · 24/5/20 | 1.025 [0.83, 1.18] · 15/2/11 | 1.075 [0.73, 1.28] · 6/1/3 | 0.957 [0.81, 1.14] · 3/2/6 |
| b6 | 0.800 [0.22, 1.02] · 0/6/39 | 0.948 [0.29, 1.03] · 0/5/17 | 0.706 [0.23, 0.92] · 0/1/9 | 0.502 [0.13, 0.64] · 0/0/13 |
| nn1 | 0.898 [0.47, 1.13] · 17/4/33 | 0.945 [0.75, 1.12] · 11/3/17 | 0.970 [0.50, 1.19] · 4/1/5 | 0.732 [0.31, 1.09] · 2/0/11 |
| knn5 | 0.977 [0.65, 1.17] · 20/4/25 | 0.988 [0.74, 1.14] · 13/2/12 | 1.015 [0.64, 1.22] · 5/0/4 | 0.879 [0.67, 1.13] · 2/2/9 |
| rule | 0.880 [0.53, 1.11] · 13/3/35 | 0.877 [0.52, 1.11] · 8/1/21 | 0.915 [0.65, 1.14] · 3/0/6 | 0.800 [0.66, 1.06] · 2/2/8 |

B3+ vs B1 on these shapes: median 1.079 [1.00, 1.43].
