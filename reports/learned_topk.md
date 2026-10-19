# Learned ranker: top-k + short online search on held-out shapes (auto-generated)

lgbm_rank (`{"objective": "regression", "num_leaves": 255, "n_estimators": 1500}`) refit on all 477 tune shapes and saved as `data/<hw>/lgbm_rank.txt/.json` (pool of 868 configs). Per held-out shape it ranks its runnable pool; best-of-k = the fastest of the first k. Two orders: **plain** (score order) and **diverse** (`diverse_order`, rule `{"group": "tile_split_bk", "delta": 0.05, "plain": 1}`: keep the top `plain` picks, then the best config of each new group within `delta` predicted log regret of the top pick, then score order). Regret = t / oracle - 1, median / p95 (mean).

## Diversification rule (tune shapes only: best-of-3 screen regret on out-of-fold scores, 20 grouped folds)

| rule | best-of-1 | best-of-3 | best-of-5 |
|---|---|---|---|
| {"group": "tile_split_bk", "delta": 0.05, "plain": 1} | 2.3% / 16.6% (4.8%) | 1.2% / 10.1% (2.8%) | 0.9% / 8.4% (2.1%) |
| {"group": "tile_split", "delta": 0.05, "plain": 1} | 2.3% / 16.6% (4.8%) | 1.3% / 10.1% (2.8%) | 0.9% / 8.8% (2.3%) |
| {"group": "tile_split_bk", "delta": 0.1, "plain": 1} | 2.3% / 16.6% (4.8%) | 1.2% / 10.6% (2.9%) | 1.0% / 8.7% (2.2%) |
| {"group": "tile_split_bk", "delta": 0.2, "plain": 1} | 2.3% / 16.6% (4.8%) | 1.2% / 10.6% (2.9%) | 1.0% / 9.1% (2.3%) |
| {"group": "tile_split_bk", "delta": 0.5, "plain": 1} | 2.3% / 16.6% (4.8%) | 1.2% / 10.6% (2.9%) | 1.0% / 9.1% (2.3%) |
| {"group": "tile_split_bk", "delta": Infinity, "plain": 1} | 2.3% / 16.6% (4.8%) | 1.2% / 10.6% (2.9%) | 1.0% / 9.1% (2.3%) |
| {"group": "tile_split", "delta": 0.05, "plain": 2} | 2.3% / 16.6% (4.8%) | 1.3% / 10.9% (2.9%) | 0.9% / 8.8% (2.2%) |
| {"group": "tile_split", "delta": 0.03, "plain": 1} | 2.3% / 16.6% (4.8%) | 1.3% / 11.0% (2.9%) | 0.9% / 9.6% (2.3%) |
| plain | 2.3% / 16.6% (4.8%) | 1.5% / 12.0% (3.3%) | 1.0% / 10.5% (2.6%) |

Distinct (BLOCK_M, BLOCK_N, SPLIT_K) among the held-out top 3: plain 1.55, diverse 2.23.

## Screen (regret vs the held-out screen oracle, full pruned space)

| order | k | all | >= 50 us | < 50 us |
|---|---|---|---|---|
| plain | 1 | 3.6% / 18.1% (5.5%) | 2.2% / 20.6% (5.6%) | 3.9% / 14.2% (5.5%) |
| plain | 2 | 2.4% / 13.1% (4.2%) | 2.0% / 19.2% (4.9%) | 3.0% / 10.0% (3.8%) |
| plain | 3 | 1.8% / 12.4% (3.6%) | 1.8% / 16.5% (4.0%) | 1.8% / 9.5% (3.4%) |
| plain | 4 | 1.4% / 12.3% (3.1%) | 1.6% / 16.5% (3.6%) | 1.0% / 9.5% (2.7%) |
| plain | 5 | 0.8% / 12.3% (2.8%) | 1.5% / 16.5% (3.5%) | 0.4% / 9.5% (2.5%) |
| diverse | 1 | 3.6% / 18.1% (5.5%) | 2.2% / 20.6% (5.6%) | 3.9% / 14.2% (5.5%) |
| diverse | 2 | 2.8% / 12.2% (4.0%) | 1.8% / 13.9% (4.0%) | 3.7% / 10.1% (4.0%) |
| diverse | 3 | 2.0% / 10.6% (3.4%) | 1.5% / 10.4% (2.8%) | 2.9% / 9.4% (3.7%) |
| diverse | 4 | 1.5% / 9.0% (2.8%) | 0.8% / 10.1% (2.4%) | 2.0% / 8.2% (3.0%) |
| diverse | 5 | 0.9% / 9.0% (2.6%) | 0.8% / 10.1% (2.2%) | 1.5% / 8.2% (2.9%) |

Online search (plain): 86 shapes from `online_lgbm_rank.jsonl`.

## Stage 2 (cold): `results_lock1755_topk.jsonl`

86 shapes, 2561 rows, 0 invalid, 120 power-capped; sessions 20261019-083638-8c2114 (harness a24d834). Oracle = fastest Triton row (screen top-8 + every pick).

| selector | all (n=86) | >= 50 us (n=30) | < 50 us (n=56) | vs B3+ median [p10, p90] |
|---|---|---|---|---|
| knn5 | 2.5% / 23.9% (9.4%) | 4.2% / 65.1% (18.9%) | 2.0% / 19.7% (4.4%) | 0.964 [0.71, 1.11] |
| knnf | 2.5% / 22.1% (6.6%) | 2.4% / 36.7% (8.7%) | 2.8% / 18.9% (5.4%) | 0.955 [0.70, 1.10] |
| plain best-of-1, full-protocol timing | 2.0% / 18.3% (4.1%) | 2.2% / 19.1% (5.2%) | 1.7% / 10.7% (3.5%) | 0.968 [0.77, 1.11] |
| plain best-of-2, full-protocol timing | 1.1% / 11.8% (3.1%) | 2.0% / 18.2% (4.5%) | 0.2% / 8.6% (2.4%) | 0.971 [0.77, 1.11] |
| plain best-of-3, full-protocol timing | 0.7% / 11.8% (2.7%) | 1.1% / 16.5% (3.7%) | 0.1% / 8.1% (2.2%) | 0.975 [0.77, 1.11] |
| plain best-of-4, full-protocol timing | 0.3% / 11.2% (2.4%) | 0.8% / 16.4% (3.3%) | 0.0% / 7.6% (1.9%) | 0.982 [0.78, 1.11] |
| plain best-of-5, full-protocol timing | 0.1% / 11.2% (2.3%) | 0.7% / 16.4% (3.1%) | 0.0% / 7.6% (1.8%) | 0.990 [0.78, 1.12] |
| diverse best-of-1, full-protocol timing | 2.0% / 18.3% (4.1%) | 2.2% / 19.1% (5.2%) | 1.7% / 10.7% (3.5%) | 0.968 [0.77, 1.11] |
| diverse best-of-2, full-protocol timing | 0.9% / 10.3% (2.9%) | 2.0% / 12.6% (3.9%) | 0.5% / 8.1% (2.3%) | 0.971 [0.77, 1.12] |
| diverse best-of-3, full-protocol timing | 0.7% / 9.8% (2.5%) | 0.9% / 12.6% (2.9%) | 0.5% / 8.1% (2.3%) | 0.982 [0.77, 1.12] |
| diverse best-of-4, full-protocol timing | 0.5% / 8.1% (2.3%) | 0.7% / 11.3% (2.5%) | 0.2% / 8.1% (2.2%) | 0.989 [0.77, 1.12] |
| diverse best-of-5, full-protocol timing | 0.2% / 8.1% (2.2%) | 0.5% / 10.5% (2.3%) | 0.0% / 8.1% (2.1%) | 0.992 [0.78, 1.12] |
| plain online top-1 (short timing picks) | 2.0% / 18.3% (4.1%) | 2.2% / 19.1% (5.2%) | 1.7% / 10.7% (3.5%) | 0.968 [0.77, 1.11] |
| plain online top-2 (short timing picks) | 1.2% / 12.5% (3.3%) | 2.0% / 18.3% (4.5%) | 0.7% / 8.6% (2.6%) | 0.971 [0.77, 1.11] |
| plain online top-3 (short timing picks) | 0.8% / 12.2% (2.9%) | 1.1% / 16.5% (3.8%) | 0.5% / 8.1% (2.4%) | 0.973 [0.77, 1.11] |
| plain online top-4 (short timing picks) | 0.8% / 11.7% (2.8%) | 0.9% / 16.5% (3.5%) | 0.5% / 8.0% (2.4%) | 0.980 [0.77, 1.11] |
| plain online top-5 (short timing picks) | 0.7% / 11.7% (2.8%) | 1.1% / 16.5% (3.3%) | 0.6% / 8.0% (2.5%) | 0.990 [0.77, 1.12] |

Worst shapes of plain top-3 (online) and the diverse top-3 there:

| shape | oracle us | plain | diverse | diverse top-3 |
|---|---|---|---|---|
| 14244x16x1257 | 72.4 | 22.6% | 7.7% | `bm32_bn32_bk64_w4_s5_g1_sk1`, `bm32_bn32_bk128_w4_s3_g1_sk1`, `bm64_bn32_bk128_w4_s2_g16_sk1` |
| 5398x196x774 | 64.0 | 19.5% | 19.4% | `bm64_bn64_bk32_w4_s2_g16_sk1`, `bm64_bn64_bk64_w4_s4_g16_sk1`, `bm64_bn64_bk32_w4_s2_g8_sk1` |
| 320x1x79 | 5.2 | 18.4% | 18.4% | `bm32_bn32_bk16_w4_s5_g1_sk1`, `bm32_bn32_bk32_w4_s5_g4_sk1`, `bm32_bn32_bk16_w4_s4_g1_sk1` |
| 51x324x5709 | 24.1 | 13.3% | 12.2% | `bm32_bn64_bk64_w4_s4_g1_sk8`, `bm32_bn32_bk64_w4_s4_g1_sk8`, `bm16_bn64_bk32_w4_s2_g1_sk16` |
| 417x512x6739 | 87.1 | 12.9% | 14.4% | `bm64_bn64_bk32_w4_s2_g4_sk8`, `bm64_bn64_bk32_w4_s2_g1_sk4`, `bm64_bn128_bk64_w4_s2_g4_sk4` |
| 4065x8842x28 | 126.1 | 10.4% | 10.4% | `bm64_bn128_bk16_w4_s2_g4_sk1`, `bm64_bn128_bk16_w4_s2_g16_sk1`, `bm64_bn128_bk16_w4_s2_g8_sk1` |
| 443x4723x393 | 52.6 | 8.4% | 1.0% | `bm64_bn64_bk32_w4_s4_g16_sk1`, `bm64_bn128_bk16_w4_s2_g1_sk1`, `bm64_bn64_bk16_w4_s3_g1_sk1` |
| 226x2141x1139 | 41.7 | 8.1% | 8.1% | `bm64_bn128_bk64_w4_s4_g8_sk1`, `bm64_bn64_bk64_w4_s4_g4_sk1`, `bm64_bn64_bk32_w4_s5_g1_sk1` |

## Stage 2 (graph): `results_lock1755_topk_graph.jsonl`

56 shapes, 1685 rows, 0 invalid, 123 power-capped; sessions 20261019-084341-c03a6e (harness a24d834). Oracle = fastest Triton row (screen top-8 + every pick).

| selector | all (< 50 us shapes) (n=56) | vs B3+ median [p10, p90] |
|---|---|---|
| knn5 | 3.8% / 21.8% (6.9%) | 0.973 [0.66, 1.17] |
| knnf | 4.7% / 26.2% (7.6%) | 0.953 [0.68, 1.15] |
| plain best-of-1, full-protocol timing | 2.3% / 13.5% (4.4%) | 0.993 [0.72, 1.17] |
| plain best-of-2, full-protocol timing | 0.3% / 12.1% (2.9%) | 1.002 [0.73, 1.17] |
| plain best-of-3, full-protocol timing | 0.1% / 12.1% (2.6%) | 1.002 [0.73, 1.18] |
| plain best-of-4, full-protocol timing | 0.0% / 11.7% (2.4%) | 1.002 [0.73, 1.18] |
| plain best-of-5, full-protocol timing | 0.0% / 11.7% (2.2%) | 1.004 [0.73, 1.18] |
| diverse best-of-1, full-protocol timing | 2.3% / 13.5% (4.4%) | 0.993 [0.72, 1.17] |
| diverse best-of-2, full-protocol timing | 0.2% / 12.1% (2.7%) | 1.002 [0.73, 1.18] |
| diverse best-of-3, full-protocol timing | 0.2% / 12.1% (2.6%) | 1.002 [0.73, 1.18] |
| diverse best-of-4, full-protocol timing | 0.1% / 11.5% (2.5%) | 1.007 [0.73, 1.18] |
| diverse best-of-5, full-protocol timing | 0.1% / 11.5% (2.4%) | 1.009 [0.73, 1.18] |
| plain online top-1 (short timing picks) | 2.3% / 13.5% (4.4%) | 0.993 [0.72, 1.17] |
| plain online top-2 (short timing picks) | 0.4% / 12.2% (2.9%) | 1.002 [0.73, 1.17] |
| plain online top-3 (short timing picks) | 0.3% / 12.2% (2.7%) | 1.002 [0.73, 1.18] |
| plain online top-4 (short timing picks) | 0.2% / 12.1% (2.5%) | 1.002 [0.73, 1.18] |
| plain online top-5 (short timing picks) | 0.1% / 12.1% (2.2%) | 1.004 [0.73, 1.18] |

Worst shapes of plain top-3 (online) and the diverse top-3 there:

| shape | oracle us | plain | diverse | diverse top-3 |
|---|---|---|---|---|
| 320x1x79 | 2.4 | 15.7% | 15.7% | `bm32_bn32_bk16_w4_s5_g1_sk1`, `bm32_bn32_bk32_w4_s5_g4_sk1`, `bm32_bn32_bk16_w4_s4_g1_sk1` |
| 381x1401x250 | 10.6 | 14.0% | 14.7% | `bm64_bn64_bk32_w4_s4_g1_sk1`, `bm32_bn64_bk32_w4_s2_g1_sk1`, `bm64_bn64_bk16_w8_s3_g1_sk1` |
| 12094x232x3 | 10.2 | 12.5% | 11.1% | `bm16_bn128_bk16_w4_s2_g16_sk1`, `bm32_bn32_bk16_w2_s2_g1_sk1`, `bm128_bn32_bk16_w4_s2_g8_sk1` |
| 1316x272x100 | 5.8 | 12.1% | 12.1% | `bm32_bn64_bk32_w4_s2_g1_sk1`, `bm64_bn32_bk16_w4_s2_g1_sk1`, `bm32_bn64_bk16_w4_s5_g1_sk1` |
| 277x3x161 | 3.7 | 12.1% | 12.1% | `bm32_bn32_bk128_w4_s2_g16_sk1`, `bm16_bn32_bk128_w2_s5_g1_sk1`, `bm16_bn32_bk64_w2_s2_g1_sk1` |
| 51x324x5709 | 19.6 | 10.8% | 8.6% | `bm32_bn64_bk64_w4_s4_g1_sk8`, `bm32_bn32_bk64_w4_s4_g1_sk8`, `bm16_bn64_bk32_w4_s2_g1_sk16` |
| 20x1x11869 | 10.3 | 9.4% | 9.4% | `bm32_bn32_bk128_w4_s5_g1_sk16`, `bm32_bn32_bk128_w4_s4_g1_sk16`, `bm16_bn32_bk64_w2_s2_g1_sk16` |
| 226x2141x1139 | 39.0 | 8.8% | 8.8% | `bm64_bn128_bk64_w4_s4_g8_sk1`, `bm64_bn64_bk64_w4_s4_g4_sk1`, `bm64_bn64_bk32_w4_s5_g1_sk1` |

## Online search cost per shape (plain; wall clock, empty Triton cache)

Compile = first call + correctness check per candidate (JIT); timing = short protocol (10 cold samples after 2 ms warmup, one placement; graph-amortized 20 x 5 hot when the fastest is < 50 us).

| k | compile s median / p90 | timing s median / p90 | total s median / p90 / sum over shapes | picked rank (0-based) counts |
|---|---|---|---|---|
| 1 | 0.14 / 0.29 | 0.01 / 0.02 | 0.16 / 0.33 / 14 | [86] |
| 2 | 0.26 / 0.52 | 0.02 / 0.03 | 0.28 / 0.56 / 25 | [51, 35] |
| 3 | 0.37 / 0.70 | 0.02 / 0.04 | 0.40 / 0.76 / 37 | [44, 25, 17] |
| 4 | 0.48 / 1.00 | 0.03 / 0.05 | 0.51 / 1.05 / 48 | [38, 20, 15, 13] |
| 5 | 0.61 / 1.18 | 0.04 / 0.07 | 0.66 / 1.28 / 61 | [31, 16, 11, 10, 18] |

Triton cache dir(s): `/home/raynerubuntu/Documents/repos/CS464/data/rtx3080_sm86_68sms_10gb/online_triton_cache/20261019-084917`. Ranking itself: median 92 ms per shape.
