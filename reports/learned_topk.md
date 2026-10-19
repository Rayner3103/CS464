# Learned ranker: top-k + short online search on held-out shapes (auto-generated)

lgbm_rank (`{"objective": "regression", "num_leaves": 255, "n_estimators": 1500}`) refit on all 477 tune shapes and saved as `data/<hw>/lgbm_rank.txt/.json` (pool of 868 configs). Per held-out shape it ranks its runnable pool; best-of-k = the fastest of the first k. Two orders: **plain** (score order) and **diverse** (`diverse_order`, rule `{"group": "tile_split_bk", "delta": 0.05, "plain": 1}`: keep the top `plain` picks, then the best config of each new group within `delta` predicted log regret of the top pick, then score order). Regret = t / oracle - 1, median / p95 (mean).

## Diversification rule (tune shapes only: best-of-3 screen regret on out-of-fold scores, 20 grouped folds)

| rule | best-of-1 | best-of-3 | best-of-5 |
|---|---|---|---|
| {"group": "tile_split_bk", "delta": 0.05, "plain": 1} | 2.3% / 16.6% (4.8%) | 1.2% / 10.1% (2.8%) | 0.9% / 8.4% (2.1%) |
| {"group": "tile_split_bk", "delta": 0.05, "plain": 1, "inner": 1, "inner_group": "warps", "inner_delta": 0.02, "inner_pos": "after"} | 2.3% / 16.6% (4.8%) | 1.2% / 10.1% (2.8%) | 0.9% / 8.4% (2.1%) |
| {"group": "tile_split_bk", "delta": 0.05, "plain": 1, "inner": 1, "inner_group": "warps", "inner_delta": 0.1, "inner_pos": "after"} | 2.3% / 16.6% (4.8%) | 1.2% / 10.1% (2.8%) | 0.8% / 8.3% (2.1%) |
| {"group": "tile_split_bk", "delta": 0.05, "plain": 1, "inner": 1, "inner_group": "stages_warps", "inner_delta": 0.02, "inner_pos": "after"} | 2.3% / 16.6% (4.8%) | 1.2% / 10.1% (2.8%) | 0.8% / 8.4% (2.1%) |
| {"group": "tile_split_bk", "delta": 0.05, "plain": 1, "inner": 1, "inner_group": "stages_warps", "inner_delta": 0.05, "inner_pos": "after"} | 2.3% / 16.6% (4.8%) | 1.2% / 10.1% (2.8%) | 0.8% / 8.4% (2.1%) |
| {"group": "tile_split_bk", "delta": 0.05, "plain": 1, "inner": 1, "inner_group": "stages_warps", "inner_delta": 0.1, "inner_pos": "after"} | 2.3% / 16.6% (4.8%) | 1.2% / 10.1% (2.8%) | 0.8% / 8.4% (2.1%) |
| {"group": "tile_split_bk", "delta": 0.05, "plain": 1, "inner": 1, "inner_group": "stages_warps", "inner_delta": 0.2, "inner_pos": "after"} | 2.3% / 16.6% (4.8%) | 1.2% / 10.1% (2.8%) | 0.8% / 8.4% (2.1%) |
| {"group": "tile_split_bk", "delta": 0.05, "plain": 1, "inner": 1, "inner_group": "stages_warps", "inner_delta": Infinity, "inner_pos": "after"} | 2.3% / 16.6% (4.8%) | 1.2% / 10.1% (2.8%) | 0.8% / 8.4% (2.1%) |
| {"group": "tile_split_bk", "delta": 0.05, "plain": 1, "inner": 1, "inner_group": "warps", "inner_delta": 0.02, "inner_pos": "before"} | 2.3% / 16.6% (4.8%) | 1.2% / 10.1% (2.8%) | 0.9% / 8.3% (2.1%) |
| plain | 2.3% / 16.6% (4.8%) | 1.5% / 12.0% (3.3%) | 1.0% / 10.5% (2.6%) |

**inner** (diagnostic) = the best rule whose inner pass takes a top-3 slot, `{"group": "tile_split_bk", "delta": 0.05, "plain": 1, "inner": 1, "inner_group": "warps", "inner_delta": 0.02, "inner_pos": "before"}`; not chosen on tune CV.

Distinct (BLOCK_M, BLOCK_N, SPLIT_K) among the held-out top 3: plain 1.55, diverse 2.23, inner 2.20.

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
| inner | 1 | 3.6% / 18.1% (5.5%) | 2.2% / 20.6% (5.6%) | 3.9% / 14.2% (5.5%) |
| inner | 2 | 2.7% / 12.2% (3.9%) | 1.8% / 13.9% (4.0%) | 3.6% / 10.1% (3.9%) |
| inner | 3 | 2.0% / 10.6% (3.3%) | 1.5% / 10.4% (2.8%) | 2.8% / 9.7% (3.5%) |
| inner | 4 | 1.5% / 10.4% (2.8%) | 0.8% / 10.1% (2.4%) | 2.0% / 9.3% (3.0%) |
| inner | 5 | 0.8% / 9.0% (2.5%) | 0.8% / 10.1% (2.2%) | 1.4% / 8.0% (2.7%) |

Online search (plain): 86 shapes from `online_lgbm_rank.jsonl`.

Online search (diverse): 86 shapes from `online_lgbm_rank_diverse.jsonl`.

Online search (inner): 86 shapes from `online_lgbm_rank_inner.jsonl`.

## Stage 2 (cold): `results_lock1755_topki.jsonl`

86 shapes, 2686 rows, 0 invalid, 128 power-capped; sessions 20261019-173740-c108f5 (harness 9e6e4be). Oracle = fastest Triton row (screen top-8 + every pick).

| selector | all (n=86) | >= 50 us (n=30) | < 50 us (n=56) | vs B3+ median [p10, p90] |
|---|---|---|---|---|
| knn5 | 2.6% / 24.1% (9.5%) | 3.8% / 65.5% (18.8%) | 2.4% / 16.9% (4.5%) | 0.970 [0.71, 1.12] |
| knnf | 2.5% / 22.3% (6.5%) | 2.2% / 36.5% (8.7%) | 2.7% / 18.4% (5.3%) | 0.967 [0.70, 1.12] |
| plain best-of-1, full-protocol timing | 2.4% / 17.8% (4.1%) | 2.6% / 19.0% (5.2%) | 2.3% / 11.8% (3.5%) | 0.968 [0.77, 1.12] |
| plain best-of-2, full-protocol timing | 1.2% / 11.9% (3.2%) | 2.0% / 18.3% (4.5%) | 0.7% / 9.3% (2.5%) | 0.971 [0.77, 1.12] |
| plain best-of-3, full-protocol timing | 0.6% / 11.6% (2.8%) | 0.9% / 16.6% (3.7%) | 0.3% / 8.3% (2.2%) | 0.981 [0.77, 1.12] |
| plain best-of-4, full-protocol timing | 0.3% / 11.0% (2.4%) | 0.7% / 16.6% (3.3%) | 0.0% / 7.2% (2.0%) | 0.988 [0.77, 1.12] |
| plain best-of-5, full-protocol timing | 0.3% / 11.0% (2.3%) | 0.7% / 16.6% (3.1%) | 0.0% / 7.2% (1.8%) | 0.993 [0.77, 1.12] |
| diverse best-of-1, full-protocol timing | 2.4% / 17.8% (4.1%) | 2.6% / 19.0% (5.2%) | 2.3% / 11.8% (3.5%) | 0.968 [0.77, 1.12] |
| diverse best-of-2, full-protocol timing | 1.2% / 10.4% (2.8%) | 1.5% / 12.5% (3.6%) | 1.0% / 8.3% (2.4%) | 0.979 [0.78, 1.12] |
| diverse best-of-3, full-protocol timing | 0.7% / 9.1% (2.3%) | 1.0% / 10.0% (2.4%) | 0.6% / 8.3% (2.2%) | 0.988 [0.78, 1.12] |
| diverse best-of-4, full-protocol timing | 0.4% / 8.5% (2.0%) | 0.5% / 8.1% (2.0%) | 0.4% / 8.3% (2.0%) | 0.991 [0.78, 1.12] |
| diverse best-of-5, full-protocol timing | 0.2% / 8.5% (1.9%) | 0.3% / 8.1% (1.8%) | 0.2% / 8.3% (1.9%) | 0.991 [0.78, 1.13] |
| inner best-of-1, full-protocol timing | 2.4% / 17.8% (4.1%) | 2.6% / 19.0% (5.2%) | 2.3% / 11.8% (3.5%) | 0.968 [0.77, 1.12] |
| inner best-of-2, full-protocol timing | 1.2% / 9.5% (2.8%) | 1.5% / 12.5% (3.6%) | 1.0% / 8.3% (2.3%) | 0.979 [0.78, 1.12] |
| inner best-of-3, full-protocol timing | 0.7% / 9.1% (2.2%) | 1.0% / 10.0% (2.4%) | 0.4% / 8.3% (2.0%) | 0.988 [0.78, 1.12] |
| inner best-of-4, full-protocol timing | 0.4% / 8.5% (1.9%) | 0.5% / 8.1% (2.0%) | 0.4% / 8.3% (1.9%) | 0.991 [0.78, 1.12] |
| inner best-of-5, full-protocol timing | 0.2% / 8.5% (1.8%) | 0.3% / 8.1% (1.8%) | 0.0% / 8.3% (1.7%) | 0.991 [0.78, 1.12] |
| plain online top-1 (short timing picks) | 2.4% / 17.8% (4.1%) | 2.6% / 19.0% (5.2%) | 2.3% / 11.8% (3.5%) | 0.968 [0.77, 1.12] |
| plain online top-2 (short timing picks) | 1.5% / 12.7% (3.5%) | 2.0% / 18.3% (4.6%) | 1.2% / 9.3% (2.9%) | 0.971 [0.77, 1.12] |
| plain online top-3 (short timing picks) | 0.9% / 12.2% (3.1%) | 0.9% / 16.7% (3.8%) | 1.0% / 8.4% (2.7%) | 0.973 [0.77, 1.12] |
| plain online top-4 (short timing picks) | 0.8% / 11.3% (2.9%) | 0.8% / 16.7% (3.4%) | 1.0% / 8.3% (2.6%) | 0.985 [0.77, 1.12] |
| plain online top-5 (short timing picks) | 1.0% / 11.3% (2.9%) | 1.0% / 16.7% (3.3%) | 1.6% / 8.3% (2.7%) | 0.986 [0.77, 1.12] |
| diverse online top-1 (short timing picks) | 2.4% / 17.8% (4.1%) | 2.6% / 19.0% (5.2%) | 2.3% / 11.8% (3.5%) | 0.968 [0.77, 1.12] |
| diverse online top-2 (short timing picks) | 1.4% / 10.4% (3.0%) | 1.5% / 12.5% (3.6%) | 1.2% / 8.4% (2.7%) | 0.971 [0.77, 1.12] |
| diverse online top-3 (short timing picks) | 1.1% / 9.1% (2.6%) | 1.0% / 10.0% (2.4%) | 1.2% / 8.4% (2.7%) | 0.980 [0.77, 1.12] |
| diverse online top-4 (short timing picks) | 0.8% / 8.5% (2.4%) | 0.6% / 8.1% (2.0%) | 1.1% / 8.4% (2.6%) | 0.982 [0.77, 1.12] |
| diverse online top-5 (short timing picks) | 0.7% / 8.5% (2.4%) | 0.3% / 8.1% (1.9%) | 0.9% / 8.4% (2.6%) | 0.986 [0.77, 1.12] |
| inner online top-1 (short timing picks) | 2.4% / 17.8% (4.1%) | 2.6% / 19.0% (5.2%) | 2.3% / 11.8% (3.5%) | 0.968 [0.77, 1.12] |
| inner online top-2 (short timing picks) | 1.4% / 10.4% (3.0%) | 1.5% / 12.5% (3.7%) | 1.2% / 8.4% (2.7%) | 0.979 [0.77, 1.12] |
| inner online top-3 (short timing picks) | 1.1% / 9.1% (2.5%) | 1.0% / 10.0% (2.4%) | 1.2% / 8.4% (2.6%) | 0.980 [0.77, 1.12] |
| inner online top-4 (short timing picks) | 1.0% / 8.5% (2.4%) | 0.8% / 8.1% (2.0%) | 1.3% / 8.4% (2.7%) | 0.986 [0.77, 1.12] |
| inner online top-5 (short timing picks) | 0.8% / 8.5% (2.3%) | 0.3% / 8.1% (1.9%) | 1.2% / 8.4% (2.6%) | 0.986 [0.77, 1.12] |

Worst shapes of plain top-3 (online) and the diverse top-3 there:

| shape | oracle us | plain | diverse | inner | inner top-3 |
|---|---|---|---|---|---|
| 14244x16x1257 | 72.3 | 22.7% | 2.9% | 2.9% | `bm32_bn32_bk64_w4_s5_g1_sk1`, `bm32_bn32_bk128_w4_s3_g1_sk1`, `bm64_bn32_bk128_w4_s2_g16_sk1` |
| 320x1x79 | 4.9 | 22.4% | 22.4% | 22.4% | `bm32_bn32_bk16_w4_s5_g1_sk1`, `bm32_bn32_bk32_w4_s5_g4_sk1`, `bm32_bn32_bk16_w4_s4_g1_sk1` |
| 5398x196x774 | 63.9 | 19.9% | 19.8% | 19.9% | `bm64_bn64_bk32_w4_s2_g16_sk1`, `bm64_bn64_bk64_w4_s4_g16_sk1`, `bm64_bn64_bk32_w4_s2_g8_sk1` |
| 51x324x5709 | 24.4 | 13.0% | 12.4% | 12.4% | `bm32_bn64_bk64_w4_s4_g1_sk8`, `bm32_bn64_bk64_w8_s4_g1_sk8`, `bm32_bn32_bk64_w4_s4_g1_sk8` |
| 417x512x6739 | 87.4 | 12.7% | 9.3% | 9.3% | `bm64_bn64_bk32_w4_s2_g4_sk8`, `bm64_bn64_bk32_w4_s2_g1_sk4`, `bm64_bn128_bk64_w4_s2_g4_sk4` |
| 4065x8842x28 | 126.1 | 10.6% | 10.6% | 10.6% | `bm64_bn128_bk16_w4_s2_g4_sk1`, `bm64_bn128_bk16_w4_s2_g16_sk1`, `bm64_bn128_bk16_w4_s2_g8_sk1` |
| 226x2141x1139 | 41.6 | 8.5% | 8.5% | 8.5% | `bm64_bn128_bk64_w4_s4_g8_sk1`, `bm64_bn64_bk64_w4_s4_g4_sk1`, `bm64_bn64_bk32_w4_s5_g1_sk1` |
| 3522x1x647 | 17.0 | 8.3% | 8.3% | 8.3% | `bm128_bn32_bk32_w8_s2_g1_sk4`, `bm64_bn32_bk32_w8_s2_g1_sk4`, `bm64_bn32_bk32_w8_s2_g4_sk4` |

## Stage 2 (graph): `results_lock1755_topki_graph.jsonl`

56 shapes, 1766 rows, 0 invalid, 166 power-capped; sessions 20261019-174513-8410d7 (harness 9e6e4be). Oracle = fastest Triton row (screen top-8 + every pick).

| selector | all (< 50 us shapes) (n=56) | vs B3+ median [p10, p90] |
|---|---|---|
| knn5 | 4.0% / 23.8% (7.0%) | 0.971 [0.65, 1.17] |
| knnf | 4.9% / 25.8% (7.7%) | 0.950 [0.69, 1.15] |
| plain best-of-1, full-protocol timing | 2.5% / 13.5% (4.4%) | 0.994 [0.72, 1.17] |
| plain best-of-2, full-protocol timing | 0.4% / 11.9% (2.9%) | 1.007 [0.73, 1.17] |
| plain best-of-3, full-protocol timing | 0.3% / 11.9% (2.6%) | 1.007 [0.73, 1.18] |
| plain best-of-4, full-protocol timing | 0.1% / 11.8% (2.4%) | 1.007 [0.73, 1.18] |
| plain best-of-5, full-protocol timing | 0.0% / 11.8% (2.2%) | 1.010 [0.73, 1.18] |
| diverse best-of-1, full-protocol timing | 2.5% / 13.5% (4.4%) | 0.994 [0.72, 1.17] |
| diverse best-of-2, full-protocol timing | 0.2% / 10.9% (2.3%) | 1.011 [0.73, 1.18] |
| diverse best-of-3, full-protocol timing | 0.2% / 9.7% (2.0%) | 1.011 [0.73, 1.18] |
| diverse best-of-4, full-protocol timing | 0.1% / 8.2% (1.8%) | 1.011 [0.73, 1.18] |
| diverse best-of-5, full-protocol timing | 0.0% / 8.2% (1.7%) | 1.011 [0.73, 1.18] |
| inner best-of-1, full-protocol timing | 2.5% / 13.5% (4.4%) | 0.994 [0.72, 1.17] |
| inner best-of-2, full-protocol timing | 0.2% / 11.6% (2.3%) | 1.007 [0.73, 1.18] |
| inner best-of-3, full-protocol timing | 0.2% / 8.2% (1.7%) | 1.011 [0.73, 1.18] |
| inner best-of-4, full-protocol timing | 0.1% / 8.2% (1.6%) | 1.011 [0.73, 1.18] |
| inner best-of-5, full-protocol timing | 0.0% / 7.8% (1.4%) | 1.011 [0.73, 1.18] |
| plain online top-1 (short timing picks) | 2.5% / 13.5% (4.4%) | 0.994 [0.72, 1.17] |
| plain online top-2 (short timing picks) | 0.4% / 12.3% (2.9%) | 1.007 [0.73, 1.17] |
| plain online top-3 (short timing picks) | 0.4% / 12.3% (2.7%) | 1.007 [0.73, 1.17] |
| plain online top-4 (short timing picks) | 0.3% / 12.6% (2.5%) | 1.007 [0.73, 1.17] |
| plain online top-5 (short timing picks) | 0.1% / 12.6% (2.3%) | 1.010 [0.73, 1.18] |
| diverse online top-1 (short timing picks) | 2.5% / 13.5% (4.4%) | 0.994 [0.72, 1.17] |
| diverse online top-2 (short timing picks) | 0.3% / 11.1% (2.3%) | 1.011 [0.73, 1.18] |
| diverse online top-3 (short timing picks) | 0.2% / 9.7% (2.1%) | 1.011 [0.73, 1.18] |
| diverse online top-4 (short timing picks) | 0.2% / 8.2% (1.9%) | 1.011 [0.73, 1.18] |
| diverse online top-5 (short timing picks) | 0.2% / 8.2% (1.9%) | 1.011 [0.73, 1.18] |
| inner online top-1 (short timing picks) | 2.5% / 13.5% (4.4%) | 0.994 [0.72, 1.17] |
| inner online top-2 (short timing picks) | 0.3% / 11.6% (2.3%) | 1.007 [0.73, 1.18] |
| inner online top-3 (short timing picks) | 0.2% / 8.2% (1.7%) | 1.011 [0.73, 1.18] |
| inner online top-4 (short timing picks) | 0.1% / 8.2% (1.6%) | 1.011 [0.73, 1.18] |
| inner online top-5 (short timing picks) | 0.1% / 7.8% (1.5%) | 1.011 [0.73, 1.18] |

Worst shapes of plain top-3 (online) and the diverse top-3 there:

| shape | oracle us | plain | diverse | inner | inner top-3 |
|---|---|---|---|---|---|
| 320x1x79 | 2.4 | 16.1% | 16.1% | 16.1% | `bm32_bn32_bk16_w4_s5_g1_sk1`, `bm32_bn32_bk32_w4_s5_g4_sk1`, `bm32_bn32_bk16_w4_s4_g1_sk1` |
| 381x1401x250 | 10.6 | 13.7% | 14.2% | 0.2% | `bm64_bn64_bk32_w4_s4_g1_sk1`, `bm64_bn64_bk32_w8_s2_g4_sk1`, `bm32_bn64_bk32_w4_s2_g1_sk1` |
| 51x324x5709 | 19.7 | 12.3% | 7.7% | 7.7% | `bm32_bn64_bk64_w4_s4_g1_sk8`, `bm32_bn64_bk64_w8_s4_g1_sk8`, `bm32_bn32_bk64_w4_s4_g1_sk8` |
| 277x3x161 | 3.7 | 12.3% | 0.0% | 0.0% | `bm32_bn32_bk128_w4_s2_g16_sk1`, `bm16_bn32_bk128_w2_s5_g1_sk1`, `bm16_bn32_bk64_w2_s2_g1_sk1` |
| 12094x232x3 | 10.3 | 11.8% | 10.8% | 10.7% | `bm16_bn128_bk16_w4_s2_g16_sk1`, `bm16_bn128_bk16_w2_s2_g8_sk1`, `bm32_bn32_bk16_w2_s2_g1_sk1` |
| 1316x272x100 | 5.7 | 11.7% | 0.0% | 0.0% | `bm32_bn64_bk32_w4_s2_g1_sk1`, `bm32_bn64_bk32_w8_s3_g1_sk1`, `bm64_bn32_bk16_w4_s2_g1_sk1` |
| 20x1x11869 | 10.4 | 9.3% | 9.3% | 9.3% | `bm32_bn32_bk128_w4_s5_g1_sk16`, `bm32_bn32_bk128_w4_s4_g1_sk16`, `bm16_bn32_bk64_w2_s2_g1_sk16` |
| 226x2141x1139 | 39.1 | 7.8% | 7.8% | 7.8% | `bm64_bn128_bk64_w4_s4_g8_sk1`, `bm64_bn64_bk64_w4_s4_g4_sk1`, `bm64_bn64_bk32_w4_s5_g1_sk1` |

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

## Online search cost per shape (diverse; wall clock, empty Triton cache)

Compile = first call + correctness check per candidate (JIT); timing = short protocol (10 cold samples after 2 ms warmup, one placement; graph-amortized 20 x 5 hot when the fastest is < 50 us).

| k | compile s median / p90 | timing s median / p90 | total s median / p90 / sum over shapes | picked rank (0-based) counts |
|---|---|---|---|---|
| 1 | 0.15 / 0.34 | 0.01 / 0.01 | 0.16 / 0.34 / 14 | [86] |
| 2 | 0.25 / 0.53 | 0.02 / 0.03 | 0.27 / 0.60 / 26 | [53, 33] |
| 3 | 0.37 / 0.74 | 0.02 / 0.04 | 0.39 / 0.86 / 37 | [45, 26, 15] |
| 4 | 0.49 / 0.94 | 0.03 / 0.05 | 0.52 / 1.02 / 50 | [42, 23, 12, 9] |
| 5 | 0.60 / 1.19 | 0.04 / 0.07 | 0.65 / 1.27 / 63 | [39, 22, 10, 8, 7] |

Triton cache dir(s): `/home/raynerubuntu/Documents/repos/CS464/data/rtx3080_sm86_68sms_10gb/online_triton_cache/20261019-113343`. Ranking itself: median 93 ms per shape.

## Online search cost per shape (inner; wall clock, empty Triton cache)

Compile = first call + correctness check per candidate (JIT); timing = short protocol (10 cold samples after 2 ms warmup, one placement; graph-amortized 20 x 5 hot when the fastest is < 50 us).

| k | compile s median / p90 | timing s median / p90 | total s median / p90 / sum over shapes | picked rank (0-based) counts |
|---|---|---|---|---|
| 1 | 0.14 / 0.27 | 0.01 / 0.01 | 0.15 / 0.28 / 14 | [86] |
| 2 | 0.25 / 0.52 | 0.02 / 0.03 | 0.26 / 0.55 / 26 | [56, 30] |
| 3 | 0.36 / 0.74 | 0.02 / 0.04 | 0.39 / 0.81 / 37 | [45, 24, 17] |
| 4 | 0.47 / 0.95 | 0.03 / 0.05 | 0.51 / 0.98 / 49 | [41, 22, 16, 7] |
| 5 | 0.60 / 1.19 | 0.04 / 0.07 | 0.65 / 1.28 / 62 | [36, 22, 13, 7, 8] |

Triton cache dir(s): `/home/raynerubuntu/Documents/repos/CS464/data/rtx3080_sm86_68sms_10gb/online_triton_cache/20261019-175103`. Ranking itself: median 95 ms per shape.
