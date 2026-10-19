# Learned ranker: top-k + short online search on held-out shapes (auto-generated)

lgbm_rank (`{"objective": "regression", "num_leaves": 255, "n_estimators": 1500}`) refit on all 477 tune shapes and saved as `data/<hw>/lgbm_rank.txt/.json` (pool of 868 configs). Per held-out shape its top 5 runnable configs are ranked; best-of-k = the fastest of the first k. Regret = t / oracle - 1, median / p95 (mean).

## Screen (regret vs the held-out screen oracle, full pruned space)

| k | all | >= 50 us | < 50 us |
|---|---|---|---|
| 1 | 3.6% / 18.1% (5.5%) | 2.2% / 20.6% (5.6%) | 3.9% / 14.2% (5.5%) |
| 2 | 2.4% / 13.1% (4.2%) | 2.0% / 19.2% (4.9%) | 3.0% / 10.0% (3.8%) |
| 3 | 1.8% / 12.4% (3.6%) | 1.8% / 16.5% (4.0%) | 1.8% / 9.5% (3.4%) |
| 4 | 1.4% / 12.3% (3.1%) | 1.6% / 16.5% (3.6%) | 1.0% / 9.5% (2.7%) |
| 5 | 0.8% / 12.3% (2.8%) | 1.5% / 16.5% (3.5%) | 0.4% / 9.5% (2.5%) |

Online search rows: 86 shapes from `online_lgbm_rank.jsonl`.


## Stage 2 (cold): `results_lock1755_topk.jsonl`

86 shapes, 2561 rows, 0 invalid, 120 power-capped; sessions 20261019-083638-8c2114 (harness a24d834). Oracle = fastest Triton row (screen top-8 + every pick).

| selector | all (n=86) | >= 50 us (n=30) | < 50 us (n=56) | vs B3+ median [p10, p90] |
|---|---|---|---|---|
| knn5 | 2.5% / 23.9% (9.4%) | 4.2% / 65.1% (18.9%) | 2.0% / 19.7% (4.4%) | 0.964 [0.71, 1.11] |
| knnf | 2.5% / 22.1% (6.6%) | 2.4% / 36.7% (8.7%) | 2.8% / 18.9% (5.4%) | 0.955 [0.70, 1.10] |
| lgbm best-of-1, full-protocol timing | 2.0% / 18.3% (4.1%) | 2.2% / 19.1% (5.2%) | 1.7% / 10.7% (3.5%) | 0.968 [0.77, 1.11] |
| lgbm best-of-2, full-protocol timing | 1.1% / 11.8% (3.1%) | 2.0% / 18.2% (4.5%) | 0.2% / 8.6% (2.4%) | 0.971 [0.77, 1.11] |
| lgbm best-of-3, full-protocol timing | 0.7% / 11.8% (2.7%) | 1.1% / 16.5% (3.7%) | 0.1% / 8.1% (2.2%) | 0.975 [0.77, 1.11] |
| lgbm best-of-4, full-protocol timing | 0.3% / 11.2% (2.4%) | 0.8% / 16.4% (3.3%) | 0.0% / 7.6% (1.9%) | 0.982 [0.78, 1.11] |
| lgbm best-of-5, full-protocol timing | 0.1% / 11.2% (2.3%) | 0.7% / 16.4% (3.1%) | 0.0% / 7.6% (1.8%) | 0.990 [0.78, 1.12] |
| lgbm online top-1 (short timing picks) | 2.0% / 18.3% (4.1%) | 2.2% / 19.1% (5.2%) | 1.7% / 10.7% (3.5%) | 0.968 [0.77, 1.11] |
| lgbm online top-2 (short timing picks) | 1.2% / 12.5% (3.3%) | 2.0% / 18.3% (4.5%) | 0.7% / 8.6% (2.6%) | 0.971 [0.77, 1.11] |
| lgbm online top-3 (short timing picks) | 0.8% / 12.2% (2.9%) | 1.1% / 16.5% (3.8%) | 0.5% / 8.1% (2.4%) | 0.973 [0.77, 1.11] |
| lgbm online top-4 (short timing picks) | 0.8% / 11.7% (2.8%) | 0.9% / 16.5% (3.5%) | 0.5% / 8.0% (2.4%) | 0.980 [0.77, 1.11] |
| lgbm online top-5 (short timing picks) | 0.7% / 11.7% (2.8%) | 1.1% / 16.5% (3.3%) | 0.6% / 8.0% (2.5%) | 0.990 [0.77, 1.12] |

## Stage 2 (graph): `results_lock1755_topk_graph.jsonl`

56 shapes, 1685 rows, 0 invalid, 123 power-capped; sessions 20261019-084341-c03a6e (harness a24d834). Oracle = fastest Triton row (screen top-8 + every pick).

| selector | all (< 50 us shapes) (n=56) | vs B3+ median [p10, p90] |
|---|---|---|
| knn5 | 3.8% / 21.8% (6.9%) | 0.973 [0.66, 1.17] |
| knnf | 4.7% / 26.2% (7.6%) | 0.953 [0.68, 1.15] |
| lgbm best-of-1, full-protocol timing | 2.3% / 13.5% (4.4%) | 0.993 [0.72, 1.17] |
| lgbm best-of-2, full-protocol timing | 0.3% / 12.1% (2.9%) | 1.002 [0.73, 1.17] |
| lgbm best-of-3, full-protocol timing | 0.1% / 12.1% (2.6%) | 1.002 [0.73, 1.18] |
| lgbm best-of-4, full-protocol timing | 0.0% / 11.7% (2.4%) | 1.002 [0.73, 1.18] |
| lgbm best-of-5, full-protocol timing | 0.0% / 11.7% (2.2%) | 1.004 [0.73, 1.18] |
| lgbm online top-1 (short timing picks) | 2.3% / 13.5% (4.4%) | 0.993 [0.72, 1.17] |
| lgbm online top-2 (short timing picks) | 0.4% / 12.2% (2.9%) | 1.002 [0.73, 1.17] |
| lgbm online top-3 (short timing picks) | 0.3% / 12.2% (2.7%) | 1.002 [0.73, 1.18] |
| lgbm online top-4 (short timing picks) | 0.2% / 12.1% (2.5%) | 1.002 [0.73, 1.18] |
| lgbm online top-5 (short timing picks) | 0.1% / 12.1% (2.2%) | 1.004 [0.73, 1.18] |

## Online search cost per shape (wall clock, empty Triton cache)

Compile = first call + correctness check per candidate (JIT); timing = short protocol (10 cold samples after 2 ms warmup, one placement; graph-amortized 20 x 5 hot when the fastest is < 50 us).

| k | compile s median / p90 | timing s median / p90 | total s median / p90 / sum over shapes | picked rank (0-based) counts |
|---|---|---|---|---|
| 1 | 0.14 / 0.29 | 0.01 / 0.02 | 0.16 / 0.33 / 14 | [86] |
| 2 | 0.26 / 0.52 | 0.02 / 0.03 | 0.28 / 0.56 / 25 | [51, 35] |
| 3 | 0.37 / 0.70 | 0.02 / 0.04 | 0.40 / 0.76 / 37 | [44, 25, 17] |
| 4 | 0.48 / 1.00 | 0.03 / 0.05 | 0.51 / 1.05 / 48 | [38, 20, 15, 13] |
| 5 | 0.61 / 1.18 | 0.04 / 0.07 | 0.66 / 1.28 / 61 | [31, 16, 11, 10, 18] |

Triton cache dir(s): `/home/raynerubuntu/Documents/repos/CS464/data/rtx3080_sm86_68sms_10gb/online_triton_cache/20261019-084917`. Ranking itself: median 92 ms per shape.
