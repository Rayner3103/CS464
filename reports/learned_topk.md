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

## Stage 2 (cold): `results_lock1755_topk.jsonl` not found


## Stage 2 (graph): `results_lock1755_topk_graph.jsonl` not found

