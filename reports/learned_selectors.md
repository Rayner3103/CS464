# Learned selectors v0 on held-out shapes (auto-generated)

Five model families x two forms, fit on the tune-split screen (477 shapes x 10840 configs) and scored on 86 held-out shapes. **cls** = shape -> config classification over the first P greedy portfolio members (labels: first member within 2.0% of the shape's best); **rank** = per-candidate scoring over each training shape's top-5 pool (trees / LightGBM regress log regret or lambdarank; LinearSVR regresses log regret and logistic regression classifies regret <= 1.05, both on standardized features plus shape x candidate products). Hyperparameters by grouped 20-fold CV on tune shapes (each fit sees 290-353 shapes; none sharing two of M, N, K with a fold shape), lowest capped mean regret. b6 / knn5 / knnf are the existing baselines (`selector_baselines.md`), CV-scored under the same folds with their weights fixed. Regret = t / oracle - 1.

## Screen: regret vs the held-out per-shape best (full pruned space)

| selector | tune CV mean | tune CV median / p95 | held-out all median / p95 | held-out >= 50 us median / p95 | held-out < 50 us median / p95 | held-out heldout_random median / p95 | held-out heldout_region median / p95 | held-out mean | max | picks the oracle | distinct picks |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b6 | 110.8% | 68.4% / 343.1% | 48.1% / 487.0% | 85.0% / 395.7% | 28.9% / 524.6% | 40.0% / 345.5% | 64.7% / 711.6% | 113.6% | 900.0% | 0.0% | 1 |
| knn5 | 8.4% | 3.5% / 28.2% | 4.7% / 25.4% | 3.6% / 65.4% | 4.8% / 21.5% | 5.0% / 24.9% | 2.1% / 27.5% | 10.8% | 277.7% | 2.3% | 67 |
| knnf | 7.5% | 2.4% / 28.3% | 4.1% / 27.3% | 2.2% / 37.4% | 4.7% / 23.2% | 5.0% / 22.4% | 1.3% / 32.3% | 7.8% | 45.4% | 3.5% | 67 |
| dt_cls | 14.6% | 4.5% / 59.2% | 7.8% / 43.9% | 7.0% / 40.3% | 8.1% / 52.9% | 9.8% / 55.3% | 3.8% / 37.3% | 15.1% | 183.1% | 2.3% | 23 |
| svm_cls | 13.5% | 4.6% / 56.4% | 8.0% / 53.6% | 6.2% / 60.0% | 8.7% / 46.6% | 9.9% / 51.8% | 7.1% / 50.3% | 15.3% | 126.7% | 2.3% | 15 |
| lgbm_cls | 10.1% | 3.4% / 32.2% | 4.1% / 37.7% | 3.5% / 71.5% | 4.3% / 25.8% | 4.3% / 29.6% | 3.4% / 71.2% | 11.0% | 137.3% | 1.2% | 23 |
| rf_cls | 9.0% | 3.3% / 30.9% | 4.3% / 32.7% | 3.2% / 28.5% | 5.3% / 32.5% | 4.3% / 29.0% | 3.4% / 61.4% | 11.4% | 137.3% | 2.3% | 22 |
| logreg_cls | 11.1% | 4.1% / 37.6% | 7.4% / 52.5% | 4.0% / 37.6% | 8.7% / 57.7% | 9.9% / 45.6% | 5.9% / 49.3% | 13.2% | 99.7% | 2.3% | 15 |
| dt_rank | 11.6% | 3.8% / 46.0% | 4.6% / 65.7% | 2.2% / 44.2% | 5.2% / 70.2% | 6.1% / 70.7% | 2.0% / 52.7% | 14.3% | 147.4% | 4.7% | 65 |
| svm_rank | 22.4% | 13.0% / 73.1% | 13.7% / 73.5% | 17.5% / 89.5% | 13.3% / 58.2% | 14.5% / 65.7% | 9.2% / 83.9% | 21.9% | 102.0% | 1.2% | 48 |
| lgbm_rank | 4.8% | 2.3% / 16.6% | 3.6% / 18.1% | 2.2% / 20.6% | 3.9% / 14.2% | 4.1% / 17.8% | 1.9% / 15.2% | 5.5% | 28.6% | 2.3% | 65 |
| rf_rank | 5.3% | 2.6% / 21.2% | 4.4% / 18.5% | 2.0% / 20.0% | 5.1% / 16.2% | 5.1% / 19.5% | 3.2% / 15.5% | 6.4% | 33.7% | 5.8% | 73 |
| logreg_rank | 22.1% | 9.0% / 77.3% | 8.5% / 63.7% | 7.0% / 85.5% | 9.3% / 41.2% | 9.0% / 47.1% | 6.3% / 77.0% | 17.3% | 124.6% | 1.2% | 58 |

## Chosen hyperparameters and candidate ceilings

Ceiling = regret of the best member of the selector's own candidate set (cls label set, rank pool) on each held-out shape: the floor any pick from that set can reach.

| selector | hyperparameters | candidates | grid points | ceiling held-out median / p95 / mean | CV mean, best / worst grid point |
|---|---|---|---|---|---|
| dt_cls | `{"P": 48, "max_depth": null, "min_samples_leaf": 5}` | 48 | 64 | 1.7% / 12.2% / 3.3% | 14.6% / 39.8% |
| svm_cls | `{"P": 16, "C": 1}` | 16 | 16 | 3.6% / 18.2% / 5.4% | 13.5% / 21.1% |
| lgbm_cls | `{"P": 32, "num_leaves": 16, "n_estimators": 300, "min_child_samples": 3}` | 32 | 36 | 2.0% / 12.5% / 3.6% | 10.1% / 14.3% |
| rf_cls | `{"P": 32, "max_features": 1.0, "min_samples_leaf": 1}` | 32 | 36 | 2.0% / 12.5% / 3.6% | 9.0% / 14.3% |
| logreg_cls | `{"P": 16, "C": 10}` | 16 | 16 | 3.6% / 18.2% / 5.4% | 11.1% / 17.1% |
| dt_rank | `{"max_depth": null, "min_samples_leaf": 5}` | 868 | 12 | 0.0% / 5.7% / 0.7% | 11.6% / 27.3% |
| svm_rank | `{"C": 0.1}` | 868 | 3 | 0.0% / 5.7% / 0.7% | 22.4% / 22.7% |
| lgbm_rank | `{"objective": "regression", "num_leaves": 255, "n_estimators": 1500}` | 868 | 12 | 0.0% / 5.7% / 0.7% | 4.8% / 7.1% |
| rf_rank | `{"max_features": 0.66, "min_samples_leaf": 5}` | 868 | 9 | 0.0% / 5.7% / 0.7% | 5.3% / 6.8% |
| logreg_rank | `{"C": 1}` | 868 | 3 | 0.0% / 5.7% / 0.7% | 22.1% / 27.6% |

## Feature importance (final fits; share of total gain / impurity decrease)

- **dt_cls**: log2_ctas_per_sm_16x64 19%, log2_N 18%, log2_M 9%, wave_eff_64x128 9%, log2_ksteps_per_split_64x64 9%, log2_K 8%, wave_eff_64x64 5%, wave_eff_16x64 4%
- **lgbm_cls**: log2_K 17%, log2_N 14%, log2_M 14%, log2_ctas_per_sm_16x64 13%, log2_ksteps_per_split_64x64 9%, log2_ctas_per_sm_16x128 5%, wave_eff_16x64 5%, log2_ctas_per_sm_32x64 4%
- **rf_cls**: log2_N 13%, log2_M 13%, log2_K 12%, log2_ksteps_per_split_64x64 10%, log2_ctas_per_sm_16x64 8%, wave_eff_64x128 6%, log2_ctas_per_sm_32x64 6%, log2_ctas_per_sm_64x64 4%
- **dt_rank**: log2_ksteps 22%, log2_waves 15%, wave_eff 14%, log2_bm 9%, log2_M 5%, log2_bn 5%, log2_bk 4%, log2_warps 3%
- **lgbm_rank**: log2_ksteps 20%, log2_waves 18%, log2_bm 10%, wave_eff 7%, log2_bn 5%, log2_K 5%, log2_M 4%, log2_bk 4%
- **rf_rank**: log2_waves 18%, log2_ksteps 18%, wave_eff 10%, log2_bm 10%, log2_bn 5%, log2_M 5%, log2_K 4%, log2_ksteps_per_split_64x64 4%

## Worst held-out shapes (screen)

| selector | shape | split | oracle us | pick | regret |
|---|---|---|---|---|---|
| b6 | 20x1x11869 | heldout_region | 15.4 | `bm32_bn32_bk32_w4_s5_g1_sk1` | 900.0% |
| b6 | 384x2x10369 | heldout_region | 26.6 | `bm32_bn32_bk32_w4_s5_g1_sk1` | 900.0% |
| b6 | 4x889x14165 | heldout_random | 50.2 | `bm32_bn32_bk32_w4_s5_g1_sk1` | 653.1% |
| knn5 | 28x3030x4012 | heldout_random | 50.2 | `bm32_bn128_bk32_w4_s5_g1_sk1` | 277.7% |
| knn5 | 2x8901x9643 | heldout_region | 255.0 | `bm16_bn128_bk32_w4_s4_g1_sk1` | 94.8% |
| knn5 | 68x6420x10359 | heldout_region | 412.7 | `bm32_bn128_bk32_w4_s3_g8_sk2` | 29.5% |
| knnf | 68x6420x10359 | heldout_region | 412.7 | `bm32_bn64_bk16_w4_s2_g1_sk2` | 45.4% |
| knnf | 79x15624x2135 | heldout_random | 222.2 | `bm32_bn64_bk16_w4_s3_g1_sk2` | 41.0% |
| knnf | 443x4723x393 | heldout_region | 52.2 | `bm128_bn64_bk32_w4_s3_g4_sk1` | 33.0% |
| dt_cls | 417x512x6739 | heldout_region | 86.1 | `bm64_bn128_bk32_w4_s5_g1_sk1` | 183.1% |
| dt_cls | 12x56x836 | heldout_random | 8.2 | `bm16_bn64_bk64_w4_s5_g1_sk1` | 99.7% |
| dt_cls | 51x324x5709 | heldout_random | 23.6 | `bm32_bn128_bk64_w4_s4_g8_sk8` | 95.3% |
| svm_cls | 79x15624x2135 | heldout_random | 222.2 | `bm16_bn64_bk64_w4_s5_g1_sk1` | 126.7% |
| svm_cls | 12x56x836 | heldout_random | 8.2 | `bm16_bn64_bk64_w4_s5_g1_sk1` | 99.7% |
| svm_cls | 384x2x10369 | heldout_region | 26.6 | `bm64_bn32_bk64_w8_s3_g1_sk4` | 96.1% |
| lgbm_cls | 37x922x10110 | heldout_region | 52.2 | `bm32_bn32_bk128_w4_s5_g1_sk1` | 137.3% |
| lgbm_cls | 417x512x6739 | heldout_region | 86.1 | `bm32_bn32_bk128_w4_s5_g1_sk1` | 74.9% |
| lgbm_cls | 2x8901x9643 | heldout_region | 255.0 | `bm16_bn32_bk16_w2_s4_g1_sk1` | 67.5% |
| rf_cls | 37x922x10110 | heldout_region | 52.2 | `bm32_bn32_bk128_w4_s5_g1_sk1` | 137.3% |
| rf_cls | 27x715x3003 | heldout_random | 19.0 | `bm32_bn32_bk128_w4_s5_g1_sk1` | 115.4% |
| rf_cls | 384x2x10369 | heldout_region | 26.6 | `bm64_bn32_bk64_w8_s3_g1_sk4` | 96.1% |
| logreg_cls | 12x56x836 | heldout_random | 8.2 | `bm16_bn64_bk64_w4_s5_g1_sk1` | 99.7% |
| logreg_cls | 384x2x10369 | heldout_region | 26.6 | `bm64_bn32_bk64_w8_s3_g1_sk4` | 96.1% |
| logreg_cls | 1316x272x100 | heldout_random | 8.3 | `bm16_bn64_bk64_w4_s5_g1_sk1` | 61.3% |
| dt_rank | 417x512x6739 | heldout_region | 86.1 | `bm32_bn128_bk64_w2_s3_g1_sk1` | 147.4% |
| dt_rank | 51x324x5709 | heldout_random | 23.6 | `bm32_bn128_bk64_w4_s4_g16_sk8` | 95.3% |
| dt_rank | 1316x272x100 | heldout_random | 8.3 | `bm64_bn128_bk16_w4_s4_g16_sk2` | 84.6% |
| svm_rank | 443x4723x393 | heldout_region | 52.2 | `bm128_bn128_bk16_w8_s3_g4_sk1` | 102.0% |
| svm_rank | 12937x577x4649 | heldout_random | 1664.0 | `bm128_bn128_bk16_w8_s3_g4_sk1` | 89.8% |
| svm_rank | 417x512x6739 | heldout_region | 86.1 | `bm16_bn128_bk32_w8_s5_g1_sk8` | 89.1% |
| lgbm_rank | 226x2141x1139 | heldout_random | 41.4 | `bm64_bn128_bk64_w4_s4_g8_sk1` | 28.6% |
| lgbm_rank | 14244x16x1257 | heldout_random | 70.7 | `bm32_bn32_bk64_w4_s5_g1_sk1` | 23.2% |
| lgbm_rank | 5398x196x774 | heldout_random | 63.5 | `bm64_bn64_bk32_w4_s2_g16_sk1` | 21.0% |
| rf_rank | 19x380x184 | heldout_random | 6.4 | `bm16_bn64_bk32_w4_s5_g1_sk4` | 33.7% |
| rf_rank | 14244x16x1257 | heldout_random | 70.7 | `bm32_bn32_bk64_w4_s4_g4_sk1` | 33.3% |
| rf_rank | 384x2x10369 | heldout_region | 26.6 | `bm32_bn32_bk64_w4_s5_g4_sk8` | 30.5% |
| logreg_rank | 8192x4096x11008 | heldout_region | 12488.0 | `bm128_bn128_bk16_w4_s4_g16_sk1` | 124.6% |
| logreg_rank | 12937x577x4649 | heldout_random | 1664.0 | `bm128_bn128_bk16_w8_s3_g4_sk1` | 89.8% |
| logreg_rank | 68x6420x10359 | heldout_region | 412.7 | `bm128_bn128_bk32_w8_s3_g1_sk2` | 80.1% |

## Stage 2 (cold): `results_lock1755_learned.jsonl` not found


## Stage 2 (graph): `results_lock1755_learned_graph.jsonl` not found

