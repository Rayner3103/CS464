# Learned selectors v0 on held-out shapes (auto-generated)

Five model families x two forms, fit on the tune-split screen (477 shapes x 10840 configs) and scored on 86 held-out shapes. **cls** = shape -> config classification over the first P greedy portfolio members (labels: first member within 2.0% of the shape's best); **rank** = per-candidate scoring over each training shape's top-5 pool (trees / LightGBM regress log regret or lambdarank; LinearSVR regresses log regret and logistic regression classifies regret <= 1.05, both on standardized features plus shape x candidate products). Hyperparameters by grouped 20-fold CV on tune shapes (each fit sees 290-353 shapes; none sharing two of M, N, K with a fold shape), lowest capped mean regret. b6 / knn5 / knnf are the existing baselines (`selector_baselines.md`), CV-scored under the same folds with their weights fixed. Regret = t / oracle - 1.

## Screen: regret vs the held-out per-shape best (full pruned space)

| selector | tune CV mean | tune CV median / p95 / p99 | held-out all median / p95 / p99 | held-out >= 50 us median / p95 / p99 | held-out < 50 us median / p95 / p99 | held-out heldout_random median / p95 / p99 | held-out heldout_region median / p95 / p99 | held-out mean | max | picks the oracle | distinct picks |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b6 | 110.8% | 68.4% / 343.1% / 1001.6% | 48.1% / 487.0% / 973.1% | 85.0% / 395.7% / 589.9% | 28.9% / 524.6% / 1081.4% | 40.0% / 345.5% / 616.8% | 64.7% / 711.6% / 1171.7% | 113.6% | 900.0% | 0.0% | 1 |
| knn5 | 8.4% | 3.5% / 28.2% / 46.4% | 4.7% / 25.4% / 122.2% | 3.6% / 65.4% / 224.7% | 4.8% / 21.5% / 26.1% | 5.0% / 24.9% / 142.3% | 2.1% / 27.5% / 75.2% | 10.8% | 277.7% | 2.3% | 67 |
| knnf | 7.5% | 2.4% / 28.3% / 66.0% | 4.1% / 27.3% / 41.7% | 2.2% / 37.4% / 44.1% | 4.7% / 23.2% / 29.4% | 5.0% / 22.4% / 33.8% | 1.3% / 32.3% / 41.7% | 7.8% | 45.4% | 3.5% | 67 |
| dt_cls | 14.6% | 4.5% / 59.2% / 146.5% | 7.8% / 43.9% / 112.2% | 7.0% / 40.3% / 143.0% | 8.1% / 52.9% / 97.3% | 9.8% / 55.3% / 97.3% | 3.8% / 37.3% / 141.6% | 15.1% | 183.1% | 2.3% | 23 |
| svm_cls | 13.5% | 4.6% / 56.4% / 153.3% | 8.0% / 53.6% / 103.7% | 6.2% / 60.0% / 108.3% | 8.7% / 46.6% / 97.7% | 9.9% / 51.8% / 112.1% | 7.1% / 50.3% / 84.0% | 15.3% | 126.7% | 2.3% | 15 |
| lgbm_cls | 10.1% | 3.4% / 32.2% / 87.3% | 4.1% / 37.7% / 84.2% | 3.5% / 71.5% / 119.2% | 4.3% / 25.8% / 33.8% | 4.3% / 29.6% / 39.3% | 3.4% / 71.2% / 118.5% | 11.0% | 137.3% | 1.2% | 23 |
| rf_cls | 9.0% | 3.3% / 30.9% / 84.1% | 4.3% / 32.7% / 118.7% | 3.2% / 28.5% / 107.5% | 5.3% / 32.5% / 104.8% | 4.3% / 29.0% / 80.1% | 3.4% / 61.4% / 124.9% | 11.4% | 137.3% | 2.3% | 22 |
| logreg_cls | 11.1% | 4.1% / 37.6% / 112.0% | 7.4% / 52.5% / 96.6% | 4.0% / 37.6% / 52.0% | 8.7% / 57.7% / 97.7% | 9.9% / 45.6% / 79.0% | 5.9% / 49.3% / 84.0% | 13.2% | 99.7% | 2.3% | 15 |
| dt_rank | 11.6% | 3.8% / 46.0% / 87.2% | 4.6% / 65.7% / 103.1% | 2.2% / 44.2% / 117.8% | 5.2% / 70.2% / 89.4% | 6.1% / 70.7% / 89.5% | 2.0% / 52.7% / 121.3% | 14.3% | 147.4% | 4.7% | 65 |
| svm_rank | 22.4% | 13.0% / 73.1% / 106.0% | 13.7% / 73.5% / 91.7% | 17.5% / 89.5% / 98.4% | 13.3% / 58.2% / 77.0% | 14.5% / 65.7% / 82.2% | 9.2% / 83.9% / 98.1% | 21.9% | 102.0% | 1.2% | 48 |
| lgbm_rank | 4.8% | 2.3% / 16.6% / 32.5% | 3.6% / 18.1% / 24.0% | 2.2% / 20.6% / 22.5% | 3.9% / 14.2% / 23.1% | 4.1% / 17.8% / 25.7% | 1.9% / 15.2% / 19.7% | 5.5% | 28.6% | 2.3% | 65 |
| rf_rank | 5.3% | 2.6% / 21.2% / 34.8% | 4.4% / 18.5% / 33.4% | 2.0% / 20.0% / 29.7% | 5.1% / 16.2% / 32.0% | 5.1% / 19.5% / 33.5% | 3.2% / 15.5% / 26.0% | 6.4% | 33.7% | 5.8% | 73 |
| logreg_rank | 22.1% | 9.0% / 77.3% / 133.7% | 8.5% / 63.7% / 95.1% | 7.0% / 85.5% / 114.5% | 9.3% / 41.2% / 53.7% | 9.0% / 47.1% / 76.5% | 6.3% / 77.0% / 111.3% | 17.3% | 124.6% | 1.2% | 58 |

## Chosen hyperparameters and candidate ceilings

Ceiling = regret of the best member of the selector's own candidate set (cls label set, rank pool) on each held-out shape: the floor any pick from that set can reach.

| selector | hyperparameters | candidates | grid points | ceiling held-out median / p95 / p99 / mean | CV mean, best / worst grid point |
|---|---|---|---|---|---|
| dt_cls | `{"P": 48, "max_depth": null, "min_samples_leaf": 5}` | 48 | 64 | 1.7% / 12.2% / 19.7% / 3.3% | 14.6% / 39.8% |
| svm_cls | `{"P": 16, "C": 1}` | 16 | 16 | 3.6% / 18.2% / 28.1% / 5.4% | 13.5% / 21.1% |
| lgbm_cls | `{"P": 32, "num_leaves": 16, "n_estimators": 300, "min_child_samples": 3}` | 32 | 36 | 2.0% / 12.5% / 19.8% / 3.6% | 10.1% / 14.3% |
| rf_cls | `{"P": 32, "max_features": 1.0, "min_samples_leaf": 1}` | 32 | 36 | 2.0% / 12.5% / 19.8% / 3.6% | 9.0% / 14.3% |
| logreg_cls | `{"P": 16, "C": 10}` | 16 | 16 | 3.6% / 18.2% / 28.1% / 5.4% | 11.1% / 17.1% |
| dt_rank | `{"max_depth": null, "min_samples_leaf": 5}` | 868 | 12 | 0.0% / 5.7% / 8.5% / 0.7% | 11.6% / 27.3% |
| svm_rank | `{"C": 0.1}` | 868 | 3 | 0.0% / 5.7% / 8.5% / 0.7% | 22.4% / 22.7% |
| lgbm_rank | `{"objective": "regression", "num_leaves": 255, "n_estimators": 1500}` | 868 | 12 | 0.0% / 5.7% / 8.5% / 0.7% | 4.8% / 7.1% |
| rf_rank | `{"max_features": 0.66, "min_samples_leaf": 5}` | 868 | 9 | 0.0% / 5.7% / 8.5% / 0.7% | 5.3% / 6.8% |
| logreg_rank | `{"C": 1}` | 868 | 3 | 0.0% / 5.7% / 8.5% / 0.7% | 22.1% / 27.6% |

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

## Stage 2 (cold): re-measured, `results_lock1755_learned.jsonl`

cold L2, per-launch, adaptive repeats, 3 input placements; all held-out shapes. Per shape: the screen's top-8 tiles, every pick above, B1, B3 and B3+ in one session. 86 shapes, 2760 rows, 0 invalid, 159 power-capped. Oracle = fastest Triton row.

Regret vs the per-shape Triton oracle, median / p95 / p99 (mean of all):

| selector | all (n=86) | >= 50 us (n=30) | < 50 us (n=56) | mean (all) |
|---|---|---|---|---|
| b6 | 47.5% / 481.6% / 937.1% | 77.4% / 387.4% / 584.9% | 25.6% / 519.7% / 1035.6% | 111.1% |
| knn5 | 3.0% / 24.2% / 122.0% | 3.6% / 65.3% / 222.7% | 2.7% / 19.6% / 22.2% | 9.7% |
| knnf | 2.3% / 23.0% / 41.2% | 1.9% / 36.4% / 45.2% | 2.6% / 20.1% / 24.1% | 6.6% |
| dt_cls | 6.4% / 41.4% / 105.7% | 5.3% / 38.3% / 141.3% | 6.5% / 50.6% / 90.5% | 14.2% |
| svm_cls | 6.5% / 51.8% / 96.3% | 5.2% / 57.9% / 107.3% | 7.0% / 44.7% / 90.0% | 14.1% |
| lgbm_cls | 3.0% / 39.0% / 81.1% | 2.3% / 70.1% / 116.6% | 3.5% / 22.3% / 33.2% | 9.9% |
| rf_cls | 3.2% / 32.3% / 116.2% | 1.7% / 28.4% / 105.7% | 4.0% / 32.3% / 100.8% | 10.3% |
| logreg_cls | 5.9% / 51.8% / 89.4% | 3.3% / 38.7% / 51.6% | 6.3% / 55.6% / 90.0% | 11.7% |
| dt_rank | 4.4% / 67.7% / 99.7% | 1.1% / 43.5% / 114.6% | 5.5% / 71.0% / 83.4% | 13.2% |
| svm_rank | 10.5% / 70.4% / 86.9% | 15.9% / 83.7% / 96.4% | 9.9% / 56.1% / 73.7% | 19.7% |
| lgbm_rank | 1.4% / 18.3% / 24.3% | 1.8% / 19.1% / 22.3% | 1.3% / 11.0% / 23.3% | 4.1% |
| rf_rank | 2.9% / 17.3% / 33.7% | 1.2% / 18.6% / 29.8% | 3.8% / 11.0% / 30.0% | 5.1% |
| logreg_rank | 5.4% / 64.6% / 88.6% | 6.8% / 81.7% / 110.3% | 4.8% / 36.0% / 58.4% | 15.3% |

Speedup vs B3+ (t_B3+ / t_pick; > 1 = faster), median [p10, p90] and shapes ahead by > 5% / within 2% / behind by > 5%:

| selector | all | >= 50 us | < 50 us |
|---|---|---|---|
| Triton oracle | 0.996 [0.83, 1.14] · 28/14/30 | 0.964 [0.83, 1.01] · 2/11/14 | 1.039 [0.84, 1.15] · 26/3/16 |
| b6 | 0.658 [0.24, 1.00] · 1/9/66 | 0.564 [0.24, 0.77] · 0/0/29 | 0.837 [0.25, 1.01] · 1/9/37 |
| knn5 | 0.959 [0.71, 1.09] · 20/12/43 | 0.893 [0.64, 1.01] · 2/7/19 | 1.008 [0.75, 1.10] · 18/5/24 |
| knnf | 0.962 [0.70, 1.09] · 18/12/41 | 0.913 [0.68, 1.01] · 2/7/19 | 0.989 [0.73, 1.09] · 16/5/22 |
| dt_cls | 0.941 [0.64, 1.07] · 18/9/47 | 0.911 [0.64, 1.01] · 2/6/19 | 0.951 [0.64, 1.08] · 16/3/28 |
| svm_cls | 0.925 [0.64, 1.06] · 11/17/52 | 0.896 [0.58, 1.01] · 2/4/21 | 0.940 [0.70, 1.07] · 9/13/31 |
| lgbm_cls | 0.950 [0.69, 1.09] · 19/13/43 | 0.913 [0.60, 1.01] · 2/6/19 | 0.971 [0.79, 1.09] · 17/7/24 |
| rf_cls | 0.950 [0.69, 1.09] · 22/10/43 | 0.918 [0.70, 1.01] · 2/6/19 | 0.972 [0.67, 1.10] · 20/4/24 |
| logreg_cls | 0.946 [0.69, 1.06] · 11/20/45 | 0.896 [0.64, 1.01] · 2/5/19 | 0.970 [0.71, 1.07] · 9/15/26 |
| dt_rank | 0.944 [0.64, 1.08] · 15/11/47 | 0.913 [0.67, 1.01] · 2/8/18 | 0.949 [0.64, 1.08] · 13/3/29 |
| svm_rank | 0.895 [0.60, 1.05] · 9/6/59 | 0.845 [0.52, 0.97] · 0/0/24 | 0.914 [0.67, 1.07] · 9/6/35 |
| lgbm_rank | 0.968 [0.76, 1.10] · 23/11/38 | 0.933 [0.78, 1.01] · 2/6/18 | 1.010 [0.76, 1.11] · 21/5/20 |
| rf_rank | 0.967 [0.75, 1.08] · 19/11/40 | 0.929 [0.74, 1.01] · 2/8/17 | 0.980 [0.76, 1.09] · 17/3/23 |
| logreg_rank | 0.932 [0.61, 1.08] · 18/11/47 | 0.870 [0.55, 1.01] · 2/7/21 | 0.968 [0.70, 1.08] · 16/4/26 |

## Stage 2 (graph): re-measured, `results_lock1755_learned_graph.jsonl`

hot L2, CUDA-graph amortized (50 launches); held-out shapes with screen oracle < 50 us. Per shape: the screen's top-8 tiles, every pick above, B1, B3 and B3+ in one session. 56 shapes, 1815 rows, 0 invalid, 99 power-capped. Oracle = fastest Triton row.

Regret vs the per-shape Triton oracle, median / p95 / p99 (mean of all):

| selector | all (n=56) | B3+ < 5 us (n=33) | B3+ 5-15 us (n=10) | B3+ >= 15 us (n=13) | mean (all) |
|---|---|---|---|---|---|
| b6 | 25.6% / 459.8% / 1236.9% | 14.4% / 226.0% / 989.8% | 59.2% / 340.2% / 385.2% | 89.2% / 798.6% / 1076.1% | 107.7% |
| knn5 | 3.7% / 22.1% / 35.9% | 2.8% / 14.4% / 28.9% | 4.2% / 18.2% / 20.4% | 5.0% / 29.4% / 36.0% | 6.8% |
| knnf | 4.7% / 25.4% / 36.5% | 2.8% / 15.4% / 31.0% | 6.7% / 23.6% / 26.5% | 6.3% / 29.2% / 34.5% | 7.5% |
| dt_cls | 10.8% / 51.9% / 115.3% | 7.7% / 71.9% / 121.2% | 15.0% / 43.6% / 53.4% | 9.2% / 37.2% / 38.9% | 17.0% |
| svm_cls | 11.1% / 57.6% / 123.7% | 11.5% / 59.1% / 111.1% | 15.1% / 30.7% / 32.0% | 9.2% / 79.3% / 111.1% | 18.2% |
| lgbm_cls | 5.1% / 31.8% / 48.3% | 4.8% / 19.2% / 41.4% | 4.4% / 22.7% / 24.9% | 6.8% / 40.2% / 45.4% | 8.9% |
| rf_cls | 5.5% / 52.1% / 111.8% | 3.1% / 28.6% / 47.4% | 11.2% / 84.2% / 101.6% | 6.2% / 67.4% / 108.8% | 13.4% |
| logreg_cls | 10.5% / 71.4% / 123.7% | 11.8% / 48.3% / 110.1% | 7.0% / 55.9% / 73.8% | 6.2% / 71.2% / 109.5% | 17.5% |
| dt_rank | 9.4% / 80.0% / 93.7% | 6.6% / 33.7% / 53.4% | 13.7% / 93.7% / 93.9% | 5.0% / 81.6% / 86.6% | 17.4% |
| svm_rank | 15.1% / 85.5% / 92.1% | 6.0% / 36.9% / 73.9% | 24.8% / 90.8% / 92.8% | 30.8% / 70.8% / 82.0% | 21.6% |
| lgbm_rank | 2.4% / 13.5% / 20.9% | 1.6% / 11.2% / 14.2% | 8.2% / 13.6% / 14.5% | 0.5% / 18.8% / 25.7% | 4.3% |
| rf_rank | 3.0% / 13.5% / 31.5% | 3.0% / 12.1% / 23.3% | 6.7% / 13.0% / 13.7% | 2.8% / 20.2% / 32.7% | 5.2% |
| logreg_rank | 8.4% / 53.2% / 89.2% | 5.1% / 41.6% / 77.8% | 15.1% / 59.4% / 82.0% | 19.2% / 49.4% / 61.3% | 15.1% |

Speedup vs B3+ (t_B3+ / t_pick; > 1 = faster), median [p10, p90] and shapes ahead by > 5% / within 2% / behind by > 5%:

| selector | all | B3+ < 5 us | B3+ 5-15 us | B3+ >= 15 us |
|---|---|---|---|---|
| Triton oracle | 1.017 [0.78, 1.18] · 24/5/19 | 1.025 [0.83, 1.18] · 15/2/10 | 1.081 [0.74, 1.29] · 6/1/3 | 0.954 [0.81, 1.15] · 3/2/6 |
| b6 | 0.800 [0.22, 1.02] · 0/7/39 | 0.950 [0.29, 1.03] · 0/6/17 | 0.700 [0.23, 0.92] · 0/1/9 | 0.505 [0.13, 0.64] · 0/0/13 |
| knn5 | 0.973 [0.66, 1.17] · 20/5/25 | 0.985 [0.74, 1.14] · 13/3/12 | 1.019 [0.64, 1.22] · 5/0/4 | 0.878 [0.68, 1.11] · 2/2/9 |
| knnf | 0.951 [0.68, 1.14] · 18/4/29 | 0.983 [0.75, 1.14] · 12/3/16 | 1.024 [0.62, 1.20] · 4/0/4 | 0.925 [0.61, 1.05] · 2/1/9 |
| dt_cls | 0.932 [0.53, 1.13] · 11/6/31 | 0.960 [0.47, 1.14] · 6/6/14 | 0.917 [0.54, 1.18] · 3/0/7 | 0.881 [0.60, 1.06] · 2/0/10 |
| svm_cls | 0.892 [0.61, 1.10] · 9/2/38 | 0.924 [0.58, 1.10] · 5/2/20 | 0.885 [0.68, 1.19] · 3/0/7 | 0.857 [0.66, 1.02] · 1/0/11 |
| lgbm_cls | 0.923 [0.69, 1.15] · 17/3/32 | 0.920 [0.74, 1.16] · 11/2/17 | 1.007 [0.68, 1.19] · 5/0/5 | 0.864 [0.61, 1.03] · 1/1/10 |
| rf_cls | 0.932 [0.61, 1.15] · 20/2/31 | 0.920 [0.70, 1.16] · 13/1/17 | 1.007 [0.43, 1.18] · 5/0/5 | 0.927 [0.57, 1.06] · 2/1/9 |
| logreg_cls | 0.910 [0.63, 1.10] · 9/5/34 | 0.915 [0.65, 1.07] · 4/5/18 | 0.885 [0.61, 1.19] · 3/0/6 | 0.858 [0.66, 1.07] · 2/0/10 |
| dt_rank | 0.896 [0.55, 1.13] · 12/5/31 | 0.920 [0.73, 1.12] · 7/1/18 | 0.876 [0.52, 1.15] · 3/2/5 | 0.884 [0.56, 1.05] · 2/2/8 |
| svm_rank | 0.874 [0.57, 1.10] · 8/7/34 | 0.969 [0.63, 1.15] · 8/5/16 | 0.765 [0.57, 1.01] · 0/2/7 | 0.743 [0.54, 0.95] · 0/0/11 |
| lgbm_rank | 0.992 [0.72, 1.17] · 18/7/22 | 0.999 [0.76, 1.17] · 12/4/12 | 1.016 [0.67, 1.18] · 3/1/3 | 0.927 [0.74, 1.14] · 3/2/7 |
| rf_rank | 0.979 [0.70, 1.16] · 18/4/25 | 0.987 [0.75, 1.16] · 11/3/14 | 0.999 [0.67, 1.21] · 4/0/3 | 0.932 [0.72, 1.08] · 3/1/8 |
| logreg_rank | 0.919 [0.59, 1.12] · 13/5/31 | 0.927 [0.64, 1.12] · 8/3/17 | 0.949 [0.57, 1.12] · 4/1/5 | 0.858 [0.61, 1.03] · 1/1/9 |
