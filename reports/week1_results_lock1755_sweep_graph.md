# Week-1 results (auto-generated)

hw_id `rtx3080_sm86_68sms_10gb` | driver 595.99.02 | torch 2.13.0+cu130 | triton 3.7.1 | harness `dd0142e6`

## B0 roofline

| spec TFLOP/s @boost | spec TFLOP/s @max clk | measured TFLOP/s | spec GB/s | measured GB/s |
|---|---|---|---|---|
| 59.5 | 73.1 | 59.7 | 760.1 | 686.9 |

## noise

### Sessions

| session | date | driver | clock | clk_obs | rounds | complete |
|---|---|---|---|---|---|---|
| noise-20261013-231817-2a31 | 10-13 23:18 | 595.99.02 | unrecorded | 1590-1755 | 2 | NO |
| noise-20261013-232542-642a | 10-13 23:25 | 595.99.02 | unrecorded | 1575-1755 | 3 | yes |
| noise-20261014-001639-lock1755 | 10-14 00:16 | 595.99.02 | locked@1755 | 1590-1755 | 3 | yes |
| noise-20261014-141218-6985 | 10-14 14:12 | 595.99.02 | locked@1755 | 1590-1755 | 3 | yes |
| noise-s1-20261009 | 10-09 11:57 | 595.91.07 | unrecorded | 1590-1965 | 3 | yes |
| noise-s2-multialloc-20261009 | 10-09 12:21 | 595.91.07 | unrecorded | 1605-1935 | 3 | yes |

Only compare sessions with the same driver and clock state; ignore sessions with complete = NO (fewer rounds understate the range).

### Noise summary per session (relative; 0.01 = 1%)

| session | cache_mode | clock | complete | n_allocs | range_p50 | range_p95 | range_p99 | range_max | range_p95_ge50us | range_p95_lt50us | within_iqr_p50 | paired_p50 | paired_p95 | paired_p99 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| noise-20261013-231817-2a31 | cold | unrecorded | NO | 3 | 0.0000 | 0.0013 | 0.0283 | 0.0532 | 0.0009 | 0.0133 | 0.0066 | 0.0000 | 0.0111 | 0.0532 |
| noise-20261013-231817-2a31 | hot | unrecorded | NO | 3 | 0.0000 | 0.0005 | 0.0048 | 0.0071 | 0.0016 | 0.0000 | 0.0057 | 0.0000 | 0.0030 | 0.0089 |
| noise-20261013-232542-642a | cold | unrecorded | yes | 3 | 0.0006 | 0.0390 | 0.0797 | 0.0837 | 0.0134 | 0.0786 | 0.0055 | 0.0038 | 0.0837 | 0.0960 |
| noise-20261013-232542-642a | hot | unrecorded | yes | 3 | 0.0000 | 0.0125 | 0.0176 | 0.0188 | 0.0120 | 0.0038 | 0.0036 | 0.0013 | 0.0162 | 0.0188 |
| noise-20261014-001639-lock1755 | cold | locked@1755 | yes | 3 | 0.0027 | 0.0361 | 0.0647 | 0.0762 | 0.0184 | 0.0615 | 0.0062 | 0.0064 | 0.0566 | 0.0981 |
| noise-20261014-001639-lock1755 | hot | locked@1755 | yes | 3 | 0.0010 | 0.0187 | 0.0267 | 0.0316 | 0.0182 | 0.0118 | 0.0056 | 0.0029 | 0.0332 | 0.0351 |
| noise-20261014-141218-6985 | cold | locked@1755 | yes | 3 | 0.0011 | 0.0359 | 0.0898 | 0.1494 | 0.0137 | 0.0642 | 0.0063 | 0.0036 | 0.0415 | 0.1494 |
| noise-20261014-141218-6985 | hot | locked@1755 | yes | 3 | 0.0001 | 0.0216 | 0.0409 | 0.0463 | 0.0185 | 0.0230 | 0.0049 | 0.0038 | 0.0251 | 0.0303 |
| noise-s1-20261009 | cold | unrecorded | yes | 1 | 0.0067 | 0.0537 | 0.1252 | 0.2000 | 0.0521 | 0.0922 | 0.0071 | 0.0207 | 0.0621 | 0.2000 |
| noise-s1-20261009 | hot | unrecorded | yes | 1 | 0.0070 | 0.0544 | 0.0647 | 0.0667 | 0.0538 | 0.0515 | 0.0067 | 0.0151 | 0.0667 | 0.0682 |
| noise-s2-multialloc-20261009 | cold | unrecorded | yes | 3 | 0.0029 | 0.0264 | 0.0633 | 0.0958 | 0.0239 | 0.0438 | 0.0091 | 0.0033 | 0.0348 | 0.1027 |
| noise-s2-multialloc-20261009 | hot | unrecorded | yes | 3 | 0.0021 | 0.0266 | 0.0396 | 0.0408 | 0.0175 | 0.0282 | 0.0090 | 0.0064 | 0.0419 | 0.0454 |

`range` = (max - min) / median of the per-round medians for the same (shape, kernel, cache mode), over 3 rounds. `paired` = the same statistic for the ratio of the two kernels' times (the quantity that decides which kernel wins). `within_iqr` = IQR / median of one run's samples.

### Noisiest 10 (session, shape, kernel, mode)

| session_id | shape | kernel | cache_mode | median_us | within_iqr_rel | across_round_range_rel |
|---|---|---|---|---|---|---|
| noise-s1-20261009 | 64x64x64 | B1 torch.matmul | cold | 5.1200 | 0.0563 | 0.2000 |
| noise-20261014-141218-6985 | 56x29x175 | B1 torch.matmul | cold | 6.3200 | 0.1620 | 0.1494 |
| noise-s2-multialloc-20261009 | 16x16x4096 | B1 torch.matmul | cold | 8.6880 | 0.1179 | 0.0958 |
| noise-20261013-232542-642a | 56x29x175 | B1 torch.matmul | cold | 6.8800 | 0.1488 | 0.0837 |
| noise-20261013-232542-642a | 1x1024x1024 | B1 torch.matmul | cold | 9.1520 | 0.1119 | 0.0769 |
| noise-20261014-001639-lock1755 | 1x1024x1024 | B1 torch.matmul | cold | 9.0240 | 0.1135 | 0.0762 |
| noise-s1-20261009 | 1000x1000x1000 | B1 torch.matmul | cold | 41.9840 | 0.0016 | 0.0732 |
| noise-s1-20261009 | 100x4093x509 | B1 torch.matmul | hot | 30.7200 | 0.0011 | 0.0667 |
| noise-s1-20261009 | 8192x64x8192 | ours:bm128_bn128_bk32_w4_s4_g8_sk1 | cold | 421.8880 | 0.0052 | 0.0655 |
| noise-s1-20261009 | 8192x64x8192 | ours:bm128_bn128_bk32_w4_s4_g8_sk1 | hot | 419.8400 | 0.0046 | 0.0634 |

### Cold / hot L2 median-time ratio over the noise set

| n | median | p10 | p90 | min | max |
|---|---|---|---|---|---|
| 360.000 | 1.003 | 0.998 | 1.120 | 0.980 | 1.277 |

Largest ratios:

| session_id | shape | kernel | hot | cold | delta_us | cold_over_hot |
|---|---|---|---|---|---|---|
| noise-20261013-232542-642a | 1x1024x1024 | B1 torch.matmul | 7.168 | 9.152 | 1.984 | 1.277 |
| noise-20261014-141218-6985 | 1x1024x1024 | B1 torch.matmul | 7.168 | 9.040 | 1.872 | 1.261 |
| noise-20261014-001639-lock1755 | 1x1024x1024 | B1 torch.matmul | 7.200 | 9.024 | 1.824 | 1.253 |
| noise-20261014-141218-6985 | 64x64x64 | B1 torch.matmul | 4.096 | 5.120 | 1.024 | 1.250 |
| noise-20261013-231817-2a31 | 64x64x64 | B1 torch.matmul | 4.096 | 5.120 | 1.024 | 1.250 |
| noise-s1-20261009 | 64x64x64 | B1 torch.matmul | 4.096 | 5.120 | 1.024 | 1.250 |
| noise-20261013-232542-642a | 64x64x64 | B1 torch.matmul | 4.096 | 5.120 | 1.024 | 1.250 |
| noise-20261014-001639-lock1755 | 64x64x64 | B1 torch.matmul | 4.096 | 5.120 | 1.024 | 1.250 |

## drift

### Thermal drift

| file | minutes | first_min_median_us | last_min_median_us | drift_rel | clock_min | clock_max | temp_start | temp_max | power_median | throttled_intervals |
|---|---|---|---|---|---|---|---|---|---|---|
| drift-20261009-120343.csv | 10.01 | 1.807e+04 | 1.807e+04 | 9.565e-05 | 1770 | 1830 | 81 | 82 | 327 | 591 |

## baselines

Source: `results_lock1755_sweep_graph.jsonl`

| session_id | rows | driver | clock | clk_obs | n_allocs |
|---|---|---|---|---|---|
| bench-20261015-125953-lock1755-sweep-graph | 2259 | 595.99.02 | locked | 1635-1755 | 3 |


## Cache mode: hot

Shapes: 74. B6 (best single hand-picked config by geomean regret): `bm16_bn64_bk128_w4_s4_g1_sk1`.

Speedup columns are time_baseline / time_ours (>1 means ours is faster).

| regime | metric | n | median | p10 | p90 | min | max |
|---|---|---|---|---|---|---|---|
| S1 square | ours/B3+ | 5 | 1.093 | 0.960 | 1.189 | 0.946 | 1.211 |
| S1 square | sweep/handpicked | 5 | 1.099 | 1.042 | 1.254 | 1.039 | 1.316 |
| S1 square | ours/B3 | 5 | 1.211 | 1.118 | 1.398 | 1.093 | 1.486 |
| S1 square | ours/B1 | 5 | 1.211 | 1.119 | 1.449 | 1.093 | 1.570 |
| S1 square | B3+/B3 | 5 | 1.000 | 1.000 | 1.458 | 1.000 | 1.571 |
| S1 square | B3+/B1 | 5 | 1.003 | 1.000 | 1.512 | 0.999 | 1.660 |
| S1 square | B3/B1 | 5 | 1.001 | 1.000 | 1.035 | 0.999 | 1.056 |
| S1 square | B2/B1 | 5 | 1.000 | 1.000 | 1.001 | 0.999 | 1.001 |
| S1 square | oracle/B6 | 5 | 1.316 | 1.124 | 2.324 | 1.099 | 2.388 |
| S1 square | ours/B5 | 5 | 5.285 | 3.039 | 7.124 | 1.927 | 7.488 |
| S2 decode (M<=16) | ours/B3+ | 2 | 0.939 | 0.936 | 0.943 | 0.935 | 0.944 |
| S2 decode (M<=16) | sweep/handpicked | 2 | 1.224 | 1.111 | 1.337 | 1.083 | 1.365 |
| S2 decode (M<=16) | ours/B3 | 2 | 1.024 | 0.990 | 1.057 | 0.982 | 1.065 |
| S2 decode (M<=16) | ours/B1 | 2 | 1.279 | 1.182 | 1.377 | 1.157 | 1.401 |
| S2 decode (M<=16) | B3+/B3 | 2 | 1.090 | 1.050 | 1.130 | 1.040 | 1.140 |
| S2 decode (M<=16) | B3+/B1 | 2 | 1.361 | 1.263 | 1.460 | 1.238 | 1.485 |
| S2 decode (M<=16) | B3/B1 | 2 | 1.257 | 1.120 | 1.393 | 1.086 | 1.427 |
| S2 decode (M<=16) | B2/B1 | 2 | 0.999 | 0.999 | 1.000 | 0.999 | 1.000 |
| S2 decode (M<=16) | oracle/B6 | 2 | 1.224 | 1.111 | 1.337 | 1.083 | 1.365 |
| S2 decode (M<=16) | ours/B5 | 2 | 20.473 | 20.078 | 20.868 | 19.979 | 20.966 |
| S2 small batch (32<=M<=128) | ours/B3+ | 1 | 0.952 | 0.952 | 0.952 | 0.952 | 0.952 |
| S2 small batch (32<=M<=128) | sweep/handpicked | 1 | 1.084 | 1.084 | 1.084 | 1.084 | 1.084 |
| S2 small batch (32<=M<=128) | ours/B3 | 1 | 1.187 | 1.187 | 1.187 | 1.187 | 1.187 |
| S2 small batch (32<=M<=128) | ours/B1 | 1 | 1.184 | 1.184 | 1.184 | 1.184 | 1.184 |
| S2 small batch (32<=M<=128) | B3+/B3 | 1 | 1.247 | 1.247 | 1.247 | 1.247 | 1.247 |
| S2 small batch (32<=M<=128) | B3+/B1 | 1 | 1.244 | 1.244 | 1.244 | 1.244 | 1.244 |
| S2 small batch (32<=M<=128) | B3/B1 | 1 | 0.998 | 0.998 | 0.998 | 0.998 | 0.998 |
| S2 small batch (32<=M<=128) | B2/B1 | 1 | 0.998 | 0.998 | 0.998 | 0.998 | 0.998 |
| S2 small batch (32<=M<=128) | oracle/B6 | 1 | 1.084 | 1.084 | 1.084 | 1.084 | 1.084 |
| S2 small batch (32<=M<=128) | ours/B5 | 1 | 19.893 | 19.893 | 19.893 | 19.893 | 19.893 |
| S3 skinny/tall | ours/B3+ | 5 | 1.107 | 0.885 | 1.132 | 0.853 | 1.140 |
| S3 skinny/tall | sweep/handpicked | 5 | 1.252 | 1.114 | 1.676 | 1.107 | 1.848 |
| S3 skinny/tall | ours/B3 | 5 | 1.123 | 0.900 | 1.172 | 0.853 | 1.176 |
| S3 skinny/tall | ours/B1 | 5 | 1.163 | 1.108 | 1.287 | 1.077 | 1.307 |
| S3 skinny/tall | B3+/B3 | 5 | 1.023 | 1.001 | 1.054 | 1.000 | 1.062 |
| S3 skinny/tall | B3+/B1 | 5 | 1.181 | 1.061 | 1.253 | 1.020 | 1.263 |
| S3 skinny/tall | B3/B1 | 5 | 1.119 | 1.043 | 1.233 | 0.997 | 1.263 |
| S3 skinny/tall | B2/B1 | 5 | 1.000 | 0.998 | 1.001 | 0.997 | 1.001 |
| S3 skinny/tall | oracle/B6 | 5 | 1.252 | 1.114 | 1.819 | 1.107 | 1.848 |
| S3 skinny/tall | ours/B5 | 5 | 15.947 | 13.912 | 18.413 | 12.691 | 18.541 |
| S4 large-K | ours/B3+ | 24 | 0.973 | 0.899 | 1.053 | 0.845 | 1.137 |
| S4 large-K | sweep/handpicked | 24 | 1.754 | 1.100 | 2.969 | 1.092 | 3.486 |
| S4 large-K | ours/B3 | 24 | 1.044 | 0.930 | 1.506 | 0.916 | 1.839 |
| S4 large-K | ours/B1 | 24 | 1.132 | 0.976 | 1.545 | 0.916 | 2.189 |
| S4 large-K | B3+/B3 | 24 | 1.078 | 1.000 | 1.483 | 1.000 | 1.850 |
| S4 large-K | B3+/B1 | 24 | 1.173 | 1.000 | 1.636 | 0.999 | 2.201 |
| S4 large-K | B3/B1 | 24 | 1.004 | 0.999 | 1.200 | 0.997 | 1.299 |
| S4 large-K | B2/B1 | 24 | 1.000 | 0.998 | 1.003 | 0.997 | 1.004 |
| S4 large-K | oracle/B6 | 24 | 3.106 | 1.928 | 6.815 | 1.483 | 7.659 |
| S4 large-K | ours/B5 | 24 | 61.000 | 30.449 | 131.058 | 19.987 | 160.414 |
| S5 unaligned | ours/B3+ | 15 | 0.976 | 0.847 | 1.110 | 0.798 | 1.230 |
| S5 unaligned | sweep/handpicked | 15 | 1.154 | 1.039 | 1.531 | 1.012 | 1.884 |
| S5 unaligned | ours/B3 | 15 | 1.124 | 0.954 | 1.251 | 0.898 | 1.297 |
| S5 unaligned | ours/B1 | 15 | 1.231 | 1.016 | 1.703 | 1.001 | 1.886 |
| S5 unaligned | B3+/B3 | 15 | 1.125 | 1.001 | 1.293 | 1.000 | 1.426 |
| S5 unaligned | B3+/B1 | 15 | 1.187 | 1.032 | 1.806 | 1.000 | 1.925 |
| S5 unaligned | B3/B1 | 15 | 1.017 | 0.999 | 1.478 | 0.998 | 1.692 |
| S5 unaligned | B2/B1 | 15 | 1.000 | 0.996 | 1.002 | 0.991 | 1.003 |
| S5 unaligned | oracle/B6 | 15 | 2.250 | 1.108 | 3.742 | 1.086 | 3.976 |
| S5 unaligned | ours/B5 | 15 | 6.773 | 2.955 | 30.077 | 2.898 | 44.363 |
| S6 wave-boundary | ours/B3+ | 3 | 0.971 | 0.947 | 0.977 | 0.942 | 0.979 |
| S6 wave-boundary | sweep/handpicked | 3 | 1.039 | 1.037 | 1.054 | 1.036 | 1.058 |
| S6 wave-boundary | ours/B3 | 3 | 0.971 | 0.954 | 1.161 | 0.950 | 1.208 |
| S6 wave-boundary | ours/B1 | 3 | 0.999 | 0.961 | 1.165 | 0.952 | 1.206 |
| S6 wave-boundary | B3+/B3 | 3 | 1.009 | 1.002 | 1.189 | 1.000 | 1.234 |
| S6 wave-boundary | B3+/B1 | 3 | 1.028 | 1.014 | 1.192 | 1.011 | 1.233 |
| S6 wave-boundary | B3/B1 | 3 | 1.002 | 0.999 | 1.023 | 0.999 | 1.028 |
| S6 wave-boundary | B2/B1 | 3 | 1.000 | 0.999 | 1.000 | 0.999 | 1.000 |
| S6 wave-boundary | oracle/B6 | 3 | 2.282 | 2.007 | 2.447 | 1.938 | 2.488 |
| S6 wave-boundary | ours/B5 | 3 | 2.536 | 2.042 | 3.277 | 1.918 | 3.462 |
| S7 random | ours/B3+ | 19 | 1.018 | 0.839 | 1.162 | 0.628 | 1.198 |
| S7 random | sweep/handpicked | 19 | 1.348 | 1.093 | 1.539 | 1.029 | 2.182 |
| S7 random | ours/B3 | 19 | 1.087 | 0.860 | 1.208 | 0.660 | 1.302 |
| S7 random | ours/B1 | 19 | 1.142 | 0.991 | 1.385 | 0.873 | 2.082 |
| S7 random | B3+/B3 | 19 | 1.000 | 1.000 | 1.184 | 1.000 | 1.305 |
| S7 random | B3+/B1 | 19 | 1.087 | 1.017 | 1.546 | 0.999 | 1.803 |
| S7 random | B3/B1 | 19 | 1.029 | 1.000 | 1.485 | 0.999 | 1.803 |
| S7 random | B2/B1 | 19 | 1.000 | 0.999 | 1.002 | 0.994 | 1.003 |
| S7 random | oracle/B6 | 19 | 1.536 | 1.093 | 2.378 | 1.029 | 5.879 |
| S7 random | ours/B5 | 19 | 8.983 | 4.871 | 21.186 | 4.188 | 69.802 |
| ALL | ours/B3+ | 74 | 0.979 | 0.863 | 1.139 | 0.628 | 1.230 |
| ALL | sweep/handpicked | 74 | 1.320 | 1.056 | 2.028 | 1.012 | 3.486 |
| ALL | ours/B3 | 74 | 1.085 | 0.922 | 1.278 | 0.660 | 1.839 |
| ALL | ours/B1 | 74 | 1.158 | 0.998 | 1.545 | 0.873 | 2.189 |
| ALL | B3+/B3 | 74 | 1.045 | 1.000 | 1.300 | 1.000 | 1.850 |
| ALL | B3+/B1 | 74 | 1.177 | 1.000 | 1.633 | 0.999 | 2.201 |
| ALL | B3/B1 | 74 | 1.021 | 0.999 | 1.401 | 0.997 | 1.803 |
| ALL | B2/B1 | 74 | 1.000 | 0.998 | 1.002 | 0.991 | 1.004 |
| ALL | oracle/B6 | 74 | 2.116 | 1.102 | 4.211 | 1.029 | 7.659 |
| ALL | ours/B5 | 74 | 15.846 | 3.374 | 75.330 | 1.918 | 160.414 |

Deep search (B3+): exhaustively screened on 74/74 shapes.


**G2** (oracle >= ~90% of B3+ on the median shape): median ours/B3+ = 0.979 -> PASS. cuBLASLt returned a median of 8 candidates per shape (min 1, max 16).

**G4** (ours beats B3+ by >5% in some identifiable regime):

| regime | n | ours > B3+ by >5% | share | ours < B3+ by >5% | within 2% |
|---|---|---|---|---|---|
| S1 square | 5 | 3 | 0.60 | 1 | 1 |
| S2 decode (M<=16) | 2 | 0 | 0.00 | 2 | 0 |
| S2 small batch (32<=M<=128) | 1 | 0 | 0.00 | 0 | 0 |
| S3 skinny/tall | 5 | 3 | 0.60 | 2 | 0 |
| S4 large-K | 24 | 4 | 0.17 | 9 | 6 |
| S5 unaligned | 15 | 3 | 0.20 | 5 | 3 |
| S6 wave-boundary | 3 | 0 | 0.00 | 1 | 0 |
| S7 random | 19 | 9 | 0.47 | 5 | 3 |

torch.matmul (B1) launches the same kernel(s) as the cuBLASLt top-1 (B2) on 74/74 shapes.


Oracle winner counts (65 distinct configs; top 10): `bm16_bn32_bk64_w2_s5_g1_sk16`: 4, `bm64_bn64_bk32_w4_s4_g8_sk8`: 3, `bm16_bn64_bk32_w4_s5_g1_sk4`: 2, `bm16_bn32_bk64_w2_s5_g1_sk1`: 2, `bm16_bn32_bk128_w2_s3_g1_sk8`: 2, `bm64_bn128_bk16_w4_s5_g8_sk1`: 2, `bm16_bn64_bk64_w4_s2_g1_sk8`: 1, `bm16_bn32_bk16_w2_s4_g1_sk1`: 1, `bm16_bn32_bk32_w2_s2_g1_sk8`: 1, `bm16_bn32_bk128_w2_s4_g1_sk16`: 1


### S1 square sweep (median us)

| M | B1 torch.matmul | B5 triton tutorial | B2 cublasLt top-1 | B3 cublasLt best-of-N | B3+ cublasLt deep | oracle(ours) | oracle_cfg | B1_tflops | oracle_tflops | ours/B1 | ours/B5 | ours/B3+ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 2.06 | 8.35 | 2.06 | 2.05 | 2.05 | 1.77 | bm16_bn32_bk32_w2_s4_g1_sk1 | 0.26 | 0.30 | 1.16 | 4.71 | 1.15 |
| 128 | 2.53 | 13.72 | 2.53 | 2.53 | 2.53 | 2.09 | bm16_bn32_bk32_w2_s4_g4_sk1 | 1.66 | 2.01 | 1.21 | 6.58 | 1.21 |
| 256 | 3.49 | 23.93 | 3.49 | 3.49 | 3.49 | 3.20 | bm32_bn32_bk32_w4_s5_g8_sk1 | 9.61 | 10.50 | 1.09 | 7.49 | 1.09 |
| 512 | 12.86 | 43.28 | 12.86 | 12.17 | 7.75 | 8.19 | bm32_bn128_bk32_w4_s4_g16_sk1 | 20.88 | 32.78 | 1.57 | 5.29 | 0.95 |
| 1024 | 54.29 | 82.58 | 54.24 | 54.24 | 42.10 | 42.86 | bm64_bn64_bk16_w2_s3_g8_sk1 | 39.55 | 50.11 | 1.27 | 1.93 | 0.98 |
