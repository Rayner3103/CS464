# Joint online search: v1 vs v2 (auto-generated)

Stage 2 (one session for every pick and baseline): `results_lock1755_v2.jsonl` (cold), `results_lock1755_v2_graph.jsonl` (graph, < 50 us). 86 held-out shapes, 56 timed graph-amortized. Ratio < 1 = faster than the reference; counts are shapes faster / slower than the reference by more than the tie margin (5% below 50 us, 2% above). Regret is vs the joint oracle (every Triton config timed in the session and B3+).

## all (n = 86)

| ranker | selector | first-use cost | vs B2 median / p10 | faster / slower | vs B3 median / p10 | faster / slower | vs B3+ median / p10 | faster / slower | regret vs joint oracle median / p95 (mean) | picks cuBLASLt |
|---|---|---|---|---|---|---|---|---|---|---|
| v1 | top-3 Triton only | 0.52 s | 0.962 / 0.701 | 41 / 18 | 0.981 / 0.834 | 32 / 27 | 1.019 / 0.852 | 24 / 38 | 4.3% / 41.7% (11.5%) | 0% |
| v1 | top-3 + LT16 | 0.59 s | 0.911 / 0.698 | 49 / 0 | 0.981 / 0.834 | 32 / 0 | 1.000 / 0.852 | 24 / 14 | 0.4% / 12.5% (3.0%) | 41% |
| v1 | top-5 + LT16 | 0.90 s | 0.901 / 0.697 | 52 / 0 | 0.980 / 0.822 | 34 / 0 | 1.000 / 0.852 | 26 / 13 | 0.3% / 12.0% (2.7%) | 41% |
| v1 | top-1 + LT16 | 0.33 s | 0.921 / 0.698 | 47 / 0 | 1.000 / 0.848 | 26 / 0 | 1.000 / 0.870 | 20 / 15 | 1.6% / 13.7% (3.9%) | 51% |
| v2 | top-3 Triton only | 0.60 s | 0.949 / 0.698 | 43 / 15 | 0.978 / 0.820 | 30 / 23 | 1.016 / 0.852 | 23 / 35 | 3.0% / 33.8% (10.2%) | 0% |
| v2 | top-3 + LT16 | 0.67 s | 0.902 / 0.680 | 49 / 0 | 0.979 / 0.820 | 30 / 0 | 1.000 / 0.852 | 23 / 14 | 0.4% / 10.4% (2.7%) | 41% |
| v2 | top-5 + LT16 | 0.96 s | 0.902 / 0.680 | 51 / 0 | 0.979 / 0.814 | 32 / 0 | 1.000 / 0.852 | 25 / 14 | 0.3% / 10.4% (2.5%) | 41% |
| v2 | top-1 + LT16 | 0.38 s | 0.923 / 0.698 | 47 / 0 | 0.987 / 0.848 | 25 / 0 | 1.000 / 0.871 | 20 / 15 | 0.6% / 14.9% (3.7%) | 44% |

## < 50 us (n = 56)

| ranker | selector | first-use cost | vs B2 median / p10 | faster / slower | vs B3 median / p10 | faster / slower | vs B3+ median / p10 | faster / slower | regret vs joint oracle median / p95 (mean) | picks cuBLASLt |
|---|---|---|---|---|---|---|---|---|---|---|
| v1 | top-3 Triton only | 0.48 s | 0.900 / 0.698 | 31 / 12 | 0.962 / 0.834 | 26 / 17 | 0.989 / 0.848 | 22 / 21 | 2.4% / 49.1% (12.7%) | 0% |
| v1 | top-3 + LT16 | 0.55 s | 0.896 / 0.648 | 35 / 0 | 0.962 / 0.834 | 26 / 0 | 0.989 / 0.848 | 22 / 7 | 0.2% / 12.5% (2.7%) | 38% |
| v1 | top-5 + LT16 | 0.82 s | 0.896 / 0.648 | 37 / 0 | 0.961 / 0.822 | 27 / 0 | 0.989 / 0.848 | 24 / 7 | 0.1% / 12.5% (2.5%) | 38% |
| v1 | top-1 + LT16 | 0.31 s | 0.897 / 0.648 | 34 / 0 | 0.987 / 0.836 | 22 / 0 | 1.000 / 0.852 | 18 / 7 | 1.1% / 13.0% (3.7%) | 46% |
| v2 | top-3 Triton only | 0.55 s | 0.901 / 0.674 | 32 / 10 | 0.965 / 0.814 | 24 / 15 | 0.993 / 0.848 | 21 / 21 | 2.6% / 49.9% (11.5%) | 0% |
| v2 | top-3 + LT16 | 0.62 s | 0.896 / 0.641 | 35 / 0 | 0.965 / 0.814 | 24 / 0 | 0.993 / 0.848 | 21 / 7 | 0.2% / 9.6% (2.3%) | 38% |
| v2 | top-5 + LT16 | 0.88 s | 0.896 / 0.641 | 37 / 0 | 0.965 / 0.814 | 26 / 0 | 0.993 / 0.848 | 23 / 7 | 0.1% / 9.6% (2.2%) | 38% |
| v2 | top-1 + LT16 | 0.35 s | 0.911 / 0.641 | 34 / 0 | 0.978 / 0.831 | 21 / 0 | 0.998 / 0.857 | 18 / 7 | 0.4% / 13.4% (3.3%) | 39% |

## >= 50 us (n = 30)

| ranker | selector | first-use cost | vs B2 median / p10 | faster / slower | vs B3 median / p10 | faster / slower | vs B3+ median / p10 | faster / slower | regret vs joint oracle median / p95 (mean) | picks cuBLASLt |
|---|---|---|---|---|---|---|---|---|---|---|
| v1 | top-3 Triton only | 0.61 s | 0.978 / 0.794 | 17 / 9 | 0.998 / 0.872 | 13 / 12 | 1.057 / 0.988 | 2 / 20 | 5.7% / 30.1% (9.4%) | 0% |
| v1 | top-3 + LT16 | 0.68 s | 0.972 / 0.792 | 21 / 0 | 0.998 / 0.872 | 13 / 0 | 1.014 / 0.988 | 2 / 15 | 1.7% / 11.4% (3.5%) | 47% |
| v1 | top-5 + LT16 | 1.03 s | 0.960 / 0.792 | 20 / 0 | 0.990 / 0.872 | 13 / 0 | 1.005 / 0.988 | 2 / 13 | 0.8% / 10.9% (3.0%) | 47% |
| v1 | top-1 + LT16 | 0.38 s | 0.978 / 0.792 | 16 / 0 | 1.000 / 0.878 | 8 / 0 | 1.025 / 0.993 | 2 / 16 | 2.5% / 17.3% (4.2%) | 60% |
| v2 | top-3 Triton only | 0.68 s | 0.977 / 0.783 | 18 / 8 | 0.990 / 0.874 | 14 / 11 | 1.039 / 0.990 | 2 / 17 | 3.9% / 26.1% (7.8%) | 0% |
| v2 | top-3 + LT16 | 0.76 s | 0.971 / 0.783 | 20 / 0 | 1.000 / 0.874 | 13 / 0 | 1.006 / 0.990 | 2 / 13 | 0.9% / 12.1% (3.3%) | 47% |
| v2 | top-5 + LT16 | 1.09 s | 0.972 / 0.783 | 20 / 0 | 0.998 / 0.862 | 13 / 0 | 1.006 / 0.991 | 2 / 13 | 0.6% / 10.2% (3.1%) | 47% |
| v2 | top-1 + LT16 | 0.42 s | 0.976 / 0.791 | 19 / 0 | 1.000 / 0.888 | 11 / 0 | 1.014 / 0.990 | 2 / 15 | 2.4% / 18.2% (4.4%) | 53% |

## Triton only: regret vs the Triton oracle of the session (median / p95 (mean))

| selector | v1 all | v1 < 50 us | v1 >= 50 us | v2 all | v2 < 50 us | v2 >= 50 us |
|---|---|---|---|---|---|---|
| top-3 Triton only | 0.4% / 9.5% (2.3%) | 0.2% / 9.8% (2.1%) | 0.9% / 9.4% (2.6%) | 0.2% / 4.6% (1.2%) | 0.1% / 5.0% (1.2%) | 0.3% / 3.3% (1.1%) |

## Paired: v2 vs v1 (chosen config's stage-2 time; geometric-mean ratio, 95% bootstrap interval)

| selector | geomean ratio | 95% interval | v2 faster / slower (> tie margin) |
|---|---|---|---|
| top-3 Triton only | 0.9897 | [0.9823, 0.9970] | 11 / 4 of 86 |
| top-3 + LT16 | 0.9970 | [0.9917, 1.0018] | 4 / 4 of 86 |
| top-5 + LT16 | 0.9985 | [0.9942, 1.0022] | 1 / 2 of 86 |
| top-1 + LT16 | 0.9977 | [0.9899, 1.0050] | 6 / 8 of 86 |
