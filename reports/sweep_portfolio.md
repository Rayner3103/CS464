# Triton config sweep: portfolio study (auto-generated)

Source: `/home/raynerubuntu/Documents/repos/CS464/data/rtx3080_sm86_68sms_10gb/sweep_screen` | 160 shapes x 10840 configs (912913 timed cells; pruned / invalid cells count as infinite regret).

Regret = t / per-shape oracle - 1, from **screen** times (few samples; slightly pessimistic, see module doc).

## G3: value of per-shape selection over the best single config (B6, full space)

B6 = `bm32_bn32_bk32_w4_s5_g1_sk1`. Regret of B6 vs oracle: median 73.9%, p90 307.5%, max 900.0%.

## G5: portfolio size vs regret (greedy on mean regret)

| size | mean | p50 | p95 | max |
|---|---|---|---|---|
| 1 | 118.13% | 73.94% | 366.71% | 900.0% |
| 2 | 59.99% | 40.19% | 151.62% | 381.8% |
| 3 | 18.28% | 12.32% | 58.33% | 114.3% |
| 4 | 13.40% | 8.37% | 45.90% | 114.3% |
| 5 | 10.32% | 4.66% | 38.35% | 114.3% |
| 6 | 8.14% | 3.66% | 26.88% | 80.0% |
| 8 | 5.98% | 2.44% | 21.31% | 38.1% |
| 10 | 4.83% | 2.12% | 20.62% | 30.3% |
| 12 | 3.92% | 1.87% | 14.42% | 24.8% |
| 16 | 2.85% | 1.59% | 8.73% | 24.3% |
| 20 | 2.16% | 1.28% | 7.64% | 16.4% |
| 25 | 1.64% | 1.11% | 5.64% | 11.3% |
| 30 | 1.31% | 0.66% | 4.99% | 7.6% |
| 40 | 0.83% | 0.31% | 3.62% | 5.6% |
| 50 | 0.55% | 0.07% | 2.03% | 3.8% |
| 60 | 0.36% | 0.00% | 1.70% | 2.0% |
| 80 | 0.15% | 0.00% | 0.82% | 1.3% |
| 100 | 0.04% | 0.00% | 0.30% | 0.5% |

**Smallest greedy portfolio with p95 regret < 5%: 29** (G5 asks for <= ~50).

Set cover (every shape has a member within eps of its oracle): eps=2%: 54 configs, eps=3%: 43 configs, eps=5%: 32 configs

First 20 portfolio members: `bm32_bn32_bk32_w4_s5_g1_sk1`, `bm32_bn32_bk64_w4_s4_g1_sk8`, `bm64_bn128_bk16_w4_s4_g1_sk1`, `bm16_bn64_bk128_w4_s4_g1_sk1`, `bm64_bn64_bk32_w4_s4_g8_sk1`, `bm32_bn32_bk128_w4_s5_g1_sk1`, `bm64_bn64_bk32_w4_s5_g16_sk8`, `bm16_bn32_bk64_w2_s5_g1_sk8`, `bm64_bn32_bk128_w8_s3_g8_sk4`, `bm128_bn64_bk32_w4_s3_g4_sk1`, `bm16_bn64_bk32_w4_s3_g1_sk1`, `bm16_bn32_bk32_w2_s2_g1_sk8`, `bm16_bn32_bk64_w2_s5_g1_sk16`, `bm64_bn64_bk16_w4_s4_g1_sk1`, `bm32_bn64_bk64_w8_s3_g1_sk1`, `bm32_bn32_bk16_w4_s2_g1_sk1`, `bm64_bn64_bk32_w4_s3_g4_sk2`, `bm16_bn64_bk64_w4_s5_g1_sk8`, `bm64_bn128_bk16_w4_s3_g8_sk1`, `bm64_bn128_bk64_w4_s2_g16_sk4`

## Which parameters matter

Mean regret if a parameter is forced to its worst value (others chosen freely per shape), and how often each value is the best choice:

| parameter | regret at worst value | best value counts (shapes) |
|---|---|---|
| SPLIT_K | 121% | 1: 115, 2: 4, 4: 9, 8: 28, 16: 4 |
| BLOCK_M | 48% | 16: 53, 32: 28, 64: 70, 128: 8, 256: 1 |
| BLOCK_K | 26% | 16: 53, 32: 46, 64: 40, 128: 21 |
| BLOCK_N | 19% | 32: 53, 64: 57, 128: 47, 256: 3 |
| num_warps | 8% | 2: 51, 4: 98, 8: 11 |
| num_stages | 5% | 2: 46, 3: 36, 4: 49, 5: 29 |
| GROUP_M | 2% | 1: 93, 4: 19, 8: 27, 16: 21 |

![portfolio](figures/portfolio_regret.png)

