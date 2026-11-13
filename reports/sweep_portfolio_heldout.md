# Triton config sweep: held-out G5 check (auto-generated)

Portfolio, B6 and set covers are fit on the tune-split screen (`/home/raynerubuntu/Documents/repos/CS464/data/rtx3080_sm86_68sms_10gb/sweep_screen`, 477 shapes) and scored on the held-out screen (`/home/raynerubuntu/Documents/repos/CS464/data/rtx3080_sm86_68sms_10gb/sweep_screen_heldout`, 86 shapes: >= 50 us 30, < 50 us 56, heldout_random 55, heldout_region 31). Held-out oracle = best of the full pruned space on that shape. Regret = t / oracle - 1, screen times.

## G5: tune-fitted greedy portfolio on held-out shapes (p95 regret)

| size | in-sample | in-sample >= 50 us | held-out all | held-out >= 50 us | held-out < 50 us | held-out heldout_random | held-out heldout_region |
|---|---|---|---|---|---|---|---|
| 1 | 343.1% | 325.3% | 487.0% | 395.7% | 524.6% | 345.5% | 711.6% |
| 5 | 42.4% | 38.7% | 31.0% | 44.8% | 27.8% | 27.1% | 43.6% |
| 10 | 19.1% | 16.0% | 21.4% | 24.8% | 20.3% | 21.3% | 23.7% |
| 16 | 13.1% | 6.0% | 18.2% | 23.4% | 14.2% | 16.4% | 23.7% |
| 20 | 11.7% | 5.8% | 17.2% | 17.9% | 13.6% | 16.4% | 18.7% |
| 25 | 8.7% | 5.1% | 16.5% | 17.9% | 10.6% | 14.0% | 17.0% |
| 29 | 7.1% | 4.9% | 13.0% | 17.9% | 10.6% | 12.5% | 17.0% |
| 30 | 7.1% | 4.9% | 12.7% | 17.9% | 10.6% | 12.4% | 17.0% |
| 40 | 6.0% | 3.5% | 12.5% | 17.6% | 10.6% | 12.4% | 13.5% |
| 50 | 4.7% | 2.5% | 12.2% | 17.5% | 10.6% | 11.7% | 13.5% |
| 60 | 3.6% | 2.1% | 11.2% | 15.6% | 10.2% | 9.4% | 13.5% |
| 100 | 1.8% | 1.5% | 11.1% | 15.6% | 9.8% | 9.4% | 13.3% |

Same table, mean regret:

| size | in-sample | held-out all | held-out >= 50 us | held-out < 50 us | held-out heldout_random | held-out heldout_region |
|---|---|---|---|---|---|---|
| 1 | 110.8% | 113.6% | 127.7% | 106.0% | 88.1% | 158.8% |
| 5 | 11.1% | 11.3% | 11.4% | 11.3% | 11.1% | 11.7% |
| 10 | 5.2% | 6.9% | 8.0% | 6.3% | 6.8% | 7.0% |
| 16 | 3.5% | 5.4% | 6.0% | 5.0% | 5.2% | 5.7% |
| 20 | 2.9% | 4.9% | 5.5% | 4.6% | 4.7% | 5.2% |
| 25 | 2.4% | 4.3% | 5.4% | 3.7% | 4.1% | 4.6% |
| 29 | 2.1% | 3.9% | 4.8% | 3.4% | 3.5% | 4.5% |
| 30 | 2.0% | 3.8% | 4.8% | 3.3% | 3.5% | 4.4% |
| 40 | 1.5% | 3.5% | 3.9% | 3.2% | 3.3% | 3.8% |
| 50 | 1.1% | 3.3% | 3.6% | 3.2% | 3.1% | 3.7% |
| 60 | 0.8% | 3.1% | 3.4% | 2.9% | 2.7% | 3.7% |
| 100 | 0.4% | 2.5% | 3.0% | 2.2% | 2.1% | 3.2% |

Same table, p99 regret (on 86 held-out shapes this is essentially the worst shape):

| size | in-sample | in-sample >= 50 us | held-out all | held-out >= 50 us | held-out < 50 us | held-out heldout_random | held-out heldout_region |
|---|---|---|---|---|---|---|---|
| 1 | 1001.6% | 368.9% | 973.1% | 589.9% | 1081.4% | 616.8% | 1171.7% |
| 5 | 87.1% | 63.1% | 69.4% | 62.5% | 67.7% | 55.3% | 81.9% |
| 10 | 28.6% | 26.6% | 29.4% | 36.6% | 21.7% | 21.8% | 36.5% |
| 16 | 22.4% | 18.7% | 28.1% | 30.7% | 20.5% | 19.7% | 30.7% |
| 20 | 18.7% | 16.2% | 22.2% | 24.8% | 20.5% | 19.7% | 25.2% |
| 25 | 14.2% | 9.4% | 19.8% | 24.8% | 14.3% | 17.4% | 24.4% |
| 29 | 13.5% | 7.2% | 19.8% | 24.8% | 12.6% | 15.5% | 24.4% |
| 30 | 13.5% | 7.2% | 19.8% | 24.8% | 12.6% | 15.4% | 24.4% |
| 40 | 10.5% | 5.5% | 19.8% | 24.8% | 12.6% | 15.4% | 24.2% |
| 50 | 7.2% | 4.3% | 19.7% | 24.8% | 12.6% | 15.3% | 24.2% |
| 60 | 6.1% | 3.6% | 18.2% | 24.3% | 11.9% | 13.3% | 24.2% |
| 100 | 2.8% | 2.4% | 18.2% | 24.3% | 11.9% | 13.3% | 24.2% |

**Smallest tune-order prefix with held-out p95 < 5%: > 100** (in-sample: 48; a greedy portfolio re-fit on the held-out shapes themselves would need 31).

B6 (`bm32_bn32_bk32_w4_s5_g1_sk1`) on held-out shapes: median regret 48.1%, p90 325.7% (G3 out of sample).

Tune-fitted set covers on held-out shapes:

| eps | size | held-out p95 | held-out p99 | held-out max | shapes within 5% | >= 50 us p95 |
|---|---|---|---|---|---|---|
| 2.0% | 102 | 9.2% | 15.0% | 27.5% | 86.0% | 7.8% |
| 3.0% | 79 | 9.9% | 18.9% | 29.5% | 84.9% | 7.8% |
| 5.0% | 59 | 9.3% | 19.1% | 27.5% | 81.4% | 9.8% |

## Worst held-out shapes for the 48-config portfolio

| shape | suite / split | oracle us | oracle config | portfolio us | portfolio config | regret |
|---|---|---|---|---|---|---|
| 37x922x10110 | S7 / heldout_region | 52.22 | `bm64_bn128_bk64_w4_s4_g1_sk16` | 66.56 | `bm64_bn32_bk64_w8_s3_g1_sk4` | 27.5% |
| 28x3030x4012 | S7 / heldout_random | 50.16 | `bm32_bn64_bk64_w4_s4_g1_sk4` | 59.33 | `bm64_bn64_bk32_w4_s4_g1_sk4` | 18.3% |
| 417x512x6739 | S7 / heldout_region | 86.08 | `bm64_bn256_bk16_w4_s2_g16_sk8` | 100.35 | `bm64_bn64_bk32_w4_s5_g8_sk8` | 16.6% |
| 15x3363x2632 | S7 / heldout_random | 38.80 | `bm16_bn32_bk32_w2_s4_g1_sk4` | 43.74 | `bm16_bn64_bk128_w4_s4_g1_sk1` | 12.7% |
| 51x324x5709 | S7 / heldout_random | 23.59 | `bm32_bn64_bk64_w4_s5_g8_sk16` | 26.52 | `bm32_bn64_bk64_w8_s5_g8_sk8` | 12.4% |
| 2125x1x1 | S7 / heldout_random | 3.53 | `bm32_bn32_bk16_w4_s5_g4_sk1` | 3.94 | `bm16_bn32_bk16_w2_s4_g1_sk1` | 11.5% |
| 384x2x10369 | S7 / heldout_region | 26.63 | `bm32_bn32_bk64_w4_s5_g4_sk16` | 29.39 | `bm16_bn32_bk64_w2_s5_g1_sk16` | 10.4% |
| 426x273x53 | S7 / heldout_region | 5.58 | `bm64_bn32_bk64_w8_s3_g4_sk1` | 6.14 | `bm32_bn32_bk32_w4_s5_g1_sk1` | 10.2% |
| 226x2141x1139 | S7 / heldout_random | 41.39 | `bm64_bn64_bk64_w8_s4_g1_sk1` | 44.94 | `bm64_bn64_bk64_w4_s4_g4_sk1` | 8.6% |
| 33x190x18 | S7 / heldout_random | 4.11 | `bm16_bn32_bk32_w2_s2_g4_sk1` | 4.46 | `bm16_bn32_bk32_w2_s3_g1_sk1` | 8.6% |
| 4065x8842x28 | S7 / heldout_random | 125.95 | `bm64_bn128_bk16_w4_s2_g1_sk1` | 136.19 | `bm64_bn64_bk32_w4_s4_g1_sk1` | 8.1% |
| 232x3389x2 | S7 / heldout_random | 6.61 | `bm32_bn256_bk16_w4_s2_g1_sk1` | 7.12 | `bm32_bn64_bk16_w4_s3_g1_sk1` | 7.7% |

## Stage-2 confirmation (re-measured, `results_lock1755_heldout477.jsonl`)

Each held-out shape re-timed cold (adaptive repeats, 3 input placements): the screen's 8 fastest tiles for that shape plus the first 60 portfolio members. Oracle = fastest of those. 86 shapes, 0 invalid rows. Sub-50 us cold times carry ~1 us event-tick quantization.

| size | all p95 / p99 / mean | >= 50 us p95 / p99 / mean | < 50 us p95 / p99 / mean | heldout_random p95 / p99 / mean | heldout_region p95 / p99 / mean |
|---|---|---|---|---|---|
| 1 | 481.3% / 921.4% / 111.2% | 388.3% / 586.6% / 125.4% | 516.7% / 1008.7% / 103.6% | 340.1% / 606.2% / 85.6% | 688.9% / 1081.5% / 156.7% |
| 5 | 26.2% / 62.0% / 10.2% | 41.5% / 57.1% / 10.7% | 25.0% / 51.2% / 10.0% | 25.0% / 40.4% / 9.6% | 40.2% / 72.9% / 11.5% |
| 10 | 21.3% / 26.9% / 5.7% | 24.3% / 34.0% / 7.4% | 17.2% / 22.8% / 4.8% | 20.0% / 24.2% / 5.5% | 23.2% / 33.9% / 6.1% |
| 16 | 17.2% / 25.7% / 3.9% | 23.4% / 28.1% / 5.3% | 12.7% / 17.8% / 3.2% | 14.7% / 18.4% / 3.5% | 23.2% / 28.1% / 4.8% |
| 20 | 14.7% / 19.8% / 3.4% | 16.3% / 20.3% / 4.3% | 12.7% / 17.8% / 2.9% | 14.7% / 18.4% / 3.2% | 13.7% / 19.5% / 3.8% |
| 25 | 13.8% / 18.1% / 2.8% | 16.3% / 20.3% / 4.1% | 8.9% / 14.6% / 2.2% | 13.4% / 16.9% / 2.7% | 12.3% / 19.5% / 3.1% |
| 29 | 11.4% / 17.1% / 2.4% | 13.6% / 19.5% / 3.2% | 7.9% / 14.6% / 1.9% | 9.9% / 14.6% / 2.1% | 12.3% / 19.5% / 2.9% |
| 40 | 8.9% / 15.1% / 1.9% | 9.3% / 13.4% / 2.2% | 7.6% / 14.6% / 1.7% | 8.7% / 14.6% / 1.7% | 8.0% / 13.3% / 2.2% |
| 48 | 8.9% / 15.1% / 1.8% | 9.3% / 13.4% / 2.0% | 7.6% / 14.6% / 1.7% | 8.7% / 14.6% / 1.6% | 8.0% / 13.3% / 2.1% |
| 60 | 8.3% / 15.1% / 1.5% | 9.3% / 13.4% / 1.9% | 7.0% / 12.1% / 1.3% | 7.7% / 12.4% / 1.2% | 8.0% / 13.3% / 2.1% |

Worst shapes at size 60 (stage 2): 93x2101x22 16.4% (5.3 us), 417x512x6739 14.9% (87.0 us), 37x922x10110 9.6% (53.2 us), 4065x8842x28 8.9% (126.0 us), 51x324x5709 8.6% (24.6 us), 226x2141x1139 7.3% (42.0 us), 490x3576x46 6.9% (13.3 us), 342x88x6772 6.5% (31.7 us)

