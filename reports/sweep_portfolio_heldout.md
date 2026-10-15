# Triton config sweep: held-out G5 check (auto-generated)

Portfolio, B6 and set covers are fit on the tune-split screen (`/home/raynerubuntu/Documents/repos/CS464/data/rtx3080_sm86_68sms_10gb/sweep_screen`, 160 shapes) and scored on the held-out screen (`/home/raynerubuntu/Documents/repos/CS464/data/rtx3080_sm86_68sms_10gb/sweep_screen_heldout`, 86 shapes: >= 50 us 30, < 50 us 56, heldout_random 55, heldout_region 31). Held-out oracle = best of the full pruned space on that shape. Regret = t / oracle - 1, screen times.

## G5: tune-fitted greedy portfolio on held-out shapes (p95 regret)

| size | in-sample | in-sample >= 50 us | held-out all | held-out >= 50 us | held-out < 50 us | held-out heldout_random | held-out heldout_region |
|---|---|---|---|---|---|---|---|
| 1 | 366.7% | 338.4% | 487.0% | 395.7% | 524.6% | 345.5% | 711.6% |
| 5 | 38.3% | 27.0% | 43.0% | 60.8% | 28.6% | 27.5% | 60.4% |
| 10 | 20.6% | 12.6% | 22.5% | 25.5% | 22.3% | 19.7% | 48.1% |
| 16 | 8.7% | 6.8% | 18.3% | 25.5% | 14.2% | 16.2% | 25.7% |
| 20 | 7.6% | 4.1% | 17.9% | 25.5% | 14.2% | 16.2% | 25.7% |
| 25 | 5.6% | 3.9% | 17.8% | 23.4% | 12.8% | 13.8% | 23.7% |
| 29 | 5.0% | 3.9% | 17.0% | 23.4% | 12.8% | 12.8% | 23.7% |
| 30 | 5.0% | 3.9% | 17.0% | 23.4% | 12.8% | 12.8% | 23.7% |
| 40 | 3.6% | 2.4% | 12.4% | 23.4% | 10.6% | 10.6% | 23.7% |
| 50 | 2.0% | 1.7% | 12.4% | 21.5% | 10.6% | 10.5% | 23.7% |
| 60 | 1.7% | 1.6% | 10.3% | 9.8% | 10.6% | 9.9% | 15.2% |
| 100 | 0.3% | 0.3% | 9.3% | 9.1% | 9.2% | 8.7% | 9.9% |

Same table, mean regret:

| size | in-sample | held-out all | held-out >= 50 us | held-out < 50 us | held-out heldout_random | held-out heldout_region |
|---|---|---|---|---|---|---|
| 1 | 118.1% | 113.6% | 127.7% | 106.0% | 88.1% | 158.8% |
| 5 | 10.3% | 12.3% | 12.4% | 12.3% | 11.8% | 13.2% |
| 10 | 4.8% | 8.8% | 8.5% | 8.9% | 8.0% | 10.1% |
| 16 | 2.9% | 5.4% | 6.7% | 4.8% | 4.8% | 6.6% |
| 20 | 2.2% | 4.8% | 5.5% | 4.5% | 4.4% | 5.5% |
| 25 | 1.6% | 4.5% | 5.1% | 4.1% | 4.1% | 5.2% |
| 29 | 1.4% | 4.2% | 4.6% | 4.0% | 3.7% | 5.2% |
| 30 | 1.3% | 4.2% | 4.6% | 4.0% | 3.7% | 5.1% |
| 40 | 0.8% | 3.6% | 4.1% | 3.4% | 3.2% | 4.4% |
| 50 | 0.5% | 3.5% | 3.9% | 3.3% | 3.0% | 4.4% |
| 60 | 0.4% | 3.1% | 3.0% | 3.2% | 2.8% | 3.7% |
| 100 | 0.0% | 2.4% | 2.5% | 2.4% | 2.3% | 2.7% |

**Smallest tune-order prefix with held-out p95 < 5%: > 100** (in-sample: 29; a greedy portfolio re-fit on the held-out shapes themselves would need 31).

B6 (`bm32_bn32_bk32_w4_s5_g1_sk1`) on held-out shapes: median regret 48.1%, p90 325.7% (G3 out of sample).

Tune-fitted set covers on held-out shapes:

| eps | size | held-out p95 | held-out max | shapes within 5% | >= 50 us p95 |
|---|---|---|---|---|---|
| 2.0% | 54 | 10.3% | 29.4% | 74.4% | 9.2% |
| 3.0% | 43 | 12.8% | 33.3% | 74.4% | 11.1% |
| 5.0% | 32 | 9.4% | 37.3% | 72.1% | 9.1% |

## Worst held-out shapes for the 29-config portfolio

| shape | suite / split | oracle us | oracle config | portfolio us | portfolio config | regret |
|---|---|---|---|---|---|---|
| 64x4096x11008 | S2 / heldout_region | 137.22 | `bm32_bn128_bk64_w4_s4_g1_sk1` | 176.13 | `bm32_bn64_bk64_w8_s5_g4_sk8` | 28.4% |
| 37x922x10110 | S7 / heldout_region | 52.22 | `bm64_bn128_bk64_w4_s4_g1_sk16` | 66.56 | `bm32_bn64_bk64_w8_s5_g4_sk8` | 27.5% |
| 226x2141x1139 | S7 / heldout_random | 41.39 | `bm64_bn64_bk64_w8_s4_g1_sk1` | 50.18 | `bm64_bn64_bk32_w4_s4_g8_sk1` | 21.2% |
| 20x1x11869 | S7 / heldout_region | 15.36 | `bm16_bn32_bk128_w2_s2_g4_sk16` | 18.43 | `bm16_bn32_bk64_w2_s5_g1_sk16` | 20.0% |
| 28x3030x4012 | S7 / heldout_random | 50.16 | `bm32_bn64_bk64_w4_s4_g1_sk4` | 59.39 | `bm32_bn32_bk64_w4_s4_g1_sk8` | 18.4% |
| 51x324x5709 | S7 / heldout_random | 23.59 | `bm32_bn64_bk64_w4_s5_g8_sk16` | 26.60 | `bm32_bn64_bk64_w8_s5_g4_sk8` | 12.8% |
| 15x3363x2632 | S7 / heldout_random | 38.80 | `bm16_bn32_bk32_w2_s4_g1_sk4` | 43.74 | `bm16_bn64_bk128_w4_s4_g1_sk1` | 12.7% |
| 14244x16x1257 | S7 / heldout_random | 70.66 | `bm128_bn32_bk64_w4_s4_g4_sk1` | 79.49 | `bm32_bn32_bk128_w4_s2_g1_sk1` | 12.5% |
| 1316x272x100 | S7 / heldout_random | 8.25 | `bm64_bn32_bk32_w4_s3_g8_sk1` | 9.22 | `bm64_bn64_bk32_w4_s4_g8_sk1` | 11.7% |
| 2125x1x1 | S7 / heldout_random | 3.53 | `bm32_bn32_bk16_w4_s5_g4_sk1` | 3.94 | `bm16_bn32_bk16_w2_s4_g1_sk1` | 11.5% |
| 384x2x10369 | S7 / heldout_region | 26.63 | `bm32_bn32_bk64_w4_s5_g4_sk16` | 29.39 | `bm16_bn32_bk64_w2_s5_g1_sk16` | 10.4% |
| 426x273x53 | S7 / heldout_region | 5.58 | `bm64_bn32_bk64_w8_s3_g4_sk1` | 6.14 | `bm32_bn32_bk32_w4_s5_g1_sk1` | 10.2% |

## Stage-2 confirmation (re-measured, `results_lock1755_heldout.jsonl`)

Each held-out shape re-timed cold (adaptive repeats, 3 input placements): the screen's 8 fastest tiles for that shape plus the first 29 portfolio members. Oracle = fastest of those. 86 shapes, 0 invalid rows. Sub-50 us cold times carry ~1 us event-tick quantization.

| size | all p95 / mean | >= 50 us p95 / mean | < 50 us p95 / mean | heldout_random p95 / mean | heldout_region p95 / mean |
|---|---|---|---|---|---|
| 1 | 484.3% / 112.2% | 388.8% / 126.4% | 518.2% / 104.6% | 340.1% / 87.1% | 694.3% / 156.8% |
| 5 | 28.0% / 10.4% | 42.7% / 10.2% | 25.2% / 10.6% | 25.0% / 10.9% | 41.7% / 9.6% |
| 10 | 25.0% / 7.6% | 16.8% / 5.3% | 25.0% / 8.9% | 25.0% / 8.4% | 19.1% / 6.2% |
| 16 | 15.8% / 3.4% | 15.6% / 4.3% | 14.0% / 2.8% | 16.1% / 3.0% | 13.9% / 4.0% |
| 20 | 13.0% / 2.8% | 11.0% / 2.8% | 14.0% / 2.7% | 16.1% / 2.8% | 8.9% / 2.6% |
| 25 | 13.0% / 2.5% | 10.6% / 2.7% | 14.0% / 2.4% | 14.7% / 2.5% | 8.9% / 2.6% |
| 29 | 11.0% / 2.2% | 9.7% / 2.3% | 12.7% / 2.1% | 11.9% / 1.9% | 8.9% / 2.6% |

Worst shapes at size 29 (stage 2): 124x28x26 21.9% (4.1 us), 226x2141x1139 19.5% (42.0 us), 15x3363x2632 13.2% (38.9 us), 20x1x11869 12.5% (16.4 us), 14244x16x1257 11.4% (71.7 us), 28x3030x4012 9.8% (50.4 us), 37x922x10110 9.6% (53.2 us), 51x324x5709 9.0% (24.6 us)

