# Zero-shot selector transfer: `rtx3090_sm86_82sms_24gb`

Source models: lgbm_rank (v1) and, when present, lgbm_rank_v2, fit on another GPU, no refit. Oracle: this GPU's held-out screen (`sweep_screen_heldout`, 27 complete shapes). Cells are median / p95 (mean) regret vs the screen oracle, the format of the 3080 `Screen` table in `reports/learned_topk.md`. A shape with no valid pick counts as 1000% regret. Scored at k = 3; the screen holds Triton configs only, so joint rows are scored on their Triton-only choice (the joint choice is scored in a stage-2 session: `picks`, then analysis.ranker_compare).

## Model `v1`, variant `as_is` (27 shapes, 0.00 invalid picks per shape out of 3)

| pick | all | >= 50 us | < 50 us | within 2% | regret > 50% | no pick | unscored |
|---|---|---|---|---|---|---|---|
| top-1 | 6.0% / 23.1% (8.1%) | 5.4% / 25.9% (8.6%) | 6.8% / 18.4% (7.4%) | 33% | 0 of 27 | 0 | 0 |
| best-of-1 | 6.0% / 23.1% (8.1%) | 5.4% / 25.9% (8.6%) | 6.8% / 18.4% (7.4%) | 33% | 0 of 27 | 0 | 0 |
| online-1 | 6.0% / 23.1% (8.1%) | 5.4% / 25.9% (8.6%) | 6.8% / 18.4% (7.4%) | 33% | 0 of 27 | 0 | 0 |
| best-of-2 | 4.2% / 13.6% (4.8%) | 2.6% / 12.7% (4.6%) | 5.4% / 12.3% (5.2%) | 37% | 0 of 27 | 0 | 0 |
| online-2 | 5.4% / 14.5% (6.4%) | 4.2% / 19.4% (6.4%) | 6.8% / 14.1% (6.4%) | 37% | 0 of 27 | 0 | 0 |
| best-of-3 | 4.2% / 12.9% (4.4%) | 2.6% / 12.3% (4.4%) | 5.0% / 10.6% (4.5%) | 41% | 0 of 27 | 0 | 0 |
| online-3 | 5.4% / 14.5% (6.1%) | 4.2% / 19.4% (6.4%) | 5.7% / 14.1% (5.8%) | 41% | 0 of 27 | 0 | 0 |
| fixed | 53.5% / 330.2% (123.4%) | 91.8% / 310.5% (116.7%) | 13.8% / 646.3% (131.9%) | 4% | 15 of 27 | 0 | 0 |

## Model `v1`, variant `native` (27 shapes, 0.00 invalid picks per shape out of 3)

| pick | all | >= 50 us | < 50 us | within 2% | regret > 50% | no pick | unscored |
|---|---|---|---|---|---|---|---|
| top-1 | 4.6% / 13.0% (5.0%) | 4.6% / 11.3% (4.7%) | 5.3% / 12.3% (5.2%) | 26% | 0 of 27 | 0 | 0 |
| best-of-1 | 4.6% / 13.0% (5.0%) | 4.6% / 11.3% (4.7%) | 5.3% / 12.3% (5.2%) | 26% | 0 of 27 | 0 | 0 |
| online-1 | 4.6% / 13.0% (5.0%) | 4.6% / 11.3% (4.7%) | 5.3% / 12.3% (5.2%) | 26% | 0 of 27 | 0 | 0 |
| best-of-2 | 3.7% / 9.1% (3.9%) | 3.7% / 8.4% (4.0%) | 3.6% / 9.2% (3.7%) | 37% | 0 of 27 | 0 | 0 |
| online-2 | 3.7% / 13.0% (4.5%) | 3.7% / 11.3% (4.6%) | 3.6% / 12.3% (4.5%) | 33% | 0 of 27 | 0 | 0 |
| best-of-3 | 3.7% / 9.1% (3.7%) | 3.7% / 8.4% (3.8%) | 3.6% / 8.5% (3.6%) | 41% | 0 of 27 | 0 | 0 |
| online-3 | 3.7% / 13.0% (4.5%) | 3.7% / 11.3% (4.5%) | 3.6% / 12.3% (4.5%) | 37% | 0 of 27 | 0 | 0 |
| fixed | 53.5% / 330.2% (123.4%) | 91.8% / 310.5% (116.7%) | 13.8% / 646.3% (131.9%) | 4% | 15 of 27 | 0 | 0 |

## Model `v2`, variant `as_is` (27 shapes, 0.00 invalid picks per shape out of 3)

| pick | all | >= 50 us | < 50 us | within 2% | regret > 50% | no pick | unscored |
|---|---|---|---|---|---|---|---|
| top-1 | 1.7% / 18.9% (5.7%) | 1.0% / 17.3% (5.8%) | 3.2% / 16.9% (5.6%) | 52% | 0 of 27 | 0 | 0 |
| best-of-1 | 1.7% / 18.9% (5.7%) | 1.0% / 17.3% (5.8%) | 3.2% / 16.9% (5.6%) | 52% | 0 of 27 | 0 | 0 |
| online-1 | 1.7% / 18.9% (5.7%) | 1.0% / 17.3% (5.8%) | 3.2% / 16.9% (5.6%) | 52% | 0 of 27 | 0 | 0 |
| best-of-2 | 0.9% / 11.2% (3.8%) | 0.2% / 11.7% (3.7%) | 1.7% / 11.2% (3.9%) | 59% | 0 of 27 | 0 | 0 |
| online-2 | 1.7% / 13.0% (4.2%) | 0.2% / 12.0% (3.9%) | 3.1% / 12.4% (4.5%) | 56% | 0 of 27 | 0 | 0 |
| best-of-3 | 0.5% / 11.2% (3.4%) | 0.2% / 11.7% (3.7%) | 1.1% / 11.2% (3.1%) | 63% | 0 of 27 | 0 | 0 |
| online-3 | 0.9% / 14.1% (4.0%) | 0.2% / 14.3% (4.3%) | 1.7% / 12.4% (3.7%) | 59% | 0 of 27 | 0 | 0 |
| fixed | 53.5% / 330.2% (123.4%) | 91.8% / 310.5% (116.7%) | 13.8% / 646.3% (131.9%) | 4% | 15 of 27 | 0 | 0 |

## Model `v2`, variant `native` (27 shapes, 0.00 invalid picks per shape out of 3)

| pick | all | >= 50 us | < 50 us | within 2% | regret > 50% | no pick | unscored |
|---|---|---|---|---|---|---|---|
| top-1 | 2.8% / 14.6% (4.9%) | 2.8% / 9.5% (4.0%) | 3.4% / 17.5% (6.1%) | 41% | 0 of 27 | 0 | 0 |
| best-of-1 | 2.8% / 14.6% (4.9%) | 2.8% / 9.5% (4.0%) | 3.4% / 17.5% (6.1%) | 41% | 0 of 27 | 0 | 0 |
| online-1 | 2.8% / 14.6% (4.9%) | 2.8% / 9.5% (4.0%) | 3.4% / 17.5% (6.1%) | 41% | 0 of 27 | 0 | 0 |
| best-of-2 | 1.9% / 10.6% (3.3%) | 1.9% / 7.9% (2.7%) | 1.9% / 12.1% (4.0%) | 56% | 0 of 27 | 0 | 0 |
| online-2 | 1.9% / 12.0% (3.5%) | 1.9% / 8.9% (2.9%) | 1.9% / 13.5% (4.2%) | 56% | 0 of 27 | 0 | 0 |
| best-of-3 | 1.6% / 7.1% (2.5%) | 0.9% / 6.7% (2.3%) | 1.9% / 8.3% (2.8%) | 63% | 0 of 27 | 0 | 0 |
| online-3 | 1.8% / 9.0% (3.1%) | 0.9% / 8.9% (2.7%) | 2.0% / 11.0% (3.6%) | 59% | 0 of 27 | 0 | 0 |
| fixed | 53.5% / 330.2% (123.4%) | 91.8% / 310.5% (116.7%) | 13.8% / 646.3% (131.9%) | 4% | 15 of 27 | 0 | 0 |

## RTX 3080 reference (same method, its own screen)

| pick | rank v1 | rank v2 |
|---|---|---|
| top-1 | 3.6% / 18.1% (5.5%) | 2.8% / 13.0% (4.6%) |
| best-of-3 (diverse) | 2.0% / 10.6% (3.4%) | 1.0% / 9.3% (2.1%) |
| best-of-5 (diverse) | n/a | 0.5% / 7.8% (1.7%) |

`fixed` row: the 3080's B6 config `bm32_bn32_bk32_w4_s5_g1_sk1` used as-is on every shape.
`regret > 50%`: scored picks at least 50% slower than the oracle, including picks the screen cut as `slow` (their one-sample time is kept).
`unscored`: a pick that is not in this GPU's screen at all (pruned or failed there), excluded from the stats: read it before trusting a row.
The 3080 reference is over its full 86-shape held-out set; for a run on a subset, compare on the shared shapes with `python -m analysis.cross_gpu`.
`online-k` is what the short search chose; `best-of-k` is the best of the same k candidates by screen time.
