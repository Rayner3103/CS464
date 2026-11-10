# Week 1: Phase 0 + first baselines (RTX 3080)

**Date:** 2026-10-09 · **hw_id:** `rtx3080_sm86_68sms_10gb` · driver 595.91.07 · torch 2.13.0+cu130 · triton 3.7.1
**Scope:** milestone M0 and the plan §13 first-week checklist. All tables are generated in
[`week1_results.md`](week1_results.md) (`python -m tune analyze`). The protocol is in [`../docs/methodology.md`](../docs/methodology.md).

## Update 2026-11-10: 3090 follow-ups on existing data — a `cross_gpu` scorer bug (ranker picks dropped) corrected, the 5% re-score, the router's held-out replay, and the bundle's coverage of the 3090 pools

CPU only, on the local checkout; no GPU job and nothing re-measured. Inputs: `pipeline/runs/rtx3090/train_best-full-slurm/data/rtx3090_sm86_82sms_24gb/` (the 3090 run directory: it has the models and the tune files, `data/rtx3090_sm86_82sms_24gb/` has only the zero-shot files). Code: `analysis/cross_gpu.py`, `tests/test_cross_gpu.py` (this commit).

- **Scorer bug (corrects the 2026-11-09 cross-GPU joint table).** `cross_gpu.stage2_times` looked up a joint pick only in the `tri` / `lth` / `ltd` stage-2 tables, and `joint_summary` counted only `family == "cublaslt"` as a cuBLASLt pick. Picks of the learned cuBLASLt ranker (`cublaslt_rank`, keys `r*`; 17 of the 86 v1 joint choices and 20 of the v2 ones) were dropped from both, so the two retrained arms showed 21 / 19 of 27 shapes timed and 4 / 3 cuBLASLt picks, and their speed was computed on the shapes where the ranker did not win. It now uses `ranker_compare.PICK_FAMILIES` and counts every `cublaslt*` family (regression test added). Corrected rows (`reports/cross_gpu_3090_retrained.md`, 27 shared shapes, joint top-3 + LT16 + ranker top-5 vs B3+):

| | picks cuBLASLt | timed in stage 2 | speed vs B3+ median / p10 | within 5% of B3+ |
|---|---|---|---|---|
| 3090 retrained v1, before | 4 | 21 | 1.018 / 0.960 | 20 / 21 |
| 3090 retrained v1, after | 10 | 27 | 1.014 / 0.957 | 25 / 27 |
| 3090 retrained v2, before | 3 | 19 | 1.022 / 0.970 | 18 / 19 |
| 3090 retrained v2, after | 11 | 27 | 1.006 / 0.964 | 25 / 27 |

  The other four arms (3080 and 3090 zero-shot) have no ranker picks and are unchanged. `analysis.ranker_compare` already looked in every pick family, so its tables (0.994 vs B3+, 26 faster / 9 slower, 61 / 0 vs B2) were not affected. The "retrained joint is slightly slower than zero-shot" reading of the old table was this artifact; with all shapes timed the retrained v2 arm (1.006) is within noise of the zero-shot ones (1.000).
- **The `StopIteration`.** The traceback in `logs/validate_compare.log` is from 01:06 on 11-09; `results_zero_shot*.jsonl` were topped up at 01:35-01:37. On the current files `analysis.ranker_compare` runs without error, finds all 258 of 258 picks (v1 and v2, k = 1, 3, 5) in the stage-2 tables, and reproduces the run directory's `ranker_compare.md` exactly. The comment at `ranker_compare.py:24` describes the failure mode (a pick from a family stage 2 did not time).
- **5% margin re-score** (`ranker_compare.EPS["large"]` set to 1.05, 3090 v2, existing stage 2; at 1.02 it reproduces the 26 / 9 and 25 / 36): the sub-50 µs rows keep their 5% margin, so the all-shapes counts do not move. At >= 50 µs (n = 25): Triton-only top-3 vs B3+ 6 faster / 15 slower -> **1 / 10**; joint top-3 + LT16 vs B3+ 6 / 9 -> **1 / 4**. G4 stays unpassed on the 3090.
- **Router held-out replay on the 3090** (`python -m analysis.router score --data <run dir data> --online zero_shot_v2_as_is.jsonl --results results_zero_shot.jsonl --graph-results results_zero_shot_graph.jsonl --model <run dir data>/router`, the 3090 router: AUC 0.803, threshold 0.055). It is a replay of the existing no-bundle run, not new timings. Regret vs the extended joint oracle (median / p95 / p99 / mean) 0.5 / 9.5 / 10.4 / 2.0% never skipping -> **0.5 / 8.6 / 10.8 / 2.0%** with the router; first use 0.875 -> **0.676 s (-23%)**, 17 of 86 shapes skipped; worse than the default pool by more than the margin on 1 shape (448×1088×4096, 71.3 -> 78.0 µs, p_win 0.013), better on 1; shapes slower than B3+ by the margin 14 -> 12. The saving is larger than the 3080's (-10%, 14 of 86 skipped) because cuBLASLt wins more often on the 3090's small shapes. Not shown: a re-run with confirmation timing on this GPU.
- **Does the 3080 bundle serve the 3090?** `python -m tune bundle status` here (a 3080, sm_86) says the bundle matches (Triton 3.7.1, target cuda / 86 / warp 32, same kernel source); the fingerprint has no SM count or device name, so the key should match on the 3090 (not run there). **Coverage is the gap:** the 3090's v1 + v2 pools hold 1,252 configs, the bundle 1,176 (the 3080's pools); 514 of the 3090's configs (41%) are not in the bundle, and 100 of the 430 Triton candidates of the 3090 v2 run (23%, ~1.2 per shape) would compile on first use (~0.13 s each, i.e. ~+0.15 s per shape). **Rebuilt the same day** as the union of the 3080's and the 3090's pools (1,690 configs, 30,327 kernels, 1,179 MiB; 335 s with the scratch cache reused): `bundle verify --per-shape 0` on the 3080 = 640,233 pairs, 20,057 distinct kernels from the bundle, **0 misses**, 6 launched kernels match torch.matmul; archive 25.4 MiB (was 19.2), unpack 14 s, byte-identical. Not yet run on the 3090 itself (`bundle verify` with the same `--pool` stems). Details: `docs/kernel_bundle.md`.
- **Gates.** None changed. **Next steps.** The 3090 follow-ups still open: a fresh-shape check, zero-shot cuBLASLt ranker / router (`TUNE_*_FOREIGN`), a second noise session. Regenerate: `python -m analysis.cross_gpu --out reports/cross_gpu_3090_retrained.md --side ...` (the command of the 2026-11-09 entry).

## Update 2026-11-09: full training run on the RTX 3090 and its noise study — the pipeline transfers within sm_86, the 3080's 2% tie margin does not (the 3090 floor is 4-5% at >= 50 µs)

**What ran.** `MODE=full sbatch remoteGPU/train_best.sh` on the RTX 3090 (`rtx3090_sm86_82sms_24gb`, 82 SMs, driver 590.48.01, torch 2.13.0+cu130, Triton 3.7.1, **clock unlocked**), everything fitted from this GPU's own data and nothing borrowed from the 3080. Run directory `pipeline/runs/rtx3090/train_best-full-slurm/` (harness `875ba36`, a second short session on `9add6b7`; clean). Its `SUMMARY.md` has the model table. Then `remoteGPU/noise.sh` (`python -m tune --clock-lock none noise --rounds 3 --repeats 50 --n-allocs 3`), session `noise-20261109-183128-5ac4`, harness `57b7a15`, clean, `data/rtx3090_sm86_82sms_24gb/noise.jsonl`.
- Rows: tune stage 2 cold 11,032 (`20261108-182541-834666`) and graph 5,249; held-out stage 2 cold 3,251 (`20261108-201758-f2a3d4`, + 94 rows `20261109-002122-9c6930`) and graph 2,344 (`20261108-203309-c4a76c`, + 78 rows). **0 invalid** in all four; 2 / 21 / 0 / 0 not converged on the four files; power-capped 70% (tune cold) / 48% (tune graph) / 60% (held-out cold) / 50% (held-out graph). Edge correctness check: 1,183 candidates, 0 failures. The `*_lock1755_tune477*` file names are a leftover of the pipeline's naming: every row says `unlocked`.
- Oracle: this GPU's own screens (`sweep_screen` 477 tune shapes, `sweep_screen_tail`, `sweep_screen_heldout` 86 shapes). The held-out set is the 3080's 86.
- Roofline (unlocked, 1680-1950 MHz): **70.2 TFLOP/s** fp16 tensor, **844 GB/s** DRAM (3080: 59.7 / 687, locked 1755). Do not compare absolute times with the 3080's.
- Not run: the router's held-out score (`analysis.router score`), a fresh-shape check, k-NN cuBLASLt proposals in the joint run (the validation pool is v2/v1 top-k + the heuristic top-16 + the learned cuBLASLt ranker's picks, which show up as `r*` keys in the log), a second noise session, a locked-clock re-run. `logs/validate_compare.log` contains a `StopIteration` traceback from `analysis/ranker_compare.py:74` (a chosen key not found in the stage-2 pick families); `ranker_compare.md` was written afterwards and is complete, but the cause was not investigated, and 19-21 of the 27 shared-shape joint picks have a stage-2 time (the "timed in the stage-2 session" column below). **Both resolved in the 2026-11-10 entry above: the `StopIteration` predates the stage-2 top-up, and the 19-21 of 27 was a `cross_gpu` scorer bug.**

**Models fitted on the 3090** (vs the 3080's):

| model | 3090 | 3080 |
|---|---|---|
| rank v1 | 477 tune shapes, pool 905, lambdarank / 127 leaves / 800 trees | 477 shapes (2026-10-19) |
| rank v2 (shipped) | 627 shapes (+ `tail`), pool 1,249 | 477 + `tail` (2026-11-01) |
| cuBLASLt ranker, tune grouped CV, closes the gap to the screen winner at top-5 / 10 / 16 | 56.8% / 68.0% / 77.6% (k-NN 8×2: 76.3%, 12 candidates; ranker + k-NN 8×2: 84-88%) | 69.4% / 78.3% / -% (k-NN 8×2: 71%) |
| family router | out-of-fold AUC **0.803**, skip threshold **0.055** | 0.823, 0.02 |

The cuBLASLt ranker is weaker here than on the 3080 and no better than k-NN at the same candidate count, the reverse of the 3080 ordering; the router's rule picks a looser threshold. Triton's screen oracle beats B3+ by more than the tie margin on 117 of 477 tune shapes (25%).

**Held-out (86 shapes), Triton-only online search vs the screen oracle** (median / p95 / p99 (mean); `reports/k3`, `reports/k5` in the run directory, `reports/learned_topk.md`):

| pick | 3090 v1 | 3090 v2 | 3080 v1 | 3080 v2 |
|---|---|---|---|---|
| online top-3 | 2.7% / 21.7% / 42.9% (5.7%) | 2.1% / 13.6% / 20.0% (4.1%) | 1.1% / 9.1% / 20.2% (2.6%) (`topki`) | not in this entry |
| online top-5 | 2.8% / 17.2% / 23.3% (4.5%) | 2.0% / 14.1% / 20.0% (3.7%) | | |
| best-of-3 (diverse) | 0.7% / 21.7% / 42.9% (4.5%) | 0.7% / 11.7% / 18.5% (2.8%) | 2.0% / 10.6% / 15.9% (3.4%) | 1.0% / 9.3% / 13.0% (2.1%) |
| best-of-5 (diverse) | | | | 0.5% / 7.8% / 13.0% (1.7%) |
| fixed 3080 B6 config | 45.2% / 584.7% / 994.2% (120.8%); 39 of 86 over 50% | | | |

No pick is more than 50% off with either model (0 of 86; the fixed config: 39). The 3090's v1 tail is twice the 3080's (p99 43%), v2 halves it (20%) but stays above the 3080's 13% best-of-3.

**Stage 2 joint search vs the baselines** (`ranker_compare.md`, joint oracle, same script as the 3080's `reports/ranker_compare.md`; v2 rows):

| | 3090 | 3080 |
|---|---|---|
| top-3 Triton only: regret vs the session's Triton oracle | 0.9% / 10.3% / 14.8% (2.3%) | 0.2% / 4.6% / 9.6% (1.2%) |
| top-3 Triton only vs B3+ median, faster / slower | 1.026, 25 / 36 | 1.016, 23 / 35 |
| top-3 + LT16: vs B2 median, faster / slower | 0.901, 61 / 0 | 0.902, 49 / 0 |
| top-3 + LT16: vs B3+ median, faster / slower | 0.994, 26 / 9 | 1.000, 23 / 14 |
| top-3 + LT16: regret vs the joint oracle | 0.5% / 9.5% / 10.4% (2.0%) | 0.4% / 10.4% / 22.1% (2.7%) |
| top-3 + LT16: first use | 0.88 s | 0.67 s |
| picks cuBLASLt, all / >= 50 µs | 24% / 8% | 41% / 47% |

v2 beats v1 paired on the Triton-only choice, 0.9822 [0.9679, 0.9951] (12 faster / 6 slower) and ties it in the joint pool (0.9975 [0.9927, 1.0025]). Regime sizes differ (3090: 61 shapes < 50 µs and 25 above; 3080: 56 and 30), so the per-regime rows are not paired. The 3090 joint search removes the Triton-only tail against the joint oracle (p99 102.4% → 10.4%; 3080 85.7% → 22.1%) and is never slower than B2.
- **Triton ceiling on the 3090** (best Triton row of the session vs the searched cuBLASLt pool incl. the ranker's configs, 86 shapes): B3+ / Triton oracle geomean 0.965; within 90% on 76%. At >= 50 µs (26 shapes) the Triton oracle is ahead by more than 2% on 8 and behind on 13; **by the 3090's own floor (5%, below) it is ahead on 2 and behind on 10**. G4 does not pass on the 3090 either; the 3080's 0 of 86 > 5% is not contradicted (2 of 26 at >= 50 µs here).

**Noise floor of the 3090** (one session, 3 rounds, `analysis.report.noise_analysis` on `noise.jsonl`; 30 shapes × torch and one Triton config × cold / hot; the raw-sample file was not copied, so only IQR spread is available). `range` = (max − min) / median over the 3 rounds. The 3080 columns are its two complete locked-1755 sessions (`noise-20261014-001639-lock1755`, `noise-20261014-141218-6985`):

| | 3090 cold | 3090 hot | 3080 cold | 3080 hot |
|---|---|---|---|---|
| range p50 | 0.41% | 0.34% | 0.11-0.27% | 0.01-0.10% |
| range p95, all | 4.29% | 2.91% | 3.6% | 1.9-2.2% |
| range p95, >= 50 µs | **4.26%** | **3.93%** | 1.4-1.8% | 1.8-1.9% |
| range p95, < 50 µs | 2.72% | 1.52% | 6.2-6.4% | 1.2-2.3% |
| paired kernel-ratio range p95 / p99 | 4.6% / 5.6% | 3.3% / 5.7% | 4.2-5.7% / 9.8-14.9% | 2.5-3.3% / 3.0-3.5% |
| within-run IQR / median p50 | 1.15% | 0.87% | 0.62-0.63% | 0.49-0.56% |
| range max | 5.6% | 6.0% | 7.6-14.9% | 3.2-4.6% |

At >= 50 µs (83 (shape, kernel, mode) groups, 24 shapes) 17% of the 3090's groups have a range over 2% (3080: 5%); p95 by kernel: Triton 4.4-4.7%, torch 3.0-3.5% (3080: 1.4-2.3% / 1.4-2.2%). The noisiest groups are the fixed Triton config on skinny memory-bound shapes: 1×5120×13824 (5.6-6.0%), 8192×64×8192 (4.4-5.1%), 32×8192×8192 (4.3%). Cold / hot L2 median ratio 1.005 (3080: 1.003). Clock wander explains little of it: the range correlates 0.2 with the clock spread within a group (spread 2.4% vs 0.3% on the locked 3080), so what produces the other part is not identified.
- **Consequence.** The 3080's ε = 2% tie margin (≥ 50 µs) **does not apply to the 3090**; a provisional margin of **~5% at ≥ 50 µs** (about the paired p95/p99), single session, unlocked. At that margin the 3090's differences in the 0.5-3 point range against the 3080 (Triton-only p95 / p99, mean regret 4% vs 2%) are within noise of the oracle itself, and its "Triton wins by > 2%" count (8 of 26) falls to 2. The large effects stand: 61 / 0 against B2, the v1 → v2 tail difference (p99 43% → 20%), 0 of 86 picks over 50%.
- **Caveats.** One session: "across-round" is within a job, no reboot or second day, so it is not the cross-session quantity of G1 (the 3080's 1.75% p95); the real 3090 floor can only be larger. The p95 is the top 4-5 of ~83 groups and p99 is one or two groups. The < 50 µs numbers use per-launch timing; no graph-amortized noise run exists.

**What transfers across GPUs of the same architecture (3080 → 3090, both sm_86; 27 shared held-out shapes, each side scored against its own GPU's screen oracle, `cross_gpu_3090_retrained.md`).** The comparison has three arms on one set: the 3080's own models, the 3080's models used zero-shot on the 3090 (2026-11-03), and the 3090-retrained models (this run):

| online top-3 (27 shapes) | 3080 v1 | 3080 v2 | 3090 zero-shot v1 | 3090 zero-shot v2 | 3090 retrained v1 | 3090 retrained v2 |
|---|---|---|---|---|---|---|
| regret median / p95 / p99 (mean) | 1.5 / 7.6 / 9.2 (2.6) | 0.8 / 6.9 / 10.2 (1.8) | 5.4 / 14.5 / 26.5 (6.1) | 0.9 / 14.1 / 14.4 (4.0) | 0.2 / 8.4 / 16.2 (2.3) | 0.9 / 11.9 / 15.2 (2.8) |
| shapes over 5% | 6 | 2 | 14 | 10 | 4 | 6 |
| mean vs 3080 v1, paired | | -0.8 [-1.8, 0.0] | **+3.5 [+1.1, +6.4]** | +1.4 [-0.7, +3.6] | -0.4 [-1.9, +1.4] | +0.1 [-1.5, +1.7] |
| picks > 50% off | 0 | 0 | 0 | 0 | 0 | 0 |
| family ceiling vs B3+, median / within 90% | 1.007 / 25 | 1.007 / 25 | 1.008 / 24 | 1.008 / 24 | 1.017 / 24 | 1.017 / 24 |

- **Transfers as it is:** the design and the code path. The diversified top-k + short online search, v2's analytic features and the joint pool with cuBLASLt's list work unchanged on the 3090, with no pick over 50% off in any of the six arms and the same Triton ceiling (24-25 of 27 within 90% of B3+, mma.sync in all 527 compiled 3090 kernels). The joint search again ties B3+ (median 0.994) and is never slower than B2.
- **Transfers with a cost, zero-shot:** the 3080's Triton rankers. v2 is within noise of the 3080 (+1.4 [-0.7, +3.6]), v1 is not (+3.5 [+1.1, +6.4], 14 shapes over 5%). Retraining on the 3090 closes v1's gap entirely (-0.4 [-1.9, +1.4]) but adds little for v2 on these 27 shapes (zero-shot +1.4 → retrained +0.1, mean 4.0 → 2.8%, shapes over 5% 10 → 6, no interval between the two arms computed and the 3090 floor is 4-5%): **the analytic-feature model is what generalises, and a retrain mainly repairs the model that overfits GPU-specific details.** The ordering v2 > v1 repeats on all 86 held-out shapes.
- **Does not transfer, rebuilt per GPU (all but the bundle refuse another `hw_id`):** the cuBLASLt ranker (tile / algorithm ids, SM count; weaker on the 3090 at 68% vs 78% at top-10), the k-NN table, the router (AUC 0.803 vs 0.823, threshold 0.055 vs 0.02), the winner cache, the kernel bundle (it matches an sm_86 + Triton 3.7.1 key, so the 3080 bundle may serve the 3090, but this was not checked). Zero-shot use of the 3080's cuBLASLt ranker or router on the 3090 (`TUNE_*_FOREIGN`) was not run in this study.
- **Does not transfer, measured per GPU:** the tie margin (2% on the locked 3080, ~5% on the unlocked 3090) and what the vendor library picks (cuBLASLt chosen on 47% of the 3080's >= 50 µs shapes, 8% of the 3090's), so the value of the Triton half is GPU-specific.
- **Same-architecture is the easy case.** The 3080 → 2060 zero-shot failed (13 of 27 picks over 50% off) for a different reason (no `mma.sync` in Triton on sm_75). Whether the rankers cross to another tensor-core generation (Ada, Hopper) is untested.

**Gates.** 3080: none changed. 3090: G1 provisional, tie margin 5% at >= 50 µs (single unlocked session); G2 holds (24 of 27 within 90% of B3+ on the shared shapes); G4 not passed (2 of 26 above 5% at >= 50 µs, 10 behind). **Next steps.** (1) A second 3090 noise job (another day, ideally after a reboot; `remoteGPU/noise.sh`) for a cross-session floor and, if the admins lock clocks, a locked run. (2) Re-score the 3090 stage 2 at the 5% margin. (3) The router's held-out score and a fresh-shape check on this GPU. (4) The cuBLASLt ranker zero-shot from the 3080 on the 3090 (`TUNE_LT_RANK_FOREIGN=1`) to separate "same-arch transfer" from "retrain needed" for that component. (5) Find the `StopIteration` in `ranker_compare` and the 3090 stage-2 untimed joint picks. (done 2026-11-10; (2) and (3) also done there as CPU replays)

Reproduce (CPU only; GPU parts were `sbatch remoteGPU/train_best.sh` and `sbatch remoteGPU/noise.sh`):
```bash
N9=pipeline/runs/rtx3090/train_best-full-slurm/data/rtx3090_sm86_82sms_24gb; D9=data/rtx3090_sm86_82sms_24gb; D3=data/rtx3080_sm86_68sms_10gb
python -m analysis.cross_gpu --out reports/cross_gpu_3090_retrained.md \
  --side 3080-v1 $D3 online_lgbm_rank_diverse.jsonl online_joint_diverse.jsonl results_lock1755_v2.jsonl results_lock1755_v2_graph.jsonl \
  --side 3080-v2 $D3 online_joint_diverse_v2.jsonl online_joint_diverse_v2.jsonl results_lock1755_v2.jsonl results_lock1755_v2_graph.jsonl \
  --side 3090-zs-v1 $D9 zero_shot_v1_as_is.jsonl zero_shot_v1_as_is.jsonl results_zero_shot.jsonl results_zero_shot_graph.jsonl \
  --side 3090-zs-v2 $D9 zero_shot_v2_as_is.jsonl zero_shot_v2_as_is.jsonl results_zero_shot.jsonl results_zero_shot_graph.jsonl \
  --side 3090-retrained-v1 $N9 zero_shot_v1_as_is.jsonl zero_shot_v1_as_is.jsonl results_zero_shot.jsonl results_zero_shot_graph.jsonl \
  --side 3090-retrained-v2 $N9 zero_shot_v2_as_is.jsonl zero_shot_v2_as_is.jsonl results_zero_shot.jsonl results_zero_shot_graph.jsonl
# noise tables: analysis.report.noise_analysis(Path(data_dir), Path(figs_dir)) on a copy of noise.jsonl (it writes noise_stats.csv next to figs_dir)
```

## Update 2026-11-08: zero-shot on the 2060 with full stage-2 coverage — the Triton-only result reproduces; the joint search lands on cuBLASLt's heuristic list (24 of 27) and the ranker no longer matters

**What ran.** The next step of the 2026-11-02 (later) entry: `python -m analysis.zero_shot picks` and its stage-2 benchmarks on the RTX 2060 (`rtx2060_sm75_30sms_6gb`, `--clock-lock 1755`, driver 595.99.02), so every zero-shot Triton pick is timed in the same session as the baselines. Run directory `data/rtx2060/quick-20261029-095746-v4/` (copied here by hand; nothing was re-measured on this machine), report [`zero_shot_rtx2060/ranker_compare.md`](zero_shot_rtx2060/ranker_compare.md). The four zero-shot row files (v1 / v2 × `as_is` / `native`, k = 5, cuBLASLt top-16 in the pool, 27 shapes) are those of v3 (search sessions `20261102-113735-d051df`, `20261102-115635-e7d450`, `20261102-124121-809c54`; unchanged); only stage 2 is new.
- Stage 2 cold `20261108-004603-dc1cf3`: 1036 rows, 27 shapes, **0 invalid**, 1% power-capped (15 rows). Graph `20261108-101016-53fee8`: 211 rows, the 6 shapes under 50 µs, 0 invalid, 0% power-capped. Lock 1755 on every row, harness `86f90b5`, clean.
- The oracle of this report is the **joint oracle** (every Triton row timed in the session ∪ B3+), not the screen of the 2026-11-02 entry; the two are different yardsticks, so regrets are not comparable across the two entries.
- `data/.../reports/zero_shot_data.md` inside the run directory was made by the scorer before the `slow`-row fix (13-21 of 27 shapes "unscored", no `regret > 50%` column); do not read it. It was not regenerated here.

**Triton-only top-3** (regret vs the session's Triton oracle, median / p95 / p99 (mean); `ranker_compare.md`, last table). Reproduces the 2026-11-02 (later) finding with full coverage:

| variant | all (n = 27) | >= 50 us (n = 21) | < 50 us (n = 6) |
|---|---|---|---|
| v1 as_is | 88.2% / 2753.6% / 3785.7% (408.9%) | 92.4% / 3656.5% / 3796.2% (514.1%) | 13.2% / 113.1% / 116.2% (40.8%) |
| v1 native | 41.8% / 603.4% / 658.9% (156.7%) | 93.4% / 607.5% / 663.0% (198.7%) | 3.7% / 33.6% / 40.2% (9.8%) |
| v2 as_is | 75.5% / 610.6% / 3224.3% (272.0%) | 88.2% / 674.6% / 3431.0% (333.8%) | 12.3% / 191.3% / 219.8% (55.9%) |
| v2 native | 41.1% / 262.4% / 569.5% (81.2%) | 54.6% / 263.7% / 594.3% (95.3%) | 7.2% / 121.4% / 148.3% (31.7%) |

(Earlier medians were 43-80%.) vs B3+ the Triton-only pick is slower on 24 of 27 shapes in every variant (median 2.9-3.6x, p10 1.26-1.38x); it is faster than B3+ on 2 shapes and ties on 1 (counts by the 2% / 5% margin).

**Why Triton cannot win (best Triton row of the session vs B3+, per-shape check on the stage-2 rows).** Median 2.3x slower, within 5% of B3+ on 4 of 27 shapes; 4-7x slower on the large shapes (8192×4096×11008: 173 ms vs 28 ms). Triton beats the heuristic cuBLASLt pool (B3) by more than 2% only on 3×2×5, 105×6×9, 369×17×12 and 5×9730×30 (4 of 27, all tiny). Consistent with the FMA-only path on sm_75 (2026-11-02): the ranker is not the limiting part.

**Joint search (Triton top-3 + cuBLASLt top-16)**, same table:

| | all (n = 27) | >= 50 us (n = 21) | < 50 us (n = 6) |
|---|---|---|---|
| vs B2 median / p10 | 0.996-1.000 / 0.692 | 1.000 / 0.695 | 0.951 / 0.56 |
| vs B3 median | 1.000 | 1.000 | 0.985 |
| vs B3+ median / p10 | 1.014 / 0.988 | 1.045 / 1.000 | 0.985 / 0.59 |
| regret vs joint oracle median / p95 / p99 (mean) | 1.4-2.2% / 49.9% / 61.6% (8.9-9.0%) | 4.5% / 57.1% / 61.9% (11.4-11.5%) | 0.0% / 0.2-1.7% / 0.3-2.2% (0.1-0.4%) |
| picks cuBLASLt | 89% | 100% | 50% |

vs B2: faster on 7 shapes, slower on 0. vs B3+ (counts as printed in `ranker_compare.md`): slower on 13 of 21 shapes at >= 50 us and faster on 0, faster on 2 and slower on 0 below 50 us; the "all" row prints 2 faster / 10 slower, which does not equal the sum of the two regimes (2 / 13). Not investigated; use the per-regime counts until the cause is found. First-use cost 10.5-15.9 s for top-3 + LT16 (top-1 + LT16: 1.6-6.6 s, top-5: 15.5-26.3 s; the 3080 is ~0.7 s), of which cuBLASLt adds ~0.4 s: the cost is compiling and short-timing slow FMA Triton kernels.
- **The ranker does not matter on this GPU.** At >= 50 µs all twelve joint rows (4 variants × top-1 / 3 / 5) give the same regret and the same picks, because the search always ends on cuBLASLt. Top-5 only costs more (26.3 vs 15.9 s on v1 `as_is`) and picks the same winners; the variants differ only below 50 µs, where Triton is chosen on 50% of the 6 shapes.
- **Where the 50-60% tail comes from.** It is cuBLASLt configs outside the heuristic list: B3 is slower than B3+ (the 20 s deep search) by 1.63x on 448×1088×4096, 1.57x on 448×1088×1024, 1.33x on 128×4096×11008 and 1.28x on 28×3030×4012 (median B3 / B3+ over the 27 shapes 1.02). The 3080's cuBLASLt k-NN table and learned ranker (2026-11-04) are 3080-only, so on the 2060 the pool is the heuristic top-16 alone (`baselines/cublaslt_knn.py`, `baselines/cublaslt_rank.py` refuse another `hw_id`).

**What the joint search chooses** (stored `choice_lt` of the four zero-shot row files; for k Triton candidates the pool rank of a cuBLASLt entry is k + its heuristic rank):
- cuBLASLt on 24 of 27 shapes in all four variants (top-1: 24-25), Triton on 3 (2-3 at top-1).
- Of the 24 cuBLASLt picks, 12-13 are cuBLASLt's own first heuristic pick (the B2 kernel) and 11-12 are another entry of the top-16 (its #2 to #7, once #10). Every cuBLASLt pick comes from the heuristic list (no proposals on this GPU).
- The winner among cuBLASLt entries is one short timing: differences within the 2% margin are arbitrary.

**Conclusion.** On this GPU the practical selector is "time cuBLASLt's heuristic list and skip Triton", except possibly for the tiny shapes under 50 µs; it is B3 by construction (1.000), 1.4% behind B3+ (median), and the learned Triton ranker adds 10-20 s of first-use cost with no change in the choice at >= 50 µs. The zero-shot joint search was already at its pool's ceiling in the 2026-11-02 (later) entry; this run confirms it with the Triton picks timed. Still a confound: Triton has no tensor-core path on sm_75, so this does not separate "the 3080 ranker does not cross GPUs" from "no mma on this GPU" (the 3090 entry, 2026-11-03, points to the second).

**Gates.** None changed. **Caveats.** n = 27 (p95 / p99 are the worst one or two shapes); the Triton oracle at >= 50 µs is itself made of slow FMA kernels, so the huge Triton-only regrets come from a few very slow picks: read the medians and the vs-B3+ counts; the "chooses" counts come from one short timing per candidate, not the stage-2 session; no `cross_gpu` re-run (the shared-shape table of 2026-11-02 (later) is unchanged). **Next steps.** (1) Retrain the cuBLASLt ranker for the 2060 (`pipeline/zero_shot.sh --lt-mode retrain`, `docs/zero_shot_runbook.md` 6b) to test whether it recovers the gap to B3+ on the four large-B3/B3+ shapes (inference, not tested). (2) Check the router guard from `8a7cec1` on this GPU: it should skip the Triton half and its 10-20 s. (3) A tensor-core GPU other than the 3080 / 3090 is no longer needed for the confound; the 3090 entry covers it.

Reproduce (CPU only, rows already in the run directory): `python -m analysis.zero_shot score --data data/rtx2060/quick-20261029-095746-v4/data --report data/rtx2060/quick-20261029-095746-v4/reports --k 3` for the screen-scored table of the 2026-11-02 entry, and `python -m analysis.zero_shot picks` then `python -m analysis.ranker_compare` (commands printed by `picks`) for `zero_shot_rtx2060/ranker_compare.md`; the per-shape B3 / B3+ / best-Triton comparison and the chosen-family counts are a short script over `results_zero_shot.jsonl` and the `choice_lt` field of `zero_shot_<variant>.jsonl`.

## Update 2026-11-06: p99 in the remaining docs, reports and generators — noise floor, fresh shapes, cross-GPU, overview; the tail the tie margin does not cover

**What.** The 2026-11-04 (audit) entry added p99 to the selector and joint-search scorers. This pass carries it to everything else that quoted a p95 without it. **Nothing was re-measured**: every number is a re-score or re-aggregation of stored rows (CPU only; no GPU job ran; GPU idle throughout). Where a generated file changed, its other cells are unchanged: with the p99 keys / columns stripped, `cross_gpu_summary.json`, the 3090 `zero_shot_*_summary.json` and `ranker_compare_summary.json`, the five `week1_summary_lock1755*.json` noise blocks and `results_overview.md` equal the committed versions cell for cell (checked by script). The one exception was the compiled-kernel count of `cross_gpu.md` (the local 3080 Triton cache has grown from 1,886 to 5,310 files since that report), which was put back to the committed value so that the diff is p99 only.

**Code.** `analysis/zero_shot.py` (`summarize` stores `p99`, `fmt` prints `median / p95 / p99 (mean)` and still prints summaries written before p99 existed; the two 3080 reference rows are now `(median, p95, p99, mean)` and were re-derived from the stored online files with `cross_gpu.screen_table` / `score_rows`: top-1 v1 3.6 / 18.1 / 24.0 / 5.5% and v2 2.8 / 13.0 / 26.2 / 4.6% reproduce the old triples exactly); `analysis/cross_gpu.py` (header text); `analysis/plots.py` (p99 columns in the overview's noise, selector and portfolio tables); `analysis/report.py` (`range_p99` / `paired_p99` next to the p95s of the noise summary); `pipeline/run.sh`, `pipeline/zero_shot.sh` (table labels). Docs: `AGENTS.md` §6 and §7, `docs/methodology.md`, `docs/zero_shot_runbook.md`, `results/*.md` (the plain-language summary; its figures still draw median / mean / p95).

**Noise floor** (`docs/methodology.md`; p99 over 44 / 43 kernels per session, i.e. essentially the worst kernel):

| | p50 | p95 | p99 | max |
|---|---|---|---|---|
| range of medians across the three sessions, ≥ 50 µs, cold | 0.24% | 1.75% | **2.3%** | 2.5% |
| same, hot | 0.09% | 1.78% | **2.6%** | 3.0% |
| across rounds, one allocation, all shapes (cold) | 0.67% | 5.4% | 12.5% | 20.0% |
| across rounds, 3 allocations pooled (cold) | 0.29% | 2.6% | 6.3% | 9.6% |
| paired kernel ratio across rounds, 1 / 3 allocations (cold) | 2.1 / 0.33% | 6.2 / 3.5% | 20.0 / 10.3% | |

Per session across rounds (≥ 50 µs, cold / hot), p95 → p99: `…642a` 1.3 / 1.2 → 2.0 / 1.7%; `…lock1755` 1.8 / 1.8 → 3.1 / 2.8%; `…6985` (2 min after a reboot) 1.4 / 1.9 → **4.3 / 4.2%** (worst cold kernel 4.8%). Under 50 µs, cold per launch: p95 6.2-6.4% in the locked sessions, p99 7.3 / 13.2%. **The tie margin ε = 2% is a p95 statement.** The worst kernel in 44 moves 2.3-2.6% between sessions and 3-4% between rounds of one session, so a 2-4% win on one shape is not distinguishable from noise even when it passes ε; G1 is unchanged (it is defined on p95). The kernel-ratio p95 of 1.94% quoted since 10-14 comes from a script that is not in the repo; the obvious definition on `noise_stats.csv` gives 2.05% cold / 2.11% hot, so no p99 is given for it.

**Fresh shapes (2026-11-05 entry, now with p99).** Regret vs the joint oracle, median / p95 / p99 / mean: heuristic-only 1.0 / 17.4 / 33.6 / 3.3% → k-NN8 1.0 / 6.7 / 27.6 / 2.5% → ranker top-5 **0.6 / 5.6 / 12.8 / 1.6%** (held-out for the same pools: p99 22.8 → 8.3 → 8.4%). Against the extended oracle: 0.7 / 5.6 / 12.9 / 1.8%. The ranker's p99 gain over the heuristic pool is −18.9 points [−36.3, 0.0] paired (the interval touches 0, as the p95's does; only the mean excludes 0). Router + confirm 3 leaves the p99 at 12.9% (A and C), B 13.9%. `python -m analysis.lt_rank_ab` and `confirm_ab` with the commands of that entry print all of it.

**Cross-GPU** (`reports/cross_gpu.md`, 27 shared shapes, p99 = the worst shape or the one before it): online top-3 p99 3080 v1 9.2%, 3080 v2 10.2%, 2060 retrained 13.3%; 3090 zero-shot v1 as_is / native 26.5 / 14.8%, v2 as_is / native 14.4 / 12.6%; 2060 zero-shot 570-3260%. So the 3090's zero-shot v2 stays within noise of the 3080 on the mean (2026-11-03 entry) but sits at 1.2-1.4x the 3080 v2's p99, and the zero-shot v1 `as_is` tail (26.5% against 9.2%) is the clearest sign that v1's tail does not transfer; at n = 27 one shape decides it. 3080 reference rows of the zero-shot reports (86 shapes, v1 / v2): top-1 24.0 / 26.2%, best-of-3 diverse 15.9 / 13.0%, v2 best-of-5 13.0%.

**Overview and Triton-only tables** (`reports/results_overview.md`, `learned_topk.md`, `selector_baselines.md`, `sweep_portfolio*.md`, all with p99 already or now): held-out cold stage 2, p95 → p99 (as regret fractions): B6 4.86 → 9.48; diverse online top-3 0.091 → 0.202; k-NN + fill 0.229 → 0.413; static portfolio of 100 configs held-out 0.111 → 0.182 (tune 0.018 → 0.028).

**Left as written.** The dated entries below keep their p95-only tables (the 2026-11-04 audit entry and this one carry the p99 of the ones the audit re-scored), the baseline tables (speed against B1-B3+, not regret distributions), `plan.md` (frozen; its gates G1/G5 are defined on p95) and the figures. Two numbers in `AGENTS.md` come from a session with no stored p99 (`topkd` cold ≥ 50 µs and graph rows) and quote the `topki` session's p99, labelled as such.

**Reproduce** (CPU only):
```bash
D=data/rtx3080_sm86_68sms_10gb
python -m analysis.plots --data $D                                  # results_overview.md with p99
python -m tune analyze --data $D --results results_lock1755.jsonl   # noise summary with range_p99 / paired_p99 (other sections of a regenerated file differ from the committed ones: the committed ones predate later data; only the noise table was spliced)
python -m analysis.zero_shot score --data data/rtx3090_sm86_82sms_24gb --report reports/zero_shot_rtx3090_sm86_82sms_24gb/k3 --k 3   # and k5
# cross_gpu: the command of the 2026-11-03 entry
```

## Update 2026-11-04 (audit): the joint-search tables rescored against an oracle that sees the proposals — ranker top-5 is 0.5% / 5.5% / 1.4%, not 0.4% / 4.7% / 1.2%; its win over the heuristic stands, its win over k-NN8 is not shown

**Why.** An audit of every number quoting the ranker's win (re-running `analysis.ranker_compare`, `lt_rank_ab`, `lt_knn_ab`, `confirm_ab` on the stored rows: all reproduce the earlier reports exactly) found one scoring flaw. The "joint oracle" of `lt_rank_ab`, `confirm_ab`, `analysis.router` and `analysis.joint_rank` was the best Triton config of the session and B3+. The stage-2 session also times the k-NN proposals and the learned ranker's top-16 (`cublaslt_knn:8`, `cublaslt_rank:16`), and those configs were not in the oracle, so a pool that picked one was scored against an oracle that cannot see it: **negative regret on 4 shapes for the ranker top-5 (worst −2.5%), 4 for k-NN8 (worst −4.9%)**. `lt_knn_ab` already printed an extended oracle next to the old one; the other scripts did not. The extended oracle (`orc_x` in `analysis.ranker_compare.references`) is now the one every script uses; it differs from the old one on 12 of 86 shapes and equals it in sessions that timed no proposals (`ranker_compare.md` is byte-identical). No measurement was re-run; this is a re-scoring of the same rows (cold `results_lock1755_rank16.jsonl`, graph `…_rank16_graph.jsonl`, online `online_joint_diverse_v2_knn8_rank16.jsonl`). `harness_dirty` is true on those rows, as before.

**Corrected held-out table** (`python -m analysis.lt_rank_ab`; v2 top-3 + heuristic LT16 + the named proposals; regret vs the extended oracle, median / p95 / p99 / mean; the old value, vs the oracle without proposals, in brackets for the rows that moved; **p99 added 2026-11-05**: on 86 shapes it is the second-worst shape plus 15% of the way to the worst, and on the 30- and 56-shape subsets it is essentially the worst shape, so read it as "the worst case", not a stable quantile):

| pool | all | ≥ 50 µs (n = 30) | < 50 µs (n = 56) | shapes > 5% / > 10% off | vs B3+ faster / slower | first-use cost |
|---|---|---|---|---|---|---|
| heuristic top-16 only | 0.7% / 11.7% / 22.8% / 2.8% (0.6 / 11.6 / 22.8 / 2.6) | 0.6 / 14.0 / 20.5 / 3.1 | 0.7 / 10.9 / 20.0 / 2.7 | 14 / 7 (12 / 7) | 22 / 17 | 0.66 s |
| + k-NN8 | 0.6% / 6.9% / 8.3% / 1.7% (0.6 / 6.8 / 8.3 / 1.6) | 0.5 / 5.3 / 6.5 / 1.6 | 0.7 / 7.2 / 8.5 / 1.8 | 9 / 0 | 23 / 13 | 0.77 s |
| **+ ranker top-5** | **0.5% / 5.5% / 8.4% / 1.4%** (0.4 / 4.7 / 8.4 / 1.2) | 0.4 / 5.0 / 8.7 / 1.4 | 0.6 / 5.2 / 7.8 / 1.3 | **5** / 0 (4 / 0) | 23 / **7** | 0.80 s |
| + ranker top-10 | 0.5% / 5.5% / 8.5% / 1.4% | 0.4 / 5.0 / 8.8 / 1.4 | same | 5 / 1 | 23 / 7 | 0.85 s |
| + ranker top-16 | 0.5% / 5.5% / 8.5% / 1.3% | same as top-10 | same | 5 / 1 | 23 / 7 | 0.92 s |
| + k-NN8 + ranker top-5 | 0.5% / 4.7% / 7.2% / 1.3% (0.5 / 4.7 / 7.2 / 1.1) | 0.5 / 3.7 / 5.2 / 1.1 | 0.6 / 5.2 / 7.5 / 1.3 | 4 / 0 | 24 / 8 | 0.90 s |

The "> 10%" column is on a knife edge: the top-5 and top-10 pools pick the same config on 68×6420×10359, which is 9.93% off in one and 10.06% in the other. Read "0 vs 1" as no difference. A fifth shape is now > 5% off for the ranker (1×709×521, 7.4%; the old oracle hid it because the best time there is a proposal that was timed in stage 2 but not chosen, which the old oracle could not see). The shapes still > 5% off: 68×6420×10359 10.1%, 226×2141×1139 8.2%, 1×709×521 7.4%, 1316×272×100 6.8%, 2048×4096×11008 5.7%.

**Is the ranker's win real?** Paired bootstrap of the regret difference a − b over the 86 shapes (4000 resamples, 95% interval, points; new in `lt_rank_ab`):

| a − b | mean | p95 | p99 |
|---|---|---|---|
| k-NN8 − heuristic | −1.08 [−1.96, −0.35] | −5.65 [−13.9, −1.2] | −12.9 [−19.3, −3.6] |
| **ranker top-5 − heuristic** | **−1.44 [−2.44, −0.65]** | **−6.81 [−13.8, −1.7]** | **−12.8 [−20.0, −4.5]** |
| ranker top-5 − k-NN8 | −0.35 [−0.82, +0.09] | −1.16 [−3.35, +1.26] | +0.2 [−2.8, +2.8] |
| k-NN8 + ranker top-5 − k-NN8 | −0.45 [−0.81, −0.16] | −1.82 [−3.56, −0.03] | −1.0 [−3.1, +0.0] |

- **Ranker vs the heuristic joint search: a real win**, in the mean and in the tail (also the paired time ratio 0.987 [0.978, 0.994], 11 faster / 1 slower). The p99 says the same: 22.8% → 8.4%; the heuristic-only pool's worst shapes are 20-30% off, the ranker's worst is 10%.
- **Ranker alone vs k-NN8: not shown.** All three intervals include 0 (at p99 the two are equal, 8.4% vs 8.3%: the same worst shapes remain, mostly 68×6420×10359); the earlier "p95 4.7% vs 6.8%" is 5.5% vs 6.9% on the extended oracle, and the p95 of 86 shapes is the 4th-5th worst shape. The case for the ranker over k-NN rests on the tune cross-validation (78% vs 71% of the gap closed at ~10 candidates) and on the shapes > 5% off (5 vs 9), not on a held-out mean.
- **The union is better than k-NN8 alone** (mean −0.45 [−0.81, −0.16], p95 borderline), at +0.13 s; shipping the ranker only is a cost choice, not a measured quality choice.
- Gap recovery against the earlier ceiling (heuristic + the best deep config: 0.4% mean / 2.4% p95, measured in another session): ranker 58% of the mean gap and 67% of the p95 gap (was 64% / 75%), k-NN8 46% / 52% (was 45% / 52%). The ceiling comes from a different session and oracle, so these are indicative.

**Where the win over cuBLAS is** (geometric-mean time ratio to B3+ of the chosen config; shapes faster / slower by the tie margin):

| pool | all (86) | ≥ 50 µs (30) | < 50 µs (56, graph-amortized) |
|---|---|---|---|
| heuristic top-16 | 0.981, 22 / 17 | 1.016, 3 / 10 | 0.963, 19 / 7 |
| + k-NN8 | 0.972, 23 / 13 | 1.002, 3 / 7 | 0.956, 20 / 6 |
| + ranker top-5 | 0.968, 23 / 7 | **1.001, 4 / 6** | 0.951, 19 / 1 |

At ≥ 50 µs the ranker pool ties B3+ (G4 unchanged); the advantage over B3+ sits in the 56 sub-50 µs shapes, which are graph-timed with the assumed 5% margin and have no graph-mode noise study (§9 item 4). The "faster than B2 / B3 on 49 / 30 shapes, slower on 0" counts of `ranker_compare.md` are partly by construction (B2 and B3's candidates are in the pool); the ranker's own contribution is the paired comparison above.

**Router and confirmation tables, rescored** (`python -m analysis.confirm_ab`, same runs and pools as the evening entry; the paired time ratios, the skipped counts and the costs are unchanged because they do not use the oracle):

| run | regret median / p95 / p99 / mean (old, without p99) | shapes > 5% / > 10% | slower than B3+ |
|---|---|---|---|
| reference: Triton first, no router, no confirm | 0.5 / 5.5 / 8.4 / 1.4 (0.4 / 4.7 / 1.2) | 5 / 0 (4 / 0) | 7 |
| router, no confirm | 0.6 / 7.3 / 10.8 / 1.8 (0.5 / 6.7 / 1.7) | 8 / 1 (7 / 1) | 9 |
| LT first, never skip, no confirm | 0.6 / 7.3 / 9.3 / 1.7 (0.6 / 6.5 / 1.5) | 8 / 0 (7 / 0) | 9 |
| A: Triton first, confirm 3 | 0.6 / 5.1 / 8.8 / 1.3 (0.5 / 4.5 / 1.2) | 5 / 0 (4 / 0) | 6 |
| B: LT first, never skip, confirm 3 | 0.6 / 6.4 / 9.3 / 1.5 (0.6 / 4.7 / 1.4) | 5 / 0 (4 / 0) | 7 |
| **C: router, confirm 3** | **0.4 / 4.9 / 8.8 / 1.3** (0.3 / 4.6 / 1.1) | **4** / 0 (3 / 0) | **5** |

Conclusions of the evening entry are unchanged: the router with confirmation is quality-neutral (C vs A worse on 0, better on 1) and saves 0.10 s (−13%) without the bundle; the cuBLASLt-first order bias is still there (B vs A: worse on 3, better on 1). At p99 the router with confirmation (8.8%) is no better than the reference (8.4%): the router saves cost, it does not improve the worst case. `analysis.router score` (simulation, never skip → router at the rule threshold): 0.5 / 5.5 / 8.4 / 1.4% → 0.5 / 4.7 / 8.8 / 1.3%. The single-score negative result (`analysis.joint_rank`) is unchanged in kind: fixed pool 0.5 / 5.5 / 8.4 / 1.4% (5 / 0 shapes) against joint top-10 0.8 / 9.1 / 26.8 / 2.4% (7 / 4), top-16 0.6 / 8.1 / 26.8 / 2.2% (6 / 4), top-24 0.6 / 5.5 / 8.5 / 1.4% (5 / 1).

**Other items from the audit.**
- *Triton-only ranker vs the trivial selectors.* The held-out win holds (screen, top-1: 3.6 / 18.1 / 5.5% for lgbm_rank vs 4.1 / 27.3 / 7.8% for knnf and 48 / 487 / 114% for b6). But the headline top-3 numbers compare a search that compiles and short-times 3 candidates (~0.4 s) with a zero-cost lookup; the equal-budget comparison is top-1, where the medians tie (2.4% vs 2.5%) and the gain is in the tail and mean. A rough paired recomputation from `results_lock1755_topki*.jsonl` (not the report's exact oracle, so only the direction is meaningful) gave a geometric-mean time ratio lgbm top-1 / knnf of about 0.97, interval about [0.95, 0.99], faster on 23 shapes and slower on 8 by 5%; it is not part of `learned_topk.md` and should be produced by the harness before it is quoted. The Triton-only selector is still slower than B3+ on 35 of 86 shapes (v2).
- *Session spread.* The same Triton diverse top-3 reads 1.0 / 9.9 / 2.7% (`topkd` session, quoted in AGENTS.md) and 1.1 / 9.1 / 2.6% (`topki`, the current `learned_topk.md`); the heuristic-only joint pool reads p95 10.4%, 10.7%, 11.6% in three sessions. A/B differences are valid within one session; a difference of 0.1-0.2 points in a median is noise (about 10% of online picks change between identical runs).
- *Held-out use in choices.* Tune CV named the ranker top-10 the primary pool; shipping top-5 was chosen after the held-out table showed top-5, 10 and 16 equal. The tail suite excluded every shape sharing two dimensions with a held-out shape, the cuBLASLt screen did not (73 of its 652 shapes share two positional dimensions with one); neither fit read a held-out label.
- *Leakage checked:* `lt_screen` shapes ∩ held-out = 0; tail ∩ held-out = 0; `lt_search` holds the held-out shapes only as the B3+ baseline (the k-NN table reads the tune ones, asserted by a test).

**p99 (2026-11-05).** The scorers (`ranker_compare`, `lt_rank_ab`, `lt_knn_ab`, `confirm_ab`, `router`, `joint_rank`) now print p99 next to p95 and `ranker_compare.md` / `_summary.json` were regenerated with it (the v2 Triton-only top-3 reads 0.2 / 4.6 / 9.6 / 1.2% vs the Triton oracle, v1 0.4 / 9.5 / 15.1 / 2.3%; the joint-oracle columns of that file are unchanged apart from the p99). The Triton-only selector reports (`learned_topk.md`, `selector_baselines.md`, `learned_selectors.md`, `sweep_portfolio.md`, `sweep_portfolio_heldout.md`) were regenerated with p99 as well, in a scratch copy of `data/<hw>/` (symlinks to the stored screens and stage-2 sessions, copies of the files the builders write) so that no saved ranker, picks file or cache in the real `data/` was touched; nothing was re-measured. Checks: with the p99 keys stripped, the new summary JSONs equal the committed ones exactly for `selector_baselines`, `learned_topk` and both portfolio reports (the only other difference is a scratch path string), and the refit `lgbm_rank` picks are identical to the saved ones. **Two non-reproducibilities, neither touching a headline number:** (1) `logreg_rank` flips 2 of 86 picks when refit with the same pinned libraries, so `learned_selectors.md` was rescored from the picks saved in `learned_picks.json` (the ones the stage-2 sessions measured) through a new opt-in `learned.build(reuse_picks=True)`; with that, all numbers match the committed summary except four `lgbm_rank` feature-importance gains that move in the 4th digit (thread-order float sums; the rendered percentages are unchanged). (2) The cached knnc fit lacked p99; `selectors` now treats a cache without it as stale and refits (about 6 minutes), which reproduced the old numbers exactly.

What p99 adds on the Triton side (held-out, 86 shapes; p99 is essentially the worst shape): the diverse online top-3 reads 1.1% / 9.1% / **20.2%** (median / p95 / p99), knnf 2.5% / 22.3% / **40.9%**, knn5 2.6% / 24.1% / 122.2%; best-of-3 with perfect timing 0.7% / 9.1% / 18.6%. On the screen, lgbm_rank top-1 is 3.6% / 18.1% / 24.0% against knnf 4.1% / 27.3% / 41.7%, and the best single config (b6) 48% / 487% / 973%. So the learned selector halves the worst case relative to knnf, but its worst shapes are still about 20% off; p95 alone (9%) hides that. The static portfolios keep a fat tail too: held-out p99 18.2% even at 100 configs (p95 11.1%), 19.7% at 50.

**Changed.** `analysis/ranker_compare.py` (`orc_x`), `analysis/lt_rank_ab.py` (extended oracle as the headline, old oracle shown once, regret bootstrap, speed by size), `analysis/confirm_ab.py`, `analysis/router.py`, `analysis/joint_rank.py` (all score against `orc_x`), `tests/test_ranker_compare.py` (one new test). Earlier entries below keep their original tables; where they quote a regret against the old oracle they carry a pointer to this entry.

**Reproduce** (CPU only):
```bash
D=data/rtx3080_sm86_68sms_10gb
python -m analysis.lt_rank_ab --data $D && python -m analysis.confirm_ab --data $D
python -m analysis.router score --data $D --model $D/router && python -m analysis.joint_rank --data $D
python -m analysis.ranker_compare --data $D --online v1=online_joint_diverse.jsonl v2=online_joint_diverse_v2.jsonl   # unchanged
```

## Update 2026-11-05: the shipped selector on 60 fresh shapes — the learned cuBLASLt ranker and the router replicate, and the ranking heuristic < k-NN < ranker holds

**Why.** Every choice behind the learned cuBLASLt ranker, the router and the confirmation pass was evaluated on the same 86 held-out shapes (the router's threshold and the ranker's model were fixed from tune data, but the pool comparisons and the order-bias finding were read off held-out). A fresh set tests the frozen systems on shapes nothing was fit, screened, selected or reported on.

**The set** (`python -m suites.fresh`, seed 465, `suites/fresh/fresh465.json`, split `fresh`): 30 `fresh_random` shapes (log-uniform M, N, K in [1, 16384], like S7) and 30 `fresh_llm` shapes (linear-layer dimensions of Qwen2-7B, Gemma-2-9B, Phi-3-mini, Mistral-Nemo-12B and Yi-6B, none in S2, with token counts {3, 24, 96, 300, 1000, 3000, 6000}, none in S2's list). Each shape is in no `suites/v1` suite, shares two of M, N, K with **no tune-split shape** (the grouping the cross-validation uses) and is ≤ 1200 GFLOP, so a Triton screen oracle exists as for the held-out shapes; `tests/test_fresh_suite.py` asserts this. The set was generated before any result on it existed. **Nothing was refit, re-thresholded or re-chosen**: v2 Triton ranker, learned cuBLASLt ranker (top-5), k-NN proposals (8 x 2), router (threshold 0.02) and `confirm = 3` are exactly the shipped objects.

**What ran.** Triton oracle: `tune sweep --shapes-file suites/fresh/fresh465.json --screen-dir sweep_screen_fresh` (60 shapes, 4,828 s, 0 skipped). Four online runs (v2 diverse k = 5, heuristic top-16): `online_fresh_all` (+ k-NN8 + ranker top-16, tagged, `20261105-000806-3ddb28`), `online_fresh_confirm3` (Triton first, `…-000957-ccbe89`), `online_fresh_router0_confirm3` (cuBLASLt first, never skip, `…-001123-863cb7`), `online_fresh_router_confirm3` (the shipped router configuration, `…-001251-7f3bd0`). ONE stage-2 session times every candidate (`torch,triton:sweep:8,triton:picks,cublaslt:16,cublaslt_deep:20,cublaslt_knn:8,cublaslt_rank:16`): cold `results_lock1755_fresh.jsonl` `20261105-001429-7498f1` (3,117 rows, 0 invalid, 0 not converged, 52% power-capped) and graph `results_lock1755_fresh_graph.jsonl` `20261105-003339-1ac6d3` (31 shapes < 50 µs, 1,637 rows, 0 invalid, 50% power-capped). `harness_dirty` is true on every row, clock locked at 1755.

**Pools** (`analysis.lt_rank_ab`; v2 top-3 Triton + heuristic top-16 + the named proposals; regret vs the joint oracle of the session = best Triton of the screen top-8 and the picks, and B3+; each pick looked up at its stage-2 time):

| pool | regret median / p95 / p99 / mean | shapes > 5% / > 10% off | slower than B3+ | `fresh_random` p95 | `fresh_llm` p95 |
|---|---|---|---|---|---|
| heuristic top-16 only | 1.0% / 17.4% / 33.6% / 3.3% | 9 / 4 | 14 | 6.2% | 17.7% |
| + k-NN8 | 1.0% / 6.7% / 27.6% / 2.5% | 7 / 2 | 13 | 5.0% | 13.0% |
| **+ ranker top-5** (shipped) | **0.6% / 5.6% / 12.8% / 1.6%** | **4 / 1** | 10 | 5.0% | **7.4%** |
| + k-NN8 + ranker top-5 | 0.6% / 5.6% / 11.3% / 1.6% | 4 / 1 | 11 | — | — |

(The held-out numbers of the same pools: 0.6 / 11.6 / 22.8 / 2.6% → 0.6 / 6.8 / 8.3 / 1.6% → 0.4 / 4.7 / 8.4 / 1.2%; 12 / 9 / 4 shapes > 5% off. With 60 shapes p99 is essentially the worst shape: 12.8% here is 96×5120×14336, whose oracle is 16% faster than B3+; the heuristic-only pool's 33.6% is the same kind of shape where a deep-search config is missing.) Against the extended oracle that also contains the proposals (the proposals can then be the oracle): ranker top-5 0.7% / 5.6% / 12.9% / 1.8%. The p99 gain over the heuristic-only pool is large (−18.9 points [−36.3, 0.0], paired) but its interval touches 0, like the p95's; the mean (−1.71 [−3.60, −0.31]) is the only one that excludes 0. Paired geometric-mean time ratios (95% bootstrap): ranker top-5 vs heuristic only **0.9851 [0.9692, 0.9973]**, faster on 5 shapes, slower on 0; k-NN8 vs heuristic only 0.9922 [0.9813, 1.0005], 4 / 2; ranker top-10 vs k-NN8 0.9929 [0.9795, 1.0020], 4 / 1 (the interval contains 1, as on held-out). Top-5, 10 and 16 give the same online choice again. By group: the ranker's gain is concentrated on the LLM-layer shapes (p95 17.7% → 7.4%, mean 4.4% → 2.2%); on the random shapes both proposal sources reach the same 5.0% p95.

**Against the vendor baselines** (shipped pool, speed = t_pick / t_baseline, < 1 faster; faster / slower by the tie margin): B2 median 0.924, p10 0.754, **42 / 1**; B3 median 0.981, 25 / 3; B3+ median 1.000, p10 0.871, 11 / 10 (≥ 50 µs: 1.007, < 50 µs: 0.998). Remaining misses (> 5% off the oracle): 96×5120×14336 (17.9%, the oracle is 16% faster than B3+ there), 24×1024×5120 (9.5%), 180×825×258 (6.7%), 469×15832×13 (5.5%). G4 is unchanged: nothing is faster than B3+ by more than the margin where the deep search is the bar.

**Router and confirmation** (`analysis.confirm_ab`, same pools, ranker top-5, k = 3 choice): the router skipped 15 of 60 shapes.

| run | regret median / p95 / p99 / mean | > 5% / > 10% | slower than B3+ | first-use cost (no bundle) | skipped |
|---|---|---|---|---|---|
| reference: Triton first, no router, no confirm | 0.7% / 5.6% / 12.9% / 1.8% | 4 / 1 | 10 | 0.954 s | 0 |
| A: Triton first, confirm 3 | 0.5% / 5.6% / 12.9% / 1.7% | 5 / 1 | 9 | 1.005 s | 0 |
| B: cuBLASLt first, never skip, confirm 3 | 0.8% / 6.4% / 13.9% / 1.9% | 5 / 1 | 7 | 1.020 s | 0 |
| **C: router, confirm 3** | **0.5% / 5.2% / 12.9% / 1.6%** | 4 / 1 | 8 | **0.855 s (−10%)** | 15 |

C vs the reference 0.9977 [0.9955, 0.9998] (worse on 1, better on 2); C vs A 0.9996 [0.9973, 1.0016] (1 / 1); C vs B 0.9977 [0.9951, 0.9999] (0 / 2); B vs A 1.0019 [0.9997, 1.0042] (the timing-order effect, worse on 2, better on 1, smaller than on held-out but the same direction). On the 15 skipped shapes the C / A time ratio has geometric mean 0.9978, worst 1.024 (6000×3072×8192: 5001 → 5121 µs, P(win) 0.012). The router's P(win) separates the shapes where Triton's oracle beats B3+ by more than the margin (12 of 60) with **AUC 0.861** (tune out-of-fold 0.823), and none of the 15 skipped shapes was one of them.

**Reading.** All three results replicate on shapes none of the models, thresholds or conclusions had seen: the ranker beats the heuristic-only pool with a paired interval that excludes 1 and is better than k-NN8 on the tail but not significantly in the mean; the router at the rule threshold saves ~10% of the no-bundle first-use cost at no measurable quality cost; the confirmation pass keeps the ordering effect small. The tail is a little heavier than on held-out (p95 5.6% vs 4.7%, p99 12.8% vs 8.4%, one shape > 10% off vs none), mostly one large LLM-layer shape. As on held-out, the router does not move the p99 (12.9% in A and C; the worst shape is not one it skips). **Limits.** 60 shapes (31 graph-amortized), one GPU, harness dirty, ~50% of the stage-2 samples power-capped; the cost column is the no-bundle regime (where the router is on); the 3080's models are the only ones exercised.

**Reproduce.**
```bash
python -m suites.fresh                                                    # suites/fresh/fresh465.json
python -m tune --clock-lock 1755 sweep --shapes-file suites/fresh/fresh465.json --screen-dir sweep_screen_fresh
# online: all (--lt-knn 8 --lt-rank 16), confirm3 (--lt-rank 5 --confirm 3), router0_confirm3 (+ --router --router-threshold 0), router_confirm3 (+ --router)
#   python -m tune --clock-lock 1755 online --shapes-file suites/fresh/fresh465.json --ranker $D/lgbm_rank_v2 --k 5 --order diverse --lt-n 16 --fresh-cache --out-name online_fresh_<name>.jsonl ...
# stage 2: TUNE_SWEEP_DIR=$D/sweep_screen_fresh TUNE_PICKS=$D/fresh_stage2_picks.json (union of the runs' Triton candidates), backends as above;
#   cold on the whole file, graph (--cache-modes hot --graph-batch 50) on the shapes with a best time < 50 us
python -m analysis.lt_rank_ab --online online_fresh_all.jsonl --results results_lock1755_fresh.jsonl --graph-results results_lock1755_fresh_graph.jsonl
python -m analysis.confirm_ab --reference online_fresh_all.jsonl --results results_lock1755_fresh.jsonl --graph-results results_lock1755_fresh_graph.jsonl \
  --run "A: Triton first, confirm 3=online_fresh_confirm3.jsonl" --run "B: LT first, never skip, confirm 3=online_fresh_router0_confirm3.jsonl" --run "C: router, confirm 3=online_fresh_router_confirm3.jsonl"
```

## Update 2026-11-04 (evening): a cost-only family router works once the timing-order bias is cancelled; a single additive score across families does not

*Regret figures in this entry are against the oracle without the proposals; the extended-oracle values are in the audit entry above (reference 0.5 / 5.5 / 1.4%, C 0.4 / 4.9 / 1.3%).*

**Questions.** (1) Can the Triton half of the online search (rank, compile or bundle load, validate, time) be skipped when cuBLASLt is clearly ahead? (2) Can one score on one scale rank Triton and cuBLASLt candidates together, so that a joint top-m pool replaces the fixed pool (3 Triton + heuristic 16 + learned-ranker 5)?

**Where the time goes** (held-out shapes, v2 + LT16 + ranker top-5; `online_joint_diverse_v2_knn8_rank16.jsonl`). Without the bundle (true first use): Triton rank 0.137 s + Triton candidates 0.455 s + cuBLASLt candidates 0.203 s per shape; the final choice is Triton on 46 of 86 shapes, so an oracle router (skip exactly when the choice is cuBLASLt) would save 0.32 s on average (41%). With the bundle installed in a fresh process and an empty Triton cache (12 held-out shapes, after the one-time 1.4 s initialisation): total 0.17 s = Triton rank 0.031 + Triton candidates 0.027 + cuBLASLt candidates 0.112 s, so an oracle router would save at most ~0.03 s on average. **The router is therefore only worth having when the Triton kernels really compile** (no bundle, or a bundle built for another Triton / GPU); `select_kernel(router="auto")` is on exactly then and off when the bundle serves them.

**Router** (`tune/router.py`, `python -m analysis.router fit|score`). The cuBLASLt candidates are timed first (no compile); a LightGBM classifier then gives P(Triton's best beats the cuBLASLt value by more than the tie margin) from the shape and log(t_cuBLASLt / roofline), and the Triton search is skipped below a threshold. Trained on the 477 tune shapes' full-protocol stage 2 (`results_lock1755_tune477[_graph].jsonl`: Triton screen top-8 vs B3+, 86 positives = 18%), grouped 5-fold out-of-fold **AUC 0.823**. The threshold was fixed by a rule written before any held-out run: the largest P(win) whose out-of-fold mean loss over all tune shapes is ≤ 0.10% (loss on a skipped shape = max(t_cuBLASLt / t_Triton-oracle − 1, 0), the worst case since the real Triton pool is the ranker's top-3): **0.02**, which skips 135 of 477 tune shapes (28%), harms 7 (Triton's oracle faster by more than the margin; worst single loss 11.3%) at 0.09% mean loss.

**Held-out, in three steps.** (a) Simulation from the existing run (`analysis.router score`): 14 of 86 skipped, 0 harmed, cost −15%. (b) The real run (`tune online --router`, session `20261104-202815-e6584c`): 13 skipped, cost 0.796 → 0.699 s (−12%; −0.74 s on a skipped shape), **but the tail got worse**: regret median / p95 / mean 0.4 / 4.7 / 1.2% → 0.5 / 6.7 / 1.7%, shapes > 5% off 4 → 7, 6 shapes worse than the reference by more than the margin and 1 better, and four of those six were shapes the router did NOT skip. (c) The control (`--router-threshold 0`: cuBLASLt timed first, nothing skipped, session `20261104-203117-e3a43b`): same pools, same stage-2 lookups, yet **worse than the Triton-first reference on 5 shapes, better on 1** (regret 0.6 / 6.5 / 1.5%). Three independent runs of the unchanged Triton-first search agree on every one of the 86 picks (0 flips), so the difference is the order: which candidates are short-timed first changes which near-tied candidate wins (e.g. 4809×2130×7527 picks a different cuBLASLt config, 3079 → 3170 µs, with the same family). The simulation cannot see this, which is why it said "no harm".

**Cancelling the order bias.** `search_shape(confirm=n)` / `tune online --confirm n` re-times the n fastest candidates once more in the reverse of their first order and ranks them by the mean of both timings (`confirm_us`; +~0.02 s). Same stage-2 lookups (session `20261104-193409-429243` cold, `…-194744-a4fa21` graph), pools = v2 top-3 + heuristic 16 + ranker top-5:

| run | regret median / p95 / mean | shapes > 5% / > 10% off | slower than B3+ | first-use cost (no bundle) | skipped | vs reference: worse / better |
|---|---|---|---|---|---|---|
| reference: Triton first, no router, no confirm | 0.4% / 4.7% / 1.2% | 4 / 0 | 7 | 0.796 s | 0 | — |
| A: Triton first, no router, confirm 3 (`20261104-203359-8e3774`) | 0.5% / 4.5% / 1.2% | 4 / 0 | 6 | 0.817 s | 0 | 2 / 3 |
| B: cuBLASLt first, never skip, confirm 3 (`…-203539-4908e5`) | 0.6% / 4.7% / 1.4% | 4 / 0 | 7 | 0.828 s | 0 | 3 / 2 |
| **C: router (threshold 0.02), confirm 3** (`…-203721-2961ff`) | **0.3% / 4.6% / 1.1%** | **3 / 0** | **5** | **0.713 s** | 14 | **1 / 3** |

C vs A (the Triton-first search with the same confirmation): worse on 0 shapes, better on 1; C vs B: worse 0, better 3; geometric-mean ratio vs the reference 0.9995. So with the confirmation the router is quality-neutral and saves 0.10 s (−13%) of first-use cost in the no-bundle regime (0.74 s on each skipped shape). The order effect is reduced, not gone (A vs B: 3 worse / 1 better). `confirm` alone does not improve the Triton-first search (A vs the reference: 2 / 3, ratio 0.9999), so it is used only with the router (`tune.online.ROUTER_CONFIRM = 3`).

**One score across families does not work** (`analysis.joint_rank`, simulation on the held-out online run, nothing tuned on held-out; the pool sizes are a printed curve). Score of a candidate = A_family(shape) + the family ranker's within-family predicted log regret, with A_f = log(best time of the family / roofline) from a small LightGBM on the tune shapes: each A_f is accurate (out-of-fold R² 0.940 Triton, 0.957 cuBLASLt, MAE 0.23 / 0.17 in log time) but the comparison that matters is not: A_Triton − A_cuBLASLt agrees in sign with the truth on only 78% of tune shapes (correlation 0.46), because a 17-23% absolute error swamps the few-percent differences between the two families' best times. Joint top-m pools (cost without bundle; regret median / p95 / mean; shapes > 5% / > 10%): top-6 0.60 s, 1.1% / 54.7% / 7.5%, 17 / 8; top-10 0.74 s, 0.7% / 9.1% / 2.2%, 7 / 3; top-16 0.88 s, 0.6% / 8.1% / 2.0%, 6 / 3; top-24 1.12 s, 0.5% / 4.5% / 1.2%, 4 / 1, against the fixed pool 0.80 s, 0.4% / 4.7% / 1.2%, 4 / 0. Adding the heuristic's first candidate to a top-10 pool helps (0.7% / 6.3% / 1.7%) but does not reach the fixed pool. The router succeeds where this fails because it uses a **measured** cuBLASLt time rather than a prediction of both. Not shipped; kept as `analysis/joint_rank.py`.

**Trimming the cuBLASLt pool** (exploratory, same simulation; not adopted). The heuristic list is still needed once the ranker is there: heuristic top-h + ranker top-5 gives regret p95 4.7% (h = 16), 5.5% (8), 6.6% (4), 8.5% (1), 17.6% (0), for an LT-side cost of 0.203 s down to 0.132 s.

**Shipped.** `select_kernel(router="auto")` (on when the bundle does not serve the Triton kernels; `True` / `False` force it), with `confirm = 3` whenever the router is active; the model is `data/<hw>/router.*` or the committed snapshot `reports/snapshot/lock1755/router.*` (150 KB, tagged with the hw_id). The skip threshold is stored in the model's json. `tune online --router [--router-threshold X] [--confirm N]` reproduces the runs. Tests: 190 pass. The winner cache keeps whatever the search picked, so a skipped shape's cached winner is the cuBLASLt one.

**Limits.** Held-out evidence is 14 skipped shapes (none worse than the same-confirmation Triton-first search); the classifier has AUC 0.82 and was trained with B3+ as the cuBLASLt value while at run time it sees the short-timed pool best, a few percent higher. One GPU, one lock state, `harness_dirty` true on all rows, 34-47% power-capped stage-2 samples. The benefit exists only in the no-bundle regime. The order bias is a property of the short-timing pass in general (any reordering of the candidates can flip near-ties), not just of the router.

**Reproduce.**
```bash
python -m analysis.router fit --save $D/router && python -m analysis.router score     # CPU; copy router.* to reports/snapshot/lock1755/
for v in "" "--router --router-threshold 0" "--router"; do python -m tune --clock-lock 1755 online --shapes-file $D/sweep_screen_heldout/subset.json \
  --ranker $D/lgbm_rank_v2 --k 5 --order diverse --lt-n 16 --lt-rank 5 --confirm 3 --fresh-cache $v --out-name online_joint_diverse_v2_rank5_<name>.jsonl; done
python -m analysis.joint_rank                                                           # CPU; the negative result
```
(The picks of every run are looked up in the stage-2 session of the previous entry; no new stage 2 was needed because the pools are subsets of the candidates it timed.)

## Update 2026-11-04 (later): a learned ranker over all valid cuBLASLt configs beats the k-NN proposals — joint search regret p95 11.6% → 4.7%, 7 shapes slower than B3+ (was 17)

*Audit correction (entry above): against an oracle that includes the proposals' own timings the held-out row for the ranker top-5 reads 0.5% / 5.5% / 1.4% (not 0.4 / 4.7 / 1.2) and the heuristic row 0.7 / 11.7 / 2.8. The win over the heuristic holds; the ranker's edge over k-NN8 is not significant (mean −0.35 [−0.82, +0.09] points, p95 −1.2 [−3.4, +1.3]).*

**What ran.** (1) A full per-config screen of the 652 tune-split shapes (`python -m baselines.cublaslt_screen`, session log `data/<hw>/logs/lt_screen.log`, 4,266 s, 1,291,554 (shape, config) rows, 624 of 652 shapes screened completely within the 40 s budget; hot-L2 screen timing, 1.0 µs grid, clock lock 1755 declared). (2) A LightGBM regressor of log(t / t_best_of_shape) per (shape, config) on those rows (`python -m analysis.lt_rank`; features: shape, config parameters with algo / tile / stages as categoricals, CTA count / padding / waves from the tile dimensions in both orientations, K per split; near-top rows weighted 3x; 300 rounds, no tuning). (3) The held-out run: the v2 Triton ranker's top-3 + the heuristic top-16 + the k-NN proposals (8 x 2) + the learned ranker's top-16 in one short-timed pool (`online_joint_diverse_v2_knn8_rank16.jsonl`, session `20261104-193204-66b37b`, 86 shapes), then ONE stage-2 session that times every candidate: cold `results_lock1755_rank16.jsonl` `20261104-193409-429243` (86 shapes, 4,828 rows incl. 1,376 ranker rows, 0 invalid, 7 not converged, 34% power-capped) and graph `results_lock1755_rank16_graph.jsonl` `20261104-194744-a4fa21` (56 shapes < 50 µs, 0 invalid, 46% power-capped). `harness_dirty` is true on every row (uncommitted changes), clock locked at 1755, driver 595.99.02.

**Tune-only grouped cross-validation, exact times** (`analysis.lt_rank`, 5 folds over shapes grouped by "shares two of M, N, K", no held-out shape read; pool = the best heuristic-seeded config of that shape's own search + the method's proposals; 250 of 652 shapes improve on the seeds by > 5%; "captured" = share of the gap between the best heuristic-seeded config and the screen winner that the pool closes):

| method | pool | captured | closes fully (≤ 2% of screen best) | mean regret vs screen best | p95 |
|---|---|---|---|---|---|
| heuristic seeds only | 0 | 0% | 0% | 6.5% | 24.8% |
| k-NN 5 x 2 | 7.8 | 63.7% | 46% | 2.2% | 11.4% |
| **k-NN 8 x 2** (shipped before) | 11.3 | 70.7% | 55% | 1.8% | 10.0% |
| ranker top-5 | 5 | 69.4% | 49% | 1.8% | 8.5% |
| **ranker top-10** | 10 | **78.3%** | 61% | 1.3% | 7.3% |
| ranker top-16 | 16 | 85.5% | 72% | 0.9% | 4.8% |
| k-NN 8 x 2 + ranker top-10 | 19.2 | 89.2% | 78% | 0.7% | 3.7% |

The k-NN row reproduces the earlier leave-one-out scan (70.7% vs 71.4%). The screen's best config is in the ranker's top 10 for 47% of shapes (median rank 13 of ~1,980 valid configs); by gain the model uses the algorithm, tile, the CTA counts and K per split. At equal pool size the ranker closes more of the gap than the neighbours (+7.6 points at ~10 candidates), and the ranker's top-5 matches k-NN 8 x 2 with half the candidates. **The held-out comparison and the primary pools (ranker top-10 vs k-NN8 at matched size, and the union) were fixed from this table before the held-out run; no hyperparameter was tuned.**

**Held-out A/B** (`analysis.lt_rank_ab`; 86 shapes, 56 timed graph-amortized; every pool = v2 top-3 Triton + heuristic top-16 + the named proposals, re-chosen with `tune.online.choose` from the one run's candidates, each pick looked up at its full-protocol time in the one stage-2 session; regret vs the joint oracle = best Triton of the session and B3+):

| pool | regret median / p95 / mean | ≥ 50 µs p95 | < 50 µs p95 | shapes > 5% / > 10% off | vs B3+ faster / slower | first-use cost | picks proposal |
|---|---|---|---|---|---|---|---|
| heuristic top-16 only | 0.6% / 11.6% / 2.6% | 13.7% | 10.9% | 12 / 7 | 22 / 17 | 0.66 s | none |
| + k-NN8 | 0.6% / 6.8% / 1.6% | 5.2% | 7.2% | 9 / 0 | 23 / 13 | 0.77 s | 22% |
| + ranker top-5 | **0.4% / 4.7% / 1.2%** | 4.8% | 4.6% | **4 / 0** | 23 / **7** | 0.80 s | 23% |
| + ranker top-10 | 0.4% / 4.7% / 1.2% | 4.8% | 4.6% | 4 / 1 | 23 / 7 | 0.85 s | 23% |
| + ranker top-16 | 0.4% / 4.7% / 1.2% | 4.8% | 4.6% | 4 / 1 | 23 / 7 | 0.92 s | 23% |
| + k-NN8 + ranker top-5 | 0.5% / 4.7% / 1.1% | **3.7%** | 5.2% | 4 / 0 | 24 / 8 | 0.90 s | 15% k-NN, 14% ranker |
| + k-NN8 + ranker top-10 | 0.5% / 4.7% / 1.1% | 3.7% | 5.2% | 4 / 0 | 24 / 8 | 0.96 s | same |

Paired geometric-mean time ratio (a / b, 95% bootstrap interval, shapes a faster / slower by the tie margin): ranker top-5 vs heuristic only **0.987 [0.978, 0.994]**, 11 / 1; k-NN8 vs heuristic only 0.990 [0.982, 0.997], 9 / 1; **ranker top-10 vs k-NN8 0.9966 [0.9925, 1.0007], 8 / 4 (not significant)**; k-NN8 + ranker top-10 vs k-NN8 0.9956 [0.9922, 0.9985], 6 / 1. The proposers' ceilings (best proposed config, full protocol): ranker top-16 is within the tie margin of B3+ on 58 of 86 shapes, beats it by more than the margin on 2, and closes the B3+ gap fully on 21 of the 26 shapes where B3+ beats the heuristic (≥ half way on 22); k-NN8 closes 12 (16). Top-5, 10 and 16 give the same online choice (the short timing never picks a rank 6-16 config), so the top-5 is enough. Remaining misses: 68×6420×10359 (10.1% off the oracle), 226×2141×1139 (8.2%), 1316×272×100 (6.8%, a 5.7 µs shape where the best is Triton and B3+ is 1.09x), 2048×4096×11008 (5.7%).

**Reading.** The learned ranker recovers about 64% of the mean gap and 75% of the p95 gap between the heuristic-only joint search and the oracle ceiling (0.4% / 2.4% mean / p95), against 45% / 52% for the k-NN8 proposals; the shapes slower than B3+ drop from 17 to 7. On the paired test it is **not** significantly faster than the k-NN8 pool (the interval touches 1), only clearly better on the tail counts (shapes > 5% off: 4 vs 9, p95 4.7% vs 6.8%), so the evidence for choosing it over k-NN is the tune cross-validation plus the tail, not a held-out mean difference. Cost: +0.14 s at top-5 (0.80 s vs 0.66 s), of which the enumeration of the valid configs (thousands of `cublasLtMatmulAlgoCheck` calls, ≈0.1-0.2 s) is the bulk and the model prediction a few ms; k-NN8 costs +0.11 s. G4 (differentiation vs B3+) is unchanged: the joint search is within the margin of B3+ on most shapes and faster on 23, but nothing is faster than B3+ by a margin that beats the deep search offline.

**Shipped.** `tune.online.select_kernel(lt_rank=5)` is now the default cuBLASLt proposal source; the k-NN proposals (`lt_knn`, 8 neighbours) are used instead only when the ranker is off or has no model for this GPU (automatic, `lt_knn=None`); `lt_knn=8, lt_rank=0` or the reverse force either. The model (1.8 MB) is committed as `reports/snapshot/lock1755/lt_rank.txt/.json` (`data/<hw>/lt_rank` wins if present); it is tagged with `hw_id` and loads only on that GPU (the CTA features also hard-code 68 SMs, so the 3090's 82 SMs need a retrained model, as the cuBLASLt tile and algorithm ids already require). The winner cache stores `lt_rank` next to `lt_knn` and treats an entry from a narrower search as a miss. The tile-id table (ids 1-33) was checked against `cublasLt.h`; ids above 33 do not occur in the screen. Tests: 187 pass.

**Limits.** One GPU; the held-out region is out of distribution for the tune shapes (k-NN8 closes the gap fully on 55% of tune-CV shapes but on 12 of the 26 held-out shapes where B3+ beats the heuristic), so the ranker's held-out advantage over k-NN8 has to be read with the paired test above. The screen labels are 3-launch hot-L2 timings with a 1.0 µs grid, so below ~10 µs the labels, and the 5% tie margin, are close to noise; 28 of the 652 tune shapes hit the 40 s budget before every config was screened. The runtime needs `lightgbm` (pinned) and the config enumeration. A new GPU needs the full screen (~70 min on the 3080) and `python -m analysis.lt_rank --save`.

**Next.** A cost-only router (skip the Triton compiles when the ranker's best cuBLASLt prediction is clearly ahead; without the bundle the compiles are about two thirds of the first-use cost); a ranker that scores Triton and cuBLASLt configs in one model (the families' regrets are now on one scale); the 3090 with its own screen and snapshot.

**Reproduce.**
```bash
TUNE_CLOCK_LOCK=1755 python -m baselines.cublaslt_screen --budget-s 40         # ~70 min, GPU idle
python -m analysis.lt_rank --folds 5 --save $D/lt_rank                         # CPU, ~1 min; copy lt_rank.txt/.json to reports/snapshot/lock1755/
python -m tune --clock-lock 1755 online --shapes-file $D/sweep_screen_heldout/subset.json --ranker $D/lgbm_rank_v2 --k 5 \
  --order diverse --lt-n 16 --lt-knn 8 --lt-rank 16 --fresh-cache --out-name online_joint_diverse_v2_knn8_rank16.jsonl
# stage 2 (cold, then the subset_lt50us.json graph run with --cache-modes hot --graph-batch 50), as in the entry below with
#   --backends torch,triton:sweep:8,triton:picks,cublaslt:16,cublaslt_deep:20,cublaslt_knn:8,cublaslt_rank:16
python -m analysis.lt_rank_ab
```

## Update 2026-11-04: cuBLASLt k-NN proposals — the deep-search winners of the nearest tune shapes close about half of the joint search's gap to B3+

**Why.** The joint search (Triton top-3 + cuBLASLt heuristic top-16) loses to B3+ only where the 20 s deep search finds a cuBLASLt config the heuristic never lists. Ceiling (`analysis.lt_ceiling`, v2 top-3 + LT16, 86 held-out shapes): the oracle winner is the heuristic list on 31 shapes, Triton on 35, a deep-search-only cuBLASLt config on **20**; adding the best deep config to the pool would take regret from 0.4% / 10.4% / 2.7% (median / p95 / mean) to 0.0% / 2.4% / 0.4%, perfect Triton only to 0.0% / 10.4% / 2.1%. Of the 22 shapes with regret over the tie margin, 21 are fixed by the deep config, 2 by perfect Triton. So the headroom is on the cuBLASLt side.

**Structure of the winners** (`analysis.lt_clusters`, 565 cached searches; 183 of 464 tune and 30 of 86 held-out shapes improve by > 5% over the best heuristic-seeded config). Algorithms 24 / 21 / 35 cover 71% of the 213 deep-needed winners and split-K > 1 is 42%, but the full config has 116 distinct values (top 5 cover 45): the winners cluster by regime (small M·N and skinny: GEMV-like algorithms 0 / 1 / 13; large K and skinny + large K: split-K with 21 / 24 / 35; regular: 35 / 21) without being a small set of recurring answers. Static portfolios fail (the 3 most frequent winners are in 30% of held-out shapes' top-8); shape-conditioned proposals work (below).

**What was built.** `baselines/cublaslt_knn.py`: for a shape, the top 2 configs of each of its k nearest tune shapes (features log2 M, N, K, 128x128 CTA fill on 68 SMs, 8-alignment, standardized on the tune shapes), rebuilt with `build_algo` (`cublasLtMatmulAlgoCheck` rejects what is invalid), no compile. Table = TUNE-split `lt_search/` caches only (478 shapes), never the shape itself; a test asserts the tune and held-out shape sets are disjoint. Wiring: `tune online --lt-knn N`, backend spec `cublaslt_knn[:k]`, family `cublaslt_knn` (key prefix `n`; counts toward no `lt_n`, so `choice_lt["<k>_lt16"]` is the heuristic-only search and `choice` the full pool), `select_kernel(lt_knn=8)` (default), winner-cache `lt_knn` field. Committed snapshot of the table: `reports/snapshot/lock1755/lt_knn_table.json` (84 KiB, GPU-tagged; other GPUs get no proposals and one warning).

**Choosing (k, per-neighbour) on tune shapes only** (`analysis.lt_knn_scan`, grouped leave-one-out: no neighbour sharing two of M, N, K; 188 of 478 tune shapes improve by > 5%; a proposal outside the shape's stored top-8 counts as no gain, so these are lower bounds). captured = share of the gap between the best heuristic-seeded config and the search winner that the pool closes.

| k, per | pool | captured (mean) | closes fully | mean regret vs search winner | p95 |
|---|---|---|---|---|---|
| heuristic seeds only | 0 | 0% | 0% | 6.3% | 23.3% |
| 3, 2 | 4.9 | 55.8% | 51% | 2.8% | 15.5% |
| 5, 2 | 7.3 | 64.8% | 61% | 2.3% | 14.0% |
| 10, 1 | 7.0 | 66.2% | 61% | 2.3% | 14.4% |
| **8, 2** | 10.6 | **71.4%** | 67% | 1.9% | 12.5% |
| 15, 1 | 9.9 | 72.4% | 68% | 1.8% | 11.1% |
| 10, 2 | 12.5 | 75.1% | 71% | 1.7% | 9.9% |
| 15, 3 | 24.4 | 85.3% | 81% | 1.0% | 5.3% |

Gains per added candidate fall off (31 points for the first neighbour, 9 for 3 → 5, 6.6 for 5 → 8, 3.7 for 8 → 10) but have no plateau by k = 20. Split-half selection over the grid with a pool ≤ 12 picked (15, 1) 119 times of 200, (8, 2) 71, never (5, 2). Ungrouped leave-one-out inflates capture by ~10 points. **(8, 2) was fixed from this table before its held-out run**; the first held-out run used (5, 2), whose k had been picked after looking at held-out hit rates (`analysis.lt_clusters` prints them), so that run is not an out-of-sample number and is kept only as a cross-check.

**Held-out A/B** (`analysis.lt_knn_ab`; 86 held-out shapes, 56 timed graph-amortized; v2 ranker, diversified top-3 + LT16 vs + proposals, both looked up at their full-protocol time in ONE stage-2 session: cold `results_lock1755_knn8.jsonl` `20261104-141749-c8bb44`, 3,452 rows, 0 invalid, 14 not converged, 40% power-capped; graph `results_lock1755_knn8_graph.jsonl` `20261104-142752-a96cd4`, 2,279 rows, 0 invalid, 48% power-capped; online `online_joint_diverse_v2_knn8.jsonl` `20261104-141605-1df6e7`. `harness_dirty` is true on every row (changes uncommitted), clock locked at 1755):

| v2 top-3 + LT16 | heuristic only | + kNN8 proposals |
|---|---|---|
| regret vs joint oracle, median / p95 / mean | 0.9% / 10.7% / 2.6% | **0.4% / 6.0% / 1.5%** |
| shapes > 5% / > 10% off | 12 / 6 | **8 / 0** |
| ≥ 50 µs (n = 30), median / p95 / mean | 1.7% / 13.5% / 3.1% | 0.5% / 4.4% / 1.4% |
| < 50 µs (n = 56), median / p95 / mean | 0.4% / 10.3% / 2.4% | 0.4% / 7.6% / 1.6% |
| vs B3+ (median; faster / slower by the margin) | 1.000; 22 / 19 | 1.000; 22 / 15 |
| first-use cost | 0.66 s | 0.77 s |
| picks a proposal | none | 20% of shapes |

Paired (proposals / heuristic only, chosen config's stage-2 time): geometric-mean ratio **0.990 [0.982, 0.997]**, faster on 7 shapes, slower on 0; top-5 Triton gives the same (0.5% / 10.7% / 2.5% → 0.3% / 6.0% / 1.4%, ratio 0.990 [0.983, 0.997]). Biggest changes: 51×324×5709 17.7 → 14.9 µs, 68×6420×10359 412 → 347 µs, 443×4723×393 51.3 → 44.0 µs, 380×6978×216 40.0 → 34.8 µs. The proposals alone (`cublaslt_knn:8`, all timed in full): never faster than B3+ by more than the margin, within it on 44 shapes; on the 27 shapes where B3+ beats the heuristic by more than the margin they close the gap fully on 13 and at least halfway on 18. The earlier (5, 2) run (session `20261104-120725-8651eb`, 0.7% / 10.5% / 2.6% → 0.4% / 7.1% / 1.6%, ratio 0.991 [0.983, 0.997], 8 faster / 1 slower, one 6% loss on 512×4096×11008) is in a different session; the baseline column moved by ~0.2 points between the sessions, so (8, 2) vs (5, 2) is "at least as good", not a measured difference.

**Gates and limits.** G4 (differentiation vs B3+ at ≥ 50 µs) is unchanged: no pick beats B3+ by more than the margin and the joint search is still slower than B3+ on 15 shapes (was 19). About half of the ceiling's gap is recovered (mean 2.6% → 1.5% vs the 0.4% ceiling; p95 10.7% → 6.0% vs 2.4%); 8 shapes remain > 5% off, mostly split-K magnitude differing from every neighbour's winner. One GPU: the table is 3080 / sm_86 data and cuBLASLt ids are per architecture; the 3090 and 2060 need their own searches. Proposals carry no compile, so the cost is validation + short timing (+0.10 s for ~6 extra candidates). The screen timings behind the table are hot-L2, 3 launches, 1.0 µs grid, and 39 of 565 searches hit the 20 s budget before screening every config.

**Next.** A LightGBM ranker over (shape, cuBLASLt config) trained on a full per-config screen of the tune shapes (`baselines/cublaslt_screen.py`, which keeps every config's time; the stored top-8 cannot train a ranker or score a proposer exactly); the k-NN result above is the bar.

**Reproduce.**
```bash
D=data/rtx3080_sm86_68sms_10gb
python -m analysis.lt_ceiling && python -m analysis.lt_clusters && python -m analysis.lt_knn_scan     # CPU only
python -m baselines.cublaslt_knn --snapshot                                                          # reports/snapshot/lock1755/lt_knn_table.json
python -m tune --clock-lock 1755 online --shapes-file $D/sweep_screen_heldout/subset.json --ranker $D/lgbm_rank_v2 --k 5 \
  --order diverse --lt-n 16 --lt-knn 8 --fresh-cache --out-name online_joint_diverse_v2_knn8.jsonl
export TUNE_SWEEP_DIR=$D/sweep_screen_heldout TUNE_PICKS=$D/v2_stage2_picks.json
python -m tune --clock-lock 1755 benchmark --shapes-file $D/sweep_screen_heldout/subset.json \
  --backends torch,triton:sweep:8,triton:picks,cublaslt:16,cublaslt_deep:20,cublaslt_knn:8 --cache-modes cold --out-name results_lock1755_knn8.jsonl
python -m tune --clock-lock 1755 benchmark --shapes-file $D/sweep_screen_heldout/subset_lt50us.json \
  --backends torch,triton:sweep:8,triton:picks,cublaslt:16,cublaslt_deep:20,cublaslt_knn:8 --cache-modes hot --graph-batch 50 --out-name results_lock1755_knn8_graph.jsonl
python -m analysis.lt_knn_ab
```

## Update 2026-11-03: zero-shot arm on the RTX 3090 — the 3080 rankers transfer to a second tensor-core GPU (v2 within noise of the 3080), unlike the 2060

**What ran.** The 3080's `lgbm_rank` (v1) and `lgbm_rank_v2`, no refit, on an RTX 3090 (`rtx3090_sm86_82sms_24gb`, Ampere sm_86, 82 SMs, 24 GB, SLURM job from `remoteGPU/zero_shot.sh`, `PER_SUITE=12`), `data/rtx3090_sm86_82sms_24gb/`. Harness `194f78b`, clean.
- Oracle: the 3090's own held-out screen, 27 of 27 shapes complete (the same 27 as the 2060 run, a subset of the 3080's 86). Four row files (v1 / v2 × `as_is` / `native`), 27 shapes each, k = 5, cuBLASLt top-16 in the pool, 0 skipped, **0 invalid candidates**. Sessions: run `20261102-200011-fd8e1e` (v1), `20261102-200129-daabf1` (v2); stage 2 (`zero_shot picks` + benchmark) cold `20261102-200238-ac2917` (962 rows), graph `20261102-200849-5ce748` (429 rows, the 12 shapes under 50 µs).
- **Not standard conditions.** The clock is **unlocked** (a job cannot run `nvidia-smi -lgc`), driver 590.48.01 (the study: 595.99.02), 95% of cold and 53% of graph stage-2 rows power-capped (throttle `0x4`). No noise study exists for this setup, so the 2% / 5% tie margins are borrowed from the 3080. Compare regret and ratios, never absolute times, with the 3080 / 2060 runs. The oracle was also screened unlocked: regrets of 1-3% are at its noise level.
- All 285 compiled kernels in the run's Triton cache contain `mma.sync` (0 FMA-only; the 2060's 571 are all FMA-only): the confound of the 2060 update is absent here.

**Zero-shot vs home GPU vs 2060** (27 shared shapes, regret vs each GPU's own screen oracle, median / p95 (mean); `analysis.cross_gpu`, full table in [`cross_gpu.md`](cross_gpu.md)):

| pick | 3080 v1 | 3080 v2 | 3090 v1 as_is | 3090 v1 native | 3090 v2 as_is | 3090 v2 native | 2060 retrained | 2060 v1 native | 2060 v2 native |
|---|---|---|---|---|---|---|---|---|---|
| online top-1 | 1.9 / 12.5 (3.6) | 0.9 / 7.2 (2.4) | 6.0 / 23.1 (8.1) | 4.6 / 13.0 (5.0) | 1.7 / 18.9 (5.7) | 2.8 / 14.6 (4.9) | 0.7 / 9.1 (2.6) | 94 / 673 (303) | 203 / 2782 (471) |
| online top-3 | 1.5 / 7.6 (2.6) | 0.8 / 6.9 (1.8) | 5.4 / 14.5 (6.1) | 3.7 / 13.0 (4.5) | 0.9 / 14.1 (4.0) | 1.8 / 9.0 (3.1) | 0.7 / 7.5 (2.5) | 43 / 604 (157) | 43 / 228 (81) |
| online top-5 | 0.8 / 7.2 (2.3) | 0.8 / 6.9 (1.9) | 5.4 / 14.5 (6.1) | 2.8 / 13.0 (4.1) | 0.9 / 13.0 (3.6) | 1.8 / 12.4 (3.5) | 0.5 / 6.7 (2.1) | 19 / 598 (111) | 15 / 228 (72) |
| top-3 picks > 50% off (of 27) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 13 | 13 |
| shapes over 5% regret (top-3) | 6 | 2 | 14 | 12 | 10 | 7 | 6 | 22 | 22 |

Paired bootstrap vs the 3080's v1 (online top-3, mean difference in points, 5,000 resamples): 3090 v1 `as_is` **+3.5 [+1.1, +6.4]**, v1 `native` +1.8 [0.0, +3.8], v2 `as_is` +1.4 [-0.7, +3.6], v2 `native` +0.5 [-1.1, +2.1]. Regime split on the 3090 oracle: 15 shapes ≥ 50 µs / 12 below (3080: 16 / 11).

**Stage 2 vs the searched cuBLASLt (B3+)** (speed = t_B3+ / t_pick, > 1 faster; all 27 shapes timed in one session, so unlike the 2060 the Triton-only picks are fully covered):
- Joint search (top-3 Triton + cuBLASLt top-16): median 0.986-1.000, p10 0.91-0.95, within 5% of B3+ on 21-24 of 27 shapes (3080: 1.005 / 0.93-0.94 / 22; 2060 zero-shot: 0.98 / 0.77 / 16-17 of 25-26). cuBLASLt is chosen on 5-9 of 27 shapes (3080: 8-9; 2060: 24). Versus B2 (`ranker_compare.md`): median 0.944-0.958 (4-6% faster), faster on 12-17 of 27 and slower on 0-1 (> margin); first-use cost 0.64-1.13 s for top-3 + LT16.
- Triton-only top-3: median 0.977-0.998; ≥ 50 µs 0.970-0.984 (3080 0.967-0.985), < 50 µs 1.04-1.07 (3080 1.15). Family ceiling (best Triton row of the session): median 1.008 [0.892, 1.171], 24 of 27 within 90% of B3+ (3080: 1.007, 25 of 27; 2060: 0.44, 3 of 27).

**Findings.**
- **Zero-shot transfer works across a same-architecture, different-size GPU.** No pick is > 50% off (0 of 27 in all four variants; the 2060: 13), and v2 `as_is` / `native` is statistically indistinguishable from the 3080's own v1 on the mean (+1.4 [-0.7, +3.6] and +0.5 [-1.1, +2.1] points). v1 `as_is` is measurably worse (+3.5 [+1.1, +6.4]). Combined with the 2060 update, the 2060 failure is now attributable to the missing tensor-core path in Triton on sm_75, not to a general inability of the ranker to cross GPUs.
- **v2 is the more robust model** (analytic features): lower mean than v1 in both variants (4.0 / 3.1% vs 6.1 / 4.5%); `native` helps v1 (median 5.4 → 3.7%) and the v2 tail (p95 14.1 → 9.0%) but not v2's median (0.9 → 1.8%, within noise at n = 27).
- **More candidates do not help much** (v1 `as_is` online-5 = online-3): the remaining error is ranker bias on this GPU, not too few candidates.
- **Joint selection** behaves as on the 3080: ties B3+ at the median and is never slower than B2 beyond the margin on more than 1 shape. G4 (differentiation vs B3+ at ≥ 50 µs) is unchanged: it still fails.

**Caveats.** n = 27 and variant differences (e.g. v2 `as_is` vs `native`) are inside their intervals. **No retrained 3090 ranker**, so the zero-shot gap cannot be separated from the unlocked-clock noise floor; a retrain would need a tune sweep (~10-17 h) on this GPU. Unlocked clock, other driver, power cap (above). The `RTX 3080 reference` table inside the generated `reports/zero_shot_rtx3090_sm86_82sms_24gb/k*/` files uses the 86-shape constants: do not compare it to these 27-shape numbers (use `cross_gpu.md`). 3090 regime mix (15 / 12) differs from the 3080's, so the < 50 µs comparison rests on 11-12 shapes per GPU.

**Gates.** None changed. **Next steps.** (1) Optionally a locked-clock 3090 re-run of the stage 2 (or `--clock-lock` at ~1500-1600 MHz if the host allows) with a short noise study, to turn the borrowed 2% margin into a measured one. (2) Retrain on the 3090 only if the 5%-regret rate (7-14 of 27) matters: the 3080 ranker is 2-6 of 27. (3) An Ada / Hopper GPU would test a different tensor-core generation.

Reproduce (CPU only; the run is `sbatch remoteGPU/zero_shot.sh`):
```bash
D9=data/rtx3090_sm86_82sms_24gb; R=reports/zero_shot_rtx3090_sm86_82sms_24gb
python -m analysis.zero_shot score --data $D9 --report $R/k3 --k 3
python -m analysis.zero_shot score --data $D9 --report $R/k5 --k 5
python -m analysis.ranker_compare --data $D9 --report $R --results results_zero_shot.jsonl --graph-results results_zero_shot_graph.jsonl \
  --online v1_as_is=zero_shot_v1_as_is.jsonl v1_native=zero_shot_v1_native.jsonl v2_as_is=zero_shot_v2_as_is.jsonl v2_native=zero_shot_v2_native.jsonl
D3=data/rtx3080_sm86_68sms_10gb; Z=data/rtx2060/quick-20261029-095746-v3/data
python -m analysis.cross_gpu \
  --side 3080-v1 $D3 online_lgbm_rank_diverse.jsonl online_joint_diverse.jsonl results_lock1755_v2.jsonl results_lock1755_v2_graph.jsonl \
  --side 3080-v2 $D3 online_joint_diverse_v2.jsonl online_joint_diverse_v2.jsonl results_lock1755_v2.jsonl results_lock1755_v2_graph.jsonl \
  --side 2060-retrained $Z online_lgbm_rank_diverse.jsonl online_lgbm_rank_diverse.jsonl results_validate.jsonl results_validate_graph.jsonl \
  --side 2060-zs-v1-as_is $Z zero_shot_v1_as_is.jsonl zero_shot_v1_as_is.jsonl results_validate.jsonl results_validate_graph.jsonl \
  --side 2060-zs-v1-native $Z zero_shot_v1_native.jsonl zero_shot_v1_native.jsonl results_validate.jsonl results_validate_graph.jsonl \
  --side 2060-zs-v2-as_is $Z zero_shot_v2_as_is.jsonl zero_shot_v2_as_is.jsonl results_validate.jsonl results_validate_graph.jsonl \
  --side 2060-zs-v2-native $Z zero_shot_v2_native.jsonl zero_shot_v2_native.jsonl results_validate.jsonl results_validate_graph.jsonl \
  --side 3090-zs-v1-as_is $D9 zero_shot_v1_as_is.jsonl zero_shot_v1_as_is.jsonl results_zero_shot.jsonl results_zero_shot_graph.jsonl \
  --side 3090-zs-v1-native $D9 zero_shot_v1_native.jsonl zero_shot_v1_native.jsonl results_zero_shot.jsonl results_zero_shot_graph.jsonl \
  --side 3090-zs-v2-as_is $D9 zero_shot_v2_as_is.jsonl zero_shot_v2_as_is.jsonl results_zero_shot.jsonl results_zero_shot_graph.jsonl \
  --side 3090-zs-v2-native $D9 zero_shot_v2_native.jsonl zero_shot_v2_native.jsonl results_zero_shot.jsonl results_zero_shot_graph.jsonl
```
`env.json` of the run: `reports/snapshot/rtx3090_unlocked/env.json`.

## Update 2026-11-02 (later): zero-shot arm on the 2060 — the 3080 rankers do not transfer for Triton-only selection; a scorer bug had hidden it

**What ran.** The 3080's `lgbm_rank` (v1) and `lgbm_rank_v2`, no refit, on the RTX 2060 (`rtx2060_sm75_30sms_6gb`, `--clock-lock 1755`, driver 595.99.02), run directory `data/rtx2060/quick-20261029-095746-v3/` (v2 plus the zero-shot rows; its `results_validate.jsonl` and retrained `online_lgbm_rank_diverse.jsonl` are byte-identical to v2's). `python -m analysis.zero_shot run --ranker v1=... v2=... --k 5 --lt-n 16 --fresh-cache`, variants `as_is` / `native`.
- Four row files, 27 shapes each (= the 2060 held-out set, 27 of the 3080's 86), k = 5, cuBLASLt top-16 in the pool, 0 skipped, **0 invalid candidates**. Sessions `20261102-113735-d051df` (v1 as_is + native), `20261102-115635-e7d450` (v2 as_is), `20261102-124121-809c54` (v2 native); lock 1755, clean. Harness `60aa667` for three files and `bed0914` for v2 native: the only code difference that matters is `4319453` (v2 `native` space lookup for pool configs the 2060 cannot hold), which v1 and v2 `as_is` never reach.
- The models in `src3080/` are byte-identical (md5) to the 3080's `lgbm_rank.txt` / `lgbm_rank_v2.txt` and differ from the 2060's own retrained model. Oracle: the retrain run's held-out screen (27 shapes); the 17.7 h sweep was not repeated.

**Scorer bug found and fixed.** `analysis.zero_shot.load_screen` kept only `ok` screen rows. The screen cuts a config as `slow` after one sample once it is > 1.5× the best so far, so every pick with regret > 50% was dropped as "unscored", and the generated table looked several times better than it was (medians 6–16% instead of 43–80% at top-3): 12–16 of 27 chosen configs per variant were `slow`. `load_screen` now keeps `slow` rows (their one-sample time is a real measurement and proves regret > 50%), `summarize` counts `over_50`, the report has a `regret > 50%` column, and three tests cover it (`tests/test_zero_shot.py`). **No earlier number changes**: in the 3080 runs and the retrained 2060 run all 27 picks are `ok` (checked pick by pick), and the `REFERENCE_3080` constants still recompute exactly. An independent script reproduced the corrected numbers.

**Zero-shot vs retrained** (27 shared shapes, regret vs the 2060's own screen oracle, median / p95 (mean); `analysis.cross_gpu`, full table in [`cross_gpu.md`](cross_gpu.md)):

| pick | 3080 v1 / v2 | 2060 retrained | v1 as_is | v1 native | v2 as_is | v2 native |
|---|---|---|---|---|---|---|
| online top-1 | 1.9 / 12.5 (3.6) / 0.9 / 7.2 (2.4) | 0.7 / 9.1 (2.6) | 198 / 6169 (810) | 94 / 673 (303) | 227 / 3600 (806) | 203 / 2782 (471) |
| online top-3 | 1.5 / 7.6 (2.6) / 0.8 / 6.9 (1.8) | 0.7 / 7.5 (2.5) | 80 / 658 (288) | 43 / 604 (157) | 58 / 636 (270) | **43 / 228 (81)** |
| online top-5 | 0.8 / 7.2 (2.3) / 0.8 / 6.9 (1.9) | 0.5 / 6.7 (2.1) | 39 / 658 (269) | 19 / 598 (111) | 38 / 392 (108) | 15 / 228 (72) |
| top-3 picks with regret > 50% | 0 / 0 of 27 | 0 of 27 | 16 of 26 (+1 absent) | 13 of 27 | 13 of 26 (+1 absent) | 13 of 27 |

Paired bootstrap vs the 3080's v1 (online top-3): zero-shot is worse on the mean by +78 [+37, +137] points (v2 native) to +286 [+84, +624] (v1 as_is); the retrained ranker is within −0.1 [−1.7, +1.6]. By regime (online top-3 median): ≥ 50 µs (n = 20–21) 88–94% for v1 and v2 as_is and v1 native, 54% for v2 native; < 50 µs (n = 5–6) 6.8–14.5%. Even the best of the 5 candidates (by screen time) is still > 50% off on 8 (v1 native) and 10 (v2 native) shapes. `as_is` → `native` helps both models (medians 80 → 43 and 58 → 43); v2 `native` has the best tail (p95 228% vs ~600%). `absent` = pick not in the 2060 screen at all (pruned there but valid when run).

**Why (partly observed).** The 2060 oracle uses BLOCK_K = 16 on all 27 shapes (the FMA-only Triton path of the earlier update). The 3080 rankers spread their top-3 over bk 16 / 32 / 64 / 128: bk 16 is 28% of the top-3 candidates for v1 `as_is`, 53% for v1 `native`, 26% / 44% for v2 `as_is` / `native`. They are asking for tiles tuned for a tensor-core GPU. This is a **confound**: the data cannot separate "different architecture" from "no tensor-core path for Triton".

**Joint search (Triton top-3 + cuBLASLt top-16).** In all four variants the joint choice is cuBLASLt on 24 of 27 shapes (the 3080: 8–9 of 27). Timed in the 2060 stage-2 session (cuBLASLt keys are deterministic; 25–26 of 27 shapes are covered, the rest are Triton picks that session did not time): speed vs B3+ median **0.979–0.983**, p10 0.770–0.772, within 5% of B3+ on 16/25–17/26; the 3080 on the same shapes 1.005, p10 0.93–0.94, 22/27. The ceiling of that pool on the 2060 (best of B3 and the Triton oracle vs B3+) is 0.986 median, p10 0.775: **the zero-shot joint search is at the pool's ceiling**, because Triton cannot beat cuBLASLt there. What is left to B3+ is the offline deep search (B3+ vs B3 on the 2060: median 0.986, p10 0.775; on the 3080 0.994 / 0.877).

**Findings.**
- The zero-shot run is valid and complete; the generated report of the first scoring was not (survivorship, fixed above).
- **Triton-only selection does not transfer** from the 3080 to this 2060 without refitting: 13 of 27 top-3 picks are ≥ 50% off even with the 2060's own limits in the features (`native`), vs 0 of 27 for the ranker retrained on 56 shapes (≈ 10 h of tune sweep).
- **Joint selection is robust to the failure**: it lands at the pool's ceiling, so on this GPU the retrain buys almost nothing for the joint selector (Triton is chosen on 2–3 shapes), and a lot for Triton-only selection. Whether retraining pays for a *tensor-core* GPU other than the 3080 is open.

**Caveats.** n = 27 (counts of picks > 50% off are robust by construction; medians and p95 of the slow picks use one-sample screen times); the 2060 Triton path is FMA-only (confound above); mixed harness commits (benign, above); v2 on the 2060 sees a pruned space of this GPU in `native`, the 3080's in `as_is`; the joint speed excludes Triton picks the stage-2 session did not time (`picks` + stage 2 would cover them); the 3080 reference numbers are on the same 27 shapes, not its 86.

**Gates.** None changed. **Next steps.** (1) `python -m analysis.zero_shot picks` and its stage-2 commands on the 2060 to time the zero-shot Triton picks and score the joint choice with full coverage. (2) A tensor-core GPU other than the 3080 (Ampere / Ada) for the transfer claim without the FMA confound. (3) Use `cross_gpu` for every subset comparison.

Reproduce (CPU only): `python -m analysis.zero_shot score --data data/rtx2060/quick-20261029-095746-v3/data --report data/rtx2060/quick-20261029-095746-v3/reports --k 3`, then `python -m analysis.cross_gpu` with the sides below (writes `reports/cross_gpu.md`):
```bash
D3=data/rtx3080_sm86_68sms_10gb; Z=data/rtx2060/quick-20261029-095746-v3/data
python -m analysis.cross_gpu \
  --side 3080-v1 $D3 online_lgbm_rank_diverse.jsonl online_joint_diverse.jsonl results_lock1755_v2.jsonl results_lock1755_v2_graph.jsonl \
  --side 3080-v2 $D3 online_joint_diverse_v2.jsonl online_joint_diverse_v2.jsonl results_lock1755_v2.jsonl results_lock1755_v2_graph.jsonl \
  --side 2060-retrained $Z online_lgbm_rank_diverse.jsonl online_lgbm_rank_diverse.jsonl results_validate.jsonl results_validate_graph.jsonl \
  --side 2060-zs-v1-as_is $Z zero_shot_v1_as_is.jsonl zero_shot_v1_as_is.jsonl results_validate.jsonl results_validate_graph.jsonl \
  --side 2060-zs-v1-native $Z zero_shot_v1_native.jsonl zero_shot_v1_native.jsonl results_validate.jsonl results_validate_graph.jsonl \
  --side 2060-zs-v2-as_is $Z zero_shot_v2_as_is.jsonl zero_shot_v2_as_is.jsonl results_validate.jsonl results_validate_graph.jsonl \
  --side 2060-zs-v2-native $Z zero_shot_v2_native.jsonl zero_shot_v2_native.jsonl results_validate.jsonl results_validate_graph.jsonl
```

## Update 2026-11-02: first second-GPU run (RTX 2060, retrained from scratch) — the selector recipe transfers, the Triton family does not (no tensor cores on sm_75)

**What ran.** `pipeline/run.sh --quick` on an RTX 2060 (`rtx2060_sm75_30sms_6gb`, 30 SMs, 6 GB, driver 595.99.02, `--clock-lock 1755`, rows at harness `6548db4`, clean). Run directory `data/rtx2060/quick-20261029-095746-v2/` (copied here by hand; the numbers below are read from it, nothing was re-measured on this machine).
- Tune sweep: 56 shapes, 35,799 s (9.9 h). Held-out screen: 27 shapes, 63,650 s (17.7 h). Held-out shapes are **27 of the 3080's 86** (all 27 keys are in the 3080 held-out set; checked). `lgbm_rank` v1 form (regression, 63 leaves, 1,500 trees) refit on the 56 tune shapes only, pool 165 configs; no held-out shape was read by any fit.
- Roofline on the 2060: 26.4 TFLOP/s fp16 tensor (fp32 accumulate), 289 GB/s (spec 336); the 3080: 62.9 TFLOP/s, 686 GB/s.
- Stage 2 cold `results_validate.jsonl` (`20261101-155413-4a1b56`): 809 rows, 0 invalid, 32 power-capped (4%), 216 `cublaslt_deep` rows (B3+ exists). Graph (`20261101-160530-155120`, shapes < 50 µs): 163 rows, 0 invalid, 0 power-capped. Online search `online_lgbm_rank_diverse.jsonl` (`20261101-160722-b3270a`): 27 shapes, k = 5, fresh Triton cache.
- **Not a zero-shot run.** `analysis.zero_shot` found nothing to score here (no `zero_shot_*.jsonl`); this is the *retrain* arm. The zero-shot arm is still open (next steps).

**Comparison method.** Both GPUs restricted to the same 27 shapes and scored with the repo's own functions (`analysis.zero_shot.score_rows` for regret vs each GPU's own screen oracle; `analysis.ranker_compare.load_stage2` / `references` for stage 2 vs B3+). 3080 rows: v1 `online_lgbm_rank_diverse.jsonl`, v2 `online_joint_diverse_v2.jsonl` (Triton-only choice `3_lt0`), stage 2 `results_lock1755_v2{,_graph}.jsonl`. The comparison is `analysis/cross_gpu.py` (CPU only; output `reports/cross_gpu.md`, command below). **Do not compare a subset run to the 86-shape `REFERENCE_3080` constants in `analysis/zero_shot.py`**: on the same 27 shapes the 3080's v1 online top-3 is 1.5% / 7.6% (2.6%), not the 2.3% / 13.7% (4.4%) of the 86. (The constants themselves recompute exactly from the 3080 rows.)

**1. Selection quality** (regret vs the GPU's own screen oracle, median / p95 (mean), 27 shared shapes):

| pick | 3080 v1 (477 tune shapes) | 3080 v2 | 2060 retrained (56 tune shapes) |
|---|---|---|---|
| top-1 | 1.9% / 12.5% (3.6%) | 0.9% / 7.2% (2.4%) | 0.7% / 9.1% (2.6%) |
| online top-3 | 1.5% / 7.6% (2.6%) | 0.8% / 6.9% (1.8%) | 0.7% / 7.5% (2.5%) |
| online top-5 | 0.8% / 7.2% (2.3%) | 0.8% / 6.9% (1.9%) | 0.5% / 6.7% (2.1%) |

Paired bootstrap on the 27 shapes (online top-3, 2060 minus 3080 v1; 5,000 resamples): mean **−0.1 points [−1.7, +1.6]**, median −0.8 [−2.1, +1.8], p90 −0.4 [−3.6, +5.5]. Six shapes exceed 5% regret on each GPU; 18 (2060) vs 16 (3080) are within 2%. Online ranking cost 37 ms on the 2060 (pool 165) vs 99 ms (3080 v1).

**2. Value against the vendor path** (stage 2, speed = t_B3+ / t_pick, > 1 = faster than the searched cuBLASLt; online top-3, Triton only):

| group | 3080 v1 | 3080 v2 | 2060 |
|---|---|---|---|
| all 27 | 1.005 | 1.005 | **0.434** [p10 0.163, p90 0.907] |
| ≥ 50 µs, cold (3080 n = 16, 2060 n = 21) | 0.967 | 0.985 | **0.242** |
| < 50 µs, graph (3080 n = 11, 2060 n = 6) | 1.147 | 1.147 | 0.928 |

Family ceiling (best Triton row in the session vs B3+, i.e. G2 / G4 on these shapes): median t_B3+/t_Triton-oracle 1.007 (3080) vs **0.440** (2060); shapes with the Triton oracle ≥ 90% of B3+: 25/27 vs **3/27**; Triton oracle faster than B3+ by more than the tie margin: 8 vs 2; slower by more: 10 vs **24**. The only 2060 wins are two launch-bound shapes of ~3 µs (105×6×9, 369×17×12, 1.7×). Largest shape, 8192×4096×11008: B3+ 28.3 ms (26.1 TFLOP/s, at the measured 26.4 peak) vs best Triton 173 ms (4.3 TFLOP/s), 6.1×. Selection cannot fix this: the oracle itself is far behind.

**Why (observed, cause not isolated).** All **193** Triton kernels in the 2060's compile cache (`online_triton_cache`, `.target sm_75`) contain **zero `mma.sync`** and only `fma.rn.f32`; the same check on the 3080's cache (`sm_86`) finds `mma.sync` in all 1,886 kernels. So with Triton 3.7.1 the `tl.dot` on Turing ran on CUDA cores (4.3 TFLOP/s is plausible for fp32 FMA on 1,920 cores at 1755 MHz); cuBLASLt uses the tensor cores. I did not test whether any Triton option or version emits tensor-core code on sm_75. The 2060's choices are also far less varied (7 distinct tiles over the 27 shapes, mostly BLOCK_K = 16, vs 14 for the 3080's v1 and 15 for v2, counting the Triton-only choice), which is a plausible reason 56 tune shapes were enough.

**Findings.**
- The pipeline (sweep → `lgbm_rank` → diversified top-k → short online search) runs end to end on a second architecture and selects as well as the 3080's within what 27 shapes can resolve (interval ±1.7 points on the mean). It does **not** show equality, and it does not show a general result for tensor-core GPUs: the 2060 space is simpler and one family is FMA-only.
- On the 2060 the Triton family fails G2 (3/27 shapes ≥ 90% of B3+) and G4: as a cuBLAS competitor it is not viable there; any value would come from a joint {Triton ∪ cuBLASLt} selector or the tiny launch-bound shapes. The joint search was not run on the 2060.
- Retrain cost on the 2060: ≈ 10 h of tune sweep (the 17.7 h held-out sweep is only the evaluation oracle).

**Caveats.** n = 27 (p95 is about the second-worst shape); a different harness commit and session than the 3080 numbers; v1 form only (no rank v2, no tail suite) on the 2060; the 2060's regime split differs because the same shapes are slower there (oracle < 50 µs: 6 shapes vs 11); ε = 2% / 5% taken from the 3080 noise study, the 2060's own noise study (`noise.jsonl`) was not analysed; 4% of the cold stage-2 rows were power-capped.

**Gates.** 3080 gates unchanged. New for the 2060 (27 shapes, indicative): G2 fails, G4 fails. **Next steps.** (1) **Zero-shot arm on the 2060** (the 3080 `lgbm_rank` / `lgbm_rank_v2`, no refit), reusing this run's held-out screen as the oracle (skips the 17.7 h sweep): `python -m analysis.zero_shot --clock-lock 1755 run --out $D --ranker v1=$D/src3080/lgbm_rank v2=$D/src3080/lgbm_rank_v2 --k 5 --lt-n 16 --fresh-cache`, then `score --data $D --report $D/../reports --k 3` (and `--k 5`), optionally `picks` for the joint stage 2; copy the 3080 models into a separate `src3080/` folder (the run's own `lgbm_rank.*` would be overwritten). Bar to beat: the retrained row above. A poor zero-shot result is confounded by the FMA-only path; a good one is the stronger evidence. (2) Joint selection and B3+ on the 2060. (3) A second tensor-core GPU for the generality claim. (4) Report a same-subset 3080 reference in `analysis.zero_shot` instead of the 86-shape constants.

Reproduce (CPU only; the run itself is `pipeline/run.sh --quick` on the 2060):
```bash
D3=data/rtx3080_sm86_68sms_10gb; D2=data/rtx2060/quick-20261029-095746-v2/data
python -m analysis.cross_gpu \
  --side 3080-v1 $D3 online_lgbm_rank_diverse.jsonl online_joint_diverse.jsonl results_lock1755_v2.jsonl results_lock1755_v2_graph.jsonl \
  --side 3080-v2 $D3 online_joint_diverse_v2.jsonl online_joint_diverse_v2.jsonl results_lock1755_v2.jsonl results_lock1755_v2_graph.jsonl \
  --side 2060 $D2 online_lgbm_rank_diverse.jsonl online_lgbm_rank_diverse.jsonl results_validate.jsonl results_validate_graph.jsonl
```
The tile counts in the Why paragraph were first quoted as 18 for the 3080 (joint choice, which can include cuBLASLt keys); the module counts the Triton-only choice.

## Update 2026-11-01: rank v2 (analytic features, near-top weights, tail suite) — Triton-only tail halves, the joint search barely moves

**What ran** (RTX 3080, `--clock-lock 1755`, driver 595.99.02).
- Tail suite (`suites/v1/tail.json`, 150 new tune shapes, commit `6280b0f`): `python -m tune --clock-lock 1755 sweep --suite tail --per-suite 1000 --screen-dir sweep_screen_tail`, env `sweep-20261031-011440`;
  150 of 150 shapes complete, 0 skipped, 8,269 s. Regimes: 30 narrow-N, 30 small-M, 20 tiny-K, 40 unaligned sub-50 µs, 20 mid-size, 10 small-M/N split-K (roofline 3–400 µs; none duplicates an existing shape, lies in the held-out regions or shares two dims with a held-out shape).
- Rank v2 (`python -m tune ranker-v2`, CPU only, commit `2a6c246`): `lgbm_rank_v2.*`, 1,174 pool configs, 627 tune shapes (477 + tail), v1's CV-chosen hyperparameters (not re-searched), diversification rule refit on v2's own out-of-fold scores.
  Tune out-of-fold best-of-3: 2.14% mean, 8.0% p95. No held-out shape was read by any fit.
- Held-out, one new stage-2 session for every pick and baseline so v1 and v2 are not confounded by session noise: v2 joint online search `online_joint_diverse_v2.jsonl` (session `20261101-020041-c23bdc`);
  stage 2 cold `results_lock1755_v2.jsonl` (`20261101-020208-bcca60`, 2,725 rows, 0 invalid, 62 not converged, ≈5% power-capped) and graph `results_lock1755_v2_graph.jsonl` (`20261101-021022-e5d06f`, 1,794 rows, 0 invalid, ≈6% power-capped), 56 shapes < 50 µs.
  The session times `torch`, `triton:sweep:8`, the union of v1's clean and v2's diverse top-5 Triton picks, `cublaslt:16` and `cublaslt_deep:20`. All at harness `2a6c246`, clean. v1 numbers are from the clean joint run `20261031-005000-437cca`.
  The full table is [`ranker_compare.md`](ranker_compare.md) (`python -m analysis.ranker_compare --online v1=online_joint_diverse.jsonl v2=online_joint_diverse_v2.jsonl`).

**What v2 is.** v1's rank form (pool = union of each training shape's top 5, LightGBM regression of capped log regret) plus (1) 8 label-free analytic features per (shape, candidate): a roofline-style time estimate, its value relative to the best runnable config of the shape and its rank among all runnable configs, compute/memory balance, bytes per flop, split-K traffic share, CTAs per SM;
(2) weight 5 on training rows within 25% of the shape's best; (3) the tail suite as extra tune data. The three were chosen on tune out-of-fold regret before any held-out look (below).

**Tune evidence** (20 grouped folds on the original 477 shapes, paired bootstrap, diverse top-3): baseline 2.77% mean / 10.1% p95 (floored at 1.02 µs: 1.73% / 8.8%);
all three: 2.13% / 9.4% (floored 0.99% / 5.3%); mean change −0.64 points [−0.99, −0.30], floored p95 −3.5 points [−5.7, −0.7]; raw p95 −0.7 [−3.2, +0.9] is not significant. Each ingredient adds: analytic features + weighting −0.29 points, tail data on top −0.35 [−0.66, −0.05].
Ablation of the analytic features (same folds, tail data in training, diverse top-3, original 477 shapes; change vs full v2 [95% interval]): no analytic features +0.37 points [+0.07, +0.68]; only the 3 peak-free features (bytes per flop, split-K traffic share, CTAs per SM) +0.02 [−0.24, +0.26];
all 8 with the compute/memory balance wrong by 4× −0.19 [−0.42, +0.03] (top-5: −0.23 [−0.44, −0.04]). On the 150 tail-suite shapes the peak-free variant is worse (+0.48 [+0.03, +0.97]) but dropping all analytic features is not (+0.05), so the tail evidence is inconclusive.
So the gain comes from the geometric part (tile reuse, padding, split-K traffic), not from a literal roofline: **v2 needs no roofline measurement**, and per-GPU peaks must not be substituted at deployment (that would shift the features away from what the model was trained on).
Learning curve (50 / 100 / 150 tail shapes): mean 2.41% / 2.23% / 2.13% vs 2.48% with none; the original-shape gain per added shape is shrinking (−0.07, −0.18, −0.10 points per 50), so a second batch of the same mix is expected to add ≲ 0.2 points.

**Held-out screen regret** (86 shapes, diverse picks, median / p95 (mean); the screen oracle is biased low): top-1 v1 3.6% / 18.1% (5.5%) → v2 2.8% / 13.0% (4.6%); top-3 2.0% / 10.6% (3.4%) → 1.0% / 9.3% (2.1%); top-5 0.9% / 9.0% (2.6%) → 0.5% / 7.8% (1.7%). Top-3: v2 better by > 2% regret on 24 shapes, worse on 9.

**Held-out stage 2, same session** (picks scored at their full-protocol time; regret vs the Triton oracle / the joint oracle of the session):

| selector | ranker | first-use cost | regret vs oracle, median / p95 (mean) | vs B2 median / p10 | faster / slower than B2 | vs B3+ median | faster / slower than B3+ |
|---|---|---|---|---|---|---|---|
| Triton top-3 only | v1 | 0.52 s | 0.4% / 9.5% (2.3%) vs Triton oracle | 0.962 / 0.701 | 41 / 18 | 1.019 | 24 / 38 |
| Triton top-3 only | **v2** | 0.60 s | **0.2% / 4.6% (1.2%)** vs Triton oracle | 0.949 / 0.698 | 43 / 15 | 1.016 | 23 / 35 |
| top-3 + cuBLASLt top-16 | v1 | 0.59 s | 0.4% / 12.5% (3.0%) vs joint oracle | 0.911 / 0.698 | 49 / 0 | 1.000 | 24 / 14 |
| top-3 + cuBLASLt top-16 | **v2** | 0.67 s | 0.4% / 10.4% (2.7%) vs joint oracle | 0.902 / 0.680 | 49 / 0 | 1.000 | 23 / 14 |
| top-5 + cuBLASLt top-16 | v1 | 0.90 s | 0.3% / 12.0% (2.7%) | 0.901 / 0.697 | 52 / 0 | 1.000 | 26 / 13 |
| top-5 + cuBLASLt top-16 | v2 | 0.96 s | 0.3% / 10.4% (2.5%) | 0.902 / 0.680 | 51 / 0 | 1.000 | 25 / 14 |

Faster / slower counts use the 5% tie margin below 50 µs and 2% above. Paired v2 vs v1, geometric-mean ratio of the chosen config's time (95% bootstrap interval): Triton-only top-3 **0.9897 [0.9823, 0.9970]** (v2 faster on 11 shapes, slower on 4);
joint top-3 0.9970 [0.9917, 1.0018] (4 / 4); joint top-5 0.9985 [0.9942, 1.0022] (1 / 2); joint top-1 0.9977 [0.9899, 1.0050] (6 / 8).

**Findings.**
- v2 does what tune predicted for Triton alone: held-out regret vs the Triton oracle falls from 2.3% to 1.2% mean and 9.5% to 4.6% p95 at top-3, in both regimes (< 50 µs 2.1% → 1.2%, ≥ 50 µs 2.6% → 1.1%), and the effect is significant in the paired time ratio.
- In the joint search the gain is **not** significant: the interval for the paired ratio includes 1, v2 is faster on 4 shapes and slower on 4, and the median regret and the B2 / B3 / B3+ counts barely move (the p95 vs the joint oracle goes from 12.5% to 10.4%).
  The cuBLASLt candidates already cover most of the regimes where the Triton ranker was weak, so a better Triton ranking changes the final pick on few shapes. The remaining joint tail is mainly B3+ (offline deep search) wins, unchanged by the ranker. G4 is unchanged.
- **First-use cost rises by ~0.08 s** (0.59 → 0.67 s at top-3 + LT16): ranking is ~40 ms slower per shape online (0.099 → 0.139 s: the analytic features over the full 10,840-config space), and v2's picks compile ~11 ms slower per candidate (slightly more warps and stages).
- Net: adopt v2 for **Triton-only** selection (where it halves the tail); for the joint search it is a small, uncertain gain for a small cost, so v1 and v2 are both acceptable there.

**Caveats.** One GPU, one stage-2 session per ranker pair, placement-pooled but single-run online choices (the choice differs between two runs on ~10% of shapes). The analytic features use fixed constants (the 3080's unlocked-clock peaks, 62.9 TFLOP/s and 686 GB/s; the clock-locked roofline is 59.7 TFLOP/s) as model units on every GPU, so they are well defined elsewhere but describe a 3080-like machine; the ablation below shows the exact values do not matter. The tail regimes were taken from the worst *tune* out-of-fold shapes (Step 0 of this work), but while relating the Triton tail to cuBLASLt wins I had also seen held-out shapes with N ≈ 1 (e.g. 384×2×10369, 7039×1×93), so the narrow-N regime was not chosen in strict isolation from held-out information; no held-out shape or label entered a fit, and the suite excludes anything sharing two dims with a held-out shape. The v2 ingredients were fixed on tune before the first held-out look.

**Gates.** None changed. **Not done / next steps.** (1) Add the joint selector and `KPolicy` to the zero-shot runbook (`analysis/zero_shot.py`), including `lgbm_rank_v2`. (2) Speed up v2 ranking (precompute the static part of the analytic features; ~40 ms is most of the cost increase). (3) A second tail batch only if it targets what is left: oracle split-K > 1, 10–50 µs runtimes and narrow N (tail rates 11%, 9%, 9% after v2).
(4) Graph-mode noise study before relying on sub-50 µs wins. (5) A cache so a shape can move from k = 3 to k = 5 without a new search.

Reproduce: `python -m tune --clock-lock 1755 sweep --suite tail --per-suite 1000 --screen-dir sweep_screen_tail`; `python -m tune ranker-v2`; `python -m tune --clock-lock 1755 online --ranker data/<hw>/lgbm_rank_v2 --shapes-file $D/sweep_screen_heldout/subset.json --k 5 --order diverse --lt-n 16 --fresh-cache --out-name online_joint_diverse_v2.jsonl`;
then `benchmark --shapes-file $D/sweep_screen_heldout/subset.json --backends torch,triton:sweep:8,triton:picks,cublaslt:16,cublaslt_deep:20 --cache-modes cold --out-name results_lock1755_v2.jsonl` with `TUNE_SWEEP_DIR=$D/sweep_screen_heldout` and `TUNE_PICKS` pointing at the union of v1 and v2 candidates (`v2_stage2_picks.json`), the same with `subset_lt50us.json --cache-modes hot --graph-batch 50` for the graph file, and `python -m analysis.ranker_compare --online v1=online_joint_diverse.jsonl v2=online_joint_diverse_v2.jsonl`.

## Update 2026-10-31: joint {Triton ∪ cuBLASLt} online selection (no router) and a tail diagnosis

**What ran** (RTX 3080, `--clock-lock 1755`, driver 595.99.02).
- Tune stage 2: `results_lock1755_tune477.jsonl`, session `20261030-141736-f6aa5f`: 477 tune shapes, cold, `torch`, `triton:sweep:8`, `cublaslt:16`, `cublaslt_deep:20`; 11,023 rows, 0 invalid, 121 not converged, ≈8% power-capped; harness `6548db4`, clean.
  Graph: `results_lock1755_tune477_graph.jsonl`, session `20261030-151839-b4fa3b`: the 198 tune shapes whose screen oracle is < 50 µs (`sweep_screen/subset_lt50us_477.json`, new; the old `subset_lt50us.json` is the 160-shape subset), `--graph-batch 50`; 4,802 rows, 0 invalid, ≈5% power-capped.
- Joint online search on the 86 held-out shapes: `online_joint_diverse.jsonl`, session `20261031-005000-437cca`, `tune online --order diverse --k 5 --lt-n 16 --fresh-cache`, harness `e9f3906`, clean; 0 skipped, 0 invalid candidates. A first run with the change uncommitted (`online_joint_diverse_dirty.jsonl`, session `20261031-003456-76db31`) agrees: the chosen config is identical on 90% of shapes and 1 shape differs by > 5% at stage-2 time.
  Every pick is scored at its full-protocol stage-2 time (held-out sessions `selectors3` / `topkd`, cold and graph). Held-out shapes were only scored, not fitted.

**Tune split: the families split the wins.** Against the Triton screen's top 8 re-timed, 477 shapes: heuristic-only cuBLASLt (B3) is > 2% faster on 124 and Triton on 254; B3+ (20 s search) is faster than Triton on 219 and slower on 99 (159 ties).
Deep search beats the heuristic by > 2% on 237 shapes. The perfect joint choice vs B2 has p10 0.675 (10% of shapes gain ≥ 32%); vs B3+ p10 0.850.

**Held-out, joint online search** (Triton diverse top-k + cuBLASLt heuristic top-16 in one short-timed pool, picks scored at stage-2 times). Ratio < 1 = faster than the reference.

| selector | first-use cost | vs B2 median / p10 | faster / slower than B2 (> 5%) | vs B3 median | faster / slower than B3 | vs B3+ median | faster / slower than B3+ | regret vs joint oracle median / p95 / mean | picks cuBLASLt |
|---|---|---|---|---|---|---|---|---|---|
| Triton top-3 only | 0.52 s | 0.961 / 0.706 | 41 / 18 | 0.981 | 31 / 27 | 1.020 | 23 / 37 | 4.4% / 41.5% / 11.6% | 0% |
| top-3 + B2 | 0.54 s | 0.961 / 0.706 | 41 / 0 | 0.983 | 31 / 13 | 1.000 | 23 / 25 | 1.4% / 21.2% / 5.3% | 33% |
| top-3 + LT4 | 0.56 s | 0.923 / 0.696 | 46 / 0 | 0.983 | 31 / 3 | 1.000 | 23 / 16 | 0.5% / 13.7% / 3.4% | 40% |
| **top-3 + LT16** | **0.59 s** | **0.916 / 0.696** | **49 / 0** | **0.983** | **31 / 0** | **1.000** | **23 / 14** | **0.5% / 13.2% / 3.1%** | 41% |
| top-5 + LT16 | 0.90 s | 0.908 / 0.696 | 51 / 0 | 0.979 | 33 / 0 | 1.000 | 25 / 13 | 0.4% / 11.9% / 2.8% | 41% |
| top-1 + LT16 | 0.33 s | 0.921 / 0.696 | 48 / 0 | 1.000 | 26 / 0 | 1.000 | 20 / 15 | 1.5% / 14.5% / 3.9% | 51% |

Split at 50 µs (top-3 + LT16): < 50 µs (n = 56, graph times) 0.894 vs B2, 36 faster / 0 slower, 0.990 vs B3+; ≥ 50 µs (n = 30, cold, ε = 2%) 0.976 vs B2, 20 / 0, 1.015 vs B3+ (B3+ faster on 15).
Caveat: choices are made by short online timings and scored by stage 2, so this is a real selection, but one placement and one run per shape; the cuBLASLt heuristic list is stage-2-timed in the same session as the Triton picks.

**Findings.**
- cuBLASLt candidates need no compile, so adding them costs ~0.08 s and removes every loss to B2 and B3 (Triton-only loses > 5% to B2 on 18 shapes, to B3 on 27).
- The remaining gap to B3+ (14 shapes > 5% slower, 15 of 30 at ≥ 50 µs) is the offline deep search, which has no online equivalent here. G4 (differentiation ≥ 50 µs) is unchanged.
- Triton top-1 + LT16 already costs 0.33 s and keeps most of the gain; top-3 buys ~0.8 points of mean regret for ~0.26 s.
- **Decision (user):** default k = 3, k = 5 for frequently called kernel patterns. Measured cost is ~0.145 s per extra Triton candidate (compile 0.134 s, timing 0.011 s). Payback of k = 1 → 3: ~40,000 calls pooled (~17,000 at ≥ 50 µs, ~870,000 at < 50 µs).
- The family router is a *cost* optimisation only (skip Triton compiles when cuBLASLt is clearly ahead, up to ~0.4 s on ~40% of shapes), not a quality lever. Deprioritised.

**Tail diagnosis (tune shapes only, 20 grouped folds, cached out-of-fold scores; no held-out data).** Diverse top-3 OOF regret 1.2% / 10.1% (mean 2.8%); 84 of 477 shapes are > 5% (63% of regret mass).
- 74 of 84 are ranker misses (the fold's pool could reach < 5%); 10 are pool misses. Pool oracle p95: 2.4% at `POOL_TOP=5`, 1.8% at 10, 0.5% at 20. 24 of the 74 already have the oracle's (tile, split, BLOCK_K) group in the picks (warps / stages miss).
- 73% of the tail has an oracle < 50 µs; 43% loses < 1.02 µs (the timing grid). Counting those as ties: p95 8.8%, mean 1.7%.
- Screen label noise is small (screen #1 vs stage-2 best among the screen's top 8, 48 shapes: median 0.2%, p95 3.9%).
- Ranker calibration is poor: corr(predicted, realised log regret of the top pick) = 0.35; on tail shapes predicted 0.058 vs realised 0.124.
- Variants (same folds and hyperparameters; paired bootstrap): analytic-cost features (padding, roofline time, relative time) and 5× near-top row weighting each lower mean top-3 regret by ~0.1–0.3 points; p95 does not move beyond noise.
  A 1.02 µs-floored target helps only the floored metric and hurts the raw one. Pool 10 is a tie. Fixed k is the main lever (k = 6: p95 7.5%, k = 8: 7.0%); an adaptive k from the predicted top-1 log regret (AUC 0.78 for tail shapes) gains ~0.1 points over fixed k at equal budget. **Nothing adopted yet.**

**Gates.** No gate changed. G5 (held-out tractability) is unaffected by this update.

**Next steps.** (1) Remaining tail work: more tune shapes in the sub-50 µs / unaligned / N ≤ 16 regimes (new `tail` suite, tune split), then re-fit with analytic features + weighting and re-score held-out once.
(2) Add the joint selector to the zero-shot runbook (the `--lt-n` change is committed in `e9f3906` and the joint search re-run clean). (3) Optional cost router. (4) Graph-mode noise study before relying on sub-50 µs wins.

Reproduce: `python -m tune --clock-lock 1755 benchmark --shapes-file $D/sweep_screen/subset.json --backends torch,triton:sweep:8,cublaslt:16,cublaslt_deep:20 --cache-modes cold --out-name results_lock1755_tune477.jsonl`;
the same with `subset_lt50us_477.json --cache-modes hot --graph-batch 50` (graph); `python -m tune --clock-lock 1755 online --shapes-file $D/sweep_screen_heldout/subset.json --k 5 --order diverse --lt-n 16 --fresh-cache`.
Tail scripts (`tail_diag.py`, `tail_exp.py`, `paired.py`, `adaptk2.py`) are scratch files, not in the repo.

## Update 2026-10-27: fp16-accumulation test was seed-dependent (found on an RTX 2060 test run)

**What ran.** A `pytest` run on an RTX 2060 (sm_75) gave 94 passed, 2 failed. No benchmark rows were produced, so no session ids apply.
- `test_detects_fp16_accumulation` failed: the fake fp16-accumulating kernel (K = 8192) passed the correctness gate. Not a 2060 bug. Re-run on the RTX 3080 over 20 seeds,
  the gate rejects that kernel on only **9 of 20** seeds (a correct fp32-accumulate kernel is accepted 20 of 20). The test used seed 0, which happens to be a rejecting seed on the 3080.
  A different GPU can draw different inputs from the same seed (PyTorch's CUDA `randn` layout depends on the SM count; not verified on the 2060).
- `test_handpicked_correct[shape0-large_square]` failed with `ptxas failed with error code -1` and an empty stderr at shape (1, 1, 1), config 128x256x32 / 8 warps for `sm_75`.
  The PTX in the log compiles cleanly with Triton's ptxas for `sm_75` on the 3080 machine (exit 0), and the same config passed on other shapes, so this looks like a killed process
  (memory or limits) on the 2060 machine. **Unresolved:** not reproduced, not fixed.

**What the gate does and does not catch.** RMS error against the fp32 reference, fp16-accumulate over fp32-accumulate, on the same inputs: about 11x at K = 8192, 15-16x at K = 16384,
about 32x at K = 65536. The gate's relative tolerance (1e-2 of |ref|) is roughly 20x above a correct kernel's error, so a kernel that accumulates in fp16 can slip through at short K.
Nonzero-mean inputs did not help (mean 1: rejected 5 of 20 at K = 8192, 0 of 20 at K = 2048).

**Change.** `tests/test_correctness.py`: the gate test now uses K = 16384 (rejected 20 of 20 seeds on the 3080), and a new test compares the RMS errors directly (`bad > 5 x good`,
seed- and architecture-independent). Gate tolerances are unchanged, so no gate in the plan moved and no earlier result is affected. Full suite on the 3080: 97 passed.

**Next steps.** Re-run `pytest tests/test_correctness.py` on the 2060 to confirm both new tests pass there. Repeat the `large_square` (1, 1, 1) test several times and check `dmesg` for an OOM kill.
Consider pruning or skipping the 128x256 tile on sm_75, where it compiles to a very large unrolled kernel.

Reproduce: `python -m pytest -q tests/test_correctness.py`.

## Update 2026-10-19 (evening): inner variety pass (warps / stages within the top tile) — fixes one shape, otherwise a tie

**What changed.** `diverse_order` gained an optional inner pass. Within the top pick's (BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K) group, it adds up to `inner` configs with a new
num_warps (or warps × stages, or warps × stages × GROUP_M) value, best-scored first and within `inner_delta` of the top pick. The pass runs before or after the outer tile pass.
**Fit on tune shapes only**, jointly with the outer rule, on the cached out-of-fold scores: 90 inner rules plus the 57 earlier ones, with "no inner pass" as a candidate.
- Placed after the outer pass, it is a no-op at k = 3, because the outer picks fill the slots. Those rules tie outer-only exactly.
- The best rule that spends a top-3 slot on it, `{"inner_group": "warps", "inner_delta": 0.02, "inner_pos": "before"}` on the same outer rule, is slightly *worse* at k = 2–3:
  tune best-of-3 mean 2.84% vs 2.77%, best-of-2 3.49% vs 3.43%. It is marginally better at k = 4–5 (2.42% vs 2.45%, 2.12% vs 2.14%).

So the CV-chosen rule stays **outer-only** (`Ranker.rank(order="diverse")`). The inner rule is saved as `meta["inner_rule"]` and measured as a diagnostic order (`--order inner`).
It changes the held-out top 3 on 10 of 86 shapes, and the online top-3 choice on 5.

**Re-measured** on commit `9e6e4be` (clean). Each session holds plain, outer-only and inner top-5 together with knnf / knn5:
- `bench-20261019-173739-lock1755-topki`: cold, 2,686 rows, 0 invalid, 4.8% power-capped.
- `-topki-graph`: 1,766 rows, 0 invalid, 9.4% power-capped.
- `online-20261019-173739-lock1755-topki`: inner, empty Triton cache, 86 × 5, 0 invalid.

The plain and outer online choices from the earlier runs are re-priced here. Tables: [`learned_topk.md`](learned_topk.md).

**Online top-3 choice re-timed with the full protocol, regret vs the per-shape Triton oracle, median / p95 (mean):**

| selector | cold, all 86 | cold, ≥ 50 µs (30) | graph, < 50 µs (56) |
|---|---|---|---|
| knnf | 2.5% / 22.3% (6.5%) | 2.2% / 36.5% (8.7%) | 4.9% / 25.8% (7.7%) |
| plain online top-3 | 0.9% / 12.2% (3.1%) | 0.9% / 16.7% (3.8%) | 0.4% / 12.3% (2.7%) |
| outer online top-3 (bar) | 1.1% / 9.1% (2.6%) | 1.0% / 10.0% (2.4%) | 0.2% / 9.7% (2.1%) |
| **inner online top-3** | 1.1% / 9.1% (2.5%) | 1.0% / 10.0% (2.4%) | **0.2% / 8.2% (1.7%)** |
| inner online top-5 | 0.8% / 8.5% (2.3%) | 0.3% / 8.1% (1.9%) | 0.1% / 7.8% (1.5%) |

- **It does what it was built for, on one shape.** 381×1401×250 goes from +14.2% to +0.2% (graph), and cold the inner pick is 7.3% faster than the outer-only pick. The second pick is `bm64_bn64_bk32_w8_s2_g4_sk1`, the
  top tile with 8 warps, which is that shape's oracle. 380×6978×216 also gains 5.6% (graph).
- **Elsewhere it is a tie.** Beyond ε, inner beats outer-only on 1 of 86 cold shapes and 2 of 56 graph shapes, and is never slower (geomean 1.001× cold, 1.004× graph). The graph p95 drop (9.7% → 8.2%)
  is that one shape leaving the tail. Cost is the same: k = 3 median 0.39 s, p90 0.81 s.
- **It cannot reach 4065×8842×28** (+10.6%). There the oracle `bm64_bn128_bk16_w4_s3_g1` differs from the top pick in stages and GROUP_M at the same warp count. The CV-best inner grouping is warps-only,
  and the stage / group variants the model ranks above it (g4, g16, g8 at s2) fill the slots. The other large misses are unchanged, and they are ranking errors no variety pass reaches:
  320×1×79 (+22% cold / +16% graph, oracle at rank 134) and 5398×196×774 (+20%, oracle tile at rank 68).
- **Outer-only reproduces** for its second session: online top-3 cold 1.1% / 9.1% / 2.6% vs 1.0% / 9.9% / 2.7%, graph 0.2% / 9.7% / 2.1% vs 0.2% / 9.9% / 2.1%.

**Decision: the bar stays outer-only diverse top-3.** Tune CV ranks it at or above every inner rule at k = 3. Promoting the inner rule because it fixed a held-out shape would fit the rule to held-out
data. It is safe to use, since it was never slower here, and `--order inner` stays available. A fair test of it needs new shapes.

**Next:** the remaining tail is ranking error, so it needs a better model, not more variety:
- Training data or features for N = 1 / tiny-K shapes (320×1×79).
- A ranking objective that weights near-optimal tiles (the 5398×196×774 tile sits at rank 68 of 868).
- Joint {Triton ∪ cuBLASLt} candidates.

Reproduce: the 2026-10-19 (late night) commands with `--out-name results_lock1755_topki[_graph].jsonl`, then
`python -m tune --clock-lock 1755 online --shapes-file $D/sweep_screen_heldout/subset.json --k 5 --order inner --fresh-cache` and
`python -m tune topk --results results_lock1755_topki.jsonl --graph-results results_lock1755_topki_graph.jsonl`.

## Update 2026-10-19 (late night): diversified top-3 — tail p95 12.8% → 9.9% at the same cost, never worse per shape

**What changed.** Plain top-k re-timed near-duplicates: the held-out top 3 covered 1.55 distinct (BLOCK_M, BLOCK_N, SPLIT_K) choices on average. `diverse_order` in `analysis/learned.py`
keeps the top-1 pick. It then takes the best-scored config of each new group, as long as its predicted log regret is within δ of the top pick, and fills any remaining slots in score order.
The groups are (BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K), and the result covers 2.23 distinct tile × split choices in the top 3. **Fit on tune shapes only.** I computed out-of-fold lgbm_rank scores
for all 477 tune shapes on the search's 20 grouped folds (cached in `data/<hw>/lgbm_rank_cv_scores.npz`). Then I scored 57 rules (4 groupings × δ ∈ {0.02 … ∞} × 1–2 plain picks, plus plain) by best-of-3 screen regret.

The winner is `{"group": "tile_split_bk", "delta": 0.05, "plain": 1}`, with an interior δ. Tune CV best-of-3 goes from 1.5% / 12.0% / 3.3% to 1.2% / 10.1% / 2.8% (median / p95 / mean). The rule is saved in the
ranker's metadata and used by `Ranker.rank(order="diverse")` and `tune online --order diverse`. The plain picks are unchanged and identical to the previous sessions'.

**Re-measured** in fresh sessions on a clean harness (commit `b56c906`). Each session times the plain and diverse top-5 together with knnf / knn5, so they share one oracle and B3+:
- `bench-20261019-112027-lock1755-topkd`: cold, 86 shapes, 2,683 rows, 0 invalid, 5.3% power-capped.
- `-topkd-graph`: 56 shapes, 1,765 rows, 0 invalid, 8.3% power-capped.
- `online-20261019-112027-lock1755-topkd`: diverse, empty Triton cache, 86 × 5 candidates, 0 invalid.

The plain online choices from `online-20261019-083637` are re-priced in these sessions. Tables: [`learned_topk.md`](learned_topk.md).

**Online top-k choice re-timed with the full protocol, regret vs the per-shape Triton oracle, median / p95 (mean):**

| selector | cold, all 86 | cold, ≥ 50 µs (30) | graph, < 50 µs (56) | vs B3+, cold all / graph |
|---|---|---|---|---|
| knnf | 2.7% / 22.8% (6.5%) | 2.5% / 36.4% (8.6%) | 4.2% / 26.5% (7.7%) | 0.960 / 0.949 |
| top-1 (no search) | 1.9% / 18.1% (4.1%) | 1.9% / 18.4% (5.0%) | 2.6% / 13.6% (4.4%) | 0.964 / 0.994 |
| plain online top-3 | 1.1% / 12.8% (3.1%) | 1.2% / 15.8% (3.7%) | 0.5% / 12.2% (2.7%) | 0.973 / 1.005 |
| **diverse online top-3** | **1.0% / 9.9% (2.7%)** | **0.9% / 11.1% (2.5%)** | **0.2% / 9.9% (2.1%)** | 0.982 / 1.011 |
| diverse online top-5 | 0.8% / 9.3% (2.5%) | 0.5% / 8.3% (1.9%) | 0.2% / 8.1% (1.9%) | 0.987 / 1.011 |
| diverse best-of-3, full-protocol timing | 0.9% / 9.3% (2.4%) | 0.9% / 11.1% (2.4%) | 0.2% / 9.8% (2.0%) | 0.983 / 1.011 |

- **Diversity strictly helps the tail.**
  - Against plain top-3, p95 drops from 12.8% to 9.9% cold, from 15.8% to 11.1% at ≥ 50 µs and from 12.2% to 9.9% under 50 µs (graph). The ≥ 50 µs mean drops from 3.7% to 2.5%.
  - Shape by shape, beyond ε, diverse is faster on 5 of 86 cold shapes and 2 of 56 graph shapes, and **never slower**. The rest tie (geomean 1.003× / 1.006×).
  - The cold < 50 µs median goes from 1.1% to 1.3%, which is within the event tick. The graph protocol for that regime improves.
- **Unlike plain, a larger k still pays.** Diverse top-5 reaches an ≥ 50 µs p95 of 8.3%, against plain top-5's 15.8%. That is now below the 48-config portfolio oracle's held-out p95 (~8.9%).
- **The cost is the same.** k = 3 takes a median 0.39 s per shape (p90 0.86 s), against 0.40 s for plain. Diverse picks are no harder to compile.
- **Fixed** (plain → diverse top-3):
  - 14244×16×1257: 18.2% → 4.6%.
  - 277×3×161: 12.8% → 0% (graph).
  - 1316×272×100: 11.7% → 0% (graph).
  - 417×512×6739: 12.9% → 9.5%.
  - 51×324×5709: 10.2% → 7.4% (graph).
- **Not fixed, and why.** These are ranker errors that no diversification within δ reaches:
  - **320×1×79** (+21% cold, +16% graph): the oracle `bm16_bn32_bk128_w2_s5` sits at rank 134, predicted 0.66 log regret behind the top. The model is simply wrong on this tiny-K, N = 1 decode-like shape.
  - **5398×196×774** (+20%): the oracle's tile (64×128, BLOCK_K 16) sits at rank 68, 0.10 behind, outside δ = 0.05.
  - **4065×8842×28** (+12%) and **381×1401×250** (+13–14%): the oracle shares the top pick's tile × BLOCK_K × split and differs only in num_stages / GROUP_M or num_warps (ranks 7–8). The grouping deliberately collapses those.
- **Against B3+:** cold all moves from 0.973 to 0.982. Under 50 µs graph it moves from 1.005 to 1.011, so the searched Triton pick is now marginally ahead of cuBLASLt's searched best at the median. G4 at ≥ 50 µs still fails (Triton oracle 0.96).

**New bar: diverse online top-3.** Cold all: 1.0% / 9.9% (2.7%). ≥ 50 µs: 0.9% / 11.1% (2.5%). Graph < 50 µs: 0.2% / 9.9% (2.1%). Cost: ~0.4 s first use per shape.

**Next:** the remaining tail needs a better ranker, not a wider search:
- An inner diversity pass over stages / warps for the top group, which would catch 4065×8842×28 and 381×1401×250.
- Training targets or features for N = 1 / tiny-K shapes (320×1×79).
- Joint {Triton ∪ cuBLASLt} candidates, now that Triton matches B3+ under 50 µs.

Reproduce:
```bash
python -m tune topk               # train, fit the diversification rule (tune CV), write plain + diverse picks
D=data/rtx3080_sm86_68sms_10gb
export TUNE_SWEEP_DIR=$D/sweep_screen_heldout TUNE_PICKS=$D/learned_topk_picks.json
python -m tune --clock-lock 1755 benchmark --shapes-file $D/sweep_screen_heldout/subset.json \
  --backends torch,triton:sweep:8,triton:picks,cublaslt:16,cublaslt_deep:20 --cache-modes cold --out-name results_lock1755_topkd.jsonl
python -m tune --clock-lock 1755 benchmark --shapes-file $D/sweep_screen_heldout/subset_lt50us.json \
  --backends torch,triton:sweep:8,triton:picks,cublaslt:16,cublaslt_deep:20 --cache-modes hot --graph-batch 50 \
  --out-name results_lock1755_topkd_graph.jsonl
python -m tune --clock-lock 1755 online --shapes-file $D/sweep_screen_heldout/subset.json --k 5 --order diverse --fresh-cache
python -m tune topk --results results_lock1755_topkd.jsonl --graph-results results_lock1755_topkd_graph.jsonl
```

## Update 2026-10-19 (night): lgbm_rank top-k + short online search — top-3 cuts p95 regret by a third for ~0.4 s per shape

**What ran.** `python -m tune topk` refits lgbm_rank on all 477 tune shapes, using the hyperparameters already chosen by tune CV. It saves the model as a standalone `analysis.learned.Ranker` in
`data/<hw>/lgbm_rank.txt/.json`: a LightGBM booster, an 868-config pool and the GPU numbers the features need. The `Ranker` ranks any (M, N, K) in a median 92 ms. On all 86 held-out shapes its top 5
match the fitted model exactly, and the build fails otherwise. The top 5 per held-out shape went into `learned_topk_picks.json` for `triton:picks`.

The new `python -m tune online` (`tune/online.py`) is the first-use search. It ranks the shape, then compiles and validates the top k from an **empty Triton cache** (`--fresh-cache`). It times
each candidate with 10 cold samples on one placement, re-times them hot with CUDA graphs (20 × 5) when the fastest is under 50 µs, and keeps the fastest.
One k = 5 run stores every candidate's short times and costs, so the choice and cost for every k ≤ 5 are recomputed offline. Each choice is then priced at its full-protocol stage-2 time.

All runs used commit `a24d834`, `harness_dirty: false`, clock locked at 1755 MHz:
- `bench-20261019-083637-lock1755-topk`: cold, 86 shapes, 2,561 rows, 0 invalid, 4.7% power-capped.
- `-topk-graph`: graph-amortized, 56 shapes under 50 µs, 1,685 rows, 0 invalid, 7.3% power-capped.
- `online-20261019-083637-lock1755-topk`: 86 shapes × 5 candidates, 0 invalid.

Tables: [`learned_topk.md`](learned_topk.md).

**Regret vs the per-shape Triton oracle, median / p95 (mean).** The online rows are what the short search actually picked, re-timed with the full protocol:

| selector | cold, all 86 | cold, ≥ 50 µs (30) | graph, < 50 µs (56) | vs B3+, cold all / graph |
|---|---|---|---|---|
| knnf | 2.5% / 22.1% (6.6%) | 2.4% / 36.7% | 4.7% / 26.2% (7.6%) | 0.955 / 0.953 |
| knn5 | 2.5% / 23.9% (9.4%) | 4.2% / 65.1% | 3.8% / 21.8% (6.9%) | 0.964 / 0.973 |
| lgbm top-1 (no search) | 2.0% / 18.3% (4.1%) | 2.2% / 19.1% | 2.3% / 13.5% (4.4%) | 0.968 / 0.993 |
| lgbm online top-2 | 1.2% / 12.5% (3.3%) | 2.0% / 18.3% | 0.4% / 12.2% (2.9%) | 0.971 / 1.002 |
| **lgbm online top-3** | **0.8% / 12.2% (2.9%)** | **1.1% / 16.5% (3.8%)** | **0.3% / 12.2% (2.7%)** | 0.973 / 1.002 |
| lgbm online top-5 | 0.7% / 11.7% (2.8%) | 1.1% / 16.5% (3.3%) | 0.1% / 12.1% (2.2%) | 0.990 / 1.004 |
| lgbm best-of-5, full-protocol timing | 0.1% / 11.2% (2.3%) | 0.7% / 16.4% (3.1%) | 0.0% / 11.7% (2.2%) | 0.990 / 1.004 |

**First-use cost per shape (wall clock, empty Triton cache):**

| k | compile + validate s, median / p90 | short timing s, median | total s, median / p90 |
|---|---|---|---|
| 1 | 0.14 / 0.29 | 0.01 | 0.16 / 0.33 |
| 3 | 0.37 / 0.70 | 0.02 | 0.40 / 0.76 |
| 5 | 0.61 / 1.18 | 0.04 | 0.66 / 1.28 |

- **Top-3 is the sweet spot.** Going from k = 1 to k = 3:
  - Cold mean regret drops from 4.1% to 2.9% and p95 from 18.3% to 12.2%. At ≥ 50 µs the median drops from 2.2% to 1.1%. Under 50 µs (graph) the median drops from 2.3% to 0.3% and p95 from 13.5% to 12.2%.
  - Top-3 beats top-1 by more than 2% on 17 of 86 cold shapes and 12 of 56 graph shapes.
  - k = 4–5 leaves p95 almost unchanged (11.7–12.1%) while costing more.
- **The short protocol is good enough.** Online top-k lands within 0.1–0.7 points (median and mean) of choosing with full-protocol times, so 10 samples per candidate are enough to rank the top k.
  Graph mode triggered on exactly the 56 shapes under 50 µs.
- **Cost is dominated by compilation, and it is small.** A config never compiled before costs a median 0.16 s (p90 0.33 s, max 0.81 s including the ~0.8 s one-time JIT startup). Timing costs
  hundredths of a second. The table's medians are measured in one process, where 186 of 430 candidates reused a kernel compiled for an earlier shape (~0 s). A cold process pays about 3 × 0.16 ≈ 0.5 s
  (median) for k = 3. That is once per new shape, against kernels of 2 µs to 12 ms.
- **The remaining tail is ranker error, not search width.** The worst shapes are the same at k = 1 and k = 5: 14244×16×1257 at +23%, 5398×196×774 at +20%, 320×1×79 at +16–18%. On average the top 5 cover
  only 1.9 distinct (BLOCK_M, BLOCK_N, SPLIT_K) tiles, and on 30 of 86 shapes all five share one tile and differ only in stages, group or BLOCK_K. A wider k mostly re-times near-duplicates.
- **Against B3+:** online top-3 moves from 0.968 to 0.973 cold (all shapes) and from 0.993 to 1.002 under 50 µs graph. Below 50 µs the searched Triton pick now ties cuBLASLt's searched best at the median.
  At ≥ 50 µs G4 still fails.
- knnf / knn5 reproduce again (knnf cold 2.5% / 22.1% / 6.6%; knn5 graph 3.8% / 21.8% / 6.9%).

**New bar: lgbm online top-3.** Cold all: 0.8% / 12.2% (2.9%). ≥ 50 µs: 1.1% / 16.5%. Graph < 50 µs: 0.3% / 12.2% (2.7%). Cost: ~0.4–0.5 s first use per shape.

**Next:** a **diversified top-k** (best-predicted config per distinct tile × split, then by score) to attack the tail that a wider plain top-k cannot reach. Then joint
{Triton ∪ cuBLASLt} candidates in the same search, since under 50 µs the margin over B3+ is now about zero at the median.

Reproduce:
```bash
python -m tune topk               # train + save data/<hw>/lgbm_rank.*, write learned_topk_picks.json
D=data/rtx3080_sm86_68sms_10gb
export TUNE_SWEEP_DIR=$D/sweep_screen_heldout TUNE_PICKS=$D/learned_topk_picks.json
python -m tune --clock-lock 1755 benchmark --shapes-file $D/sweep_screen_heldout/subset.json \
  --backends torch,triton:sweep:8,triton:picks,cublaslt:16,cublaslt_deep:20 --cache-modes cold --out-name results_lock1755_topk.jsonl
python -m tune --clock-lock 1755 benchmark --shapes-file $D/sweep_screen_heldout/subset_lt50us.json \
  --backends torch,triton:sweep:8,triton:picks,cublaslt:16,cublaslt_deep:20 --cache-modes hot --graph-batch 50 \
  --out-name results_lock1755_topk_graph.jsonl
python -m tune --clock-lock 1755 online --shapes-file $D/sweep_screen_heldout/subset.json --k 5 --fresh-cache
python -m tune topk               # re-score with the stage-2 and online files
```

## Update 2026-10-19 (late): learned selectors v0 — per-candidate LightGBM sets a new bar

**What ran.** `analysis/learned.py` (`python -m tune learned`) fits five model families in two forms, on the tune screen only:
- **cls:** shape → config classification over the first P members of a greedy portfolio, refit per fold. A shape's label is the first member within 2% of its best member.
  Inputs are the 22 shape features (dims, fill, align). Models: decision tree, LinearSVC, LightGBM, random forest, logistic regression.
- **rank:** per-candidate scoring over the union of each training shape's top-5 configs (868 candidates). Each row has the 22 shape features, the 7 config parameters and the 6
  `candidate_features`. The trees, the forest and LightGBM regress capped log regret (LightGBM could also pick lambdarank). LinearSVR regresses log regret and logistic regression
  classifies "within 5% of the oracle", both on standardized inputs plus every shape × candidate product.

Hyperparameters come from grouped 20-fold CV on the tune shapes, lowest capped mean regret. Each fit sees 290–353 shapes, and drops every tune shape that shares two of M, N, K with a
fold shape. The share-two-dims graph has one component of 258 shapes, so GroupKFold was not an option. 198 grid points, about 2 h of CPU, cached per model in
`data/<hw>/learned_search.json`. The grids were widened once, on tune CV only, wherever the first pass's winner sat on an edge. No held-out shape informed any fit or choice.

**Re-measured** in fresh sessions on a clean harness (commit `3bbc349`, `harness_dirty: false`). Each session also re-times b6 / knn5 / knnf, so everything shares one oracle and B3+:
- `bench-20261019-032812-lock1755-learned`: cold, 86 shapes, 2,760 rows, 0 invalid, 5.8% power-capped.
- `-learned-graph`: graph-amortized, 56 shapes under 50 µs, 1,815 rows, 0 invalid, 5.5% power-capped.

Tables: [`learned_selectors.md`](learned_selectors.md).

**Regret vs the per-shape Triton oracle, median / p95 (mean):**

| selector | tune CV mean | held-out screen | stage 2 cold, all 86 | cold, ≥ 50 µs (30) | graph, < 50 µs (56) |
|---|---|---|---|---|---|
| knn5 (bar < 50 µs) | 8.4% | 4.7% / 25.4% (10.8%) | 3.0% / 24.2% (9.7%) | 3.6% / 65.3% | 3.7% / 22.1% (6.8%) |
| knnf (bar ≥ 50 µs) | 7.5% | 4.1% / 27.3% (7.8%) | 2.3% / 23.0% (6.6%) | 1.9% / 36.4% | 4.7% / 25.4% (7.5%) |
| dt_cls | 14.6% | 7.8% / 43.9% (15.1%) | 6.4% / 41.4% (14.2%) | 5.3% / 38.3% | 10.8% / 51.9% (17.0%) |
| svm_cls | 13.5% | 8.0% / 53.6% (15.3%) | 6.5% / 51.8% (14.1%) | 5.2% / 57.9% | 11.1% / 57.6% (18.2%) |
| lgbm_cls | 10.1% | 4.1% / 37.7% (11.0%) | 3.0% / 39.0% (9.9%) | 2.3% / 70.1% | 5.1% / 31.8% (8.9%) |
| rf_cls | 9.0% | 4.3% / 32.7% (11.4%) | 3.2% / 32.3% (10.3%) | 1.7% / 28.4% | 5.5% / 52.1% (13.4%) |
| logreg_cls | 11.1% | 7.4% / 52.5% (13.2%) | 5.9% / 51.8% (11.7%) | 3.3% / 38.7% | 10.5% / 71.4% (17.5%) |
| dt_rank | 11.6% | 4.6% / 65.7% (14.3%) | 4.4% / 67.7% (13.2%) | 1.1% / 43.5% | 9.4% / 80.0% (17.4%) |
| svm_rank | 22.4% | 13.7% / 73.5% (21.9%) | 10.5% / 70.4% (19.7%) | 15.9% / 83.7% | 15.1% / 85.5% (21.6%) |
| **lgbm_rank** | **4.8%** | **3.6% / 18.1% (5.5%)** | **1.4% / 18.3% (4.1%)** | 1.8% / 19.1% | **2.4% / 13.5% (4.3%)** |
| rf_rank | 5.3% | 4.4% / 18.5% (6.4%) | 2.9% / **17.3%** (5.1%) | **1.2% / 18.6%** | 3.0% / 13.5% (5.2%) |
| logreg_rank | 22.1% | 8.5% / 63.7% (17.3%) | 5.4% / 64.6% (15.3%) | 6.8% / 81.7% | 8.4% / 53.2% (15.1%) |

lgbm_rank: regression objective, 255 leaves, 1,500 trees, learning rate 0.05. rf_rank: 100 trees, max_features 0.66, min leaf 5. Tune CV for knn5 / knnf uses the same folds,
with their weights fixed as fit on all tune shapes, which slightly favours them.

- **Per-candidate tree ensembles are the first selectors to beat the k-NN bar, and they do it in every regime.** Against knnf in cold stage 2, lgbm_rank's mean drops from 6.6% to 4.1% and its p95 from 23% to 18%.
  At ≥ 50 µs its p95 is 19% against knnf's 36%, about half. Under 50 µs (graph) its mean drops from 6.8% to 4.3% and its p95 from 22% to 13.5% against knn5. The ranking also holds shape by shape:
  - Cold, beyond ε (2% ≥ 50 µs, 5% below): lgbm_rank is faster than knnf on 21 shapes, slower on 10 and tied on 55 (geomean 1.022×). Against knn5 it is 23 / 7 / 57 (1.036×).
  - Graph, ε 5%: 17 / 6 / 33 against knnf and 16 / 10 / 32 against knn5.
  
  Its worst stage-2 shape is 226×2141×1139 at +29%, where knnf's worst is +45%.
- **CV picks the same winner as held-out does.** lgbm_rank has the lowest tune-CV mean (4.8%), rf_rank is second (5.3%). The held-out screen mean (5.5%, 6.4%) and stage-2 cold mean (4.1%, 5.1%) rank them the same, with every other learned selector at ≥ 9.9%. So naming
  lgbm_rank the bar is not a choice made on held-out data.
- **The form matters more than the model.** The same features fed to classifiers over a static label set (cls) stay at a 32–54% held-out p95. That is despite label-set ceilings of 12.5% p95 for P = 32,
  because P has to be small for the labels to be learnable, and a misclassification costs the full regret. Per-candidate scoring lets the model reuse what it learned about one config on others.
- **Linear models fail in both forms**, even with every shape × candidate product (svm_rank / logreg_rank CV about 22%). Regret is not close to linear in these features. Single decision trees overfit (rank p95 66–68%).
- **What the trees use.** In both lgbm_rank and rf_rank, the candidate's own K steps per split and its waves (from `candidate_features`) carry 36–38% of the importance. Next come
  BLOCK_M, wave efficiency and BLOCK_N. The per-candidate features that k-NN voting could not exploit (knnc) do work when a model can weigh them against the shape.
- **Against B3+ nothing changes** (median t_B3+/t_pick, cold ≥ 50 µs): lgbm_rank 0.933, knnf 0.913, Triton oracle 0.964. G4 still fails at ≥ 50 µs. Under 50 µs (graph), lgbm_rank reaches
  0.992, against 0.951 for knnf and 1.017 for the oracle, so a joint {Triton ∪ cuBLASLt} selector is where the remaining margin is.
- **knnf / knn5 reproduce for the fourth session in a row** (knnf cold 2.3% / 23.0% / 6.6% vs 1.8% / 22.9% / 6.5% in `selectors3`; knn5 graph 3.7% / 22.1% vs 3.7% / 22.3%).

**Caveats.** lgbm_rank's winner is again on the edge of its grid (255 leaves, 1,500 trees), but CV gains have flattened (4.9% → 4.8% from 127 to 255 leaves). Lambdarank has the best
CV median (1.6–1.9%) but a worse mean, so it lost under the mean rule. The cold ≥ 50 µs column has only 30 shapes, so its p95 is about the second-worst shape.

**New bar:** lgbm_rank, cold all median 1.4% / p95 18.3% / mean 4.1%, ≥ 50 µs 1.8% / 19.1%, graph < 50 µs 2.4% / 13.5% / 4.3%.

**Next:** top-k from lgbm_rank plus a short online search (time its top 3–5, AGENTS next step 2), which now has a calibrated ranker to draw from. Add cuBLASLt algorithms as
candidates in the same ranker (joint selection). Check the ranker on a lambdarank / regression blend, or try a pool beyond the top 5.

Reproduce:
```bash
python -m tune learned            # CV search (cached in data/<hw>/learned_search.json), held-out picks -> data/<hw>/learned_picks.json
D=data/rtx3080_sm86_68sms_10gb
export TUNE_SWEEP_DIR=$D/sweep_screen_heldout TUNE_PICKS=$D/learned_picks.json
python -m tune --clock-lock 1755 benchmark --shapes-file $D/sweep_screen_heldout/subset.json \
  --backends torch,triton:sweep:8,triton:picks,cublaslt:16,cublaslt_deep:20 --cache-modes cold --out-name results_lock1755_learned.jsonl
python -m tune --clock-lock 1755 benchmark --shapes-file $D/sweep_screen_heldout/subset_lt50us.json \
  --backends torch,triton:sweep:8,triton:picks,cublaslt:16,cublaslt_deep:20 --cache-modes hot --graph-batch 50 \
  --out-name results_lock1755_learned_graph.jsonl
python -m tune learned            # re-score with both stage-2 files
```

## Update 2026-10-19 (later): per-candidate fill and split features (knnc) do not beat knnf

**What changed.** `candidate_features` computes the fill and split features for **each candidate's own tile and split** rather than for fixed
reference tiles. For every (shape, config) it has:
- **waves:** CTAs × SPLIT_K / (68 SMs × resident CTAs per SM), with residency set by shared memory, threads and the 16-CTA limit.
- **last-wave efficiency**, **K steps per split**, **M × N padding** to whole tiles, **K padding** to whole BLOCK_K × SPLIT_K steps, and **EVEN_K**.
- **log2 M, N, K** as shape context.

The new selector **knnc** scores every runnable config on the target. A config's score is its own mean log regret on the k tune shapes where that config saw
the closest conditions, so it can pick a config that was no neighbour's winner. The dims weight and k were fit on tune shapes only, by grouped leave-one-out over
16 settings: dims weight 0.5, k = 10. In-sample that gives mean 6.6%, median 2.5% and p95 25.6%, against knnf's 5.9%, 2.1% and 22.5%. The fit takes about 6 min and is cached in
`data/<hw>/selector_knnc_fit.json` (snapshot without picks in `reports/snapshot/lock1755/`).

**Re-measured** with all seven selectors in fresh sessions on a clean harness (commit `5424930`, `harness_dirty: false`):
- `bench-20261019-003241-lock1755-selectors3`: cold, 86 shapes, 2,526 rows, 0 invalid, 3.4% power-capped.
- `-selectors3-graph`: 56 shapes under 50 µs, 1,670 rows, 0 invalid, 6.1% power-capped.

Tables: [`selector_baselines.md`](selector_baselines.md).

**Regret vs the per-shape Triton oracle, median / p95 (mean):**

| selector | held-out screen | stage 2 cold, all 86 | cold, ≥ 50 µs (30) | graph, < 50 µs (56) |
|---|---|---|---|---|
| knn5 | 4.7% / 25.4% (10.8%) | 2.7% / 22.6% (9.5%) | 3.9% / 65.4% | **3.7% / 22.3%** (**6.7%**) |
| **knnf** | **4.1% / 27.3% (7.8%)** | **1.8% / 22.9% (6.5%)** | **1.9%** / 36.6% | 4.7% / 26.2% (7.3%) |
| **knnc** | 5.3% / 29.3% (10.7%) | 4.4% / 26.7% (9.2%) | 3.8% / **31.9%** | 7.1% / 36.0% (10.5%) |

Against B3+ (median t_B3+ / t_pick): ≥ 50 µs cold, knnc 0.899, knnf 0.916, knn5 0.894 and oracle 0.964. Under 50 µs graph, knnc 0.940, knnf 0.955 and knn5 0.972.

- **knnc is worse than knnf almost everywhere.** Its median is higher in every column, from 1.3× on the screen to 2.4× in cold stage 2, and below 50 µs it is the worst of the three. The one gain is the ≥ 50 µs tail (p95 31.9%
  vs 36.6%), which is about one shape out of 30, and it is still well short of the 48-config portfolio oracle (8.9%).
- **It regresses on the shape that motivated the features.** 2×8901×9643 goes to +120% on the screen. knnc picks `bm16_bn256_bk32_w8_s4_g1_sk1`, whose record on decode shapes with
  similar waves and K steps is good, but here it is 2.2× slower than the oracle.
- **The cause is not the size of the candidate set.** I checked two explanations as diagnostics on held-out data; neither was used for fitting:
  - Restricting knnc to the 48-config portfolio (25 candidates per shape) or to the 284 tune-oracle configs (144 per shape) does not help: screen p95 31% / 38%.
  - Its picks are not over-optimistic: predicted median regret 6.2% vs actual 5.3%. Only the tail is under-predicted (mean 7.6% predicted vs 10.7% actual).
  
  So it is the transfer itself that fails. A config's regret on tune shapes with similar waves, K steps and padding does not predict its regret on a new shape well enough.
  The ranking among near-equal configs depends on things these features omit, such as memory- vs compute-boundness, L2 reuse and per-warp work.
- **knnf and knn5 reproduce across sessions.** knnf cold is 1.8% / 22.9% (6.5%) here vs 2.7% / 22.9% (6.4%) on `selectors2`, and knn5 2.7% / 22.6% vs 2.9% / 23.7%. Medians differ by under 1 point
  and p95 by about 1 point.

**The bar is unchanged:** knnf (cold ≥ 50 µs median 1.9%, p95 37%) and knn5 (graph < 50 µs median 3.7%, p95 22%). The per-candidate features stay in the code as inputs
for a trained ranking model (next step 1 in `AGENTS.md`), which can weigh them against shape features rather than averaging neighbours. Nearest-neighbour voting over them is not enough.

Reproduce: the 2026-10-18 commands with `--out-name results_lock1755_selectors3[_graph].jsonl`, then `python -m tune selectors`, which scores those files by default.

## Update 2026-10-19: selector baselines rebuilt with GPU-fill and alignment features

**What changed.** The k-NN distance now uses derived shape features beyond (log2 M, log2 N, log2 K). The groups are built in `shape_features` and weighted in `embed`:
- **fill:** for 6 reference tiles (16×64 … 128×128), log2(CTAs / 68 SMs) and wave efficiency, which is CTAs / (waves × SMs) and is low when the last wave is nearly empty.
  It also includes log2 of the K steps per split if a 64×64 grid were split to fill the SMs.
- **align:** whether M, N and K are divisible by 16 and by 8.

The group weights and k come from a grid of 120 settings, scored only on the tune shapes with **grouped leave-one-out**: a tune shape may not use any neighbour that shares two
of M, N, K with it. Plain leave-one-out let S2's M sweeps over one layer act as near-copies of each other. The best setting gives **knnf** (fill 1.5, align 0.5,
k = 3), and the best k = 1 setting gives **nn1f** (fill 0.5, align 0.5). Both were fixed before any held-out shape was scored. The four earlier selectors and their picks are unchanged.
I chose the feature groups after seeing the 2026-10-18 held-out misses, which were split-K and wave choices, but no weight or k was fit on held-out data.

**Re-measured** in two new sessions with the same protocol as 2026-10-18, so all six selectors share each session's oracle and B3+:
- `bench-20261018-235334-lock1755-selectors2`: cold, 86 shapes, 2,459 rows, 0 invalid.
- `-selectors2-graph`: graph-amortized, 56 shapes under 50 µs, 1,625 rows, 0 invalid.

5.0% and 4.4% of rows hit the power cap. Rows record `harness_dirty: true` because the feature code was uncommitted.
Tables: [`selector_baselines.md`](selector_baselines.md).

**Regret vs the per-shape Triton oracle, median / p95 (mean):**

| selector | held-out screen | stage 2 cold, all 86 | cold, ≥ 50 µs (30) | graph, < 50 µs (56) | tune, grouped LOO |
|---|---|---|---|---|---|
| nn1 | 5.4% / 132% (30.7%) | 3.8% / 129% (29.0%) | 1.5% / 194% | 9.3% / 113% (26.5%) | 4.9% / 53% |
| **nn1f** | 4.5% / 40% (10.2%) | 3.5% / **40%** (8.8%) | 1.5% / 40% | 8.2% / 41% (13.0%) | 2.9% / 30% |
| knn5 | 4.7% / 25.4% (10.8%) | 2.9% / 23.7% (9.4%) | 3.7% / 65% | **3.5% / 20.9%** (6.5%) | 2.8% / 29% |
| **knnf** | **4.1%** / 27.3% (**7.8%**) | **2.7% / 22.9% (6.4%)** | **1.9% / 37%** | 3.8% / 26.7% (7.3%) | 2.1% / 22.5% |
| rule | 10.3% / 40.5% | 7.4% / 37.1% (13.2%) | 4.7% / 37% | 12.9% / 58.7% (18.4%) | 7.5% / 44% |

**Against B3+, median t_B3+ / t_pick:** at ≥ 50 µs (cold), the Triton oracle is 0.964, knnf **0.922**, nn1f 0.927, knn5 0.896 and rule 0.879. Below 50 µs (graph), the oracle is 1.018,
knn5 0.975 and knnf 0.952.

- **The features fix 1-NN's catastrophic misses.** Its p95 drops from 129% to 40% (cold) and from 113% to 41% (graph). 14244×16×1257 goes from +466% to +46%, and 2×8901×9643 from +95%
  to +9%.
- **For k-NN they help on the larger kernels and not below 50 µs.** At ≥ 50 µs, knnf halves the median (3.7% → 1.9%) and the p95 tail (65% → 37%), and moves from 0.896 to
  0.922 of B3+. The two motivating shapes are fixed: 2×8901×9643 goes from +95% to +9% and 28×3030×4012 from +272% to +17%, both now with split-K picks. Below 50 µs knnf is
  slightly worse than knn5 (graph p95 26.7% vs 20.9%, mean 7.3% vs 6.5%). Across all 86 shapes it is a modest gain: mean 9.4% → 6.4%, p95 23.7% → 22.9%.
- **The tail is not closed.** p95 is still about 23% overall. The worst ≥ 50 µs shapes for knnf are now different ones: 68×6420×10359 +46% (knn5 +29%), 79×15624×2135 +40% and
  443×4723×393 +32%. These features stop the selector from transferring a config that is wrong for the target's grid, but they do not make it find the winner. With only
  shape-level features, every neighbour vote still ranks whole configs.
- **No selector differentiates from B3+ at ≥ 50 µs**, and the oracle itself cannot (0.964). That is unchanged.

**The bar for a learned selector** is now knnf on large kernels (cold ≥ 50 µs: median 1.9%, p95 37%) and knn5 on small ones (graph: median 3.5%, p95 21%), with an
overall p95 of about 23% and mean of about 6%. The next step is unchanged: per-candidate features (the fill and K-step features computed for each candidate's own tile and split,
not for fixed reference tiles), so a model can score configs rather than vote between neighbours' winners.

Reproduce: the commands at the end of the 2026-10-18 section with `--out-name results_lock1755_selectors2[_graph].jsonl`; `python -m tune selectors`
now scores those files by default (pass `--results results_lock1755_selectors.jsonl --graph-results results_lock1755_selectors_graph.jsonl` for the first run).

## Update 2026-10-18: trivial selector baselines on held-out shapes (plan 8.1 item 7)

**What ran.** Three selectors, each mapping (M, N, K) to one Triton config. All were fit only on the 477-shape tune screen and scored on the 86 held-out shapes
(`analysis/selectors.py`, `python -m tune selectors`):
- **nn1**: the nearest tune shape in (log2 M, log2 N, log2 K), Euclidean distance; use that shape's fastest config.
- **knn5**: the 5 nearest tune shapes; use the config with the lowest mean log regret over them.
- **rule**: a hand-written rule (`rule_config`). BLOCK_M follows M and BLOCK_N follows N. Split-K doubles while the grid has fewer CTAs than the 68 SMs.
  Warps follow the tile. I wrote it while looking only at the tune-split oracle configs (5 iterations), then froze it.
- **b6** (best single tune config) is included as the floor.

Every pick passes the pruning rules for its target shape, so a neighbour's best config is skipped when it is pruned on the target. The new backend
`triton:picks` times the picks (`data/<hw>/selector_picks.json`; snapshot in `reports/snapshot/lock1755/selector_picks.json`).
- **Stage 2, cold:** session `bench-20261018-230423-lock1755-selectors` (10 min, 2,375 rows, 0 invalid). It covered all 86 held-out shapes, each timed over
  its 8 best screen tiles, every selector pick, B1, B3 (`cublaslt:16`) and B3+ (`cublaslt_deep:20`), all in one session.
- **Stage 2, graph:** session `bench-20261018-230423-lock1755-selectors-graph` (5 min, 1,567 rows, 0 invalid). It ran the same backends, hot and CUDA-graph
  amortized (`--graph-batch 50`), on the 56 held-out shapes whose screen oracle is under 50 µs.
- Both rows record `harness_dirty: true`, because the `triton:picks` backend was not yet committed when they ran; it is committed with this update.
  3.3% (cold) and 5.1% (graph) of rows hit the power cap, all on the largest shapes in each set.

Tables: [`selector_baselines.md`](selector_baselines.md).

**Regret vs the per-shape Triton oracle, median / p95:**

| selector | held-out screen (full-space oracle) | stage 2 cold, all 86 | stage 2 cold, ≥ 50 µs (29) | stage 2 graph, < 50 µs (56) | in-sample (tune, leave-one-out), screen |
|---|---|---|---|---|---|
| b6 | 48.1% / 487% | 48.8% / 484% | 74.8% / 327% | 24.7% / 455% | 68.4% / 343% |
| nn1 | 5.4% / 132% | 3.6% / 129% | 1.1% / 123% | 9.2% / 116% | 4.1% / 60% |
| **knn5** | **4.7% / 25.4%** | **2.1% / 23.7%** | 3.1% / 69.0% | **3.6% / 21.7%** | 2.3% / 23.2% |
| rule | 10.3% / 40.5% | 7.4% / 37.4% | 4.7% / 37.9% | 13.0% / 59.2% | 7.5% / 44.3% |

On the screen, the k-NN neighbourhood size matters for the tail: k = 3 gives p95 43%, k = 5 25%, k = 10 25% (median 5.0–5.4% for all three).

**Against B3+ (t_B3+ / t_pick, > 1 = the pick is faster), median [p10, p90], and shapes ahead by > 5% / behind by > 5%:**

| | cold, ≥ 50 µs (29) | cold, < 50 µs (57) | graph, < 50 µs (56) |
|---|---|---|---|
| Triton oracle (top-8 screen tiles + picks) | 0.954 [0.83, 1.02] · 2 / 14 | 1.033 [0.83, 1.16] · 24 / 19 | 1.017 [0.77, 1.18] · 24 / 20 |
| knn5 | 0.897 [0.64, 1.01] · 2 / 18 | 1.000 [0.74, 1.14] · 20 / 24 | 0.977 [0.65, 1.17] · 20 / 25 |
| nn1 | 0.920 [0.51, 1.01] · 2 / 18 | 0.952 [0.52, 1.12] · 15 / 29 | 0.898 [0.47, 1.13] · 17 / 33 |
| rule | 0.877 [0.67, 0.98] · 2 / 18 | 0.940 [0.68, 1.11] · 12 / 32 | 0.880 [0.53, 1.11] · 13 / 35 |
| b6 | 0.566 [0.27, 0.77] · 0 / 28 | 0.821 [0.22, 1.03] · 3 / 39 | 0.800 [0.22, 1.02] · 0 / 39 |

- **k-NN-5 is the baseline to beat.** It is the best on almost every column, with a held-out median regret of 2–5% and p95 of 22–25%. That is far better than any fixed
  config (b6 median 48%), but the tail is roughly 3× worse than the 48-config portfolio oracle (8.9% p95, stage 2). A learned selector has to close the gap
  between "pick 1 of 10,840 from the neighbours" (p95 ≈ 24%) and "best of the 48-portfolio" (≈ 9%).
- **1-NN has a good median and a bad tail.** It is often exactly right (stage-2 median 1.1% at ≥ 50 µs), but when the neighbour's winner does not transfer it fails badly
  (14244×16×1257: +469%, 37×922×10110: +135%). Averaging over 5 neighbours fixes most of those. nn1 and knn5 agree on only 7% of shapes.
- **The hand rule is robust but imprecise.** Its median is 2–4× knn5's. At ≥ 50 µs its tail is better than knn5's (38% vs 69% p95; with 29 shapes, p95 is the
  2nd-worst shape, so this is one shape's difference), and below 50 µs it is worse. Out-of-sample regret matches in-sample regret (7.5% → 7.4%), so it did not overfit the tune shapes. It also never wins by luck.
- **The worst misses are split-K and wave-quantization decisions that (log M, log N, log K) cannot express.** 2×8901×9643 (255 µs, +95% for both knn5 and rule)
  wants `bn64 … sk2`. A BLOCK_N = 128 tile gives 70 CTAs on 68 SMs, which is 2 waves with the second one nearly empty. 28×3030×4012 (+277% for knn5) wants split-K 4,
  but its neighbours picked split-K 1. 68×6420×10359 wants split-K 4 as well. The obvious next features are the tile and wave counts per candidate tile
  (ceil(M/BM)·ceil(N/BN) / SMs) and K steps per split.
- **Against B3+, no trivial selector differentiates at ≥ 50 µs.** Even the Triton oracle is behind (median 0.954, ahead of B3+ by > 5% on 2 of 29), consistent
  with G4 at ≥ 50 µs on the tune shapes. knn5 loses another ~6% on top (0.897). Below 50 µs, knn5 is at parity with B3+ (cold 1.000, graph 0.977) and ahead on
  20 of 56 shapes, but B3+ is ahead on just as many.

**Gate implications.** G3 holds out of sample for selection: even the simplest selector recovers most of the per-shape value (knn5 median 2–5% vs b6 48%).
The selector problem is the tail. The bar for a learned model is set: on held-out shapes it must beat **knn5 at p95 ≈ 24% / median ≈ 2–5% regret**, and to be
useful against cuBLAS it must pick from {Triton candidates ∪ cuBLASLt configs}, since neither family dominates.

**Next steps.**
1. Learned selector v0: rank candidate configs per shape, using shape features plus per-tile derived features (CTA count, waves, K steps per split, padding waste).
   Evaluate it with exactly this harness (`triton:picks` + `python -m tune selectors`) against knn5.
2. Top-k + short online search: time the selector's top 3–5 at first use. This tests whether the tail closes once the model only has to get the winner into a short list.
3. Joint {Triton, cuBLASLt} selection at < 50 µs, where the two families split the wins.

```bash
python -m tune selectors        # writes data/<hw>/selector_picks.json and scores the screen (and stage 2 if present)
D=data/rtx3080_sm86_68sms_10gb
TUNE_SWEEP_DIR=$D/sweep_screen_heldout python -m tune --clock-lock 1755 benchmark --shapes-file $D/sweep_screen_heldout/subset.json \
  --backends torch,triton:sweep:8,triton:picks,cublaslt:16,cublaslt_deep:20 --cache-modes cold --out-name results_lock1755_selectors.jsonl
TUNE_SWEEP_DIR=$D/sweep_screen_heldout python -m tune --clock-lock 1755 benchmark --shapes-file $D/sweep_screen_heldout/subset_lt50us.json \
  --backends torch,triton:sweep:8,triton:picks,cublaslt:16,cublaslt_deep:20 --cache-modes hot --graph-batch 50 --out-name results_lock1755_selectors_graph.jsonl
python -m tune selectors        # re-score with both stage-2 files
```

## Update 2026-10-16: 3× more tune shapes does not fix held-out G5; the tail is not a coverage problem

**What ran.** Option 1 from the previous update: build the portfolio from more tune shapes.
- **Tune screen, all eligible shapes:** session `sweep-tune2-20261015-235431-lock1755` (`python -m tune sweep --per-suite 1000`, 4.5 h).
  It screened every tune-split shape up to 1.2 TFLOP: **477 shapes**, the original 160 (reused from cache) plus 317 new
  (S7 103, S2 84, S6 66, S3 36, S5 22, S4 6). It used the same space and protocol as before: 2,558,593 cells, 0 invalid, 0 crashes. The subset is in
  `reports/snapshot/lock1755/sweep_subset_477.json`. The 160-shape summaries are kept as `*_160.json` in that directory.
- **Stage-2 confirmation:** session `bench-20261016-042636-lock1755-heldout477` (20 min, 5,730 rows, all valid). It re-timed the 86 held-out shapes over
  each shape's 8 best screen tiles plus the **first 60** members of the new portfolio (`triton:sweep:8,triton:portfolio:60`). The portfolio it ran with
  is `data/<hw>/results_lock1755_heldout477.portfolio.json`. 6.5% of rows hit the power cap, on 10 shapes, mostly the largest. One of them, 4065×8842×28, is among the worst.

Tables: [`sweep_portfolio.md`](sweep_portfolio.md), [`sweep_portfolio_heldout.md`](sweep_portfolio_heldout.md)
(`python -m tune portfolio --heldout --results results_lock1755_heldout477.jsonl`).

| p95 regret (mean) | 160-shape portfolio | **477-shape portfolio** |
|---|---|---|
| in-sample size for p95 < 5% | 29 | **48** |
| held-out screen, 29 configs | 17.0% (4.2%) | 13.0% (3.9%) |
| held-out screen, 60 configs | 10.3% (3.1%) | 11.2% (3.1%) |
| held-out screen, 100 configs | 9.3% (2.4%) | 11.1% (2.5%) |
| held-out stage 2, 29 configs | 11.0% (2.2%) | 11.4% (2.4%) |
| held-out stage 2, 48 configs | – | **8.9% (1.8%)** |
| held-out stage 2, 60 configs | – | 8.3% (1.5%) |

Stage-2 oracles differ between the two runs: each is the fastest of that run's candidates, and the new run has 60 portfolio members instead of 29. Compare the 29-config rows only roughly.

- **G5 still fails out of sample.** The best stage-2 result is 8.3% p95 with 60 configs. In-sample, 3× the shapes needs a larger portfolio for the same bar (29 → 48 configs).
  The set cover at ε = 5% grows from 32 to 59 configs.
- **More tune shapes barely help.** A learning curve on screen data (greedy fit on random tune subsets, mean of 5 seeds; held-out p95 at 29 / 48 / 60 configs)
  gives 80 shapes: 15.7 / 12.5 / 12.3%, 160: 14.8 / 13.9 / 12.7%, 320: 13.9 / 12.9 / 11.3%, 477: 13.0 / 12.2 / 11.2%. A portfolio fit only on
  the 127 S7 tune shapes (the same distribution as `heldout_random`) is no better (14.0 / 13.3 / 13.3%).
- **This rejects the coverage explanation from the previous update.** Allowing *every* config that is the screen oracle on some tune shape (284 distinct configs)
  still leaves held-out screen p95 at **7.7%**. The configs that win on the tail held-out shapes are not the best config on any tune shape. Only the
  top-5 union (896 configs) gets to 4.0%. The held-out tail is idiosyncratic: many configs sit within a few % of each other, and which one wins on a given shape
  does not repeat across neighbouring shapes.
- **The tail is a handful of shapes.** p95 over 86 shapes is about the 5th-worst shape, and over the 30 shapes ≥ 50 µs it is the 2nd-worst. At 60 configs (stage 2) the
  worst shapes are 93×2101×22 at +16% (5.3 µs, about one event tick), 417×512×6739 at +15% (`heldout_region`, outside the tuned K range), 37×922×10110 at +10% (region),
  4065×8842×28 at +9% (power-capped) and 51×324×5709 at +9%. Several sit at the limit of what cold per-launch timing can resolve.
- **The typical shape is fine.** Mean stage-2 regret is 1.5–1.8% at 48–60 configs.

**Implication for the selector.** Static portfolio coverage plateaus at about 8–9% p95 out of sample whatever the number of tune shapes, so option 1 is closed.
The remaining levers are at selection time: option 3, a short online search over the top-k predicted configs, or a larger candidate set (hundreds of configs)
that the selector ranks rather than a fixed portfolio. The top-5 union result says such a candidate set exists; whether a model can rank it well enough is the
next experiment. G5 as written (a static portfolio of ≤ 50 with p95 < 5%) should be reported as failed
out of sample, with mean regret under 2%.

## Update 2026-10-15 (night): held-out G5 check; the 29-config portfolio does not generalize to p95 < 5%

**What ran.** The portfolio, B6 and the set covers were fit only on the 160 tune-split screen shapes. They were then scored on shapes they never saw:
- **Held-out screen:** session `sweep-heldout-20261015-210132-lock1755` (`python -m tune sweep --split heldout`, 1.3 h). It covered all
  **86 held-out shapes** up to 1.2 TFLOP: 55 `heldout_random` (S7) and 31 `heldout_region` (K between 9000 and 12000, or M between 300 and 450; from S2, S6 and S7).
  It used the same full pruned space and screen protocol as the tune screen: 307,648 cells, 0 invalid, 0 crashes. The data is in `data/<hw>/sweep_screen_heldout/`,
  kept apart from the tune screen; the subset is in `reports/snapshot/lock1755/sweep_subset_heldout.json`.
- **Stage-2 confirmation:** session `bench-20261015-222149-lock1755-heldout` (10 min, 3,102 rows, all valid). Each held-out shape was re-timed with
  the full cold protocol over its 8 best screen tiles plus the 29 portfolio members (`triton:sweep:8,triton:portfolio:29`). The oracle is the fastest
  of those. 7.7% of rows hit the power cap (throttle 0x4); all of them are on the 11 largest shapes, and none is among the worst-regret shapes.

Tables: [`sweep_portfolio_heldout.md`](sweep_portfolio_heldout.md) (`python -m tune portfolio --heldout`).

### G5 out of sample

| 29-config portfolio | p95 regret | mean regret | shapes within 5% of oracle |
|---|---|---|---|
| tune shapes (in-sample, screen) | 5.0% | 1.4% | 95% |
| held-out, screen vs full-space oracle | **17.0%** | 4.2% | 76% |
| held-out, stage 2 (re-measured) | **11.0%** | 2.2% | 84% |
| held-out ≥ 50 µs, stage 2 | 9.7% | 2.3% | 83% |
| held-out `heldout_region`, stage 2 | 8.9% | 2.6% | 77% |
| held-out `heldout_random`, stage 2 | 11.9% | 1.9% | 87% |

- **G5 fails out of sample.** Following the tune-fitted greedy order, held-out p95 regret stays above 5% even at 100 configs on screen data (9.3%).
  In stage 2, 29 configs give 11.0%. The two numbers bracket the truth: the screen oracle is the minimum over about 10,000 noisy few-sample times,
  so it is biased low and regret is overstated. The stage-2 oracle only sees 37 candidates, so regret is understated.
- **The typical shape is fine; the tail is the problem.** Median regret is 0–1%, mean 2.2%. About 16% of held-out shapes (14 of 86) are more than 5% off.
- **This is overfitting, not a harder shape set.** A greedy portfolio re-fit on the held-out shapes themselves needs 31 configs for p95 < 5%,
  almost the same as the in-sample 29. Held-out shapes are not intrinsically harder. Their best configs are just not in a portfolio chosen
  from 160 shapes: the tail of the greedy list mostly memorizes individual tune shapes.
- **The worst shapes** (stage 2) are skinny shapes (one of M or N small) with K over 1000, where the right split-K and tile combination is missing.
  Examples: 226×2141×1139 at +19.5%, 15×3363×2632 at +13.2%, 20×1×11869 at +12.5%, 14244×16×1257 at +11.4%, 37×922×10110 at +9.6%.
  The largest regret is 124×28×26 at +21.9%, but that is only 0.9 µs on a 4.1 µs kernel, inside the cold event tick.
- **G3 holds out of sample.** B6 (`bm32_bn32_bk32_w4_s5_g1_sk1`) has median regret 48% and p95 487% on held-out shapes, so per-shape selection is still worth a lot.

**Implication for the selector.** A fixed small portfolio with a perfect picker is not enough to stay within 5% at the tail on unseen shapes.
Options, cheapest first:
1. Build the portfolio from more tune shapes. The screen is about 55 s per shape, so a few hundred more tune shapes is an overnight run.
2. Allow a larger portfolio (about 60–100), since the selector only needs to rank members, not cover the space.
3. Fall back to a short online search over the top-k predicted configs for shapes outside the training distribution.

Option 2 can be checked now: on screen data, held-out p95 is 10.3% at 60 configs and 9.3% at 100. So size alone does not close
the gap; coverage of shape space does.

## Update 2026-10-15 (evening): full Triton config sweep (M3/M4); G2, G3 and G5 pass, G4 only for tiny GEMMs

**Stage 1 (screen):** session `sweep-20261015-075746-lock1755` (`python -m tune sweep`, 4.1 h). It covered a coarse subset of
**160 shapes**: about 23 per suite from S1–S7, tune split only, at most 1.2 TFLOP per shape, stratified by FLOPs
(`reports/snapshot/lock1755/sweep_subset.json`). Every config in the pruned space was tried on each shape: up to 10,840 per shape,
**912,913 shape × config cells** in total. Each cell was compiled, validated against the fp32 reference and screen-timed with a cold L2
(1 sample, cut at 1.5× the best so far, otherwise 5 samples; 15 samples with a trimmed mean under 50 µs). Results:
0 invalid configs, 0 crashes. Per-shape files are in `data/<hw>/sweep_screen/`.

**Stage 2 (full protocol):** session `bench-20261015-075746-lock1755-sweep` (38 min, 9,226 rows, all valid). For each shape, the 8 best
distinct tiles from the screen were timed alongside the 5 hand-picked configs, B1, B5, B2/B3 and B3+, in one session.
Tables: [`week1_results_lock1755_sweep.md`](week1_results_lock1755_sweep.md).

**Stage 2b (µs kernels):** the 74 subset shapes whose best kernel runs under 50 µs were re-timed with **CUDA-graph amortized timing** (hot, 50 launches
per graph, `--graph-batch 50`), as the methodology prescribes for kernels below the event tick. Session `bench-20261015-125953-lock1755-sweep-graph`, 2,259 rows, all valid.
Tables: [`week1_results_lock1755_sweep_graph.md`](week1_results_lock1755_sweep_graph.md).

### Headline (cold L2, per-launch, 160 shapes)

| | median | p10 | p90 | min | max |
|---|---|---|---|---|---|
| ours (sweep oracle) vs **B3+** | **0.988** | 0.85 | 1.07 | 0.69 | 1.27 |
| sweep oracle vs the 5 hand-picked configs | **1.075** | 1.00 | 2.20 | 1.00 | 4.72 |
| ours vs B1 | 1.075 | 0.95 | 1.36 | 0.78 | 1.94 |
| B3+ vs B1 | 1.078 | 1.01 | 1.36 | 0.99 | 1.95 |

The sweep's gain over the hand-picked configs is largest in S4 large-K (median 2.16×, from split-K variants), S3 skinny (1.26×) and S7 random (1.16×).

### Where ours stands against B3+

| kernel size | timing | n | median ours/B3+ | ours ahead >5% | within 2% | ours behind >5% |
|---|---|---|---|---|---|---|
| ≥ 50 µs | cold, per-launch | 86 | 0.981 | **0** | 39 | 28 |
| ≥ 50 µs, all of M, N, K divisible by 16 | cold, per-launch | 50 | 0.988 | 0 | | |
| ≥ 50 µs, some dimension unaligned | cold, per-launch | 36 | **0.904** | 0 | | |
| < 50 µs | hot, graph-amortized | 74 | 0.979 | 22 | 13 | 25 |
| B3+ < 5 µs | hot, graph-amortized | 20 | 1.087 | **12** | | 4 |
| B3+ 15–50 µs | hot, graph-amortized | 17 | 0.954 | 0 | | 6 |

- **For kernels at or above 50 µs (the range G1 covers), ours never beats B3+ by more than 5%.** On aligned shapes we are at parity (0.988). All 10 worst
  losses are unaligned shapes (3000³ 0.75×, 4088³ 0.76×, 4094³ 0.78×, 8191³ 0.84×, and S5/S7 odd shapes). This is a kernel-level limitation, not a
  selection one: with row strides that are not 16-byte aligned, Triton cannot issue vectorized 16-byte loads. A fix needs a different kernel
  (padding or a copy-to-aligned prologue), not better config choice.
- **Ours does win on tiny GEMMs (B3+ under about 5 µs).** With graph-amortized timing (the event tick does not limit it), ours is 1.08–1.23× faster
  on 12 of 20 such shapes: 64³, 128³, 256³, 127×2039×7, 4093×7×7 and seven S7 shapes with K ≤ 449. The absolute savings are only
  0.23–0.59 µs per call. Just above the band (5–15 µs) a few more wins persist: 8–16 × 1024 × 1024 at 1.11–1.12×, 1024×64×1024 at 1.14×, 16 × {128, 256} × 4096
  with split-K at 1.12–1.14×, and 518×813×87 at 1.20×. Losses there are just as common. The likely cause is cuBLAS's fixed per-kernel cost on problems
  that use only a handful of CTAs.
- **Losses under 50 µs** cluster in large-K split-K with tiny M and N (14×2×5131 0.63×, 7×7×2039 0.80×, 16×64×16384 0.85×). Our split-K writes fp32 partials
  and runs a separate reduction kernel, while cuBLAS uses a fused or atomic reduction. That is also a kernel-level fix.

### Portfolio study (G3, G5) from the screen data

From [`sweep_portfolio.md`](sweep_portfolio.md) (`python -m tune portfolio`). Regret is measured against the per-shape screen oracle.
- **G3, best single config over the full space:** B6 = `bm32_bn32_bk32_w4_s5_g1_sk1`. Its regret vs the per-shape oracle is median **74%**, p90 307%.
  In stage 2, the hand-picked B6 vs the sweep oracle has median 1.23×, p90 6.2×. Per-shape selection is clearly worth it.
- **G5, portfolio size:** a greedy portfolio of **29 configs** reaches p95 regret under 5% (mean 1.3%). 16 configs give a p95 of 8.7%, 50 configs 2.0%.
  Set cover needs 32 configs at ε = 5%, 43 at 3% and 54 at 2%.
- **Which parameters matter**, ranked by the regret from forcing each one to its worst value: SPLIT_K (121%) ≫ BLOCK_M (48%) > BLOCK_K (26%) >
  BLOCK_N (19%) ≫ num_warps (8%), num_stages (5%), GROUP_M (2%). The selector's features should capture the split-K and tile-size
  decisions. Stages and GROUP_M barely matter.
- **Caveats:** this is in-sample: the portfolio is chosen and evaluated on the same 160 tune-split shapes. Screen times are few-sample, so the regrets
  are slightly pessimistic. The held-out check is still to do: screen the held-out shapes, then evaluate the tune-shape portfolio on them.

### Gate status (supersedes earlier tables)

| Gate | Status |
|---|---|
| G1 measurement validity | **Pass** for kernels ≥ 50 µs, cold per-launch. The graph-amortized mode used for < 50 µs has no across-session noise study yet |
| G2 family ≥ 90% of B3+ | **Pass**: median 98.8% over 160 shapes from all 7 suites (aligned shapes at or above 50 µs: 98.8%; unaligned: 90.4%) |
| G3 selection value | **Pass**: full-space B6 regret median 74%, p90 307% (the gate asks for > 10% median or > 30% p90) |
| G4 differentiation | **Fails at or above 50 µs** (0 of 86 shapes beat B3+ by > 5%). **Narrow pass for tiny GEMMs** (B3+ < ~5 µs: 12 of 20 shapes, 8–23%, 0.23–0.59 µs absolute), pending a repeat session in graph mode |
| G5 tractability | **Pass (in-sample)**: 29 configs reach p95 regret < 5%. Held-out validation pending |

### Conclusion (plan §8: written down explicitly, because G4 is mostly failing)

On the RTX 3080, a well-searched cuBLASLt (B3+) is at least as fast as the best of our Triton family for every GEMM large enough
to matter (at least 50 µs). The family reaches parity on aligned shapes and loses about 10% on unaligned ones. The only regime where we beat B3+ is tiny
GEMMs under about 5 µs, by fractions of a microsecond. That matters only where launches are amortized (CUDA graphs) and such GEMMs dominate.
So **the project is not viable as a cuBLAS competitor on this GPU. It is viable as a selection project:** G3 and G5 both pass strongly
(selection is worth 74% at the median over any single config, and about 30 configs suffice). The practical framing that follows is a selector that
picks from {our portfolio ∪ cuBLASLt configs} (B3+ is itself a selection over a config space), or a non-NVIDIA backend target where no B3+ exists.

### Next steps
1. **Held-out G5 check:** screen the held-out S1/S2/S6/S7 shapes and evaluate the 29-config tune-split portfolio on them.
2. **Trivial selector baselines (plan 8.1.7):** nearest-neighbour on (log M, log N, log K) and a hand-written rule, evaluated on held-out shapes
   against the oracle and against B3+.
3. **Repeat the graph-mode µs measurement in a second session** (and after a reboot) to establish its noise floor before relying on the tiny-GEMM wins.
4. Kernel-level fixes, if differentiation still matters: an aligned-copy or padded path for unaligned strides, and a fused or atomic split-K reduction.

## Update 2026-10-15 (later): deeper B3 (B3+) closes the K = 4096 gap; G4 does not hold with 5 configs

Run: `bench-20261015-023430-lock1755-b3plus` (standard conditions, commit `08e2a2c`, 5624 rows, all valid), in
`results_lock1755_b3plus.jsonl`. It re-measures B1, B5, ours, B2/B3 and B3+ in the same session. Full tables:
[`week1_results_lock1755_b3plus.md`](week1_results_lock1755_b3plus.md).
**B3+** = the best of B3 and the 8 fastest configs from an enumerate-and-screen search over cuBLASLt's algorithm space
(algo × tile × stages × split-K × reduction × swizzle × custom; `baselines/cublaslt_search.py`). Each search ran in its own process with a 20 s budget per shape.
Every shape was screened exhaustively except 33 large ones (median 27% of valid configs screened, minimum 1% on 16384³; the heuristic's seeds were always included).

### Headline (cold L2, 143 shapes)

| | median | p10 | p90 | min | max |
|---|---|---|---|---|---|
| ours (5-config oracle) vs **B3+** | **0.959** | 0.83 | 1.01 | 0.66 | 1.15 |
| ours vs B3 (same session) | 0.991 | 0.88 | 1.08 | 0.66 | 1.36 |
| **B3+ vs B3** (what the deeper search buys) | 1.028 | 1.00 | 1.15 | 1.00 | **1.67** |
| B3+ vs B1 | 1.055 | 1.00 | 1.44 | 1.00 | 1.67 |

| regime (cold) | ours/B3+ median | B3+/B3 median | ours > B3+ by >5% | within 2% | ours < B3+ by >5% |
|---|---|---|---|---|---|
| S1 square (22) | 0.921 | 1.011 | 3 (64³, 128³, 256³; all < 7 µs) | 1 | 14 |
| S2 decode, M ≤ 16 (55) | 0.989 | 1.028 | **0** | 33 | 13 |
| S2 small batch, 32–128 (33) | 0.901 | 1.060 | **0** | 5 | 25 |
| S2 prefill, M ≥ 512 (33) | 0.957 | 1.026 | **0** | 0 | 10 |

Hot L2 gives the same picture: overall 0.947, with a single win over B3+ (128³, under 5 µs).

- **The K = 4096 small-M wins were cuBLAS heuristic misses.** On 2–16 × {11008, 12288} × 4096, ours/B3 was 1.07–1.17. Against
  B3+ it is **1.000–1.007**, i.e. a tie. The deep search finds split-K / tile configs that the heuristic never offers (e.g.
  8×11008×4096: B3 159 µs → B3+ 136 µs, ours 136 µs). The 128×1024×4096 split-K win (1.34× over B3) turns into a 0.80× loss: B3+ is
  1.67× faster than B3 there.
- **No regime survives G4 with the 5 hand-picked configs.** The only shapes where we beat B3+ by more than 5% are µs-scale squares, which G1 does not cover
  (per-launch timing has ~1 µs granularity) and which do not count.
- **G2 still passes, with less margin.** Median 95.9% of B3+. The weakest regime is small batch (median 0.901, right at the bar).
  Prefill sits at a uniform ~4% gap (0.93–0.97), and B3+ was budget-limited on most of those shapes, so the true gap may be slightly larger.
- **Largest remaining losses:** non-multiple-of-16 squares (1000³, 1023³, 1025³ at 0.66–0.71; 3000³ at 0.67), M = 128 small batch
  (0.80–0.87), and M = 1 GEMV (0.76 on 1×1024×4096).
- **The vendor oracle is a moving target.** Timing cuBLAS's own heuristic list gives +0.1% at the median (B3/B1). A 20 s enumerate-and-screen
  search adds a further 2.8% at the median and up to 1.67×. Any claim of "beats cuBLAS" has to be made against B3+.

### What G4 failing means here

G4 is defined on **our family's oracle**, and so far that oracle covers 5 configs. This result shows that 5 configs do not differentiate
from a well-searched cuBLAS. It does not yet show that the full Triton space cannot. That is settled by the full sweep (M3/M4), compared against B3+ under
the same budget treatment.

If G4 still fails after the sweep, then per plan §8, **the project remains viable as a research/learning project on selection (G3, G5) or as a
non-NVIDIA backend project, not as a cuBLAS competitor on the RTX 3080.** In that case the selector's target becomes "pick the best of
{our configs ∪ cuBLASLt configs}", and B3+ is the oracle to approach.

### Gate status (supersedes earlier tables)

| Gate | Status |
|---|---|
| G1 measurement validity | **Pass** for kernels ≥ 50 µs under standard conditions |
| G2 family ≥ 90% of B3+ | **Pass**: median 95.9% with 5 hand-picked configs (small batch 90.1%, at the bar) |
| G3 selection value | Positive: oracle/B6 median 1.04, p90 2.19; decode median 1.60 |
| G4 differentiation | **Fails with 5 configs** against B3+ (0 regimes; earlier K = 4096 wins are ties). Final verdict after the full sweep |
| G5 tractability | Not started |

### Next steps
1. Full pruned Triton config sweep (M3/M4) on a coarse subset of all suites, judged against B3+. This is the deciding input for G4 and
   the input to the portfolio study (M5, G5).
2. Targeted configs for the loss clusters: a GEMV-style config for M = 1, split-K variants for M = 128 × small N, and the
   unaligned-stride / masking path for non-multiple-of-16 squares.
3. If the sweep is run on large shapes, raise the B3+ budget there (e.g. 60 s) or note the coverage, since B3+ was budget-limited on 33 shapes.

## Update 2026-10-15: B2/B3 (cuBLASLt) added; G2 passes, G4 narrows sharply

Run: `bench-20261015-015054-lock1755-b3` (standard conditions, commit `8c896ab`, 3336 rows, all valid), in
`results_lock1755_b3.jsonl`. Full tables: [`week1_results_lock1755_b3.md`](week1_results_lock1755_b3.md).
B2 = cuBLASLt heuristic top-1. B3 = best of the heuristic's top-16 after timing each, which in practice is **a median of 4 candidates (1–9)**.

### Headline (cold L2, 143 shapes)

| | median | p10 | p90 | min | max |
|---|---|---|---|---|---|
| ours (5-config oracle) vs **B3** | **0.993** | 0.88 | 1.09 | 0.66 | 1.36 |
| ours vs B1 (for reference) | 1.008 | 0.92 | 1.31 | 0.66 | 1.51 |
| **B3 vs B1** (what timing cuBLAS's own candidates buys) | 1.001 | 1.00 | 1.18 | 0.99 | **1.57** |
| B2 vs B1 | 1.000 | 1.00 | 1.00 | 0.96 | 1.05 |

- **B1 is B2.** `torch.matmul` launches exactly the cuBLASLt top-1 kernel on 143/143 shapes (recorded kernel names).
- **G2 passes.** Median ours/B3 = 0.993, well above the 0.90 bar. Five hand-picked Triton configs reach vendor-oracle parity
  on the median shape.
- **Most of our earlier "wins over cuBLAS" were cuBLAS heuristic misses, not kernel advantages.** B3 recovers them by timing ≤ 8
  of cuBLAS's own candidates:
  - The 5120/13824 family (512×5120×5120: B1 751 µs → B3 483 µs): the heuristic ranks split-K algorithms 5th–6th.
  - M=1 decode (1×1024×4096): a dedicated `gemvx` kernel, ranked 4th, beats the GEMM that B1 uses.
  - 32×13824×5120 (the 38% "regression"): the heuristic picks a 256×64-tile kernel for 32 rows; its 2nd candidate is 11% faster.

  Against B3, all of these sit at 0.93–1.03×.

### G4: where ours still beats B3 by more than 5%

| regime | n | ours > B3 by >5% | within 2% | ours < B3 by >5% |
|---|---|---|---|---|
| S2 decode (M ≤ 16) | 55 | **11 (20%)** | 20 | 3 |
| S2 small batch (32–128) | 33 | 5 (15%) | 7 | 16 |
| S1 square | 22 | 5 (23%) | 3 | 11 |
| S2 prefill (M ≥ 512) | 33 | 1 (3%) | 16 | 1 |

The surviving wins cluster in **one identifiable sub-regime: K = 4096 with N ∈ {1024, 11008, 12288} at M ≤ 128**
(LLaMA-7B gate/up, the QKV projection, 1024-wide KV). There we are 1.06–1.17× faster than B3 with `bm16_bn64_bk128` (decode) and
`bm64_bn64` (M = 32–64). Two shapes also show split-K wins: 128×1024×4096 at 1.34×, and 512×11008×4096 at 1.15×.

**Caveats that keep G4 open:**
- **B3 is shallowest exactly where we win.** On the 11008 shapes cuBLASLt returns 1–2 near-identical candidates (both the
  older `s1688` 256×64 kernel). A full algorithm enumeration (`cublasLtMatmulAlgoGetIds`/`AlgoInit`/`AlgoCheck` over
  tiles × stages × split-K) may close these gaps. Until it is tried, these wins mean "beats cuBLAS's best heuristic candidate", not
  "beats cuBLAS".
- The square-shape wins (512³ at 1.36×, 1024³ at 1.22×, 128³, 256³, 64³) are all under 50 µs except 1024³, which is outside G1's coverage.
- Losses to B3 are concentrated in unaligned squares (4097³ 0.79×, 2049³ 0.84×, 4095³ 0.84×, 1000³ 0.66×), M = 128 small batch, and
  M = 1 GEMV (0.79×; we have no GEMV-style config).

### Gate status (supersedes earlier tables)

| Gate | Status |
|---|---|
| G1 measurement validity | **Pass** for kernels ≥ 50 µs under standard conditions |
| G2 family ≥ 90% of B3 | **Pass**: median 99.3% with only 5 hand-picked configs |
| G3 selection value | Positive: oracle/B6 median 1.04, p90 2.19; decode median 1.53 |
| G4 differentiation | **Narrow and provisional**: a K = 4096, M ≤ 128 sub-regime, 6–17% over a shallow B3. Needs a deeper B3 |
| G5 tractability | Not started |

**Implications for the project:**
1. The honest bar is B3, not B1. A trivial "time cuBLAS's own top-8" selector already captures up to 1.57× over `torch.matmul`.
   Any learned selector must be compared against that, not against B1.
2. Per-shape selection is valuable for both families. Whether *our* family adds anything beyond the vendor's candidates
   depends on the deeper-B3 check and on the full config sweep (M3/M4), which may widen the K = 4096 / small-M advantage.

### Next steps
1. **Deeper B3:** enumerate cuBLASLt algorithms beyond the heuristic (AlgoGetIds → AlgoInit → set tile/stages/split-K →
   AlgoCheck), capped per shape, at least on the G4 shapes. This decides whether the K = 4096 wins are real.
2. Full pruned Triton config sweep (M3/M4) on a coarse subset of all suites, then the portfolio study (M5) for G5.
3. Add a GEMV-style config (M = 1) and fix unaligned strides; both are among our main losses to B3.

## Update 2026-10-14: G1 closed under standard conditions; baselines re-run locked

> The sections below this one are the original 2026-10-09 results, measured **unlocked on driver 595.91.07**. For any
> new comparison, use the locked numbers in this section. Full tables: [`week1_results_lock1755.md`](week1_results_lock1755.md).

**Standard conditions from now on:** SM clock locked at 1755 MHz (`sudo nvidia-smi -lgc 1755,1755`, then
`python -m tune --clock-lock 1755 ...`), driver 595.99.02, 3 allocations per measurement, cold L2 primary. Every session now
writes `data/<hw>/env/<session_id>.json`, and every row carries its commit, clock-lock state and driver.

### G1: passes for kernels ≥ 50 µs

There were three complete locked sessions, the last one 2 minutes after a reboot and about 15 h after the first:

| | p50 | p95 | max |
|---|---|---|---|
| range of medians across the 3 sessions, cold | 0.24% | **1.75%** | 2.5% |
| range of medians across the 3 sessions, hot | 0.09% | **1.78%** | 3.0% |
| range of the kernel-vs-kernel ratio across the 3 sessions | 0.38% | **1.94%** | 2.2% |

Within each session, the p95 range across rounds is 1.3–1.9%. The median shift between any two sessions is ≤ 0.05%.
**The tie margin is now 2%** for kernels ≥ 50 µs (`docs/methodology.md`). Kernels under 50 µs are still about 6% p95 in cold
per-launch mode; rank them with graph-amortized timing. The lock is a ceiling only: under heavy load the power cap still pulls the clock
to about 1455–1590 MHz on some rows, and it does so the same way every session.

### B0 under the lock

60.0 TFLOP/s (59.7 on the clean-commit re-run) vs 62.9 unlocked, and 687 GB/s (unchanged). The unlocked file is kept as
`data/<hw>/roofline-20261009-unlocked.json`.

### S1 + S2 re-run under the lock (`results_lock1755.jsonl`, 1960 rows, all valid)

| (cold L2) | unlocked, 10-09 | **locked, 10-14** |
|---|---|---|
| ours vs B1, median [p10, p90] | 1.006 [0.92, 1.27] | **1.008 [0.92, 1.34]** |
| ours vs B5, median | 2.12 | **2.21** |
| oracle vs B6, median [p90] | 1.08 [2.13] | **1.04 [2.20]** |
| decode (M ≤ 16): ours vs B1 / oracle vs B6, median | 1.03 / 1.53 | **1.05 / 1.58** |
| shapes: win > 5% / within 2% / loss > 5% vs B1 | 42 / – / 26 | **44 / 45 / 28** |
| shapes beating B5 by > 5% (losing by > 5%) | 107 (0) | **111 (0)** |

**The conclusions hold.** Comparing unlocked to locked per shape, no shape flipped between win and loss. 23 of 143 moved between
neighbouring classes (tie ↔ win, tie ↔ loss), almost all of them already near the 5% boundary. Clocking down 5% slowed B1 by a median 3.1% and our
oracle by 1.5%.

The largest wins and losses are unchanged:
- Wins: 512×5120×5120 at 1.51×, 512×5120×13824 at 1.48×, M = 64 on the 13824/14336 layers at about 1.44×, decode M = 8 at 1.40×.
- Losses: 1000³ at 0.66×, 3000³ at 0.70×, M = 128 on large N/K at 0.84–0.86×.

**One unexplained cuBLAS change:** 32×13824×5120 went from parity to 1.36× because **B1** slowed from 229 µs to 316 µs (+38%,
the same on all 3 allocations). Our kernels moved less than 2%. A 5% lower clock cannot explain that on a memory-bound shape, so cuBLAS most likely
picked a different algorithm under the new driver or clock. torch and its bundled cuBLAS are unchanged. B1 rows don't record which cuBLAS
kernel ran, so this can't be confirmed from the data. That is one more reason to build B3, and to log the chosen cuBLAS kernel per row.

### Gate status (supersedes the table below)

| Gate | Status |
|---|---|
| G1 measurement validity | **Pass** for kernels ≥ 50 µs under standard conditions (p95 < 2% across 3 sessions and a reboot). Not covered: kernels under 50 µs |
| G2 family ≥ 90% of B3 | Not testable yet (no B3). Against B1: median 100.8% |
| G3 selection value | Positive early signal. Oracle/B6 median 1.04, p90 2.20; decode median 1.58 |
| G4 differentiation | Candidates: decode, the 5120/13824 family, M = 64. All need re-checking against B3 |
| G5 tractability | Not started |

### Next steps (in order)
1. **B3 (cuBLASLt exhaustive) and B2 (heuristic top-1)**, plus logging the chosen cuBLAS kernel name on B1 rows.
2. Kernel fix for unaligned strides (1000³, 3000³), then re-check S1 and S5.
3. M3/M4: full pruned config sweep on a coarse subset of all suites under standard conditions, then the portfolio study (M5).
4. Re-run `python -m tune --clock-lock 1755 noise --rounds 3 --repeats 50 --n-allocs 3` after any driver update.

## TL;DR

1. **The harness works and is validated.** It has env capture, adaptive CUDA-event timing (cold/hot L2, backlogged launches,
   graph-amortized mode), a correctness gate, resumable JSONL records, and a CLI. 67 unit tests pass. All 7 kernels
   (B1, B5, 5 hand-picked configs) pass correctness on all 169 edge shapes: 1183/1183.
2. **Noise floor ≈ 3% (p95), 0.3% (median), after a methodology fix.** The dominant noise source was **input allocation
   placement**, not heat or clocks. The same data at a different address shifts memory-bound kernels by up to 3.7%.
   Pooling every measurement over 3 placements halves the p95. **Tie margin ε = 3%.**
3. **Even 5 hand-picked Triton configs are competitive with cuBLAS on this card.** The per-shape oracle over them
   beats `torch.matmul` (B1) by more than 5% on 42/143 S1+S2 shapes, ties within 3% on 61, and loses by more than 5% on 26. The median is 1.006×,
   the range 0.67×–1.42×. It beats the Triton tutorial kernel (B5) by more than 5% on 107/143 and never loses to it by more than 5%.
4. **Per-shape selection already matters.** The best single hand-picked config (B6) gives up a median 7.7% vs the
   5-config oracle, and **53% (median), up to 3×, on decode shapes**. That is a first, small-scale signal for gate G3.

## Checklist status (plan §13)

| Item | Status |
|---|---|
| Pin environment; `env.json` capture | ✅ `requirements.txt`; `python -m tune env` (GPU and NVML state, versions, git commit, display/processes) |
| Timing utility + unit tests | ✅ `tune/timing.py`, `tests/test_timing.py` |
| Correctness checker + edge-shape suite | ✅ `tune/correctness.py`, `suites/v1/edge.json` (169 shapes); 1183/1183 pass |
| Noise-floor study on ~30 shapes (multi-session) | ⚠️ **2 sessions, same day** (the second one tests the allocation fix). Needs re-runs on other days for the across-session component |
| B0, B1 on S1 + S2, first sanity plot | ✅ `figures/roofline.png`, `figures/speedup_{cold,hot}.png` |
| Minimal Triton matmul with 3–5 hand-picked configs; beats B5 somewhere | ✅ 5 configs; beats B5 on 107/143 shapes by more than 5% |
| Decide primary cache mode | ✅ **cold L2** (reasons in methodology.md; the data shows median cold/hot = 1.003) |
| Shape suites S1–S7 as versioned JSON with held-out splits | ✅ `suites/v1/` with 608 shapes; region holdout (9000 ≤ K ≤ 12000, 300 ≤ M ≤ 450) plus 30% of S7 |

## B0: roofline

| | spec | measured |
|---|---|---|
| fp16 → fp32-acc tensor TFLOP/s | 59.5 (68 SMs × 1710 MHz × 512 FLOP/clk) | **62.9** (cuBLAS 8192³ at about 1875 MHz) |
| DRAM GB/s | 760 | **686** (1 GiB D2D copy, 90% of spec) |

The card boosts above its spec clock (up to 1965 MHz seen), which is why measured compute exceeds spec. Under sustained load
it is **power-capped** (NVML throttle reason 0x4 on about 90% of rows) and settles at about 1800 MHz. See `figures/roofline.png`:
decode GEMMs from both B1 and ours sit on the bandwidth roof, and large prefill or square GEMMs reach 55–62 TFLOP/s.

## Noise floor (Q4 / gate G1)

| | 1 allocation (session 1) | 3 allocations pooled (session 2) |
|---|---|---|
| across-round range of medians, cold, p50 / p95 / max | 0.67% / 5.4% / 20% | **0.29% / 2.6%** / 9.6% |
| same, kernels ≥ 50 µs, p95 | 5.2% | **2.4%** |
| paired ratio (B1 / ours) range, cold, p50 / p95 | 2.1% / 6.2% | **0.33% / 3.5%** |
| within-run IQR / median, p50 | 0.7% | 0.9% |

- **Thermal drift is negligible.** Continuous 8192³ for 10 min: +0.01% first minute → last minute, std 0.1%
  (`figures/thermal_drift.png`).
- **Allocation placement is the dominant term.** 8 fresh allocations × 3 timings each, for 8192×64×8192: 0.0% within an
  allocation, 3.7% across allocations. It does not cancel in kernel ratios. The fix is now the default
  (`--n-allocs 3`). Session 1 was run before the fix and is kept in `noise.jsonl` for comparison.
- **µs-scale kernels** (under about 20 µs) are quantized to a ~1.02 µs grid in per-launch mode, even though event resolution is 32 ns.
  The worst remaining outlier (9.6%) is one of these. Rank them with graph-amortized timing.
- **G1 verdict: provisionally passes at the median (0.3% ≪ 2%) and is borderline at p95 (2.6–3.5%).** Not yet shown
  reproducible across sessions. Clock locking (`sudo nvidia-smi -lgc`) was not available to the harness. It is the
  obvious next lever if p95 must come below 2%.

## Baselines on S1 + S2 (cold L2, 3 placements; 143 shapes, the 22 shapes with dims > 16384 excluded)

Speedups are time_baseline / time_ours (>1 means ours is faster). "ours" = per-shape best of the 5 hand-picked configs.

| regime | n | ours vs B1, median [p10, p90] | ours vs B5, median | oracle vs B6, median [p90] |
|---|---|---|---|---|
| S1 square | 22 | 0.94 [0.90, 1.16] | 1.87 | 1.08 [1.27] |
| S2 decode (M ≤ 16) | 55 | 1.03 [0.99, 1.33] | 4.14 | **1.53 [2.54]** |
| S2 small batch (32 ≤ M ≤ 128) | 33 | 0.98 [0.88, 1.11] | 2.94 | 1.00 [1.13] |
| S2 prefill (M ≥ 512) | 33 | 1.00 [0.95, 1.10] | 1.00 | 1.05 [1.08] |
| **all** | 143 | **1.006 [0.92, 1.27]** | **2.12** | 1.08 [2.13] |

The hot-L2 numbers are in `week1_results.md` and tell the same story.

**Where we win (more than 5% outside the noise floor):**
- Skinny decode GEMMs (`bm16_bn64_bk128`): up to 1.39× (M = 4–16, N = K = 5120).
- M = 64 and M = 512 on LLaMA-13B/8B-style dims: up to 1.42× (512×5120×5120: cuBLAS reaches 38 TFLOP/s, ours 54).

  The 5120/13824 family looks like a **cuBLAS heuristic miss**. B3 (cuBLASLt exhaustive) may close it, so these
  wins are not yet evidence for G4.

**Where we lose:**
- Unaligned squares: 1000³ at 0.67×, 3000³ at 0.71×, and 2^k ± 1 at about 0.89×. The likely cause (not yet profiled): Triton
  specializes integer args on divisibility by 16. Strides of 1000 or 3000 (≡ 8 mod 16) and odd strides miss that
  specialization, so it cannot prove alignment for wide vectorized loads. That is a kernel-level fix, not a selection
  problem (plan §8.1 item 5).
- M = 128 with large N and K: 0.84–0.87×. No hand-picked config targets 128×(large) tiles with deep K pipelining. The full sweep
  will tell whether the family covers it.

**B5 is a weak baseline.** It is 2× slower than cuBLAS at the median, and up to 14× slower on decode, because a 128×256 tile is
mostly padding for M ≤ 16. Beating B5 says little about the family's quality. B3 is the real bar.

## Gate status after week 1

| Gate | Status |
|---|---|
| G1 measurement validity | **Provisional pass** at the median; p95 borderline (≈3%); across-session not yet measured |
| G2 family ≥ 90% of B3 | Not testable yet (no B3). Against B1, 5 configs already reach a median of 100.6% |
| G3 selection value | **Early positive signal.** Oracle/B6 median 1.08, p90 2.13 with only 5 configs. Needs the full config space |
| G4 differentiation | Candidate regimes: decode (M ≤ 16) and the 5120 family. Must be re-checked against B3 |
| G5 tractability | Not started (needs the full sweep, M4–M5) |

## Recommended next steps

1. **Repeat the noise study on 2+ other days**, including one after a cold boot:
   `python -m tune noise --rounds 3 --repeats 50 --n-allocs 3`. That closes G1.
2. If you are OK with root, test clock locking: `sudo nvidia-smi -lgc 1755,1755`, re-run the noise study, then
   `sudo nvidia-smi -rgc`. Expect p95 to drop if clock variation is a meaningful part of the residual.
3. **Build B3** (cuBLASLt heuristic top-N enumeration via a ctypes/C++ shim) before trusting any "beats cuBLAS" claim.
   Add B2 (top-1 heuristic) from the same shim.
4. Kernel fix for unaligned K/N strides (e.g. a padded-K or an EVEN_N/aligned-pointer specialization), then re-check
   S1 and S5.
5. M3/M4: run the full pruned config space (`kernels/config_space.py`: static pruning keeps 10,840 of 19,200, and per-shape rules cut more; the GROUP_M × SPLIT_K axes alone are a 20× multiplier, so a coarse first pass should fix those) on a
   coarse subset of all suites, compute the real B6 and oracle, then do the portfolio study (M5).

## Reproduce

```bash
python -m tune env && python -m tune check --suite edge && python -m tune roofline
python -m tune noise --rounds 3 --repeats 50 --n-allocs 3
python -m tune drift --minutes 10
python -m tune benchmark --suite S1,S2 --backends torch,triton_tutorial,triton:handpicked --cache-modes cold,hot
python -m tune analyze
```
