# Week 1: Phase 0 + first baselines (RTX 3080)

**Date:** 2026-10-09 · **hw_id:** `rtx3080_sm86_68sms_10gb` · driver 595.91.07 · torch 2.13.0+cu130 · triton 3.7.1
**Scope:** milestone M0 and the plan §13 first-week checklist. All tables are generated in
[`week1_results.md`](week1_results.md) (`python -m tune analyze`). The protocol is in [`../docs/methodology.md`](../docs/methodology.md).

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
