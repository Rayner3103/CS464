# Measurement methodology (Phase 0)

The timing protocol is set here once and used for every number in the project. The code is in `tune/timing.py`.
Plan sections 3.1–3.4 cover the rationale.

## Timing protocol

| Item | Choice |
|---|---|
| Timer | CUDA events recorded on the launch stream, around each sample. No Python wall-clock. |
| First call | Always untimed: Triton JIT compile, cuBLAS heuristic, and allocator warmup happen here. |
| Warmup | ≥3 runs, then repeat until ≥25 ms of device time has elapsed. |
| Repeats | Adaptive. Collect samples in chunks of about 50 ms. Stop when ≥10 samples **and** relative std error (std/√n/mean) < 1%, or at 1000 samples or 2 s wall-clock. `converged` is stored on every row. |
| Statistic | **Median** is primary. p10/p25/p75/p90/p99, min, max, n and RSE are stored too. Raw samples go to `*.raw.jsonl`. |
| Launch overhead | Before each chunk, a device-side `torch.cuda._sleep` is queued so the CPU enqueues the whole chunk while the GPU is still busy. Events then measure device time back-to-back, not Python launch latency. Chunks are capped at 256 launches so the driver's pending-launch queue (about 1k entries) never fills; a full queue would make the measurement CPU-bound again. |
| Per-launch vs amortized | `batch=1` gives per-launch device time. That includes about 1–2 µs of event overhead, which matters only for kernels under ~20 µs. `batch=N, use_graph=True` captures N launches into a CUDA graph and reports time/N (amortized). Use it for µs-scale kernels. |
| L2 flush (cold mode) | Before every sample, `zero_()` a 64 MiB buffer (≥8× L2), outside the timed region. Cold mode forbids `batch>1`, because launches 2..N would run hot. |
| Hot mode | No flush. Operands stay L2-resident across samples. |
| Input placement | **Every measurement pools raw samples over 3 fresh input allocations** (`--n-allocs 3`, the default). Each allocation is shifted by a seeded 1–64 MiB pad. Values are identical across placements, only physical addresses differ. See "Allocation noise" below. |
| CPU | The process is pinned to one core (`os.sched_setaffinity`), so the launch thread does not migrate. |

### Sanity checks (unit-tested in `tests/test_timing.py`)
- A device spin of known cycle count (about 200 µs) is recovered to within ±25%.
- Rep caps are honored exactly.
- For L2-resident operands, hot ≤ cold.
- The CUDA-graph per-launch time is ≤ the single-launch event time.

## Allocation noise (found in the week-1 noise study)

Re-timing the same buffers is very stable. Timing the same kernel on a *fresh allocation of identical
data* is not:

| 8 fresh allocations × 3 back-to-back timings, cuBLAS, cold | within-allocation range | across-allocation range |
|---|---|---|
| 8192×64×8192 (memory-bound) | 0.0% | 3.7% |
| 4096×4096×4096 (compute-bound) | 0.5% | 0.8% |
| 32×8192×8192 | 0.0% | 0.0% |

(Data: `data/<hw>/alloc_variance.json`.) The likely cause is how operand pages map onto DRAM channels and banks.
The effect is not thermal: a 10-minute continuous run drifts by 0.01%. It does not cancel in kernel-vs-kernel
ratios either, because different kernels' access patterns respond to placement differently.

Pooling the samples over 3 placements cuts the noise floor roughly in half. All numbers are over 30 shapes × 2 kernels × 3 rounds:

| | 1 allocation | 3 allocations pooled |
|---|---|---|
| across-round range of medians, p50 / p95 / p99 (cold) | 0.67% / 5.4% / 12.5% | 0.29% / 2.6% / 6.3% |
| paired kernel-ratio range, p50 / p95 / p99 (cold) | 2.1% / 6.2% / 20.0% | 0.33% / 3.5% / 10.3% |

(p99 is essentially the worst of the 60 (shape, kernel) ranges, or of the 30 ratios: pooling cuts the median and p95 by about half and the tail
by about half too, but a tail kernel still moves 6-10% between rounds. The 1-allocation paired p99 is the single worst ratio.
Reproduce: `python -m tune analyze` prints `range_p99` / `paired_p99` per session next to the p95s.)

**Tie margin for analysis (revised 2026-10-14): ε = 2% for kernels ≥ 50 µs** measured under the standard conditions below.
Kernels within 2% of each other count as tied. Kernels under 50 µs keep ε = 5% in per-launch mode; rank them with
graph-amortized timing instead. Wins under 5% are reported but not relied on for go/no-go decisions.

## Noise floor under standard conditions (gate G1)

**Standard conditions:** SM clock locked at 1755 MHz (`sudo nvidia-smi -lgc 1755,1755`, run with `--clock-lock 1755`),
driver 595.99.02, 3 allocations pooled per measurement, cold L2 primary. There were three complete sessions, one of them right after a
reboot:

| session | when | p95 range across rounds (≥ 50 µs), cold / hot | p99 (≈ worst of 44 / 43 kernels), cold / hot |
|---|---|---|---|
| `noise-20261013-232542-642a` | 10-13 23:25 (lock inferred from observed max clock) | 1.3% / 1.2% | 2.0% / 1.7% |
| `noise-20261014-001639-lock1755` | 10-14 00:16 | 1.8% / 1.8% | 3.1% / 2.8% |
| `noise-20261014-141218-6985` | 10-14 14:12, 2 min after a reboot | 1.4% / 1.9% | 4.3% / 4.2% |

Across the three sessions, for the same (shape, kernel, mode) at ≥ 50 µs, the range of medians is p50 0.24%, **p95 1.75%**, p99 2.3%, max 2.5% (cold),
and **p95 1.78%**, p99 2.6%, max 3.0% (hot). For the ratio between the two kernels, which is what decides selection, it is **p95 1.94%**. The median shift between any
two sessions is ≤ 0.05%. **G1 passes for kernels ≥ 50 µs.**

**What ε = 2% does not cover.** ε is set from the p95. The tail is wider: the worst one or two of 44 kernels move 2.3-2.6% across sessions (p99) and,
within one session across rounds, 3-4% (the p99 of the 10-14 sessions; the post-reboot session has a 4.8% worst cold kernel). A pick that wins by 2-4% on one
kernel is therefore not a reliable win, which is why the tables report p99 and the worst shape next to p95 and name the shapes in the tail. (The kernel-ratio
p99 is not given: the 1.94% p95 above comes from a script that is not in the repo and could not be reproduced from `reports/noise_stats.csv`, which gives
2.05% cold / 2.11% hot for the ratio range by the obvious definition.)

Caveats:
- Kernels under 50 µs remain at about 6% p95 in cold per-launch mode (p99 7-13%, 16 kernels per session, essentially the worst one), so G1 does not cover them.
- The lock is a ceiling, not a floor. Under sustained heavy load the card still hits the power cap (throttle 0x4) on 14–19% of rows
  and dips to about 1590 MHz. It happens the same way in every session, so it does not hurt reproducibility.
- These numbers hold only under the standard conditions. A driver update, a different lock, or unlocked clocks means re-running the noise study
  before comparing results.

## Noise floor with the SM clock unlocked (2026-11-13)

Same protocol (`python -m tune --clock-lock none noise --rounds 3 --repeats 50 --n-allocs 3`), same driver 595.99.02, desktop idle, no
`nvidia-smi -lgc`. Under load the SM clock runs at 1575-1965 MHz (p50 ≈ 1860) and 82-92% of rows are power-capped (locked: 14-19%). Three
complete sessions, the third after a reboot (34 min after boot, not 2 min as for `…6985`):

| session | when | p95 range across rounds (≥ 50 µs), cold / hot | p99 (≈ worst of 42 kernels), cold / hot |
|---|---|---|---|
| `noise-20261113-185827-22be` | 11-13 18:58 | 3.2% / 2.5% | 4.2% / 4.6% |
| `noise-20261113-190809-53f0` | 11-13 19:08, right after the first | 1.9% / 1.6% | 2.8% / 2.5% |
| `noise-20261113-205714-0ab2` | 11-13 20:57, after a reboot | 6.1% / 4.5% | 7.8% / 6.7% |

Across the three sessions, at ≥ 50 µs, the range of the session medians is p50 0.42%, **p95 1.35%**, p99 2.3%, max 2.9% (cold) and p50 0.60%,
**p95 1.90%**, p99 2.7%, max 3.0% (hot); for the kernel ratio p95 1.49% / 1.96%. The session speed shift is ≤ 0.12%. Unlocked kernels run
4.4% faster than locked at the median (time ratio 0.956).

**Tie margins unlocked (same rule as above: p95 rounded up):**
- **ε = 2% for kernels ≥ 50 µs still holds** for comparisons between sessions (cross-session p95 1.90% vs 1.78% locked; bootstrap intervals over
  shapes [0.86, 2.95] vs [0.82, 2.94], not distinguishable).
- **ε = 5% under 50 µs per launch is unchanged** (cross-session p95 4.2% cold / 5.3% hot; locked 5.3% / 2.4%). By the rule both conditions read 6%.
- **A single measurement is about twice as noisy.** The rounds of one session disagree more (p95 4.25% / 3.83% vs 1.65% / 1.67% locked), and one
  full-protocol measurement (one benchmark row) is off the cross-session median by p95 1.84% / 2.01% (p99 4.1% / 3.8%), vs 1.08% / 1.04% locked.
  The cross-session figure stays at 2% only because each session contributes the median of 3 rounds. On unlocked data a 2% margin is all noise,
  and a 2-4% win between two single rows is not a reliable win.
- **The online search's short timing** (10 cold samples, one placement, `tune/online.py`), emulated from the raw samples: a kernel-vs-kernel
  comparison at ≥ 50 µs is off by p95 **3.5%** (p99 6.1%) vs 2.7% (6.9%) locked. **Use ~4% as the near-tie margin of a deployed, unlocked
  selector** (3% locked).

The extra spread sits in particular rounds (the first round of each session, at 95-100% power-capped, and the last round of the post-reboot
session), is per kernel rather than a common drift (it does not cancel in the kernel ratio), and is not explained by the median clock of the round.
Tables, definitions and the bootstrap: `reports/noise_compare_unlocked.md`, `python -m analysis.noise_compare` (docstring); write-up
`reports/week1.md` 2026-11-13. Graph-amortized timing is not covered.

## µs-scale kernels

CUDA event timestamps have 32 ns resolution. Still, per-launch medians of kernels under ~20 µs cluster on a ~1.02 µs grid
(4.10, 5.12, 6.14, 7.17 µs …). Treat per-launch differences of about 1 µs on such kernels as unresolvable. To rank them,
use the amortized mode (`batch=N, use_graph=True`, hot L2). Cold-mode timing of µs kernels is limited to this
granularity by construction.

## Primary cache mode

**Decision: cold L2 is the primary mode. Hot is recorded alongside it in the baseline runs.**

Reasons:
1. It is the conservative case. Hot numbers can only be faster.
2. It matches real inference better for the shapes that matter most (S2). One layer's weight matrix (e.g. 4096×4096 fp16 = 32 MiB) is far larger than the 5 MiB L2. Consecutive GEMMs in a model touch different weights, so a decode GEMM almost never finds its B operand in L2. Hot-mode timing of a skinny GEMM measures an L2-bandwidth-bound kernel that does not occur in practice.
3. The data agrees: over the noise set the cold/hot ratio has median 1.003 and p90 1.14. The large ratios (up to 1.25) are all µs-scale kernels whose operands fit in the 5 MiB L2. For large GEMMs the two modes match within noise, so choosing cold costs nothing there.

Hot mode stays the right model for genuinely back-to-back reuse, such as the same small GEMM in a tight loop. Every row is tagged with `cache_mode`, so both stay available to analysis.

## Environment control

| Item | Status on the RTX 3080 box |
|---|---|
| Clock locking (`nvidia-smi -lgc`) | Set by the user with root; the harness never runs as root. NVML has no getter for locked clocks, so every session **probes** the state. An idle GPU still at a high clock, or a clock capped well below max under a light spin load with no power or thermal limit, means `likely_locked`. Declare the lock with `python -m tune --clock-lock 1755 ...` (or `TUNE_CLOCK_LOCK=1755`). The declaration is recorded, and a warning is printed if it disagrees with the probe. Unlocked, the card runs **power-capped** under sustained load (throttle reason 0x4), with the SM clock between 1590 and 1965 MHz. **Only compare sessions with the same lock state and driver.** |
| Idle GPU | Compute processes are checked through NVML and logged in `env.json`. **This GPU also drives the desktop display** (`display_active=1`). Compositor work is a background noise source and is measured as part of the noise floor. |
| Persistence mode | Off (recorded). |
| ECC | Not supported on GeForce (recorded as N/A). |
| Power limit | 340 W default (recorded). |
| Fingerprint | `hw_id = <gpu>_sm<cc>_<n>sms_<mem>gb`. |
| Provenance | Every session writes `data/<hw>/env/<session_id>.json` (driver, versions, git commit and dirty flag, clock-lock probe, argv), and `env.json` is refreshed to the latest session. Every result row also carries `session_id`, `driver`, `harness_commit`, `harness_dirty`, `clock_lock_state`, `clock_lock_mhz` and `env_file`. Rows written before 2026-10-14 lack the last five fields. |

## Correctness gate

The code is in `tune/correctness.py`.

- **Reference.** fp32 cuBLAS GEMM of the fp16-rounded inputs, with TF32 explicitly disabled. Inputs are N(0,1), so output elements have std ≈ √K.
- **Pass criterion.** `|C − ref| ≤ rtol·|ref| + atol_frac·√K` elementwise. fp16 uses rtol = atol_frac = 1e-2; bf16 uses 3e-2. This passes fp32-accumulate kernels and fails garbage tiles. In the unit tests it also fails fp16 accumulation over long K.
- **Unwritten elements.** The output buffer is pre-filled with NaN, so a kernel that skips a row or tile fails.
- **Errors.** Compile errors, OOM and launch errors are caught and recorded as `valid=false` with a `failure_reason`. Invalid configs are never timed.
- **Crashes.** If a config faults the CUDA context, the process dies. The runner's `inflight.json` marker makes the next run record that candidate as `crashed_previous_run` and skip it.
- **Edge suite.** `suites/v1/edge.json` has 169 shapes: M/N/K ∈ {1, 2, 3, primes, 2^k±1}, K < BLOCK_K, and K up to 16384 with M = N = 1.

## Known limitations of this phase
- Per-candidate subprocess isolation and hang timeouts (plan 7.2) are not implemented yet. Crash recovery works through restart-and-resume instead.
- Clocks are recorded, not locked. Clock-normalized analysis is possible later from the per-row `clock_mhz`.
- The across-session noise data covers about 15 hours and one reboot, all on one driver. Re-run `python -m tune --clock-lock 1755 noise --rounds 3 --repeats 50 --n-allocs 3` after any driver update.
