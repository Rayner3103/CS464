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
| across-round range of medians, p50 / p95 (cold) | 0.67% / 5.4% | 0.29% / 2.6% |
| paired kernel-ratio range, p50 / p95 (cold) | 2.1% / 6.2% | 0.33% / 3.5% |

**Tie margin for analysis: ε = 3%.** Kernels within 3% of each other count as tied. Wins under 5% are reported but never relied on. Revisit once across-session data exists.

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
- Both noise sessions ran on the same day with different methods. The across-session component needs `python -m tune noise --rounds 3 --repeats 50 --n-allocs 3` re-run on other days, ideally including one cold boot.
