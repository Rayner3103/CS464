# AGENTS.md — CS464: learned kernel selection for GPU GEMM

Guide for agents working in this repo. It summarizes what the baseline phase (branch `baseline`) built, what it found,
and how to run, extend and not break it. Source of truth for details: [`plan.md`](plan.md) (the plan and gates),
[`docs/methodology.md`](docs/methodology.md) (measurement protocol), [`reports/week1.md`](reports/week1.md) (dated
findings, newest first).

## 1. What this project is

Phase 1 of a learned kernel selector for fp16 GEMM (`C = A·B`, row-major NN, fp32 accumulate) on one GPU
(RTX 3080, `hw_id = rtx3080_sm86_68sms_10gb`). Before any ML, it answers with data:

- How fast is the vendor path (cuBLAS / cuBLASLt), including a searched cuBLASLt?
- How fast is a parameterized Triton family at its per-shape oracle?
- Does per-shape selection beat a single fixed config, and can a small portfolio cover the space?
- How noisy are the measurements (the smallest trustworthy difference)?

Scope is frozen (plan §2): one NVIDIA GPU, Triton, fp16 (bf16 supported in the code paths), NN layout, plain GEMM,
M, N, K in [1, 16384]. No fused epilogues, batched/grouped GEMM, or quantized dtypes.

## 2. Repository layout

```
tune/        harness package and CLI (python -m tune ...)
  cli.py         subcommands: env check roofline benchmark baseline sweep noise drift analyze portfolio selectors
  env.py         hw fingerprint, NVML telemetry, clock-lock probe, git provenance, CPU pinning, env.json
  timing.py      CUDA-event timer: warmup, adaptive repeats, cold/hot L2, backlogged launches, CUDA-graph amortized
  correctness.py fp32 reference, tolerance check, NaN-prefilled output, placed (address-shifted) inputs
  records.py     append-only JSONL schema (plan 5.3), raw-sample side file, resume keys, alignment flags
  runner.py      backend spec -> candidates, validate-then-time loop, inflight.json crash recovery, budget
  sweep.py       stage-1 screen of the full pruned Triton space (subprocess per shape, parallel compile)
  noise.py       noise-floor study and thermal-drift run
  profile.py     records the device kernel names that vendor backends launch
kernels/
  triton_matmul.py  parameterized Triton GEMM (MatmulConfig, EVEN_K specialization, deterministic split-K)
  config_space.py   SPACE (19,200 configs), static + shape pruning rules, HANDPICKED configs
baselines/
  roofline.py         B0: measured peak TFLOP/s and DRAM GB/s
  torch_mm.py         B1: torch.matmul (default cuBLAS)
  cublaslt.py         B2/B3: cuBLASLt heuristic top-N through ctypes (same libcublasLt torch loads)
  cublaslt_search.py  B3+: enumerate-and-screen over the cuBLASLt algorithm space (subprocess, cached per shape)
  triton_tutorial.py  B5: Triton tutorial matmul, default config
suites/
  generate.py   deterministic suite builder (seed 464) and the tune/held-out split
  v1/*.json     S1-S7, edge (169 shapes), noise (30 shapes); 608 suite shapes
analysis/
  report.py          analyze: baseline/noise/drift tables and figures
  portfolio.py       B6 over the full space, greedy portfolio, set cover, parameter importance, held-out G5
  selectors.py       trivial selectors (b6, nn1, knn5, rule, nn1f, knnf, knnc), picks file, held-out scoring
  alloc_variance.py  allocation-placement noise experiment
tests/       pytest unit tests (43 tests; most need a CUDA GPU)
reports/     write-ups, generated tables (*.md, *_summary.json, *.csv), figures/, snapshot/ (small committed copies of data)
data/<hw>/   all raw results, screens, env records, logs (gitignored, machine-specific)
learning/    background study notes (not part of the harness)
```

## 3. Environment

- Use the existing Python environment; versions are pinned in `requirements.txt` (torch 2.13.0+cu130, triton 3.7.1,
  numpy, pandas, pyarrow, matplotlib, nvidia-ml-py, pytest). Run everything **from the repo root** as `python -m ...`.
- **Standard conditions** (every gate number assumes them): SM clock locked at 1755 MHz by the user
  (`sudo nvidia-smi -lgc 1755,1755`; reset with `sudo nvidia-smi -rgc`), declared to the harness with
  `python -m tune --clock-lock 1755 <cmd>` (or `TUNE_CLOCK_LOCK=1755`), driver 595.99.02, 3 input placements pooled,
  cold L2 primary. The harness never runs as root; it probes the lock and warns on mismatch.
- Only compare sessions with the same lock state and driver. After a driver change, re-run the noise study:
  `python -m tune --clock-lock 1755 noise --rounds 3 --repeats 50 --n-allocs 3`.
- The GPU also drives the desktop display; compositor noise is part of the measured floor.

## 4. CLI reference (`python -m tune [--clock-lock MHZ] [--no-pin] <cmd>`)

Measuring commands pin the process to one CPU core and write under `data/<hw_id>/` (override with `--out`).

| Command | What it does | Output |
|---|---|---|
| `env` | Capture fingerprint, versions, NVML state, clock-lock probe, git commit/dirty | `env.json`, `env/<session>.json` |
| `check --suite edge --backends ...` | Correctness gate for every candidate over a suite | `check-<suite>.json`; exit 1 on any failure |
| `roofline` | B0: measured peak tensor TFLOP/s and DRAM bandwidth vs spec | `roofline.json` |
| `benchmark --suite S1,S2 --backends ... --cache-modes cold[,hot]` | Validate, then time every (shape, candidate). Resumable, `--budget 30m`, `--n-allocs 3`, `--graph-batch N` (hot only: CUDA-graph amortized), `--shapes-file`, `--out-name`, `--prune-shapes` | `<out-name>.jsonl` + `.raw.jsonl` |
| `baseline` | Alias of `benchmark` with `--which` defaulting to `torch,triton_tutorial` | same |
| `sweep [--split tune\|heldout] [--per-suite N] [--max-gflop G] [--workers W]` | Stage-1 screen: every pruned Triton config on a stratified shape subset (`--list-only` writes the subset only) | `sweep_screen[_heldout]/` (one JSON per shape + `subset.json`) |
| `noise --rounds 3 --repeats 100` | Noise-floor study over `suites/v1/noise.json` | `noise.jsonl` |
| `drift --minutes 10` | Continuous 8192³ thermal-drift run | `drift-*.csv` |
| `analyze --results results_lock1755.jsonl` | Tables + figures; output names get the results tag as a suffix | `reports/week1_results*.md`, `reports/figures/` |
| `portfolio [--heldout --results ...]` | B6 over the full space, greedy portfolio curve, set cover, parameter importance; `--heldout` scores the tune-fitted portfolio on held-out shapes | `reports/sweep_portfolio[_heldout].md` + summary JSON |
| `selectors [--results ... --graph-results ...]` | Fit trivial selectors on the tune screen, write picks, score on held-out shapes (screen + stage 2) | `data/<hw>/selector_picks.json`, `reports/selector_baselines.md` |

### Backend spec strings (`--backends`, comma-separated; `runner.candidates_for`)

| Spec | Meaning |
|---|---|
| `torch` | B1, `torch.matmul` |
| `triton_tutorial` | B5 |
| `triton:handpicked` | the 5 `HANDPICKED` configs |
| `triton:<config_key>` | any config, e.g. `triton:bm64_bn64_bk64_w4_s3_g1_sk8` |
| `triton:sweep[:k]` | the k (default 8) fastest screened configs for this shape (`TUNE_SWEEP_DIR`) |
| `triton:portfolio[:n]` | first n (default 29) members of the greedy portfolio (`TUNE_PORTFOLIO`, default `reports/sweep_portfolio_summary.json`) |
| `triton:picks` | selector picks for this shape (`TUNE_PICKS`, default `data/<hw>/selector_picks.json`) |
| `cublaslt[:N]` | cuBLASLt heuristic top-N (default 16; often returns 1–9). Rank 0 = B2, per-shape best = B3 |
| `cublaslt_deep[:budget_s]` | B3+ search (default 20 s/shape, cached in `TUNE_LT_SEARCH_DIR` = `data/<hw>/lt_search/`); best of `cublaslt` ∪ `cublaslt_deep` = B3+ |

## 5. Capabilities built in this phase

### Measurement harness (M0)
- **Timing** (`tune/timing.py`): CUDA events on the launch stream; untimed first call (JIT/heuristic/allocator); ≥25 ms
  warmup; adaptive repeats until ≥10 samples and RSE < 1% (caps: 1000 samples / 2 s; `converged` is stored);
  median is primary and p10/p25/p75/p90/p99/min/max/IQR/RSE are stored. A device-side `_sleep` backlog makes events measure device
  time rather than Python launch latency (chunks ≤256 launches). Cold mode zeroes a 64 MiB buffer before every sample. Hot mode does no flush.
  Graph mode captures N launches in a CUDA graph and reports time/N, for µs kernels.
- **Allocation pooling**: every measurement pools samples over 3 fresh input allocations shifted by seeded 1–64 MiB pads.
  This was the dominant noise source (up to 3.7% on memory-bound shapes), not heat or clocks.
- **Correctness gate** (`tune/correctness.py`): reference is fp32 cuBLAS on fp16-rounded inputs with TF32 off.
  The check is `|C−ref| ≤ rtol·|ref| + atol_frac·√K` (fp16 1e-2, bf16 3e-2), on a NaN-prefilled output. Compile, OOM and launch errors are recorded as
  `valid=false` + `failure_reason` and never timed.
- **Crash safety**: `inflight.json` marks the candidate in flight. After a context-killing fault the next run records it
  as `crashed_previous_run` and continues. Completed rows are skipped on resume. The sweep isolates each shape in a subprocess.
- **Provenance**: every row carries `session_id`, `driver`, `harness_commit`, `harness_dirty`, `clock_lock_state`,
  `clock_lock_mhz`, `env_file`, plus clock/temp/power/throttle at measurement time. Vendor rows record the actual device kernel names.
- **Records** (`tune/records.py`): append-only JSONL, fsync per row, schema in `FIELDS`, raw samples in `*.raw.jsonl`
  keyed by `record_id`.

### Kernel family and config space (M3)
- `TritonMatmul(MatmulConfig)`: tunables BLOCK_M/N/K, num_warps, num_stages, GROUP_M (L2 swizzle) and SPLIT_K. EVEN_K
  drops the K-mask when divisible. Split-K is deterministic: fp32 partials go to a `[SPLIT_K, M, N]` workspace and a separate reduce kernel combines them, with no atomics.
- Config key format: `bm{BM}_bn{BN}_bk{BK}_w{warps}_s{stages}_g{GROUP_M}_sk{SPLIT_K}` (`tune.sweep.parse_key` inverts it).
- Pruning is named and auditable. **Static** rules (per GPU): smem limit, accumulator regs > 128, < 8 accumulator elements per thread, tile too
  small for the warp count. They keep 10,840 of 19,200. **Shape** rules: oversized tiles, starved split-K, split-K workspace too large, no-op GROUP_M.

### Baselines (plan §4.1)
B0 roofline · B1 torch.matmul · B2 cuBLASLt heuristic top-1 · B3 best of the heuristic top-N · **B3+** best of B3 and a
20 s/shape enumerate-and-screen over algo × tile × stages × split-K × reduction × swizzle × custom · B5 Triton tutorial ·
B6 best single config from the family. B4 (CUTLASS) was not built.

### Shape suites (`suites/v1`, regenerate with `python -m suites.generate`)
S1 square sweep · S2 LLM layers (LLaMA-2/3 7B–70B dims × M ∈ {1…8192}) · S3 skinny/tall · S4 large-K · S5 unaligned/primes ·
S6 wave-boundary (built from the SM count) · S7 random log-uniform · `edge` (169) · `noise` (30). The split was fixed before
any results: **heldout_region** (9000 ≤ K ≤ 12000 or 300 ≤ M ≤ 450), **heldout_random** (30% of S7), the rest is **tune**, and edge/noise
are `aux`. Never fit anything on held-out shapes.

### Two-stage full sweep (M4)
1. **Screen** (`tune sweep`): every pruned config per shape gets compiled in parallel workers, validated and given a cheap cold timing
   (1 sample, cut if > 1.5× the best so far, else 5 samples; 15 with a trimmed mean under 50 µs).
2. **Stage 2** (`benchmark --backends triton:sweep:8,...`): the screen's top tiles re-timed with the full protocol in
   the same session as the baselines.

The screen biases the oracle low (winner's curse); stage 2 under-covers it. Report both.

### Analysis
- `portfolio`: regret = t(c,s)/min_c t(c,s), with inf (capped at 10) where pruned or invalid. Outputs B6 (lowest geomean regret), a greedy
  portfolio curve (size vs mean/p95), set cover at ε = 2/3/5%, parameter importance (regret from forcing each param to its worst value), and
  held-out scoring with a learning curve.
- `selectors`: b6, nn1 / knn5 on (log2 M, log2 N, log2 K), the hand rule `rule_config`, and **nn1f / knnf**, which add GPU-fill
  features (CTA count and wave efficiency for 6 reference tiles, K steps per split) and 8/16-alignment. Weights and k are fit by grouped
  leave-one-out on tune shapes only (no neighbour sharing two of M, N, K). **knnc** computes the same fill and split features per
  candidate (`candidate_features`: waves at that config's residency, last-wave efficiency, K steps per split, padding, EVEN_K) and scores
  each config by its own regret on the k tune shapes where it saw the closest conditions (fit cached in `data/<hw>/selector_knnc_fit.json`,
  ~6 min to refit). The selectors are scored against the screen oracle and the stage-2 cold and graph sessions, and against B3+.

## 6. Key results so far (RTX 3080, standard conditions)

| Gate (plan §8.2) | Status |
|---|---|
| G1 measurement validity | **Pass** for kernels ≥ 50 µs (cross-session p95 ≈ 1.75%; tie margin **ε = 2%**). Under 50 µs use ε = 5% per-launch, or rank with graph-amortized timing (no cross-session noise study yet) |
| G2 family ≥ 90% of B3+ | **Pass**: median 98.8% over 160 shapes (aligned 98.8%, unaligned 90.4%) |
| G3 selection value | **Pass**: B6 `bm32_bn32_bk32_w4_s5_g1_sk1` median regret 68–74%, p90 ≈ 275–307%; holds out of sample (median 48%) |
| G4 differentiation vs B3+ | **Fails ≥ 50 µs** (0 of 86 shapes > 5% faster). Narrow pass for tiny GEMMs (B3+ < ~5 µs: 12/20 shapes, 0.23–0.59 µs saved) |
| G5 tractability | In-sample pass (29 configs on 160 shapes, 48 on 477 for p95 < 5%). **Fails held-out**: static portfolios plateau at ~8–9% p95 (mean < 2%) whatever the number of tune shapes |

Other findings to keep in mind:
- Conclusion written per plan §8: **not viable as a cuBLAS competitor on this GPU, viable as a selection project.** The target is selecting
  from {Triton portfolio ∪ cuBLASLt configs}, or a non-NVIDIA backend.
- Losses ≥ 50 µs are kernel-level: unaligned strides (no 16-byte vector loads) and split-K reduction cost on tiny M, N. Better selection cannot fix them.
- Parameter importance: SPLIT_K ≫ BLOCK_M > BLOCK_K > BLOCK_N ≫ warps, stages, GROUP_M.
- Measured roofline: 62.9 TFLOP/s fp16 tensor, 686 GB/s DRAM. Under sustained load the card power-caps (throttle 0x4).
- **Current selector bar** (held-out, latest sessions `selectors3` / `selectors3-graph`): knnf at ≥ 50 µs cold (median 1.9%,
  p95 37%), knn5 under 50 µs graph (median 3.7%, p95 22%), knnf overall cold (median 1.8%, p95 23%, mean 6.5%). A learned model must beat these with this same harness. Per-candidate k-NN (knnc) does
  not: same per-candidate features, worse median (cold 4.4% vs knnf 1.8%); only its ≥ 50 µs tail is a little better (p95 32% vs 37%).

## 7. Rules for agents

**GPU job hygiene (has cost real data before):**
- Before any GPU measurement, run `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` and confirm the GPU is idle.
- Never run two GPU jobs at once. That includes pytest while a benchmark or sweep runs. Overlapping runs contaminate data, and it then has to be discarded.
- Track the real python PID (`pgrep -f "^python -u -m tune.*<pattern>"`), not a `setsid`/shell parent. Wait with a
  background `while kill -0 PID` loop.
- Killing a `tune sweep` parent orphans its spawn compile workers (~218 MiB GPU context each). Kill them by PID. Never use `pkill -f`
  with a pattern that also matches your own shell command.

**Measurement discipline:**
- Use the standard conditions and `--clock-lock 1755` for anything that goes into a report. Long runs belong in the background
  with a log in `data/<hw>/logs/`.
- Commit harness changes before measuring when possible. Rows record `harness_dirty`, and reports call it out when true.
- Kernels under 50 µs: rank them with `--cache-modes hot --graph-batch 50`. Per-launch times sit on a ~1.02 µs grid.
- Treat differences within ε (2% ≥ 50 µs, 5% below) as ties. Report distributions (median, p10/p90/p95, worst) and regimes, never only means.
- Compare against **B3+**, not B1/B2, before claiming any win over cuBLAS.
- Keep held-out shapes out of every fitting step (portfolio, selector weights, rule tuning). Say so explicitly when a choice was informed by held-out misses.

**Code and reports:**
- Match the existing style: dense module docstrings explaining the *why*, `from __future__ import annotations`, small pure
  functions, numpy/pandas for analysis, and no new dependencies without pinning them in `requirements.txt`.
- New backends go through `candidates_for` and expose `make_runner(a, b, out=None)`, so they are validated and timed like everything else.
- `data/` is gitignored. Copy small, report-relevant artifacts to `reports/snapshot/lock1755/`. Generated tables are named
  with the results tag (`week1_results_lock1755_<tag>.md`, `*_summary_*.json`).
- Add each new finding as a dated `## Update YYYY-MM-DD: ...` section at the top of `reports/week1.md`. Include what ran (session ids,
  row counts, invalid count, % power-capped), the tables, what changed in the gates, next steps and reproduce commands.
- Run `python -m pytest -q` after code changes (GPU required, GPU idle).

## 8. Reproduce the main pipeline

The selector, sweep-screen, portfolio and noise commands match the ones recorded in `reports/week1.md`. The B3+ baseline and
sweep stage-2 invocations were rebuilt from the backend specs and output names. Check the session logs in
`data/<hw>/logs/` for the exact arguments of a past run. The held-out portfolio stage 2 also used
`triton:sweep:8,triton:portfolio:60` with `TUNE_PORTFOLIO` pointing at that run's portfolio file.

```bash
python -m tune env && python -m tune check --suite edge && python -m tune --clock-lock 1755 roofline
python -m tune --clock-lock 1755 noise --rounds 3 --repeats 50 --n-allocs 3
python -m tune drift --minutes 10
# baselines + hand-picked family, cold and hot
python -m tune --clock-lock 1755 benchmark --suite S1,S2 \
  --backends torch,triton_tutorial,triton:handpicked,cublaslt:16,cublaslt_deep:20 --cache-modes cold,hot \
  --out-name results_lock1755_b3plus.jsonl
python -m tune analyze --results results_lock1755_b3plus.jsonl
# full sweep: screen (hours), then stage 2 on the screened subset
python -m tune --clock-lock 1755 sweep                         # tune split, 24/suite (use --per-suite 1000 for all 477)
D=data/rtx3080_sm86_68sms_10gb
python -m tune --clock-lock 1755 benchmark --shapes-file $D/sweep_screen/subset.json \
  --backends torch,triton_tutorial,triton:handpicked,triton:sweep:8,cublaslt:16,cublaslt_deep:20 \
  --out-name results_lock1755_sweep.jsonl
python -m tune portfolio
# held-out: screen, stage 2 with the portfolio, then the selectors
python -m tune --clock-lock 1755 sweep --split heldout
python -m tune portfolio --heldout --results results_lock1755_heldout477.jsonl
python -m tune selectors       # writes selector_picks.json; then re-time picks:
TUNE_SWEEP_DIR=$D/sweep_screen_heldout python -m tune --clock-lock 1755 benchmark \
  --shapes-file $D/sweep_screen_heldout/subset.json \
  --backends torch,triton:sweep:8,triton:picks,cublaslt:16,cublaslt_deep:20 --cache-modes cold \
  --out-name results_lock1755_selectors3.jsonl
TUNE_SWEEP_DIR=$D/sweep_screen_heldout python -m tune --clock-lock 1755 benchmark \
  --shapes-file $D/sweep_screen_heldout/subset_lt50us.json \
  --backends torch,triton:sweep:8,triton:picks,cublaslt:16,cublaslt_deep:20 --cache-modes hot --graph-batch 50 \
  --out-name results_lock1755_selectors3_graph.jsonl
python -m tune selectors       # re-score with both stage-2 files
```

## 9. Open next steps

1. **Learned selector v0**: rank candidate configs per shape with a trained model over shape features plus
   `candidate_features` (already computed per candidate). A nearest-neighbour vote over these features (knnc) is not enough:
   similar per-candidate conditions on tune shapes do not predict a config's regret on a new shape. Evaluate with
   `triton:picks` + `tune selectors` against knnf/knn5.
2. **Top-k + short online search**: time the model's top 3–5 at first use to close the tail.
3. **Joint {Triton ∪ cuBLASLt} selection**, especially under 50 µs where the families split the wins.
4. Noise study for graph-amortized mode (second session plus a reboot) before relying on tiny-GEMM wins.
5. Kernel-level fixes if differentiation matters: an aligned-copy or padded path for unaligned strides, and a fused or atomic split-K reduction.
6. Not built yet: B4 (CUTLASS), per-candidate subprocess isolation with hang timeouts (plan 7.2), adaptive sampling.
