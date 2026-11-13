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
  cli.py         subcommands: env check roofline benchmark baseline sweep noise drift analyze portfolio selectors bundle
  env.py         hw fingerprint, NVML telemetry, clock-lock probe, git provenance, CPU pinning, env.json
  timing.py      CUDA-event timer: warmup, adaptive repeats, cold/hot L2, backlogged launches, CUDA-graph amortized
  correctness.py fp32 reference, tolerance check, NaN-prefilled output, placed (address-shifted) inputs
  records.py     append-only JSONL schema (plan 5.3), raw-sample side file, resume keys, alignment flags
  runner.py      backend spec -> candidates, validate-then-time loop, inflight.json crash recovery, budget
  sweep.py       stage-1 screen of the full pruned Triton space (subprocess per shape, parallel compile)
  noise.py       noise-floor study and thermal-drift run
  profile.py     records the device kernel names that vendor backends launch
  online.py      top-k + short online search with the saved Ranker (compile, validate, short-time, pick); `select_kernel` adds the winner cache and the kernel bundle.
                 `search_shape` = `propose_lt` + `propose_triton` (ranking: needs the models / lightgbm) + `measure` (GPU half: needs no model);
                 `proposal_keys` / `backends_from_keys` carry a proposal to a process running another interpreter
  router.py       cost-only family router: after the cuBLASLt pool is timed, skip the Triton half when P(Triton still wins) < threshold (LightGBM, `data/<hw>/router.*` or `reports/snapshot/lock1755/router.*`); on in `select_kernel` only when the bundle does not serve the Triton kernels
  winner_cache.py per-shape winner cache: append-only JSONL (schema 2), one searched shape per line with the environment it was timed in; current / soft-stale / hard-stale tiers, `mark_verified`, `stale_entries`, `compact` (prunes superseded lines only inside an `env_id` scope), `model_id`; `backend_for` turns a choice into a runnable backend
kernels/
  triton_matmul.py  parameterized Triton GEMM (MatmulConfig, EVEN_K specialization, deterministic split-K)
  config_space.py   SPACE (19,200 configs), static + shape pruning rules, HANDPICKED configs
  bundle.py         prebuilt-kernel bundle, runtime side: Triton cache manager that serves compiled kernels from `kernels/bundle/` and falls back to compiling; `install()`, `status()`; the limitations are in its docstring
  bundle_build.py   bundle build side: enumerate every (config, argument specialization) the rankers' pools can launch, compile in workers, keep only cubin + json; `verify_bundle` is the zero-compile coverage gate
baselines/
  roofline.py         B0: measured peak TFLOP/s and DRAM GB/s
  torch_mm.py         B1: torch.matmul (default cuBLAS)
  cublaslt.py         B2/B3: cuBLASLt heuristic top-N through ctypes (same libcublasLt torch loads)
  cublaslt_search.py  B3+: enumerate-and-screen over the cuBLASLt algorithm space (subprocess, cached per shape)
  cublaslt_knn.py     cuBLASLt proposals: the deep-search winners of the nearest tune shapes (table from tune-split `lt_search/`, committed snapshot `reports/snapshot/lock1755/lt_knn_table.json`); family `cublaslt_knn`
  cublaslt_rank.py    learned ranker over every valid cuBLASLt config (LightGBM, `data/<hw>/lt_rank`, committed snapshot `reports/snapshot/lock1755/lt_rank.*`): runtime enumeration + features + top-m; family `cublaslt_rank`
  cublaslt_screen.py  full per-config screen of the tune shapes (every config's time, `data/<hw>/lt_screen/*.json.gz`): the training set of a learned cuBLASLt ranker
  triton_tutorial.py  B5: Triton tutorial matmul, default config
suites/
  generate.py   deterministic suite builder (seed 464) and the tune/held-out split
  fresh.py      fresh evaluation shapes (`python -m suites.fresh`, seed 465 -> `suites/fresh/fresh465.json`, 60 shapes in no suite and sharing two of M, N, K with no tune shape, <= 1200 GFLOP); `tune sweep --shapes-file` screens them
  v1/*.json     S1-S7, edge (169 shapes), noise (30 shapes); 608 suite shapes
analysis/
  report.py          analyze: baseline/noise/drift tables and figures
  portfolio.py       B6 over the full space, greedy portfolio, set cover, parameter importance, held-out G5
  selectors.py       trivial selectors (b6, nn1, knn5, rule, nn1f, knnf, knnc), picks file, held-out scoring
  learned.py         learned selectors v0: DT, linear SVM, LightGBM, RF, logistic regression; cls and rank forms; grouped CV;
                     Ranker (saved lgbm_rank), top-k scoring (build_topk)
  alloc_variance.py  allocation-placement noise experiment
  plots.py           figures + reports/results_overview.md for the current state (baselines, noise, selectors, kernel-selection stats); CPU-only, reads existing results (`python -m analysis.plots`)
  zero_shot.py       zero-shot transfer: the saved lgbm_rank on another GPU, scored vs that GPU's own screen (`python -m analysis.zero_shot run|score`); `load_screen` keeps `slow` screen rows
  lt_ceiling.py / lt_clusters.py / lt_knn_scan.py / lt_knn_ab.py / lt_rank.py / lt_rank_ab.py / router.py / confirm_ab.py / joint_rank.py  the cuBLASLt k-NN, learned-ranker, router and joint-score study (router: fit + held-out score; confirm_ab: paired scoring of the router / confirmation online runs; joint_rank: the negative result) (CPU only): where the joint search's regret comes from, winner structure, tune-only leave-one-out choice of k, held-out A/B from one stage-2 session
  cross_gpu.py       cross-GPU comparison on the held-out shapes the runs share: regret vs each GPU's own screen, speed vs B3+, family ceiling, PTX mma check, paired bootstrap; CPU-only (`python -m analysis.cross_gpu --side LABEL DIR ...`, docstring has the command). Never compare a subset run to the 86-shape `REFERENCE_3080` constants in `zero_shot.py`
lscli/       the learned-selector CLI and daemon (docs/lscli/spec.md is the contract, docs/lscli/plan.md the PR plan):
  client.py      opt-in torch patch (LSCLI_PATCH=1 via lscli.pth, or lscli.enable()): dispatches cached winners, counts shapes
  worker.py      measure / env / verify under the USER's interpreter (line protocol on a private fd)
  daemon.py      idle-time loop: gate.py (NVML + window + power + locks), scheduler.py, jobs.py (process trees, preemption),
                 models.py (native / zero-shot sets), propose.py (cuBLASLt ranker proposals), training.py (train_best stages)
  cli.py         `lscli` commands; config.py, files.py, envs.py, usage.py, search_queue.py, control.py, doctor.py
install/     install.sh (curl | bash), lscli.service (systemd --user), lscli-clock (optional root clock helper),
             zero_shot.tar.xz (the 3080 zero-shot model set the installer unpacks; rebuilt with pack_zero_shot.py, reproducibly)
tests/       pytest unit tests (~300 tests; most need a CUDA GPU)
reports/     write-ups, generated tables (*.md, *_summary.json, *.csv), figures/, snapshot/ (small committed copies of data)
data/<hw>/   all raw results, screens, env records, logs (gitignored, machine-specific)
learning/    background study notes (not part of the harness)
```

## 3. Environment

- Use the existing Python environment; versions are pinned in `requirements.txt` (torch 2.13.0+cu130, triton 3.7.1,
  numpy, pandas, pyarrow, matplotlib, nvidia-ml-py, pytest, scikit-learn, lightgbm). Run everything **from the repo root** as `python -m ...`.
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
| `sweep [--split tune\|heldout] [--per-suite N] [--max-gflop G] [--workers W]` | Stage-1 screen: every pruned Triton config on a stratified shape subset (`--list-only` writes the subset only). `--workers 0` (default) = auto: up to 10, capped by free host RAM and VRAM, re-read before every shape (budget ~1.25 GiB + ~450 MiB per worker; each ~200-config chunk compiles in a fresh process because a worker's RSS grows ~0.6 MiB per config, 1.4 GiB+ over a whole shape, which OOM-killed a 16 GB machine's desktop session); a killed worker halves the pool, a shape that does not fit in VRAM is recorded as `skipped` (retried on the next run) | `sweep_screen[_heldout]/` (one JSON per shape + `subset.json`) |
| `noise --rounds 3 --repeats 100` | Noise-floor study over `suites/v1/noise.json` | `noise.jsonl` |
| `drift --minutes 10` | Continuous 8192³ thermal-drift run | `drift-*.csv` |
| `analyze --results results_lock1755.jsonl` | Tables + figures; output names get the results tag as a suffix | `reports/week1_results*.md`, `reports/figures/` |
| `portfolio [--heldout --results ...]` | B6 over the full space, greedy portfolio curve, set cover, parameter importance; `--heldout` scores the tune-fitted portfolio on held-out shapes | `reports/sweep_portfolio[_heldout].md` + summary JSON |
| `selectors [--results ... --graph-results ...]` | Fit trivial selectors on the tune screen, write picks, score on held-out shapes (screen + stage 2) | `data/<hw>/selector_picks.json`, `reports/selector_baselines.md` |
| `learned [--results ... --graph-results ...]` | Grouped-CV grid search (cached, ~2 h first run) and held-out scoring of the 10 learned selectors; reads `selector_picks.json` for b6/knn5/knnf | `data/<hw>/learned_picks.json`, `learned_search.json`, `reports/learned_selectors.md` |
| `topk [--results ... --graph-results ... --online ...]` | Refit lgbm_rank on all tune shapes, save `data/<hw>/lgbm_rank.txt/.json`, write top-5 picks per held-out shape, score best-of-k and the online search | `data/<hw>/learned_topk_picks.json`, `reports/learned_topk.md` |
| `ranker-v2 [--extra-screens sweep_screen_tail]` | CPU only. Fit rank v2 (v1's rank form + analytic roofline features + near-top row weights + the tail suite as extra tune shapes) on all tune shapes; hyperparameters are v1's CV-chosen ones (v2 is always a regression: when v1's search chose lambdarank, v2 takes the best regression entry of that search, `learned.v2_hp`); diversification rule refit on v2's own out-of-fold scores. Reads no held-out shape. Use with `online --ranker data/<hw>/lgbm_rank_v2` | `lgbm_rank_v2.txt/.json`, `lgbm_rank_v2_cv_scores.npz` |
| `bundle build\|status\|verify` | The prebuilt Triton kernel bundle (`kernels/bundle.py`). `build` compiles every kernel the rankers' pools can launch (v1 ∪ v2 pools + hand-picked, fp16, every specialization) into a scratch cache (`data/<hw>/bundle_build_cache`, resumable) and keeps only cubin + json in `kernels/bundle/` (gitignored); GPU idle, ~1 GiB host RAM + ~360 MiB VRAM per worker. `verify` warms every runnable (pool config, suite shape) pair with the bundle installed and an empty user cache and requires 0 compiles, then runs a few kernels against torch.matmul; exit 1 on any miss. `status` prints whether the bundle matches this Triton / GPU. `pack` writes the committed archive `kernels/bundle.tar.xz` (19 MiB; `unpack` is what `install()` does on a fresh clone, ~11 s, atomic). `--pool`, `--dtypes`, `--workers`, `--limit` | `kernels/bundle/<cache key>/{_matmul_kernel,_splitk_reduce_kernel}.{cubin,json}`, `manifest.json`; `kernels/bundle.tar.xz` |
| `online --shapes-file ... --k 5 [--order diverse\|inner] [--lt-n 16] [--k-frequent 5 --frequent-calls 1000000] --fresh-cache` | First-use search: rank (plain or diversified top-k), compile + validate + short-time the top k (empty Triton cache), keep the fastest; choices for every k' <= k stored. `--lt-n N` also short-times cuBLASLt's heuristic top N in the same pool (joint selection, no compile; `choice_lt` stores k' x lt_n' in {0, 1, 4, N}). `--lt-knn N` (with `--lt-n`) adds cuBLASLt proposals from the deep searches of the N nearest tune shapes (top 2 each, no compile; `baselines/cublaslt_knn.py`); they join `choice` but no `choice_lt` entry, so `choice_lt["<k>_lt16"]` stays the heuristic-only search for A/B. `--router [--router-threshold X]` times the cuBLASLt candidates first and skips the Triton half when the family router (`tune/router.py`) says Triton cannot win (`--router-threshold 0` = LT first, never skip: the ordering control); `--confirm N` re-times the N fastest candidates in reverse order and ranks by the mean of both timings (cancels the timing-order bias; used with the router). `--lt-rank M` (with `--lt-n`) adds the M configs the learned cuBLASLt ranker (`baselines/cublaslt_rank.py`) predicts fastest; same family-subset rule. `--k-frequent` makes k per shape (call-count policy `tune.online.KPolicy`, default k 3 / frequent 5): shapes in the file with `calls` >= `--frequent-calls` or `frequent: true` get `--k-frequent` candidates, the rest `--k`; one shape from Python: `tune.online.select_kernel(ranker, M, N, K, calls=...)` | `online_lgbm_rank[_diverse].jsonl` (`online_joint_*` with `--lt-n`) |

Not part of `python -m tune`: `python -m analysis.plots [--theme light|dark] [--only baselines,noise,selectors,selection]` redraws every overview figure into `reports/figures/overview/` and rewrites
`reports/results_overview.md` from the files already in `data/<hw>/` and `reports/`. It never touches the GPU (safe next to a running job), so it is not a `tune` subcommand, which imports torch. `python -m analysis.zero_shot` runs the saved ranker on another GPU with no refit (variants `as_is` with the source GPU's numbers and
`native` with this device's; diversified top-3 + the short online search) and scores it against that GPU's own held-out screen (`reports/zero_shot_<hw_id>.md`).
Runbook (standalone version with the prerequisites, scoring and pitfalls: [`docs/zero_shot_runbook.md`](docs/zero_shot_runbook.md); how to read the report is in the module docstring), same GPU job hygiene as section 7:
1. Copy `lgbm_rank.txt/.json` (rank v1) and `lgbm_rank_v2.txt/.json` (rank v2) from the source GPU's `data/<hw>/` by hand (`data/` is gitignored; ~35 MB each). Match `requirements.txt` versions. v2's analytic-feature constants are fixed model units: do not change them for the new GPU.
2. Pick an SM clock lock the target GPU holds under load and use it for every command (`--clock-lock X`); only compare sessions with the same lock and driver.
3. The oracle: `python -m tune --clock-lock X sweep --split heldout --per-suite 1000` (the same 86 shapes as the 3080 run; ~1.3 h there).
4. `python -m analysis.zero_shot --clock-lock X run --ranker v1=<dir>/lgbm_rank v2=<dir>/lgbm_rank_v2 --k 5 --lt-n 16 --fresh-cache`, then
   `python -m analysis.zero_shot --clock-lock X score --k 3 --report <dir>/k3` (and `--k 5 --report <dir>/k5` for the frequent tier; each call writes one fixed file name, so use separate `--report` folders: a k = 5 run stores every k' <= 5; with `--k 3 --k-frequent 5` shapes carrying `calls` >= 1M or `frequent: true` get k = 5)
   (`--clock-lock` goes before the subcommand; `run` resumes per shape and k; never overlap it with the sweep; a bare `--ranker <dir>/lgbm_rank --k 3` is the original Triton-only run and keeps its file names).
   The screen oracle is Triton-only, so `score` rates the Triton-only choice of a joint (`--lt-n`) run. The joint choice is scored against B2 / B3 in a short stage-2 session:
   `python -m analysis.zero_shot picks` writes its inputs and prints the `tune benchmark` and `python -m analysis.ranker_compare` commands (~0.5 h; add `cublaslt_deep:20` for B3+).
5. Add a dated update to `reports/week1.md` with the table. Read the `regret > 50%` and `unscored` columns first: the screen cuts a config as `slow` after one sample once it is > 1.5x the best so far, and `load_screen` keeps those rows (their regret is > 50%); a scorer that drops them hides the bad picks (this happened once, `reports/week1.md` 2026-11-02 later). For a run on a subset of the 86 shapes (a `pipeline/run.sh --quick` GPU has 27), compare on the shared shapes with `python -m analysis.cross_gpu`, not with the 86-shape reference rows.

### Backend spec strings (`--backends`, comma-separated; `runner.candidates_for`)

| Spec | Meaning |
|---|---|
| `torch` | B1, `torch.matmul` |
| `triton_tutorial` | B5 |
| `triton:handpicked` | the 5 `HANDPICKED` configs |
| `triton:<config_key>` | any config, e.g. `triton:bm64_bn64_bk64_w4_s3_g1_sk8` |
| `triton:sweep[:k]` | the k (default 8) fastest screened configs for this shape (`TUNE_SWEEP_DIR`) |
| `triton:portfolio[:n]` | first n (default 29) members of the greedy portfolio (`TUNE_PORTFOLIO`, default `reports/sweep_portfolio_summary.json`) |
| `triton:picks` | selector picks for this shape (`TUNE_PICKS`, default `data/<hw>/selector_picks.json`; `learned_picks.json` for the learned selectors) |
| `cublaslt[:N]` | cuBLASLt heuristic top-N (default 16; often returns 1–9). Rank 0 = B2, per-shape best = B3 |
| `cublaslt_knn[:k]` | the proposals for this shape: top 2 configs of each of the k (default 5) nearest tune shapes' deep searches (family `cublaslt_knn`; key rank = rank among kept proposals) |
| `cublaslt_rank[:m]` | the m (default 10) configs the learned ranker predicts fastest among all configs valid for this shape (family `cublaslt_rank`, key prefix `r`) |
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

### Prebuilt kernels and the winner cache (shipping the selector)
- **Bundle** (`kernels/bundle.py`, `kernels/bundle_build.py`, `docs/kernel_bundle.md`): the online search spends ~2/3 of its first-use cost compiling its top-k Triton configs (~0.13 s each). The bundle holds the compiled binary of every kernel the saved rankers can pick, so a fresh clone loads them instead.
  Triton's own cache cannot just be copied: its group files store absolute paths. `BundleCacheManager` (a `FileCacheManager` selected through `TRITON_CACHE_MANAGER`) looks in this machine's cache, then the read-only bundle, then lets Triton compile (into the user cache, never the repo). `select_kernel(use_bundle=True)` installs it; `tune online --fresh-cache`, `analysis.zero_shot` and the benchmark commands do not, so every measured compile cost stays a true first use.
- **Winner cache** (`tune/winner_cache.py`): `select_kernel(cache=True)` returns the stored winner of an earlier search at least as wide (k and lt_n) with no ranking, compiling or timing, and stores the winner of every search it runs. Default file `data/<hw>/winner_cache.jsonl` (`TUNE_WINNER_CACHE=<path>` moves it, `=off` disables it). The default cache is one instance per process (re-read only when the file changed: a hit costs ~0.1 ms) and declares the model set (`tune.online.model_ids`: the Triton ranker passed in, this GPU's cuBLASLt ranker and router files), so a retrained model makes the old winners soft-stale. The experiments pass `cache=None`.
  Each line carries its environment (GPU, Triton, torch, CUDA runtime, cuBLASLt, driver, VBIOS, Triton kernel source hash, declared clock lock, optional model ids). Another GPU, or a changed kernel source for a Triton winner, is **hard-stale** (never served); any other difference is **soft-stale**: a miss by default, served by `select_kernel(serve_stale=True)` after one correctness check in the new environment (`mark_verified`), with `reason: "cached_stale"` so the caller can queue a re-search. `compact()` drops lines a current-environment search replaced.

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
  portfolio curve (size vs mean/p95/p99), set cover at ε = 2/3/5%, parameter importance (regret from forcing each param to its worst value), and
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
| G1 measurement validity | **Pass** for kernels ≥ 50 µs (cross-session p95 ≈ 1.75%; tie margin **ε = 2%**; p99 of the two-session cold / hot difference 2.1% / 2.2% over 44 / 43 kernels, i.e. essentially the worst kernel sits at the margin, `reports/results_overview.md`). Under 50 µs use ε = 5% per-launch, or rank with graph-amortized timing (no cross-session noise study yet) |
| G2 family ≥ 90% of B3+ | **Pass**: median 98.8% over 160 shapes (aligned 98.8%, unaligned 90.4%) |
| G3 selection value | **Pass**: B6 `bm32_bn32_bk32_w4_s5_g1_sk1` median regret 68–74%, p90 ≈ 275–307%, p99 ≈ 1000% (477 tune shapes); holds out of sample (median 48%, p99 973%) |
| G4 differentiation vs B3+ | **Fails ≥ 50 µs** (0 of 86 shapes > 5% faster). Narrow pass for tiny GEMMs (B3+ < ~5 µs: 12/20 shapes, 0.23–0.59 µs saved) |
| G5 tractability | In-sample pass (29 configs on 160 shapes, 48 on 477 for p95 < 5%; the same portfolios have p99 7.2% at 50 configs and 2.8% at 100). **Fails held-out**: static portfolios plateau at ~8–9% p95 (mean < 2%) whatever the number of tune shapes, and the tail is fatter: held-out p99 19.7% at 50 configs and 18.2% at 100 (p95 11.1%) |

Other findings to keep in mind:
- Conclusion written per plan §8: **not viable as a cuBLAS competitor on this GPU, viable as a selection project.** The target is selecting
  from {Triton portfolio ∪ cuBLASLt configs}, or a non-NVIDIA backend.
- Losses ≥ 50 µs are kernel-level: unaligned strides (no 16-byte vector loads) and split-K reduction cost on tiny M, N. Better selection cannot fix them.
- Parameter importance: SPLIT_K ≫ BLOCK_M > BLOCK_K > BLOCK_N ≫ warps, stages, GROUP_M.
- Measured roofline: 62.9 TFLOP/s fp16 tensor, 686 GB/s DRAM. Under sustained load the card power-caps (throttle 0x4).
- **Joint selection and rank v2 (2026-11-01).** `tune online --lt-n 16` puts cuBLASLt's heuristic top-16 in the same short-timed pool as the Triton picks: top-3 + LT16 costs ~0.6 s first use and is > 5% faster than B2 on 49 of 86 held-out shapes and slower on 0
  (median 0.91 vs B2, 1.00 vs B3+; B3+ stays an offline bar). `KPolicy` / `select_kernel` (`tune/online.py`) choose k = 3, or 5 for shapes the caller marks frequent (`calls` >= 1M). Rank v2 (`python -m tune ranker-v2`, analytic features + near-top weights + the `tail` suite as tune data) halves the Triton-only held-out tail
  (regret vs the Triton oracle at top-3: 0.4% / 9.5% / 15.1% / 2.3% -> 0.2% / 4.6% / 9.6% / 1.2% median / p95 / p99 / mean) but moves the joint search only within noise, for +0.08 s. Details and caveats: `reports/week1.md` 2026-11-01, `reports/ranker_compare.md`. Score rankers side by side with `python -m analysis.ranker_compare`.
- **Second GPU (RTX 2060, Turing, 2026-11-02).** Triton's `tl.dot` compiles to FMAs on sm_75 (no `mma.sync` in any cached kernel): the best Triton config is 0.44x of B3+ (3080: 1.0x), so the family fails G2 and G4 there. Retrained on 56 shapes (~10 h of sweep) the ranker selects as well as on the 3080 (online top-3 0.7% / 7.5% / 13.3% / 2.5% median / p95 / p99 / mean, 0 of 27 picks > 50% off); the 3080 rankers used zero-shot do not transfer for Triton-only selection (13 of 27 top-3 picks > 50% off in `native`, the 2060 oracle uses BLOCK_K = 16 everywhere), but the joint search with cuBLASLt top-16 lands at its pool's ceiling (0.98 of B3+, cuBLASLt chosen on 24 of 27). Confounded by the missing tensor-core path; a second tensor-core GPU is open. `reports/week1.md` 2026-11-02, `reports/cross_gpu.md`.
- **Third GPU (RTX 3090, Ampere sm_86, 2026-11-03, unlocked clock, driver 590.48.01).** The 3080 rankers used zero-shot on the same 27 held-out shapes transfer: 0 of 27 top-3 picks > 50% off in all four variants (2060: 13), online top-3 regret 5.4 / 3.7 / 0.9 / 1.8% median and 26.5 / 14.8 / 14.4 / 12.6% p99 (v1 as_is / v1 native / v2 as_is / v2 native) vs 1.5% and 9.2% (3080 v1; p99 is essentially the worst of 27 shapes); v2 is within noise of the 3080 (+0.5 to +1.4 points mean), v1 as_is is worse (+3.5 [+1.1, +6.4]). Joint search with cuBLASLt top-16: 0.99-1.00 of B3+ median, same as the 3080. No retrained 3090 ranker and no noise study (unlocked), so margins are borrowed. `reports/week1.md` 2026-11-03, `reports/cross_gpu.md`.
- **cuBLASLt k-NN proposals (2026-11-04).** The joint search's gap to B3+ is cuBLASLt configs the heuristic never lists (20 of 86 held-out oracle winners). `select_kernel` now also times the top-2 deep-search configs of the 8 nearest tune shapes (`tune.online.KNN_DEFAULT`, picked on a tune-only grouped leave-one-out scan, `analysis.lt_knn_scan`): v2 top-3 + LT16 regret vs the joint oracle 0.9% / 10.7% / 22.6% / 2.6% -> 0.4% / 6.0% / 8.3% / 1.5% (median / p95 / p99 / mean; > 10% off on 6 -> 0 shapes; paired 0.990 [0.982, 0.997], 7 faster / 0 slower), +0.10 s first use, about half of the ceiling's gap (0.4% / 2.4%). G4 unchanged. One GPU's table (3080 / sm_86); the (5, 2) first run's k was seen against held-out hit rates, the (8, 2) run was not. `reports/week1.md` 2026-11-04.
- **Learned cuBLASLt ranker (2026-11-04, later).** A LightGBM regressor of log regret per (shape, valid cuBLASLt config), trained on the full screen of the 652 tune shapes (`baselines/cublaslt_screen.py`, `analysis/lt_rank.py`), is now the default cuBLASLt proposal source in `select_kernel` (`lt_rank=5`; k-NN8 only as the fallback when it has no model for the GPU). Tune grouped CV (exact times): at 10 candidates it closes 78% of the gap to the screen winner vs 71% for k-NN 8 x 2. Held-out (v2 top-3 + LT16 + proposals, one stage-2 session, regret vs the **extended oracle** that includes the proposals' own timings; the first write-up used an oracle without them, which gave the proposals negative regret on 4 shapes and read 0.4 / 4.7 / 1.2% for the ranker): regret median / p95 / p99 / mean 0.7% / 11.7% / 22.8% / 2.8% (heuristic only) -> 0.6% / 6.9% / 8.3% / 1.7% (k-NN8) -> **0.5% / 5.5% / 8.4% / 1.4%** (ranker top-5) (p99 on 86 shapes is essentially the worst shape; ranker and k-NN8 tie there); shapes > 5% off 14 / 9 / 5; slower than B3+ on 17 / 13 / 7; +0.14 s first use. Paired time ratio vs the heuristic 0.987 [0.978, 0.994]; regret difference ranker - heuristic -1.44 points mean [-2.44, -0.65], p95 -6.8 [-13.8, -1.7], p99 -12.8 [-20.0, -4.5]: a real win. **Ranker vs k-NN8 is not shown on held-out**: time ratio 0.9966 [0.9925, 1.0007], regret difference -0.35 [-0.82, +0.09] mean, -1.2 [-3.4, +1.3] p95, +0.2 [-2.8, +2.8] p99; the evidence for the ranker over k-NN is the tune CV and the shape counts, and k-NN8 + ranker top-5 is better than k-NN8 alone (mean -0.45 [-0.81, -0.16]). Against B3+ the pool ties at >= 50 us (geomean 1.001, 4 faster / 6 slower of 30); its advantage (0.951, 19 / 1) is in the 56 graph-timed sub-50 us shapes. 3080 / sm_86 model only (tile ids, 68-SM features). `reports/week1.md` 2026-11-04 (later) and (audit).
- **Family router and the timing-order bias (2026-11-04, evening).** With the Triton kernels really compiling (no bundle) the Triton half is ~0.6 of ~0.8 s and the final choice is cuBLASLt on 40 of 86 held-out shapes; with the bundle in a fresh process the whole search is ~0.17 s and the Triton half ~0.06 s. `select_kernel(router="auto")` therefore turns the router on only without a serving bundle. Router: LightGBM classifier of P(Triton's best beats the cuBLASLt value by > the tie margin) from the shape and the measured cuBLASLt time / roofline (tune out-of-fold AUC 0.82, threshold 0.02 by a rule fixed before held-out), cuBLASLt timed first. Held-out (v2 + LT16 + ranker top-5, extended oracle): regret 0.5 / 5.5 / 8.4 / 1.4% -> 0.4 / 4.9 / 8.8 / 1.3% (router with confirmation; median / p95 / p99 / mean: the router does not improve the p99), first-use cost 0.796 -> 0.713 s (14 of 86 skipped, none worse than the same-confirmation Triton-first search). **Finding: the order in which candidates are short-timed changes which near-tied candidate wins** (cuBLASLt-first with no skipping: worse than Triton-first on 5 shapes, better on 1; three identical-order runs flip 0), so the router needs the reverse-order `confirm` pass (`ROUTER_CONFIRM = 3`), which brings the control back to 3 / 2. A single additive cross-family score (family best-time model + within-family rank) does not work: sign of "who wins" right on 78% of tune shapes, joint top-10 p95 9.1% vs 5.5% for the fixed pool (`analysis/joint_rank.py`, not shipped). `reports/week1.md` 2026-11-04 (evening).
- **Fresh-shape check of the shipped selector (2026-11-05).** 60 new shapes (30 random, 30 LLM-layer dimensions of models not in S2; disjoint from every suite shape and from any tune shape sharing two dims; `suites/fresh.py`), frozen models, one stage-2 session, real Triton screen oracle. Regret vs the joint oracle (median / p95 / p99 / mean): heuristic-only 1.0 / 17.4 / 33.6 / 3.3% -> k-NN8 1.0 / 6.7 / 27.6 / 2.5% -> learned ranker top-5 **0.6 / 5.6 / 12.8 / 1.6%** (shapes > 5% off 9 / 7 / 4; paired vs heuristic 0.9851 [0.9692, 0.9973], 5 faster / 0 slower); vs B2 median 0.924 (42 faster / 1 slower), vs B3+ 1.000. Router (+ confirm 3): same quality (0.5 / 5.2 / 12.9 / 1.6% vs 0.7 / 5.6 / 12.9 / 1.8%; the router leaves the p99 alone), first-use cost -10% (no bundle), 15 of 60 skipped, none truly Triton-better; AUC 0.861 on the fresh shapes (tune OOF 0.823). `reports/week1.md` 2026-11-05.
- **Triton-only selector bar (v1): lgbm diverse online top-3** (`tune online --order diverse`, sessions `topkd` / `topkd-graph` / `online-…-topkd`). lgbm_rank (`analysis/learned.py`) is a LightGBM regressor
  of log regret per (shape, candidate) over each tune shape's top-5 pool, using shape features + config params + `candidate_features`. It ranks the shape. `diverse_order` keeps the top pick, then takes the best config
  of each new (BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K) group within 0.05 predicted log regret. That rule was fit on out-of-fold tune scores and is saved in `lgbm_rank.json`. The 3 picks are compiled, validated and short-timed (graph when < 50 µs), and the fastest is kept.
  Cold all: 1.0% / 9.9% / 2.7% (median / p95 / mean; the same rule in the later `topki` session reads 1.1% / 9.1% / 20.2% / 2.6% with p99, so differences of ~0.1-0.2 points in a median are session noise; the p99s below are all from the `topki` session, the current `reports/learned_topk.md`, and p99 is essentially the worst shape). ≥ 50 µs: 0.9% / 11.1% / 2.5% (`topki` p99 17.1%). Graph < 50 µs: 0.2% / 9.9% / 2.1% (`topki` p99 15.0%). Cost: ~0.4 s first use per shape (JIT compile).
  Plain top-3 in the same sessions: 1.1% / 12.8% / 3.1% cold, 0.5% / 12.2% graph (`topki` p99 22.4% cold, 14.8% graph). Top-1: 1.9% / 18.1% / 4.1% (`topki` p99 23.8%). knnf: 2.7% / 22.8% / 6.5% (`topki` p99 40.9%). Diverse is never slower than plain per shape beyond ε.
  An inner pass over warps / stages of the top tile (`--order inner`, fit on tune CV, not chosen there) ties it except for fixing 381×1401×250 (graph p95 9.7% → 8.2%). It is never slower, but it is not adopted,
  because that would fit the rule to held-out data. Held-out p99 (essentially the worst shape): diverse online top-3 20.2%, knnf 40.9%, knn5 122%; a 100-config static portfolio still 18.2%. Remaining tail (+10–22%) is ranker error: 320×1×79, 5398×196×774, 4065×8842×28. Shape → config classifiers and every linear model are worse than k-NN.
  Against B3+: 1.01 under 50 µs graph; G4 still fails at ≥ 50 µs. The Triton-only selector is slower than B3+ on 35 of 86 held-out shapes (v2). Its win over knnf at equal budget (top-1, no timing) is in the tail and mean, not the median (2.4% vs 2.5%); the top-3 numbers pay ~0.4 s of compile and timing that a table lookup does not.

## 6b. lscli (the selector as a service)

`lscli` turns `select_kernel` into a per-user service (spec: `docs/lscli/spec.md`). Pieces an agent must not break:
- **The daemon process never creates a CUDA context.** NVML only, LightGBM on the CPU; everything GPU runs in a job
  (`lscli.propose`, `lscli.worker` under the user's interpreter, `train_best.sh` stages). Otherwise NVML lists the daemon
  as a compute process and the idle gate never opens. `tests/test_lscli_daemon.py` checks the end-to-end path.
- **Winners are timed where they run**: the worker runs under the registered env's interpreter; the daemon writes the
  entry only after a final gate check. Entries carry `env_extra` (clock policy, model ids, `env_id`); see the
  winner-cache tiers in `tune/winner_cache.py`.
- **No model set = cuBLASLt-only searches**: `select_kernel(tune.online.LT_ONLY, ...)` / a daemon with no models
  searches cuBLASLt alone and stores `k = 0`; lookups ask for `k = 0` so such an entry counts, and it is widened once a
  ranker exists. Do not pass `None` for "no ranker": `None` is the tests' placeholder for a faked search.
- **Daemon measurements are not study data**: they run at stock or helper-locked clocks with no noise study and are
  never written into `data/<hw>/` result files or reports.
- **Branches**: nothing is committed directly to `learned-selector-cli` (or `learned-selectors`); every change is a PR
  from its own branch, reviewed first.

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
- Treat differences within ε (2% ≥ 50 µs, 5% below) as ties. Report distributions (median, p10/p90/p95/p99, worst; at n ≈ 60-90 shapes p99 is essentially the worst shape, so name it) and regimes, never only means.
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

Full training of the shipped selector on another GPU (rank v1 + v2, cuBLASLt ranker, router; optional bundle; held-out validation; models packaged into the run directory):
`pipeline/train_best.sh [--smoke] [--clock-lock X] [--with-bundle] [--skip-validation]` (stage list: `--list-stages`; resumable with `--resume <run dir>`; ~35 h on a 3080, extrapolated). Output: `pipeline/runs/train_best-<mode>-<ts>/{data/<hw>,reports,models,SUMMARY.md}`; `models/INSTALL.md` says where the files go. Its CPU stages (`lt_fit`, `router_fit`, `package`, `summary`) were rehearsed on the 3080's data and reproduce the committed fits (router AUC 0.823, threshold 0.02); the GPU stages were not run end to end.

## 9. Open next steps

1. **Learned selector v0: done** (lgbm_rank, see §6 and `reports/week1.md` 2026-10-19 late). Next: lambdarank / regression blends, a larger candidate pool.
2. **Top-k + short online search: done** (`tune online --order diverse`, top-3; inner warps / stages pass tried, a tie). Next: better ranking for N = 1 / tiny-K shapes and near-optimal tiles ranked low.
3. **Joint {Triton ∪ cuBLASLt} selection: done without a router, with k-NN cuBLASLt proposals** (2026-11-04; learned cuBLASLt ranker and the cost-only router done; one additive score over both families tried and not adopted. Next: the 3090 screen, a second, wider order-bias fix (interleaved timing); the shipped selector was checked on fresh shapes 2026-11-05) (`--lt-n 16`, see §6). Left: a cost-only family router (skip Triton compiles when cuBLASLt is clearly ahead), the joint selector in the zero-shot runbook, and faster v2 ranking (+40 ms). The cache that moves a shape from k = 3 to k = 5 is the winner cache (§5).
4. Noise study for graph-amortized mode (second session plus a reboot) before relying on tiny-GEMM wins.
5. Kernel-level fixes if differentiation matters: an aligned-copy or padded path for unaligned strides, and a fused or atomic split-K reduction.
6. Not built yet: B4 (CUTLASS), per-candidate subprocess isolation with hang timeouts (plan 7.2), adaptive sampling.

## 10. Limitations of the shipped selector (prebuilt kernels and winner cache)

**Prebuilt kernel bundle** (`kernels/bundle.py`; every miss falls back to compiling, so these cost time, never correctness):
- **One Triton build and one GPU architecture.** The bundle is valid for the Triton version (3.7.1) and the compute capability it was built on (sm_86: RTX 3080, also the 3090). Another Triton version, another architecture (the 2060 is sm_75), a different `ptxas`, or any edit to `kernels/triton_matmul.py` changes Triton's cache keys: every lookup misses and every kernel compiles on first use, exactly as without a bundle. `install()` warns once with the reason; `python -m tune bundle status` shows it. Rebuild with `python -m tune bundle build` (~10 min on 10 workers).
- **Only what the rankers can propose.** The union of the v1 and v2 pools (+ the 5 hand-picked configs), fp16 only (bf16 needs `--dtypes fp16,bf16` and compiles on first use otherwise), for tensors with 16-byte-aligned base pointers and the contiguous row-major strides `TritonMatmul` passes. A config outside the pool (`triton:<any key>`), a retrained ranker with a new pool, or a non-aligned view compiles on first use; rebuild after retraining a ranker.
- **The C launcher is not bundled.** Triton also builds a small launcher per argument signature with gcc on first launch, cached in the user cache; it depends on the Python ABI and host compiler. The first launches in a fresh user cache pay it.
- **A bundle hit skips compilation, not validation**: the correctness gate still runs on every candidate. The bundle is read only; a miss writes to the user cache (`TRITON_CACHE_DIR`).
- **Size and shipping.** 21,187 kernels, 799 MiB as files (986 MiB on disk): not committable as files, so `kernels/bundle/` is gitignored and the repo carries the 19 MiB archive `kernels/bundle.tar.xz`, unpacked by the first `install()`. An archive for another Triton / architecture is not unpacked. Every committed rebuild adds ~19 MiB to git history (use Git LFS if it will change often).

**cuBLASLt k-NN proposals** (`baselines/cublaslt_knn.py`):
- The table is one GPU's tune-split deep searches (3080, sm_86, cuBLASLt ids are per architecture). `table()` uses local `lt_search/` caches, else the committed snapshot only when its `hw_id` matches; any other GPU proposes nothing and warns once, so the joint search falls back to the heuristic top-16. A fresh GPU needs `python -m tune ... benchmark --backends cublaslt_deep:20` over the tune shapes (or `baselines.cublaslt_screen`) and `python -m baselines.cublaslt_knn --snapshot`.
- Proposals are validated (`cublasLtMatmulAlgoCheck` + the correctness gate) and short-timed, never trusted. A winner-cache entry from a search without proposals (`lt_knn` 0) is a miss when proposals are requested.

**cuBLASLt learned ranker** (`baselines/cublaslt_rank.py`):
- One GPU's model (3080 / sm_86; CTA features assume 68 SMs; tile / algorithm ids are per architecture and cuBLASLt version). `ranker()` loads `data/<hw>/lt_rank` or the committed snapshot and refuses a model whose `hw_id` differs (warns once; `select_kernel` then falls back to the k-NN proposals, which fall back to the heuristic top-16). Needs `lightgbm` at run time and enumerates every valid config of the shape (≈0.1-0.2 s). A new GPU: `python -m baselines.cublaslt_screen` (~70 min on the 3080), `python -m analysis.lt_rank --save data/<hw>/lt_rank`, copy to the snapshot. The SM count of the fill features comes from the model's json (`sms`, saved from the device at fit time; models without it are 68). Another GPU, by hand: `pipeline/zero_shot.sh --lt-mode zero_shot|retrain` (3090: either, 2060: retrain; `docs/zero_shot_runbook.md` section 6b), or per process `TUNE_LT_RANK=<stem> TUNE_LT_RANK_FOREIGN=1 [TUNE_LT_RANK_SMS=native]` to use another GPU's model zero-shot.
- Picks are validated and short-timed like any candidate; a winner-cache entry from a narrower search (`lt_rank` 0) is a miss when the ranker is requested.

**Family router** (`tune/router.py`):
- One GPU's model (3080), refused on another `hw_id` (warns once; the search then runs both halves). Only useful when the Triton kernels compile for real; it skips ~16% of shapes at the rule threshold 0.02 on the held-out set. The cuBLASLt half is timed before Triton, which changes the timing order: always run it with `confirm` (select_kernel does). A skipped shape's winner is cuBLASLt and is what the winner cache stores. Zero-shot on another GPU only with `TUNE_ROUTER_FOREIGN=1` (+ `TUNE_ROUTER_SMS=native`); retraining needs that GPU's whole tune stage 2, so it is a transfer test (runbook section 6b).

**Winner cache** (`tune/winner_cache.py`):
- Entries are per machine and per environment (schema 2: GPU id, Triton, torch, CUDA runtime, cuBLASLt version, driver, VBIOS, kernel source hash, declared clock lock, optional model ids). A soft-stale entry is a miss unless `serve_stale=True`; a hard-stale one (other GPU; Triton winner of other kernel source) is never served. Schema 1 lines (4-field environment) read as soft-stale, so the first lookup after upgrading re-searches once; so do lines the default cache wrote before it declared model ids. A field the current process does not declare (no `TUNE_CLOCK_LOCK`, a `WinnerCache` built without model ids) is not compared: such a process serves winners timed under any clock policy as current. Nothing is shared across GPUs.
- `compact()` drops a line that a later line of exactly the same environment replaced, and a superseded stale line only inside the compactor's own `env_id` scope (`environment(env_id=...)`; lscli sets one per registered interpreter). Without an `env_id` several interpreters may share the file, so nothing else is dropped: the file keeps one line per shape per environment it has seen (delete it to start over).
- Nothing expires by age: a winner is reused until the environment changes, a wider search is requested for that shape, or the file is deleted. A stored winner is the fastest of one short online timing (10 cold samples, one placement, no tie margin), so near-ties are as arbitrary as in `choose`.
- A cuBLASLt winner is a heuristic key; if the library no longer returns it the entry is a miss and the shape is searched again.
- A cache hit returns no `candidates` and `rank_s = 0`; callers that read the candidate list (experiments) must pass `cache=None`.
