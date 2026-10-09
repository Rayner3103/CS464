# Plan: Baseline Benchmarking & Ground Setting
### Learned kernel selection for GPU matmul: parameterized kernel family + install-time benchmarking harness

**Status:** Draft v0.1
**Scope of this document:** everything *before* building the selector model. The goal is to establish what is measurable, what the baselines are, and whether there is enough headroom to justify a learned selector.

---

## 1. Purpose and Guiding Questions

This phase answers four questions with data, not intuition:

| # | Question | Why it matters |
|---|----------|----------------|
| Q1 | How fast is the best available vendor path (cuBLAS / cuBLASLt) across our target shape distribution? | Defines the bar we must beat or match |
| Q2 | How fast is our parameterized kernel family at its **per-shape oracle best**? | Upper bound for any selector built on this family |
| Q3 | How large is the gap between the oracle and (a) a single best fixed config, (b) cuBLAS? | Determines whether a selector adds value at all |
| Q4 | How noisy are our measurements, and what is the smallest difference we can trust? | Sets the resolution limit of any model we train |

**Core insight:** a selector can never beat the oracle of its kernel family. If the oracle does not meaningfully beat both a single fixed config and cuBLAS, the project needs to change direction before any ML is written.

### Non-goals for this phase
- No ML model training (beyond trivial sanity baselines).
- No multi-GPU or multi-vendor support.
- No packaging or installer work.
- No fused epilogues, grouped GEMM, or quantized dtypes.

---

## 2. Initial Scope (deliberately narrow)

| Dimension | Phase-1 choice | Deferred |
|-----------|----------------|----------|
| GPU | Single NVIDIA GPU you have physical access to | Other generations, AMD, Intel |
| Kernel language | **Triton** (fast to parameterize) | CUTLASS templates, raw CUDA |
| Dtype | **fp16** (fp32 accumulate), then bf16 | fp32/TF32, int8, fp8 |
| Layout | Row-major A, row-major B (NN) | NT, TN, TT, strided inputs |
| Op | Plain GEMM, C = A·B | Batched, grouped, bias/activation fusion |
| Shape range | M, N, K in [1, 16384] | Larger, memory-bound extremes |

Record in the repo exactly which GPU, driver, CUDA, PyTorch, and Triton versions every result came from (see 4.5).

---

## 3. Phase 0: Measurement Methodology (do this first)

Bad timing infrastructure invalidates everything downstream, so this is the first deliverable.

### 3.1 Timing protocol
- Use **CUDA events** around the kernel launch, not Python wall-clock.
- **Warmup** runs before timing (discard the first N, enough for clocks and caches to settle).
- **Repeat** until a confidence criterion is met (e.g. relative std error < 1%) or a max repeat cap is hit, rather than a fixed count.
- Store **raw samples** (or at least median, IQR, min, max, n), not only the mean.
- Report the **median** as the primary statistic. It is robust to outliers.
- **Flush L2** between runs in one mode, and don't flush in another. Both matter: cold-cache numbers are the conservative case, hot-cache numbers match back-to-back inference calls. Decide which is primary and document it.
- For very small kernels (microseconds), time a **batch of back-to-back launches** (or use CUDA graphs) so launch overhead does not dominate. Report both per-launch and amortized numbers.

### 3.2 Environment control
- Lock or at least **record GPU clocks** (`nvidia-smi -lgc` where permitted); log power limit, temperature, and clock speed during runs.
- Run on an **idle GPU** (check no other processes via `nvidia-smi`).
- Disable or log persistence mode and ECC settings.
- Pin CPU affinity and avoid heavy CPU contention during benchmarking.
- Record a **hardware fingerprint** (GPU name, SM count, compute capability, memory size, driver, CUDA, library versions). This becomes the key for the tuning database later.

### 3.3 Noise-floor study (Q4)
- Choose ~30 representative shapes. For each, time one fixed kernel **100+ times across several sessions** (different days, cold and warm GPU).
- Compute within-run and across-session variance.
- Deliverable: a **noise-floor number**, e.g. "differences under X% are indistinguishable". All later analysis treats kernels within this margin as ties.
- Check thermal drift: run a 10-minute continuous workload and plot clock/time over duration.

### 3.4 Correctness gate
- Every kernel config is validated against a reference (cuBLAS fp32 or a high-precision CPU/GPU reference) with dtype-appropriate tolerances.
- Configs that fail correctness on any validation shape are excluded and logged, never silently timed.
- Include edge shapes: M/N/K = 1, primes, non-multiples of tile sizes, K smaller than BLOCK_K.

---

## 4. Phase 1: Baselines

### 4.1 Baselines to measure

| ID | Baseline | Purpose |
|----|----------|---------|
| B0 | Theoretical roofline (peak TFLOPs, peak bandwidth from spec and measured) | Absolute ceiling and efficiency context |
| B1 | `torch.matmul` (default cuBLAS path) | What a typical user gets today |
| B2 | **cuBLASLt heuristic**, top-1 algorithm | Vendor's own selector |
| B3 | **cuBLASLt exhaustive**: time the top-N heuristic candidates (N≈8-32) and take the best | Vendor-library oracle |
| B4 | CUTLASS profiler best config (if available) | Strong open-source reference |
| B5 | Triton's stock `matmul` tutorial kernel, default config | Naive baseline for our own family |
| B6 | Best **single fixed config** from our family over the whole shape set | Tests whether per-shape selection helps at all |

B3 matters most. Beating cuBLAS's default heuristic is much easier than beating cuBLAS if it were allowed to benchmark. The latter is the honest ceiling.

### 4.2 Metrics (define once, reuse everywhere)

- **Runtime** (µs) and **achieved TFLOP/s** = 2·M·N·K / time.
- **Efficiency** = achieved / roofline peak.
- **Speedup vs baseline X** = time_X / time_ours.
- **Regret** of a choice = time_chosen / time_oracle − 1 (0% is perfect).
- Always report **distributions**: median, p10, p90, p99, worst-case. Never only the mean.
- Report results **grouped by shape regime** (see 5.2), since averages hide regime-specific wins and losses.

---

## 5. Phase 2: Shape Distribution & Dataset Design

The test set determines what conclusions are possible. Build it deliberately.

### 5.1 Principles
- Separate **training/tuning shapes** from **held-out evaluation shapes** from the start. Never evaluate the selector on shapes it tuned on.
- Hold out *regions* (e.g. a whole range of K), not only random points, to test extrapolation.
- Include shapes people actually run, not just uniform grids.

### 5.2 Shape suites

| Suite | Description | Why |
|-------|-------------|-----|
| S1 Square sweep | M=N=K ∈ {64 … 16384}, powers of two and off-by-small values (e.g. 1000, 1025) | Classic benchmark; exposes tile quantization |
| S2 LLM-realistic | Hidden/FFN dims of common models (e.g. 4096, 5120, 8192, 11008, 14336, 28672) × token counts M ∈ {1, 2, 4, 8, 16, 32, 64, 128, 512, 2048, 8192} | Real workloads; decode-phase skinny GEMMs |
| S3 Skinny / tall | One dim small (1-64), others large; both orientations | Where vendor heuristics are weakest |
| S4 Large-K | Small M,N with large K | Split-K regime |
| S5 Unaligned / odd | Dims not divisible by 8/16/64, primes | Alignment and vectorization cliffs |
| S6 Wave-boundary | Shapes straddling the tile-count at which the number of CTA waves changes (compute from SM count) | Directly probes wave quantization |
| S7 Random log-uniform | Random (M,N,K) sampled log-uniformly | Unbiased coverage and a clean held-out set |

Rough budget: a few hundred to ~2,000 shapes total for the baseline study. Size it to the time budget in 8.

### 5.3 Dataset record schema (one row per shape × config × dtype × layout)

```
hw_fingerprint, driver, cuda, torch, triton versions
M, N, K, dtype, layout, alignment_flags
kernel_family, config (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, GROUP_M, SPLIT_K, ...)
valid (bool), failure_reason
time_median_us, time_p10, time_p90, n_samples, cache_mode
clock_mhz, temp_c, power_w (at measurement time)
timestamp, session_id
```

Store as Parquet or JSONL, append-only, with raw samples in a side file for the noise study.

---

## 6. Phase 3: Parameterized Kernel Family

### 6.1 Tunable parameters (Triton starting point)

| Parameter | Typical values | Notes |
|-----------|----------------|-------|
| BLOCK_M | 16, 32, 64, 128, 256 | Large values hurt small M (waste) |
| BLOCK_N | 32, 64, 128, 256 | |
| BLOCK_K | 16, 32, 64, 128 | Affects shared-memory footprint |
| num_warps | 2, 4, 8 | |
| num_stages | 2, 3, 4, 5 | Software pipelining depth; shared-mem limited |
| GROUP_M | 1, 4, 8, 16 | L2 locality via swizzling |
| SPLIT_K | 1, 2, 4, 8, 16 | Needs a reduction path; adds nondeterminism if atomics used |
| (optional) tensor-core instr shape, EVEN_K constexpr specialization | | Removes masking when dims divide evenly |

### 6.2 Config-space management
- Full Cartesian product is large (thousands). **Prune before benchmarking:**
  - Remove configs violating shared-memory or register limits for this GPU.
  - Remove configs where tile exceeds the problem size by a large factor.
  - Remove dominated configs after an initial coarse pass (see 6.3).
- Record the pruning rules so they are reproducible and portable to other GPUs later.
- Track **compile time per config**. In a JIT setting this is a real cost at install time and at first use.

### 6.3 Config-space reduction experiment
Goal: find a **small portfolio** (e.g. 10-50 configs) that covers nearly all of the oracle's value.
1. Benchmark the full pruned config set on a coarse shape subset.
2. Greedy set cover: pick configs so that for every shape at least one selected config is within ε (e.g. 3%, tied to the noise floor) of the oracle.
3. Plot portfolio size vs mean/p95 regret. A steep curve that flattens quickly means the selection problem is small and tractable. If it needs hundreds, the selector problem is hard.

This experiment is likely the single most informative result of the whole phase, and it also drastically cuts install-time cost.

---

## 7. Phase 4: Install-Time Benchmarking Harness

### 7.1 Responsibilities
1. Detect hardware and build the fingerprint.
2. Generate a candidate (shape, config) work list under a **time budget**.
3. Compile, validate, and time each candidate with the protocol from section 3.
4. Persist results incrementally (crash-safe, resumable).
5. Emit a dataset file consumable by the later model-training step.

### 7.2 Design requirements
- **Resumable:** kill and restart without losing completed measurements.
- **Time-budgeted:** accept `--budget 10m/30m/2h`; sampler prioritizes useful measurements first.
- **Isolated:** run each candidate in a subprocess (or at least guard CUDA errors) so one illegal-memory-access config does not kill the session. Add timeouts for hanging kernels.
- **Deterministic seeds** for shape sampling and input data.
- **Observable:** progress, ETA, and live clock/thermal readout; abort or pause if the GPU throttles heavily.
- **Pluggable backends:** abstract `Kernel` interface (`compile`, `run`, `validate`) so cuBLAS/CUTLASS/Triton all go through the same harness.

### 7.3 Sampling strategy (v0 → later)
- **v0:** fixed suites from section 5 (no cleverness; just get data).
- **v1 (later):** adaptive sampling. Densify near regime boundaries and where candidate configs are close.

### 7.4 CLI sketch

```
tune benchmark --suite all --dtype fp16 --budget 30m --out data/<hw_id>/
tune baseline  --which cublas,cublaslt,torch --suite all
tune analyze   --data data/<hw_id>/ --report reports/
tune noise     --shapes noise_set.json --repeats 100
```

### 7.5 Environment capture
Every run writes `env.json` (hardware fingerprint, library versions, clock settings, git commit of the harness). Results without it are treated as invalid.

---

## 8. Phase 5: Headroom Analysis & Go / No-Go Gates

### 8.1 Analyses to produce
1. **Oracle vs cuBLASLt-exhaustive (B3)** per shape; heatmaps over (M, N) at fixed K, per regime.
2. **Oracle vs best single config (B6):** value of per-shape selection.
3. **Oracle vs cuBLAS default (B1/B2):** value over what users get today.
4. **Portfolio-size vs regret curve** (6.3).
5. **Where do we lose?** Shapes where even the oracle is slower than cuBLAS. Likely needs kernel-level fixes (better instructions, epilogue) rather than selection.
6. **Which parameters matter?** Feature importance or ablation. Does BLOCK_K or SPLIT_K drive variance?
7. **Trivial selector baselines:** nearest-neighbor on (M,N,K), and a hand-written rule-based heuristic. Any learned model must beat these.

### 8.2 Decision gates (suggested thresholds; adjust after seeing the noise floor)

| Gate | Condition | If it fails |
|------|-----------|-------------|
| G1 Measurement validity | Noise floor < ~2% on median; reproducible across sessions | Fix methodology before anything else |
| G2 Family quality | Oracle of our family ≥ ~90% of B3 on the median shape | Improve kernels, or reposition vs. cuBLAS (target other regimes/backends) |
| G3 Selection value | Oracle beats best-single-config by a meaningful margin (e.g. >10% median or >30% p90) | Use a static config; selector adds little |
| G4 Differentiation | Oracle beats B3 in some identifiable regime (e.g. skinny, odd-shaped) by >5% on a meaningful share of shapes | Narrow the project to where the gap exists, or pivot target hardware |
| G5 Tractability | A portfolio of ≤ ~50 configs achieves p95 regret < ~5% | Selection problem is hard; revisit features / approach |

Passing G1-G3 and G5 with G4 failing means the project is still viable as a **research/learning project or a non-NVIDIA backend project**, not as a cuBLAS competitor. Write that conclusion down explicitly if it happens.

---

## 9. Milestones & Deliverables

| Milestone | Deliverable | Gate |
|-----------|-------------|------|
| M0 | Repo skeleton, env capture, timing utility, correctness checker | Timing reproducible |
| M1 | Noise-floor report | G1 |
| M2 | Baselines B0-B5 across all suites | Baseline tables + roofline plot |
| M3 | Parameterized Triton kernel with validated config space and pruning | Correctness on edge-shape suite |
| M4 | Full benchmark run → dataset; B6 computed | Dataset schema frozen |
| M5 | Portfolio reduction study | G5 |
| M6 | Headroom report and go/no-go memo | G2, G3, G4 |

Suggested repo layout:

```
repo/
  tune/            # harness package (hw detection, runner, samplers, backends)
  kernels/         # triton family, config space, pruning rules
  baselines/       # cublas, cublaslt, torch, cutlass wrappers
  suites/          # shape suites as versioned JSON
  analysis/        # notebooks/scripts for headroom, regret, portfolio
  data/<hw_id>/    # results + env.json (gitignored or LFS)
  reports/
  plan.md
```

---

## 10. Risks (specific to this phase)

| Risk | Mitigation |
|------|------------|
| Timing noise swamps real differences | Section 3; treat sub-noise differences as ties |
| Triton JIT compile time makes the sweep very slow | Parallel compile workers, cache compiled artifacts, prune hard, coarse pass first |
| cuBLASLt API access from Python is awkward | Use PyTorch's cuBLAS path first; add a thin C++/ctypes wrapper for algo enumeration (needed for B3) |
| Kernel family underperforms cuBLAS everywhere | Caught at G2; consider CUTLASS-based family instead of Triton |
| Overfitting to a benchmark suite | Held-out regions and random log-uniform set from day one |
| Results do not transfer to another GPU | Expected. The point of the tool is to re-tune; record per-hardware and compare later |
| Scope creep (more dtypes, layouts, fusion) | Section 2 scope is frozen until G2-G5 are decided |

---

## 11. Next Steps (after this phase passes the gates)

### 11.1 Selector model (Phase B)
1. **Feature engineering** that encodes hardware discontinuities rather than raw M, N, K:
   - tiles per dimension for candidate tile sizes (ceil(M/BM), …), total CTAs, waves = CTAs / (SM count × occupancy), tail-wave utilization
   - alignment flags (divisible by 2/4/8/16/64), M mod tile, arithmetic intensity, log M/N/K, aspect ratios
   - hardware-normalized quantities (SM count, bandwidth/FLOP ratio) for future cross-GPU work
2. **Problem formulations to compare:**
   - Regression on log-runtime per config, then argmin
   - Direct classification over the portfolio
   - Learning-to-rank (pairwise or listwise)
   - Nearest-neighbor / kernel regression baseline
3. **Models:** gradient-boosted trees (LightGBM/XGBoost), random forest, small MLP, kNN. Trees are the likely winners given discontinuities and tiny inference cost.
4. **Evaluate by regret**, on held-out shapes and held-out regions, plus end-to-end speedup vs B1/B3.

### 11.2 Hybrid runtime design (Phase C)
- Model predicts top-k, micro-benchmark those k on first sight of a shape, cache the winner (persistent LRU keyed by shape/dtype/layout).
- Confidence-based fallback to cuBLAS when the model is uncertain or the shape is out-of-distribution.
- Compiled/low-overhead dispatcher; measure dispatch overhead in µs and verify CUDA graph compatibility.

### 11.3 Install-time efficiency (Phase D)
- Adaptive / active sampling to hit a 10-30 minute install budget.
- Ship a **pre-pruned portfolio** instead of the full config space.
- Warm-start from priors shipped per GPU architecture, then refine locally.

### 11.4 Robustness (Phase E)
- Drift detection: re-tune triggers on driver/CUDA/GPU change.
- Periodic lightweight re-validation of the chosen kernels on live shapes.
- Handling multi-GPU heterogeneity.

### 11.5 Scope expansion (Phase F)
bf16 and fp32/TF32 → transposed layouts and strides → batched and grouped GEMM → fused epilogues (bias, GELU, residual) → int8/fp8.

---

## 12. Research Directions

1. **Cost models and learned performance prediction:** compare to TVM/Ansor-style learned cost models and tile-level analytical models. Can a hybrid (analytical wave/tile model + learned residual) beat a pure learned model with less data?
2. **Active learning / Bayesian optimization for sampling:** how few measurements are needed for a given regret? Uncertainty-driven sampling near decision boundaries.
3. **Transfer across GPUs:** train on GPU A, adapt to GPU B with a handful of measurements (hardware-normalized features, meta-learning, fine-tuning). Would turn the install step from "tune from scratch" into "calibrate".
4. **Uncertainty and out-of-distribution detection:** conformal prediction or ensemble disagreement to decide when to micro-benchmark or fall back to cuBLAS.
5. **Config-space structure:** is the optimal config a low-dimensional function of a few shape quantities? Discover simple interpretable rules that a tree distills (and compare to cuBLAS's heuristic behavior).
6. **Portfolio construction as a theoretical problem:** submodular coverage / minimax-regret portfolio selection with guarantees; relationship to algorithm selection literature.
7. **Workload-aware tuning:** weight shapes by real usage frequency (e.g. from traced model inference) rather than uniform coverage; online adaptation as the shape distribution shifts.
8. **Noise-aware learning:** model measurement variance explicitly; train with ties and rank-based losses so the model is not penalized for distinctions below the noise floor.
9. **Non-NVIDIA targets:** Intel (SYCL/oneAPI), Apple Metal, AMD ROCm, Vulkan. This is where the vendor-library gap is largest and the project's value proposition strongest. Evaluate after the CUDA study.
10. **Beyond GEMM:** apply the same harness to attention (FlashAttention-style tile params), convolution, reductions, and fused kernels, where vendor coverage is thinner.
11. **Reproducibility and fairness in autotuner benchmarking:** a publishable methodology contribution (timing protocol, noise floors, regret reporting) given how inconsistent autotuning evaluations are in the literature.

---

## 13. Immediate Action Checklist (first week)

- [ ] Pin the environment; script `env.json` capture
- [ ] Implement timing utility (events, warmup, adaptive repeats, L2 flush option) + unit tests
- [ ] Implement correctness checker and edge-shape suite
- [ ] Run noise-floor study on ~30 shapes (multi-session)
- [ ] Measure B0, B1 on S1 (square sweep) and S2 (LLM shapes) as a first sanity plot
- [ ] Write a minimal Triton matmul with 3-5 hand-picked configs; confirm it beats B5 on at least some shapes
- [ ] Decide primary cache mode (cold vs hot L2) and document it
- [ ] Draft shape suites S1-S7 as versioned JSON, with held-out splits fixed up front
