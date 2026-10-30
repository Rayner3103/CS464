# Pipeline: install → tune → train → validate, in one command

```bash
sudo nvidia-smi -lgc 1755,1755        # once, before the run (see "Clock lock"); undo with: sudo nvidia-smi -rgc
pipeline/run.sh --quick               # about 4–5 hours; --full reproduces the whole study (a day or more)
```

The script builds a virtualenv, checks the machine, measures the GPU, sweeps the Triton GEMM config space (tune),
trains the learned kernel selector (LightGBM `lgbm_rank` plus a diversification rule), and validates it on held-out
shapes against `torch.matmul` and a searched cuBLASLt (B3+). Everything it writes goes to
`pipeline/runs/<mode>-<timestamp>/`. The repo's own `data/` and `reports/` are never touched.

## Prerequisites

| Need | Detail |
|---|---|
| OS | Linux (developed on Ubuntu, kernel 7.0) |
| GPU | One NVIDIA GPU, fp16 tensor cores. Developed and measured only on an RTX 3080 (`sm_86`). Other GPUs should run, but the sweep takes different times and the numbers are not comparable to the study |
| Driver | NVIDIA driver ≥ 580 (the pinned wheels are CUDA 13.0; the study used 595.99.02). The script does not install drivers |
| Python | 3.10 or newer with `venv` and development headers (`Python.h`); 3.12 is what was tested. Ubuntu: `sudo apt install python3-venv python3-dev` |
| C compiler | `gcc` or `cc`. Triton builds its launcher stubs with it. Ubuntu: `sudo apt install build-essential` |
| git | The harness records the commit and dirty state on every row |
| Network | First run downloads torch (cu130 wheel, about 3 GB with its CUDA libraries) and the other pinned packages. Override the wheel index with `TORCH_INDEX_URL` |
| Disk | ≥ 20 GB free (venv ≈ 8 GB; a full run's data is several GB) |
| CPU / RAM | The sweep compiles in parallel (`--workers`, default auto: up to 10, capped so each worker (~1.25 GiB RAM budget, ~450 MiB VRAM) fits in free memory, re-checked before every shape; chunks run in fresh processes so a worker peaks at ~1 GiB). On a 16 GB RAM / 6 GB VRAM machine with a desktop running expect roughly 6-8 workers; use `--workers 4` to leave more headroom |
| Idle GPU | No other compute process may be using the GPU. Overlapping jobs contaminate timings. The script checks before every GPU stage and stops if it finds one |
| sudo | Only for you to lock the clock (below). The script never calls sudo |

`pipeline/run.sh --check-only` runs the prerequisite checks and exits.

### Clock lock

All gate numbers in the study assume the SM clock is locked (RTX 3080: 1755 MHz). The harness cannot lock clocks
without root, so you do it once:

```bash
sudo nvidia-smi -lgc 1755,1755     # before the run
sudo nvidia-smi -rgc               # after the run, to restore normal boost
```

Pass the same value with `--clock-lock MHZ` (default 1755). The `env` stage probes the real state, and the run stops
with the command above if the GPU does not look locked. On a different GPU choose a clock it can sustain under
load without power-capping (roughly 80–85% of its maximum SM clock). To run unlocked anyway:
`--clock-lock none --allow-unlocked`. Results are then not comparable with the study.

## Modes

| Mode | Tune shapes | Held-out shapes | Noise study | Time (RTX 3080, estimate) | Use it for |
|---|---|---|---|---|---|
| `--smoke` | ~21 | ~9 | skipped | 1h20m (measured) | checking the plumbing; the numbers mean little |
| `--quick` (default) | ~50 | ~27 | 1 round × 20 repeats | roughly 4–5 h (extrapolated from the smoke run's ≈ 2.3 min per tune shape and ≈ 2.4 min per held-out shape) | a real, smaller run of the whole flow |
| `--full` | all 477 | all 86 | 3 × 50, plus 10 min drift | about 20 h (the tune sweep alone took ≈ 17 h in the study) | reproducing `reports/week1.md` |

Only the smoke time is measured; quick and full are extrapolations. Stage logs print elapsed time. The sweep dominates every mode.

## Stages

Run order, with the `tune` command behind each. Logs are `pipeline/runs/<run>/logs/<stage>.log`.

| Stage | What it does |
|---|---|
| `preflight` | Prerequisite checks above |
| `install` | Creates `pipeline/.venv`, installs `requirements.txt` (pinned), checks that torch sees the GPU, writes `environment.txt` |
| `tests` | `pytest -q` (GPU needed, runs alone) |
| `env` | `tune env`: hardware fingerprint, versions, clock-lock probe. **Stops here if the clock is not locked** |
| `check` | `tune check`: every candidate against the fp32 reference (S1 in smoke, the 169-shape `edge` suite otherwise) |
| `roofline` | `tune roofline`: measured peak TFLOP/s and DRAM bandwidth |
| `noise`, `drift` | Noise-floor study (sets the tie margin) and a thermal drift run |
| `tune_sweep` | **Tune.** `tune sweep --split tune`: every pruned Triton config (up to ~10,800) compiled, validated and timed on each tune shape |
| `portfolio` | Best single config (B6), greedy portfolio curve, set cover and parameter importance |
| `heldout_sweep` | The same screen on held-out shapes (never used for fitting) |
| `portfolio_heldout` | Scores the tune-fitted portfolio on the held-out shapes |
| `selectors` | Trivial selectors (b6, nn1, knn5, rule, knnf, knnc); writes the reference picks |
| `train` | **Train.** `tune topk`: grouped-CV hyperparameter search for the learned models (cached), refits `lgbm_rank` on all tune shapes, fits the diversification rule on out-of-fold scores, saves `data/lgbm_rank.{txt,json}` |
| `validate_cold` | **Validate.** Re-times every pick on held-out shapes with the full protocol (cold L2) next to the screen's best tiles, `torch.matmul`, cuBLASLt top-16 (B3) and the searched cuBLASLt (B3+) |
| `validate_graph` | Same for held-out shapes under 50 µs, CUDA-graph amortized |
| `online_plain`, `online_diverse` | First-use search: rank, compile (empty Triton cache), check and short-time the top 5, keep the fastest |
| `score` | `tune topk` again with the timing files: best-of-k, online search, regret vs the oracle and vs B3+ |
| `summary` | Writes `SUMMARY.md` |

## Options

```
--smoke | --quick | --full     mode (default --quick)
--clock-lock MHZ|none          declared lock (default 1755); --allow-unlocked to continue unlocked
--run-dir DIR                  output directory
--resume DIR                   continue a run; finished stages are skipped and the mode is read from the directory
--from STAGE / --only A,B      start at a stage / run only some stages
--force                        re-run stages even if marked done
--skip-install                 use the current `python` on PATH (checks the pinned versions, installs nothing)
--skip-tests                   skip pytest
--venv DIR, --python BIN       virtualenv location, interpreter used to create it
--workers N                    parallel sweep compile workers (default 0 = auto, capped by free RAM / VRAM)
--list-stages, --check-only
```

A stage that fails stops the run and prints the resume command. Completed shapes are kept at every level (the sweep
writes one file per shape, benchmarks append and skip finished rows), so `--resume` continues rather than restarts.
Ctrl-C stops the child processes; if you kill the script some other way, check `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`
for orphaned sweep workers (about 218 MiB each) and kill them by PID before resuming.

## Output

```
pipeline/runs/<run>/
  SUMMARY.md           what ran and where the tables are
  environment.txt      commit, pinned package versions
  mode, stamps/        run bookkeeping
  logs/<stage>.log
  data/                env.json, sweep_screen*/, results_validate*.jsonl, online_*.jsonl, learned_*.json, lgbm_rank.{txt,json}
  reports/             sweep_portfolio*.md, selector_baselines.md, learned_topk.md (+ summaries)
```

The trained selector is `data/lgbm_rank.txt` + `.json`. Load it with
`analysis.learned.Ranker.load(Path("<run>/data/lgbm_rank"))` and call `.rank(M, N, K, k=3, order="diverse")`.

## Not included

The comparison of all ten learned model families (`tune learned`, about 2 h of CPU for the full grid plus two
stage-2 timing sessions), the S1/S2 baseline benchmark with `tune analyze` (G1/G2 tables), the first-pass stage 2 of
the tune split, and the held-out stage 2 of the greedy portfolio. They are listed in `AGENTS.md` §8. `train` already
runs the full CV grid for every model (it needs it to pick the LightGBM hyperparameters), so `tune learned --data
<run>/data --report <run>/reports` afterwards only adds the held-out scoring and is cheap.
