# `lscli` implementation plan

What gets built, in what order, and how it lands. The contract (formats, commands, behaviour) is [`spec.md`](spec.md);
this file is the route to it. Decisions were taken in the 2026-11-08 design interview and are summarised in section 1.

## 1. Decisions

| Topic | Decision |
|---|---|
| Name | `lscli` (learned-selector CLI); unit `lscli.service`; home `~/.local/share/lscli` |
| What is learned | This GPU's own models (rank v1/v2, cuBLASLt ranker, router) via `pipeline/train_best.sh` stages in idle slices, plus a usage record of the user's shapes. The models are LightGBM, not a linear regression |
| Shape capture | Opt-in torch patch, NN `torch.matmul` / `@` / `torch.mm` only, fp16 (bf16 dispatched but unvalidated); `F.linear` (NT) shapes are logged, never dispatched |
| Cache miss | Fall back to torch and queue the shape; search inline only for shapes the usage record counts at ≥ 1M decayed calls. A winner is dispatched only when torch's kernel is ≥ 50 µs and the winner saves ≥ max(5 µs, 4 %) (3 % under a clock lock) |
| Idle gate | GPU idle (no foreign compute process, low utilization) + time window + AC power |
| Preemption | Abort, discard the unit in flight, back off, resume later from per-shape checkpoints |
| Clock | Optional sudoers helper (MHz chosen at install) locks the SM clock only around idle jobs; otherwise stock clocks, recorded |
| Install | `curl … \| sh`; Linux + systemd user service; no root except the optional clock helper |
| Process split | Thin client in the user's env, heavy daemon in its own venv; files only between them |
| Where winners are timed | In the user's interpreter (a worker the daemon starts), so the measured kernel is the one that runs. Triton ranking and the router run in the daemon (CPU); the cuBLASLt ranker's proposals in a short-lived daemon-venv child (needs the GPU); the daemon process never holds a CUDA context |
| Cold start | The 3080 rankers zero-shot + cuBLASLt heuristic top-16, until native models exist; then re-searched. The zero-shot set is committed (`install/zero_shot.tar.xz`), not a release asset. With no model set at all, searches are cuBLASLt-only (`k = 0`) and are widened once a ranker exists |
| Staleness | Hard (other GPU; Triton kernel source changed) = miss. Soft (driver, torch, Triton, CUDA, cuBLASLt, VBIOS, clock policy, models) = keep serving after one correctness check, re-search in idle time |
| Retraining | Only the stages an environment change affects (spec 4.5) |
| Scope | One GPU, one user |

## 2. Branches and pull requests

Rules:

* Nothing is committed directly to `learned-selectors` or `learned-selector-cli`. Every change lives on its own branch
  and lands through a PR.
* Before each PR is raised, a Sonnet reviewer subagent reads the diff. Its findings are fixed (or answered in the PR)
  and listed in the PR body.
* Each PR runs `python -m pytest -q` on an idle GPU.

```
learned-selectors
 └─ lscli/refactor ── PR #1 → learned-selectors
     = learned-selector-cli  (the lscli main branch: branched from lscli/refactor, no commits of its own)
        └─ lscli/spec ────── PR → learned-selector-cli
            └─ lscli/core ───── PR → lscli/spec
                └─ lscli/client ── PR → lscli/core
                    └─ lscli/daemon ─ PR → lscli/client
                        └─ lscli/train ── PR → lscli/daemon
                            └─ lscli/install  PR → lscli/train
```

Merging goes bottom-up. Once a PR merges into `learned-selector-cli`, the next one is retargeted to
`learned-selector-cli`. A change requested in a lower PR is made on that branch, and the branches above it are
rebased.

| PR | Branch → base | Content | Done when |
|---|---|---|---|
| [#1](https://github.com/Rayner3103/CS464/pull/1) | `lscli/refactor` → `learned-selectors` | `tune.online`: `propose_lt` / `propose_triton` / `measure`, `proposal_keys` / `backends_from_keys`. `tune.winner_cache` schema 2: wider environment, current / soft / hard tiers, `serve_stale`, `mark_verified`, `stale_entries`, `compact` | Behaviour of `search_shape` pinned identical; concurrency test; full suite green (opened) |
| [#2](https://github.com/Rayner3103/CS464/pull/2) | `lscli/spec` → `learned-selector-cli` | `docs/lscli/spec.md`, this plan | Reviewed |
| [#3](https://github.com/Rayner3103/CS464/pull/3) | `lscli/core` → `lscli/spec` | `lscli/` package: paths and `LSCLI_HOME`, config (`tomllib`, defaults, `client.json` export), locked-JSONL helpers, `envs.json`, profile merge with decay, queue (append, de-duplicate, consume), `state.json` / `control.json` readers. `lscli` entry point with `status`, `config`, `envs`, `profile`, `prewarm`, `cache ls/verify/compact/clear`. `python -m lscli.worker measure` (spec 7) | Unit tests for every file format; a worker round trip on the GPU writes a current entry |
| [#4](https://github.com/Rayner3103/CS464/pull/4) | `lscli/client` → `lscli/core` | `lscli.enable()` / `disable()`, the patch rules (spec 5.2), the per-call path (spec 5.3) with stale verification and the inline-frequent rule, usage-log flushing, autograd wrapper. `enable-env` / `disable-env`, `lscli.pth`, the package-name conflict check | Overhead test within the budget (spec 5.4); outputs equal `torch.matmul` within the correctness tolerance; a miss is never slower than torch; backward works |
| [#5](https://github.com/Rayner3103/CS464/pull/5) | `lscli/daemon` → `lscli/client` | `lscli daemon run`: idle gate (opening vs staying-open rules, job descendant tracking), preemption watcher (process-group kill + descendant kill), scheduler (spec 4.3, `gpu.lock`), search jobs (Triton ranking in the daemon, `lscli.propose` child for cuBLASLt ranker proposals, measure in the worker, router round trip over the protocol channel), maintenance (profile merge, queue consumption, daily compaction). `pause` / `resume`, `logs`, `daemon start/stop/restart`, `install/lscli.service` | A fake foreign GPU process triggers kill + discard + backoff with no row written; the daemon's own job never preempts itself; gate unit tests (window wrap, battery, utilization); end to end: a program's miss → queued → searched → the next run hits |
| [#6](https://github.com/Rayner3103/CS464/pull/6) | `lscli/train` → `lscli/daemon` | Training orchestration over `train_best.sh --only <stage>` (`--skip-install --skip-validation`, run from the `src/` clone), model registry (`MODELS.json`, ids, atomic activation), zero-shot → native switch, `superseded` re-search, retraining in a new run directory seeded with the unaffected stages' outputs, CPU-stage scheduling. `train status/start/retrain` | `train_best --smoke` completes under repeated preemptions; simulated cuBLASLt / Triton / driver changes select the right stages and start them from empty outputs |
| [#7](https://github.com/Rayner3103/CS464/pull/7) | `lscli/install` → `lscli/train` | `install/install.sh` (git clone, pinned model hashes, gcc / headers preflight), hardened `lscli-clock` + sudoers + `clock.conf` (MHz chosen at install), `doctor`, `uninstall`, a script that packs the zero-shot models | Fresh install → working daemon on a clean user account; re-install is idempotent; uninstall leaves nothing (except `$LSCLI_HOME` without `--purge`); clocks reset after `systemctl --user stop` and after a SIGKILL'd daemon restarts |
| [#8](https://github.com/Rayner3103/CS464/pull/8) | `lscli/zero-shot` → `lscli/install` | No release assets: the 3080 zero-shot set is committed as `install/zero_shot.tar.xz` (reproducible pack) and unpacked from the clone. cuBLASLt-only fallback with no model set (`select_kernel(LT_ONLY, ...)`, daemon and client inline search; `k = 0` entries, any-width lookups, widen once a ranker exists) | A sandboxed install without `--models-url` gets the committed set; the set loads on sm_86 (all three models) and sm_75 (Triton only); a daemon with no models searches and stores a `k = 0` cuBLASLt winner and does not search it again |
| [#10](https://github.com/Rayner3103/CS464/pull/10) | `lscli/gate-idle` → `lscli/zero-shot` | Fix found by a real install: a job's own GPU utilization, still reported by NVML when it exits, restarted the idle clock at the search's final check and a training stage's quick-failure check (one search per `gpu_idle_minutes`). The post-job grace now starts when the job's processes exit | Back-to-back searches on consecutive ticks with utilization reported during jobs; CPU-only daemon tests fail on the old code |
| [#11](https://github.com/Rayner3103/CS464/pull/11) | `lscli/train-lock` → `lscli/gate-idle` | Fixes found by running training on a real install (`lscli train start`, RTX 2060, quick mode): the daemon's own train_best lock (taken by a `flock` child) preempted every GPU stage; a stage killed while creating `hw_id` left it empty and every resume failed; an exiting job child still listed by NVML preempted `tune_sweep` | preflight → roofline and `tune_sweep` run under the daemon with no false preemption; a foreign CUDA process preempts the sweep with no orphans and the stage resumes per shape; the new tests fail on the old code |
| [#13](https://github.com/Rayner3103/CS464/pull/13) | `lscli/v2-objective` → `lscli/train-lock` | The same training reached `train_v2` and failed it twice: v1's grid search chose lambdarank and v2 copied it, but v2 is a regression (LightGBM: "Should set group for ranking task"). v2 now takes the best regression entry of v1's search | `tune ranker-v2` completes on the failing run's data; tests cover each source of v2's hyperparameters |

## 3. Things to know while building

* **Dev GPU.** The development machine has an RTX 2060 (sm_75), so the 3080 models are refused there by `hw_id`
  (zero-shot needs the foreign-GPU flags). That exercises the cold-start path, with the cuBLASLt half carrying the
  search. The native-model path is tested with the 2060's own `pipeline/runs/quick-*` models. An sm_86 run (remote GPU
  box) checks the bundle path.
* **GPU job hygiene** (AGENTS.md section 7) applies to the daemon itself. It must never overlap a user's job (that is
  the gate's whole purpose), it kills sweep compile workers by PID, and it never runs two GPU jobs at once (`gpu.lock`
  for its own jobs; `train_best.sh` takes `src/pipeline/.lock` itself, and a lock held by a manual run keeps the gate
  closed).
* **Daemon measurements are not study data.** They run at stock or helper-locked clocks (the stock-clock noise floor
  is measured on the 3080 only, `reports/week1.md` 2026-11-13), and they are never written into `data/<hw>/` result files or reports.
* **Version skew.** Bundle cubins only serve a user env with Triton 3.7.1 on sm_86. In other envs the worker compiles
  in idle time, and the compiled kernels land in the user's own Triton cache.
* **Packaging.** The client installs the existing top-level `tune`, `kernels` and `baselines` packages plus `lscli`. The
  daemon installs the repo checkout in editable mode, because `analysis.learned.Ranker` and `pipeline/` are needed only
  there.

## 4. Open after v1

* NT / TN layouts (`nn.Linear`): the cuBLASLt transpose flag, an NT axis in the Triton training suites, layout as a ranker feature.
* Multi-GPU (per-device models, caches, gates) and a shared multi-user service.
* Moving the vendored packages under an `lscli._vendor` namespace to remove the name-conflict limitation.
* Noise floor for daemon conditions: done for stock clocks on the 3080, desktop idle (2026-11-13; the dispatch margin is
  4 % unlocked / 3 % locked). Open: other GPUs, measuring while the desktop is in use, graph timing under 50 µs.
