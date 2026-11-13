# `lscli` specification (v1)

`lscli` (learned-selector CLI) puts the learned GEMM kernel selector (`tune.online.select_kernel`) to work on one
user's machine without code changes beyond an opt-in. It learns which GEMM shapes the user runs, trains this GPU's
own selector models when the machine is not being used, and uses the remaining idle GPU time to run the first-use
search for those shapes ahead of time, so a program finds the winner already chosen.

This document is the contract: file formats, commands, the daemon's behaviour and the client's behaviour. The
implementation plan (branches, PRs, order of work) is [`plan.md`](plan.md). Anything not written here is not promised.

## 1. Scope

| In v1 | Not in v1 |
|---|---|
| Linux with systemd (user session), one NVIDIA GPU (device 0) | macOS / Windows; multi-GPU (device 0 is used, a warning is printed) |
| One user: a `systemd --user` service run by whoever installed it | A shared system service for several users |
| fp16, 2-D, contiguous row-major NN `torch.matmul` / `@` / `torch.mm`, M, N, K in [1, 16384] | NT / TN layouts (`F.linear`, `nn.Linear`): logged, never dispatched |
| bf16: dispatched (every winner passes the correctness gate) but **unvalidated**: the rankers, the cuBLASLt ranker and the bundle were trained / built on fp16 only | Shapes outside [1, 16384]: torch |
| Inference and training (backward runs on torch) | Dispatch inside `torch.compile` graphs or CUDA-graph capture |

## 2. Components

| Component | Runs in | Needs | Does |
|---|---|---|---|
| **client** (`lscli.enable()`, the patch; `src/` on the env's path via `lscli.pth`) | the user's own Python env | torch, triton, numpy, nvidia-ml-py (+ lightgbm / pandas for the optional inline search) | counts shapes, dispatches cached winners, falls back to torch, queues misses |
| **worker** (`python -m lscli.worker`) | the user's interpreter, started by the daemon | same as the client | compiles, validates and times one proposed pool (`tune.online.measure`), writes the winner |
| **daemon** (`lscli daemon run`, unit `lscli.service`) | `~/.local/share/lscli/venv` (pinned versions) | the repo's `requirements.txt` | idle gate, scheduler, Triton ranking and the router (CPU: LightGBM), training, model registry, cache compaction. The daemon process itself never creates a CUDA context |
| **proposer** (`python -m lscli.propose`) | the daemon venv, started per search job | same as the daemon | the GPU-needing part of ranking: the learned cuBLASLt ranker's proposals (`propose_lt`), which build descriptors and call `cublasLtMatmulAlgoCheck` |
| **CLI** (`lscli ...`) | the daemon venv (on `PATH` via `~/.local/bin/lscli`) | same as the daemon | user commands, section 6 |
| **clock helper** (`/usr/local/libexec/lscli-clock`, optional) | root, via one sudoers rule | `nvidia-smi` | locks / resets the SM clock around idle jobs |

The client and the daemon never import each other at run time; they communicate only through the files in section 3.
Nothing leaves the machine: there is no telemetry, and the usage record (shapes, interpreter paths) stays in
`$LSCLI_HOME`, which is created with mode 0700.

## 3. Files

`LSCLI_HOME` defaults to `~/.local/share/lscli`; the config is `~/.config/lscli/config.toml` (`LSCLI_CONFIG` overrides).
Every JSONL file is append-only with one fsync'd line per write under `flock`; a torn last line is ignored by readers.

```
$LSCLI_HOME/
  src/                       the repo checkout at the installed tag (pipeline/train_best.sh lives here)
  venv/                      daemon env
  winners.jsonl              winner cache (tune/winner_cache.py, schema 2)
  profile/<pid>-<t0>.jsonl   client usage logs, one file per process
  profile.json               merged, decayed usage record (daemon-written)
  queue.jsonl                search requests (clients and CLI append, daemon consumes)
  queue.offset               how far the daemon has consumed queue.jsonl (daemon-written)
  gpu.lock                   held (flock) by the daemon while one of its jobs uses the GPU
  envs.json                  registered client interpreters
  models/<hw_id>/            this GPU's models + MODELS.json
  models/zero_shot/          the 3080 models used until native ones exist
  train/<hw_id>/<n>/         train_best run directories (n = 1 for the first training, +1 per retraining)
  state.json                 daemon status (daemon-written, read by `lscli status`)
  control.json               pause / resume requests (CLI-written)
  client.json                settings the client needs (daemon-written from config.toml)
  logs/daemon.log, logs/jobs/<job-id>.log
```

### 3.1 `config.toml` (all keys optional; defaults shown)

```toml
[idle]
window = "01:00-07:00"      # local time, may wrap midnight; "" = any time of day
gpu_idle_minutes = 10       # no foreign compute process and utilization below gpu_util_max for this long
gpu_util_max = 5            # percent
require_ac = true           # no battery may be discharging
backoff_minutes = 10        # after a preemption

[search]
k = 3                       # Triton candidates per shape (KPolicy.k)
k_frequent = 5              # for shapes at or above frequent_calls (KPolicy.k_frequent)
frequent_calls = 1000000    # decayed call count that makes a shape frequent
lt_n = 16                   # cuBLASLt heuristic candidates
lt_rank = 5                 # learned cuBLASLt ranker proposals (0 = off)
half_life_days = 30         # decay of the usage record

[client]
dispatch = true             # false = count shapes only, never dispatch
inline_search = true        # search inline on a miss of a frequent shape (section 5.3)
min_kernel_us = 50          # never dispatch when torch's kernel is shorter than this (host launch cost dominates)
min_saving_us = 5           # nor when the winner saves less than this per call vs torch (also needs >= 4 %, 3 % locked)

[clock]
lock_mhz = 0                # 0 = measure at stock clocks; > 0 must equal the helper's /etc/lscli/clock.conf (set at install)

[train]
enabled = true
mode = "full"               # "full" | "quick" (train_best.sh modes)
```

### 3.2 Usage log `profile/<pid>-<t0>.jsonl` (client-written, every 30 s and at exit)

```json
{"t": 1762560000.0, "env_id": "3f2a91c0d4e5", "M": 4096, "N": 4096, "K": 4096, "dtype": "fp16", "layout": "NN",
 "calls": 1200, "hits": 1150, "fallbacks": 50}
```
`calls` etc. are increments since the previous flush. `layout` is `NN` (dispatchable) or `NT` (`F.linear`, logged only).

### 3.3 `profile.json` (daemon-written; merged from the logs)

```json
{"updated": 1762560000.0, "half_life_days": 30, "offsets": {"4242-1762559000.jsonl": 18234},
 "shapes": [{"M": 4096, "N": 4096, "K": 4096, "dtype": "fp16", "layout": "NN", "calls": 1.3e6, "calls_total": 2000000,
             "hits": 1900000, "fallbacks": 100000, "envs": ["3f2a91c0d4e5"],
             "first_seen": 1762000000.0, "last_seen": 1762560000.0}]}
```
`calls` is the decayed count as of `updated`: each increment is weighted by `0.5 ** (age_days / half_life_days)`.
The merge reads each log from its saved offset (only complete lines), so a log still being written is read
incrementally; a log is deleted only when its process (the pid in its name) has exited and it was read to the end.
The client resets its counters in a forked child (`os.register_at_fork`), so a DataLoader worker does not
double-count its parent's calls.

### 3.4 `queue.jsonl`

```json
{"t": 1762560000.0, "op": "search", "M": 4096, "N": 4096, "K": 4096, "dtype": "fp16", "env_id": "3f2a91c0d4e5",
 "reason": "miss"}
```
`reason` is one of `manual` (`lscli prewarm`), `miss` (client fallback), `stale` (soft-stale entry served),
`superseded` (new models), in that priority order. Duplicates are allowed: the daemon de-duplicates on (shape, dtype,
env_id), keeping the most urgent reason, and drops requests a current cache entry already satisfies. It records how
far it consumed in `queue.offset`. When everything is consumed it truncates the file **in place, under the same
`flock` the appenders take** (the inode never changes, so no writer can append to an unlinked file), and resets the
offset to 0.

### 3.5 `envs.json`

```json
{"3f2a91c0d4e5": {"python": "/home/u/miniconda3/envs/llm/bin/python", "prefix": "/home/u/miniconda3/envs/llm",
                  "torch": "2.13.0+cu130", "triton": "3.7.1", "cuda_runtime": "13.0", "registered": 1762560000.0}}
```
`env_id` = first 12 hex of sha256 of the interpreter's absolute path, **not** resolved: a venv's `bin/python` is a
symlink to the base interpreter, and resolving it would lose the venv. Written by `lscli enable-env`; the worker reports its
versions with every result and the daemon updates the record when they changed.

### 3.6 Winner cache `winners.jsonl`

The format of `tune/winner_cache.py` schema 2, at `$LSCLI_HOME/winners.jsonl`. Each entry's environment is the
worker's: GPU, its torch / Triton / CUDA runtime / cuBLASLt, driver, VBIOS, kernel source hash, plus the fields lscli
adds (`env_extra`, written to `client.json` so the client and the worker declare the same values):

* `clock_lock`: the daemon's measurement policy (`"unlocked"` or the MHz from `[clock] lock_mhz`), **not** the clock the
  user's program runs at. Entries timed under one policy are soft-stale under another, for the client as well, because
  the client declares the policy from `client.json`;
* `models`: `{"triton": <id>, "lt_rank": <id>, "router": <id>, "source": "native" | "zero_shot"}` from `MODELS.json`
  (an absent model is `null`);
* `env_id`: the registered env that timed it. Another env's winner is soft-stale here: served after the correctness
  check, then re-searched in this env. `env_id` also scopes `compact()`, so compacting with one env's environment
  never drops another env's lines.

lscli entries also carry `torch_us` (torch.matmul's short cold time for the shape, measured by the worker in the same
session) for the client's dispatch rule (section 5.3).

Tiers (from `tune/winner_cache.py`):

| Tier | When | Client | Daemon |
|---|---|---|---|
| current | all declared fields match | serve | widen if needed (section 4.3) |
| soft-stale | driver / torch / Triton / CUDA / cuBLASLt / VBIOS / clock policy / models differ, or the line predates a field | serve after one correctness check (`mark_verified`); a failed check is a miss | re-search (reason `stale` / `superseded`) |
| hard-stale | other GPU, or a Triton winner whose kernel source changed | miss | re-search if the shape is still in the profile |

`lscli cache compact` (and the daemon, daily) runs `WinnerCache.compact()`: a line that a current search replaced is
dropped unless its own search was wider (k, lt_n, lt_knn, lt_rank); soft-stale lines stay until re-searched.

### 3.7 `models/<hw_id>/MODELS.json`

```json
{"hw_id": "rtx3080_sm86_68sms_10gb", "source": "native",
 "models": {"triton": {"stem": "lgbm_rank_v2", "sha256": "..."}, "lt_rank": {"stem": "lt_rank", "sha256": "..."},
            "router": {"stem": "router", "sha256": "..."}},
 "trained_env": {"triton": "3.7.1", "cublaslt": 130101, "driver": "595.99.02", "kernel_src": "711287a76b6c5768"},
 "stages": {"tune_sweep": "done", "lt_screen": "running", "...": "..."},
 "activated": 1762560000.0}
```
A model id is `<stem>:<first 12 hex of sha256>`. A new set becomes active by an atomic rename of the directory.
`zero_shot/MODELS.json` has the same form with `"source": "zero_shot"` and the source GPU's `hw_id`.

### 3.8 `state.json` (daemon-written every 5 s)

```json
{"pid": 4242, "version": "0.2.0", "state": "waiting", "since": 1762560000.0,
 "gate": {"open": false, "reasons": ["outside window 01:00-07:00", "foreign compute process 12345"]},
 "job": null, "backoff_until": null, "paused_until": null,
 "models": {"source": "zero_shot", "training": {"stage": "tune_sweep", "done": 3, "total": 12}},
 "queue": {"pending": 14}, "cache": {"current": 120, "soft": 6}}
```
`state` ∈ `starting`, `waiting` (gate closed), `running` (a job holds the GPU), `backoff` (after a preemption),
`paused`, `error`.

## 4. Daemon

### 4.1 Idle gate (evaluated every 5 s)

The gate **opens** when all of these hold:

1. **GPU idle.** NVML lists no compute process on the daemon's GPU, and the GPU utilization stayed below
   `gpu_util_max` for `gpu_idle_minutes`. Graphics-only clients (the desktop compositor) are not compute processes. An
   idle process that holds a CUDA context (a Jupyter kernel, a paused training script) **is** one and keeps the gate
   closed; `lscli status` and `lscli doctor` name it.
2. **Window.** The local time is inside `[idle] window` (a window may wrap midnight).
3. **Power.** With `require_ac`, no `/sys/class/power_supply/*` battery reports `Discharging`.
4. **Not paused, no backoff, no foreign pipeline run.** `control.json` asks for no pause, no preemption backoff is
   running, and nobody else holds `src/pipeline/.lock` (a manual `train_best.sh` run).

While one of the daemon's jobs runs, the gate **stays open** as long as conditions 2-4 hold and no compute process
outside the job appears. Utilization is not checked then, because the job itself drives it up.

A job's processes are the job's root process and all its descendants, found by walking `/proc/<pid>/task/*/children`
each poll (sweep compile workers are spawned grandchildren). NVML reports host PIDs: processes in other PID namespaces
(containers) may appear with PIDs that match nothing, and they count as foreign. This errs on the side of yielding.

### 4.2 Preemption

While a job runs, a watcher polls every second. When the gate closes (section 4.1):

1. The job's process group gets SIGTERM, then SIGKILL after 5 s. Descendants that left the group (spawned compile
   workers) are killed by PID from the last descendant walk.
2. The clock is reset if it was locked.
3. The unit in flight is discarded. A search writes nothing, because the worker writes its winner only as its last
   step and is killed before that. A training stage keeps only the shapes it completed (train_best stages are
   resumable per shape).
4. `backoff_minutes` of backoff starts.

No measurement taken while a foreign compute process existed is ever written.

### 4.3 Scheduler

One job at a time, holding `gpu.lock`. Jobs, highest priority first; within a class, ordered by `calls × t_est`
(decayed calls from `profile.json` × the roofline estimate of one call):

1. **search**: queued `manual` / `miss` requests, and every NN profile shape with no usable entry for an env that ran
   it.
2. **widen**: shapes whose decayed calls reached `frequent_calls` but whose entry was searched with `k < k_frequent`.
3. **re-search**: soft-stale entries (`stale`).
4. **train**: the next unfinished training stage (section 4.5), if training is enabled and needed.
5. **re-search**: entries superseded by newly activated models (`superseded`).
6. **maintenance**: profile merge, queue consumption and daily cache compaction. These are CPU only and need no gate.

The CPU-only training stages (`train_v2`, `lt_fit`, `router_fit`, `package`, `summary`) need the window, power and
pause conditions, but not GPU idleness. Every other stage is a GPU stage.

### 4.4 One search

For a (shape, dtype, env_id):

1. **Triton proposals** (daemon process, CPU): `propose_triton` with k = `k_frequent` when the shape's decayed calls
   are at least `frequent_calls`, else `k`.
2. **cuBLASLt proposals** (`python -m lscli.propose`, daemon venv, a child of the job): the learned cuBLASLt ranker's
   top `lt_rank` configs (`propose_lt(..., lt_n, lt_rank=...)`), as keys. They are built with the daemon venv's
   cuBLASLt. The worker rebuilds them with its own library (`cublaslt_knn.from_key`) and drops any its library
   rejects, so a user env with another cuBLASLt gets a subset, never a wrong kernel. Skipped when there is no
   cuBLASLt ranker for this GPU.
3. **Measurement**: `<env python> -m lscli.worker measure` (protocol in section 7) with all the proposal keys. The
   worker queries its own library's heuristic top `lt_n`, rebuilds the pool (`backends_from_keys`), short-times
   torch.matmul for the shape (`torch_us`), runs `tune.online.measure` (with the router round trip when a router is
   active), and writes the winner to `winners.jsonl` with its own environment plus `env_extra`.

The worker uses the user's normal `TRITON_CACHE_DIR`, so the kernels it compiles are already cached when the user's
program first runs them.

### 4.5 Training

* **Models in use.** Until `models/<hw_id>/MODELS.json` lists every model, the daemon uses `models/zero_shot/` (the
  3080 rankers) plus cuBLASLt's heuristic top `lt_n`. The zero-shot cuBLASLt ranker and router are used only on a GPU
  with the source's architecture (they are refused elsewhere, as in `select_kernel`). The zero-shot Triton rankers are
  used on any GPU.
* **First training.** The daemon runs `pipeline/train_best.sh --run-dir train/<hw_id>/1 --skip-install
  --skip-validation [--clock-lock MHZ | --clock-lock none --allow-unlocked] --full|--quick --only <stage>`
  (`--full` or `--quick` from `train.mode`; later stages use `--resume train/<hw_id>/1 --only <stage>`). That is one stage per job, in the script's order: `preflight`, `env`, `check`,
  `roofline`, `tune_sweep`, `heldout_sweep`, `selectors`, `train_v1`, `tail_sweep`, `train_v2`, `lt_screen`,
  `lt_fit`, `tune_stage2_cold`, `tune_stage2_graph`, `router_fit`, `bundle` (only when the daemon venv's Triton
  matches a registered env), `package`, `summary`.
  * `train_best.sh` runs from `src/`, which is a git clone (`preflight` checks the history) using the daemon venv's
    `python` (`--skip-install`).
  * It takes `src/pipeline/.lock` itself. The daemon does not hold that lock; its own jobs hold `gpu.lock`.
  * When `package` finishes, `models/` of the run is copied into `models/<hw_id>/` and activated. Every existing
    entry then becomes `superseded` (soft-stale by `models`).
  * `heldout_sweep` only screens the held-out shapes that `selectors` / `tune topk` need. No fit sees them.
* **Retraining after an environment change.** The daemon compares the active `trained_env` with the daemon venv's
  environment once a day:

| Change | Stages re-run |
|---|---|
| cuBLASLt version | `lt_screen`, `lt_fit`, `tune_stage2_cold`, `tune_stage2_graph`, `router_fit`, `package` |
| Triton version or kernel source | `tune_sweep`, `heldout_sweep`, `selectors`, `train_v1`, `tail_sweep`, `train_v2`, `tune_stage2_cold`, `tune_stage2_graph`, `router_fit`, `bundle`, `package` |
| driver only | none: every winner becomes soft-stale and is re-searched; `lscli train retrain` forces a full retraining |

  A retraining runs in a **new** run directory `train/<hw_id>/<n+1>/`. The directory is seeded with hard links to the
  previous run's outputs and stamps of the stages that are **not** re-run, so the stages that are re-run start empty.
  Reusing the old directory would keep their old rows, because the stages resume per shape and `--only` skips
  stamped stages.

### 4.6 Clock

* The lock MHz is chosen **at install time**. The installer suggests about 85 % of the card's maximum SM clock,
  validated against `nvidia-smi -q -d SUPPORTED_CLOCKS`. It writes the root-owned `/etc/lscli/clock.conf` and sets
  `[clock] lock_mhz` to the same value.
* With `lock_mhz > 0`, the daemon runs `sudo -n /usr/local/libexec/lscli-clock lock` before a GPU job and
  `sudo -n /usr/local/libexec/lscli-clock reset` after it, on preemption and on exit. The unit runs the reset in
  `ExecStartPre` and `ExecStopPost`, so a crash or SIGKILL is cleaned up at the next start.
* The helper:
  * is root-owned (0755) in root-owned directories;
  * uses an absolute `/usr/bin/nvidia-smi` and a scrubbed environment;
  * takes exactly one argument, `lock` or `reset`;
  * reads the MHz only from `clock.conf` and re-validates it.
* `/etc/sudoers.d/lscli` allows exactly those two command lines, for the installing user only, and is checked with
  `visudo -c`.
* Without the helper, `lock_mhz` must be 0.

## 5. Client

### 5.1 Enabling

* `import lscli; lscli.enable()` patches for the rest of the process; `lscli.disable()` restores torch.
* `LSCLI_PATCH=1` in the environment enables it through `lscli.pth` (installed by `lscli enable-env`). The hook
  imports nothing unless the variable is set. It then enables the client right after the program's own `import
  torch`: it never imports torch or initializes CUDA itself, so `CUDA_VISIBLE_DEVICES` set in-script still works and
  programs without torch pay nothing.
* `LSCLI_DISABLE=1` overrides both (a kill switch).

### 5.2 Which calls are dispatched

`torch.matmul(a, b)`, `a @ b`, `torch.mm(a, b)`, `a.matmul(b)` and `a.mm(b)` are handled when all of these hold:

* exactly two positional arguments: keyword calls (`input=`, `out=`, ...) go to torch with their arguments as given;
* both are plain tensors or `nn.Parameter`s (other subclasses keep their own dispatch), on CUDA;
* the tensors' device is the current device;
* that device's UUID equals the daemon's GPU (`gpu_uuid` in `client.json`), so `CUDA_VISIBLE_DEVICES` cannot route a
  winner to another card;
* the dtype is fp16 or bf16, both operands have the same dtype, and both are 2-D;
* M, N and K are in [1, 16384];
* both are contiguous with 16-byte-aligned data pointers;
* `out=` is not given;
* the call is not inside `torch.compile` tracing (compiled code uses torch's op), CUDA-graph capture, `torch.autocast`
  (torch's cast semantics apply), a `torch.func` transform (`grad`, `vmap`, `jvp`) or forward-mode AD.

The kernel runs inside an `autograd.Function` only when an operand requires grad. Its backward calls the **original**
(unpatched) `torch.matmul`.

Every other call goes to the original torch function untouched. `F.linear` is wrapped only to count its shape
(`layout: "NT"`).

### 5.3 Per-call behaviour

1. Count the shape (an in-process counter).
2. Look the shape up in the winner table.
   * The client loads `winners.jsonl` once with the `env_extra` of `client.json`, re-checks its mtime at most once
     per second and reloads on change.
   * Any search width is accepted (`get(k=1, lt_n=0, allow_stale=True)`), because widening is the daemon's job
     (section 4.3).
3. **Current entry:** dispatch if the rule in step 6 allows it. The backend is built once per shape and kept.
4. **Soft-stale entry:** the first time in this process, run the correctness check, once per environment, recorded
   with `mark_verified`.
   * On a pass, dispatch (step 6) and queue `stale` once.
   * On a fail, treat it as a miss.
5. **Miss:**
   * **Search inline** when all of these hold: `inline_search` is on, the shape's decayed calls in `profile.json` are
     at least `frequent_calls`, and lightgbm and the models in `client.json` can be loaded. The search is
     `select_kernel(..., calls=calls, cache=<the lscli WinnerCache>, use_bundle=False, serve_stale=True)` and runs
     once per process. `use_bundle=False` keeps a Triton cache manager out of the user's process.
   * **Otherwise**, call torch and queue `miss` once per process.
6. **Dispatch rule.** Dispatch only when the entry has `torch_us` and both conditions hold:
   * `torch_us >= min_kernel_us`: under 50 µs the Python-side Triton launch costs as much as the kernel;
   * `torch_us - winner_us >= max(min_saving_us, f · torch_us)` with f = 0.04 when the entry was timed at stock clocks
     (`env.clock_lock` "unlocked", or not declared) and f = 0.03 under a declared clock lock. Both times come from
     the short online timing (10 cold samples, one placement), whose kernel-vs-kernel error at >= 50 µs is p95 3.5 %
     unlocked and 2.7 % locked on the 3080 (`reports/week1.md` 2026-11-13, `analysis/noise_compare.py`). The
     study's 2 % margin is for full-protocol medians and is too small for these.
   Otherwise call torch, which is still counted as a hit: the selector chose torch.

A dispatched call never raises a new exception type. Routing errors are contained and the call goes to torch. On
a kernel error the client calls torch, disables dispatch for that shape for the rest of the process, and logs one
warning.

* **Threads.** Decisions run outside the client's lock with a per-shape guard. While one thread decides a shape,
  other threads calling it get torch. Reloading the table swaps it in atomically.
* **Large soft-stale entries.** A soft-stale entry whose correctness check would allocate more than 512 MiB in the
  user's process is not checked there: the call uses torch and a `stale` re-search is queued.

### 5.4 Overhead budget

`tests/test_lscli_client.py::test_overhead` measures the host time per call (median of 10,000 calls; the CPU is
printed):

* **Undispatched call** (wrong dtype, layout, device, ...): at most +5 µs over the unpatched function.
* **Dispatched hit:** at most +20 µs over calling the selected kernel directly (`backend.make_runner(a, b)()`). That
  launch itself costs more than torch's (a Triton launch from Python takes tens of µs). This is why dispatch is limited
  to kernels of at least `min_kernel_us`, where the launch runs ahead of the GPU.

Measured on the RTX 2060 box (x86_64), after all the scope checks of 5.2:

* +2.3 µs for an early exit (fp32; torch takes 11.5 µs);
* +3.7 µs for a late exit (fp16, non-contiguous);
* +13.5 µs for a dispatched call over a 36.9 µs Triton launch.

The dispatched budget was raised from 15 to 20 µs once measured. The routing checks cost 4.7 µs timed alone but
about 13 µs when interleaved with a Triton launch: each check costs more next to the launcher (the usage count goes
from 0.75 to 3.3 µs), which points to CPU cache effects, so the cost is real. Measured runs give 13-14 µs; the budget
leaves room for machine noise.

The test syncs often enough that the launch queue never fills, because a full queue would count GPU time as host
overhead.

## 6. CLI

`lscli [--home DIR] [--json] <command>`. With `--json`, a command prints one JSON document instead of text.

| Command | Synopsis | Effect |
|---|---|---|
| `status` | `lscli status [--watch]` | Daemon state, gate and reasons, current job, queue, training progress, cache tiers, client hit rate (from the profile) |
| `enable-env` | `lscli enable-env [PYTHON] [--no-deps]` (default: `python3` on `PATH`) | Needs torch + triton importable there. Refuses if the env has its own `lscli` / `tune` / `kernels` / `baselines` / `analysis` package. pip-installs numpy / nvidia-ml-py if missing (`--no-deps`: fail instead). Writes `lscli.pth` into the env's site-packages: it puts `src/` on the path and runs the `LSCLI_PATCH` hook. Registers the env in `envs.json` and queues the shapes the profile already knows for it |
| `disable-env` | `lscli disable-env PYTHON` | Remove `lscli.pth`; unregister |
| `envs` | `lscli envs` | List registered interpreters and their versions |
| `pause` / `resume` | `lscli pause [--for DURATION]`, `lscli resume` | Stop starting jobs (a running one is preempted) / clear the pause |
| `prewarm` | `lscli prewarm (--shapes FILE \| MxNxK ...) [--dtype fp16] [--env ID]` | Queue shapes (`manual`); FILE is a shapes JSON (`suites/` format) |
| `profile` | `lscli profile [--top N] [--layout NN\|NT]` | The usage record, hottest first |
| `cache` | `lscli cache ls [--stale]` / `verify` / `compact` / `clear [--yes]` | List entries with tiers; run the correctness check on every entry served here; compact; delete the file |
| `train` | `lscli train status` / `start` / `retrain [--stages S,...]` | Training progress; allow training now (window ignored, gate still required); force stages |
| `config` | `lscli config [KEY [VALUE]]` | Show / get / set a config key (`idle.window`, ...) and tell the daemon to reload |
| `logs` | `lscli logs [-f] [--job ID]` | Daemon or job log |
| `doctor` | `lscli doctor` | Checks: GPU and driver, systemd user session and linger, unit state, clock helper, versions of each registered env (and gcc + `Python.h`, which Triton's launcher needs), file permissions, models present, processes holding a CUDA context (they keep the gate closed) |
| `daemon` | `lscli daemon run` (used by the unit) / `start` / `stop` / `restart` | Foreground loop / `systemctl --user` wrappers |
| `uninstall` | `lscli uninstall [--purge] [--yes]` | Stop and remove the unit, reset clocks, remove the clock helper (sudo), `disable-env` every registered env, remove `~/.local/bin/lscli`; keep `$LSCLI_HOME` unless `--purge` |

Exit codes: `0` success, `1` failure, `2` usage error, `3` the daemon is needed but not running, `4` unsupported
system (no NVIDIA GPU, no systemd user session).

## 7. Worker protocol

`<python> -m lscli.worker measure` reads one JSON request from stdin:

```json
{"M": 4096, "N": 4096, "K": 4096, "dtype": "fp16", "k": 3, "lt_n": 16, "lt_rank": 5, "confirm": 3, "router": true,
 "proposals": [{"family": "triton", "key": "bm128_bn128_bk32_w4_s3_g1_sk1", "setup_s": 0.0},
               {"family": "cublaslt_rank", "key": "r0_a21_t20_s8_sk1_r0_sw0_c0", "setup_s": 0.12}],
 "cache": "/home/u/.local/share/lscli/winners.jsonl",
 "env_extra": {"clock_lock": "1755",
               "models": {"triton": "lgbm_rank_v2:0123456789ab", "lt_rank": "lt_rank:3c4d5e6f7a8b",
                          "router": "router:9a8b7c6d5e4f", "source": "native"}},
 "order": "diverse", "ranker": "lgbm_rank_v2", "reply_timeout_s": 30}
```

It writes JSON lines to its **protocol channel**:

* `{"op": "route", "t_lt_us": 41.2}` when `router` is true, after the cuBLASLt half is timed. The daemon answers on
  stdin with `{"skip": false, "p": 0.31, "threshold": 0.02}`. With no answer within `reply_timeout_s`, the worker
  does not skip.
* `{"op": "result", "choice": {...}, "candidates": [...], "router": {...}, "torch_us": 812.0, "written": true,
  "env": {...}}` as the last line. `env` is the worker's environment, used to update `envs.json`.
* `{"op": "error", "error": "..."}` for any failure (the exit status is non-zero too).

The channel is the worker's original stdout. At start-up the worker duplicates fd 1 to a private descriptor, then
points fd 1 and `sys.stdout` at stderr. A library print or warning therefore cannot corrupt the protocol.

The daemon reads the channel line by line (never `communicate()`) while it can still write stdin, and it kills a
worker that sends nothing for 10 minutes.

Other subcommands, all with JSON on stdout:

* `env` prints the interpreter's `environment()`;
* `verify --cache PATH --env-extra JSON` runs the correctness check on every entry this env would serve.

## 8. Install

```bash
curl -fsSL https://raw.githubusercontent.com/Rayner3103/CS464/<tag>/install/install.sh | sh
```

1. **Preflight.** Exit 4 if any of these fails:
   * Linux;
   * `systemctl --user` is reachable;
   * `nvidia-smi` works;
   * Python ≥ 3.11 for the daemon venv (`tomllib`);
   * gcc and the Python headers are present (Triton builds its launcher with them);
   * `git` is present;
   * one GPU (with more: warn and use device 0).
2. `git clone` the repo at `<tag>` into `$LSCLI_HOME/src`. It must be a clone, because `train_best.sh`'s `preflight`
   checks the history. Create `venv/` and install the pinned `requirements.txt` (torch from the cu130 index) and
   `src` in editable mode. `$LSCLI_HOME` is created with mode 0700.
3. Download the zero-shot models (GitHub release assets of `<tag>`) into `models/zero_shot/`. Their sha256 sums are
   **pinned in `install.sh`** itself, not fetched from the release. Unpack `kernels/bundle.tar.xz`.
4. Write `~/.config/lscli/config.toml` if absent. Link `~/.local/bin/lscli`. Install and start `lscli.service`
   (`systemctl --user enable --now`). Run `loginctl enable-linger` when the user agrees (prompt; `--linger` /
   `--no-linger`).
5. Clock helper: ask "install the clock helper (needs sudo)? [y/N]" (`--clock-helper` / `--no-clock-helper`).
   On yes:
   * ask for the lock MHz (`--clock-mhz`, default about 85 % of the maximum SM clock, validated against the supported
     clocks);
   * install the helper, `/etc/lscli/clock.conf` and `/etc/sudoers.d/lscli` (checked with `visudo -c`);
   * set `[clock] lock_mhz`.
6. Print the next step: `lscli enable-env $(command -v python3)`.

The installer is idempotent; re-running it upgrades `src/` and the venv and keeps the data.

The window only matters while the machine is awake. lscli never wakes it from suspend, so a machine that sleeps at
night trains in whatever window it is awake for. Without linger, the service stops at logout.

## 9. Known limitations (v1)

* NN only: workloads whose GEMMs go through `nn.Linear` are counted but not sped up.
* The client puts `src/` on the user's env path (`lscli.pth`), so lscli's top-level `tune`, `kernels`, `baselines` and
  `analysis` packages become importable there. A user package with one of these names conflicts; `enable-env` refuses
  if it finds one. Code updates reach every env with the next `src/` update; there is no per-env copy.
* Winners are timed in idle conditions and served under load; near-ties (within 4 % at ≥ 50 µs at stock clocks, 3 %
  locked; 5 % below, borrowed: graph timing has no noise study) are arbitrary.
* A user env whose Triton is not the bundle's version gets no prebuilt kernels: its compiles happen in the worker,
  in idle time.
* Zero-shot quality depends on the GPU: close to native on sm_86 (3090), Triton half useless on sm_75 (2060), where
  the cuBLASLt half carries the search.
* The gate cannot tell an idle process that holds a CUDA context from a busy one: such a process keeps background
  work from running at all.
* The cuBLASLt ranker's proposals come from the daemon venv's cuBLASLt. A user env with another cuBLASLt build gets
  only the subset its library accepts.
