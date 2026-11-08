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
| fp16 and bf16, 2-D, contiguous row-major NN `torch.matmul` / `@` | NT / TN layouts (`F.linear`, `nn.Linear`): logged, never dispatched |
| Inference and training (backward runs on torch) | Dispatch inside `torch.compile` graphs or CUDA-graph capture |

## 2. Components

| Component | Runs in | Needs | Does |
|---|---|---|---|
| **client** (`lscli` package: `lscli.enable()`, the patch) | the user's own Python env | torch, triton, numpy, nvidia-ml-py | counts shapes, dispatches cached winners, falls back to torch, queues misses |
| **worker** (`python -m lscli.worker`) | the user's interpreter, started by the daemon | same as the client | compiles, validates and times one proposed pool (`tune.online.measure`), writes the winner |
| **daemon** (`lscli daemon run`, unit `lscli.service`) | `~/.local/share/lscli/venv` (pinned versions) | the repo's `requirements.txt` | idle gate, scheduler, ranking (`propose_*`), training, model registry, cache compaction |
| **CLI** (`lscli ...`) | the daemon venv (on `PATH` via `~/.local/bin/lscli`) | same as the daemon | user commands, section 6 |
| **clock helper** (`/usr/local/libexec/lscli-clock`, optional) | root, via one sudoers rule | `nvidia-smi` | locks / resets the SM clock around idle jobs |

The client and the daemon never import each other at run time; they communicate only through the files in section 3.

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
  envs.json                  registered client interpreters
  models/<hw_id>/            this GPU's models + MODELS.json
  models/zero_shot/          the 3080 models used until native ones exist
  train/<hw_id>/             train_best run directory (resumable)
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
min_saving_us = 5           # do not dispatch a winner that saves less than this per call vs torch

[clock]
lock_mhz = 0                # 0 = measure at stock clocks; > 0 needs the clock helper

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

### 3.3 `profile.json` (daemon-written; merged from the logs, which are then deleted)

```json
{"updated": 1762560000.0, "half_life_days": 30,
 "shapes": [{"M": 4096, "N": 4096, "K": 4096, "dtype": "fp16", "layout": "NN", "calls": 1.3e6, "calls_total": 2.0e6,
             "first_seen": 1762000000.0, "last_seen": 1762560000.0}]}
```
`calls` is the decayed count: each increment is weighted by `0.5 ** (age_days / half_life_days)`.

### 3.4 `queue.jsonl`

```json
{"t": 1762560000.0, "op": "search", "M": 4096, "N": 4096, "K": 4096, "dtype": "fp16", "env_id": "3f2a91c0d4e5",
 "reason": "miss"}
```
`reason` is one of `miss` (client fallback), `stale` (soft-stale entry served), `superseded` (new models), `manual`
(`lscli prewarm`). Duplicates are allowed; the daemon de-duplicates on (shape, dtype, env_id) and drops requests a
current cache entry already satisfies. The daemon records consumed offsets in `state.json` and rewrites the file when
everything in it is consumed.

### 3.5 `envs.json`

```json
{"3f2a91c0d4e5": {"python": "/home/u/miniconda3/envs/llm/bin/python", "prefix": "/home/u/miniconda3/envs/llm",
                  "torch": "2.13.0+cu130", "triton": "3.7.1", "cuda_runtime": "13.0", "registered": 1762560000.0}}
```
`env_id` = first 12 hex of sha256 of the interpreter's real path. Written by `lscli enable-env`; versions are re-read at
the start of every worker run (a changed env updates its record).

### 3.6 Winner cache `winners.jsonl`

The format of `tune/winner_cache.py` schema 2, at `$LSCLI_HOME/winners.jsonl`. Each entry's environment is the
worker's: GPU, its torch / Triton / CUDA runtime / cuBLASLt, driver, VBIOS, kernel source hash, plus:

* `clock_lock`: the daemon's measurement policy (`"unlocked"` or the MHz from `[clock] lock_mhz`), **not** the clock the
  user's program runs at: entries timed under one policy are soft-stale under another;
* `models`: `{"triton": <id>, "lt_rank": <id>, "router": <id>, "source": "native" | "zero_shot"}` from `MODELS.json`.

Tiers (from `tune/winner_cache.py`):

| Tier | When | Client | Daemon |
|---|---|---|---|
| current | all declared fields match | serve | nothing |
| soft-stale | driver / torch / Triton / CUDA / cuBLASLt / VBIOS / clock policy / models differ, or the line predates a field | serve after one correctness check (`mark_verified`); a failed check is a miss | re-search (reason `stale` / `superseded`) |
| hard-stale | other GPU, or a Triton winner whose kernel source changed | miss | re-search if the shape is still in the profile |

`lscli cache compact` (and the daemon, daily) runs `WinnerCache.compact()`: an entry a current search replaced is
dropped; soft-stale entries stay until re-searched.

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

The gate is open when all of these hold:

1. **GPU idle.** NVML lists no compute process on device 0 other than the daemon's own job processes, and the GPU
   utilization stayed below `gpu_util_max` for `gpu_idle_minutes`. Graphics-only clients (the desktop compositor)
   do not count as compute processes.
2. **Window.** The local time is inside `[idle] window`.
3. **Power.** With `require_ac`, no `/sys/class/power_supply/*` battery reports `Discharging`.
4. **Not paused / no backoff.** `control.json` asks for no pause, and no preemption backoff is running.

### 4.2 Preemption

While a job runs, a watcher polls NVML every second. If a compute process that is not one of the job's processes
appears (or the gate closes for any other reason):

1. The job's process group gets SIGTERM, then SIGKILL after 5 s. Orphaned compile workers are killed by PID.
2. The clock is reset (if it was locked).
3. The unit in flight is discarded: a search writes nothing; a training stage keeps only the shapes it completed
   (train_best stages are resumable per shape).
4. `backoff_minutes` of backoff starts.

No measurement taken while a foreign compute process existed is ever written.

### 4.3 Scheduler

Jobs, highest priority first; within a class, by `calls × t_est` (decayed calls from `profile.json` × the roofline
estimate of one call):

1. **search**: queued `miss` / `manual` requests, and every profile shape with no usable entry.
2. **re-search**: soft-stale entries (`stale`).
3. **train**: the next unfinished training stage (section 4.5), if training is enabled and needed.
4. **re-search**: entries superseded by newly activated models (`superseded`).
5. **maintenance**: profile merge, queue rewrite and cache compaction (daily; CPU only, gate not required).

CPU-only training stages (`train_v1`, `train_v2`, `lt_fit`, `router_fit`) need the window and power conditions but
not GPU idleness.

### 4.4 One search

For a (shape, dtype, env_id), the daemon:

1. Ranks in its own venv: `propose_triton` (k = `k_frequent` when the shape's decayed calls are at least
   `frequent_calls`, else `k`) and the model-dependent cuBLASLt proposals (`lt_rank`).
2. Starts `<env python> -m lscli.worker measure` (protocol in section 7) with the proposal keys.
3. The worker rebuilds the pool in the user's env (`backends_from_keys`; it queries cuBLASLt's heuristic itself),
   measures it, and writes the winner to `winners.jsonl` with its own environment.

The worker uses the user's normal `TRITON_CACHE_DIR`, so the kernels it compiles are already cached when the user's
program first runs them.

### 4.5 Training

* **Models in use.** Until `models/<hw_id>/MODELS.json` exists with every model, the daemon uses
  `models/zero_shot/` (the 3080 rankers, foreign-GPU flags set) plus cuBLASLt's heuristic top `lt_n`. On a GPU of
  another architecture the cuBLASLt ranker and router are refused, as in `select_kernel`.
* **First training.** `pipeline/train_best.sh --resume train/<hw_id> --only <stage> --skip-validation
  [--clock-lock MHZ | --clock-lock none --allow-unlocked]`, one stage per job, in the script's dependency order,
  skipping the `validate_*` stages. When the last fit finishes, the new set is packaged into `models/<hw_id>/`,
  activated, and every existing entry becomes `superseded` (soft-stale by `models`).
* **Retraining after an environment change.** Only the affected stages re-run:

| Change | Stages |
|---|---|
| cuBLASLt version | `lt_screen`, `lt_fit`, `tune_stage2_cold`, `tune_stage2_graph`, `router_fit` |
| Triton version or kernel source | `tune_sweep`, `tail_sweep`, `selectors`, `train_v1`, `train_v2`, `tune_stage2_*`, `router_fit`, `bundle` |
| driver | `tune noise --rounds 1`; if the noise floor moved past 2 %, as for a cuBLASLt change, else nothing |

Retraining compares the active `trained_env` with the daemon venv's environment once a day.

### 4.6 Clock

With `lock_mhz > 0` and the helper installed, the daemon runs `sudo -n /usr/local/libexec/lscli-clock lock` before a
GPU job and `... reset` after it, on preemption, on exit, and from the unit's `ExecStopPost`. The helper reads the
MHz from the root-owned `/etc/lscli/clock.conf`. It never takes a frequency argument, so the sudoers rule allows
exactly two fixed command lines. Without the helper, `lock_mhz` must be 0.

## 5. Client

### 5.1 Enabling

* `import lscli; lscli.enable()` patches for the rest of the process; `lscli.disable()` restores torch.
* `LSCLI_PATCH=1` in the environment enables it at interpreter start-up, through `lscli.pth` (installed by
  `lscli enable-env`), which imports nothing unless the variable is set.
* `LSCLI_DISABLE=1` overrides both (a kill switch).

### 5.2 Which calls are dispatched

`torch.matmul(a, b)`, `a @ b` and `torch.mm(a, b)` are handled when all of these hold:

* both are CUDA tensors on device 0;
* the dtype is fp16 or bf16, both operands have the same dtype, and both are 2-D;
* both are contiguous with 16-byte-aligned data pointers;
* `out=` is not given;
* the call is not inside `torch.compile` tracing or CUDA-graph capture.

Under autograd the kernel runs inside an `autograd.Function` whose backward uses `torch.matmul`.

Every other call goes to the original torch function untouched. `F.linear` is wrapped only to count its shape
(`layout: "NT"`).

### 5.3 Per-call behaviour

1. Count the shape (an in-process counter).
2. Look the shape up in the winner table. The client loads `winners.jsonl` once, then re-checks its mtime at most
   once per second and reloads on change.
3. **Current entry:** run the winner's backend (built once per shape and kept).
4. **Soft-stale entry:** the first time in this process, run the correctness check (once per environment, recorded
   with `mark_verified`). On a pass, serve it and queue `stale` once; on a fail, treat it as a miss.
5. **Miss:** if `inline_search` and the shape's decayed calls in `profile.json` are at least `frequent_calls`, run
   `select_kernel(..., calls=calls, serve_stale=True)` inline, once per process (needs lightgbm in the user's env; if
   it is missing, behave as below). Otherwise call torch and queue `miss` once per process.
6. **No dispatch** of a winner whose `winner_us` is within `min_saving_us` of torch's time, where known (the
   per-process dispatch overhead is larger than the gain).

A dispatched call never raises a new exception type. On any error the client calls torch, disables dispatch for
that shape for the rest of the process, and logs one warning.

### 5.4 Overhead budget

The patched path for an undispatched call (wrong dtype, layout, device, ...) adds at most 2 µs; a dispatched hit adds
at most 5 µs over the kernel launch. Measured by `tests/lscli/test_overhead.py`.

## 6. CLI

`lscli [--home DIR] [--json] <command>`. With `--json`, a command prints one JSON document instead of text.

| Command | Synopsis | Effect |
|---|---|---|
| `status` | `lscli status [--watch]` | Daemon state, gate and reasons, current job, queue, training progress, cache tiers, client hit rate (from the profile) |
| `enable-env` | `lscli enable-env [PYTHON]` (default: `python3` on `PATH`) | Install the client into that interpreter's env (`pip install --no-deps` from `src/`, then numpy / nvidia-ml-py if missing), write `lscli.pth`, register in `envs.json`, queue the env's shapes for search |
| `disable-env` | `lscli disable-env PYTHON` | Remove `lscli.pth` and the client package; unregister |
| `envs` | `lscli envs` | List registered interpreters and their versions |
| `pause` / `resume` | `lscli pause [--for DURATION]`, `lscli resume` | Stop starting jobs (a running one is preempted) / clear the pause |
| `prewarm` | `lscli prewarm (--shapes FILE \| MxNxK ...) [--dtype fp16] [--env ID]` | Queue shapes (`manual`); FILE is a shapes JSON (`suites/` format) |
| `profile` | `lscli profile [--top N] [--layout NN\|NT]` | The usage record, hottest first |
| `cache` | `lscli cache ls [--stale]` / `verify` / `compact` / `clear [--yes]` | List entries with tiers; run the correctness check on every entry served here; compact; delete the file |
| `train` | `lscli train status` / `start` / `retrain [--stages S,...]` | Training progress; allow training now (window ignored, gate still required); force stages |
| `config` | `lscli config [KEY [VALUE]]` | Show / get / set a config key (`idle.window`, ...) and tell the daemon to reload |
| `logs` | `lscli logs [-f] [--job ID]` | Daemon or job log |
| `doctor` | `lscli doctor` | Checks: GPU and driver, systemd user session and linger, unit state, clock helper, versions of each registered env, file permissions, models present |
| `daemon` | `lscli daemon run` (used by the unit) / `start` / `stop` / `restart` | Foreground loop / `systemctl --user` wrappers |
| `uninstall` | `lscli uninstall [--purge] [--yes]` | Stop and remove the unit, reset clocks, remove the clock helper (sudo), `disable-env` every registered env, remove `~/.local/bin/lscli`; keep `$LSCLI_HOME` unless `--purge` |

Exit codes: `0` success, `1` failure, `2` usage error, `3` the daemon is needed but not running, `4` unsupported
system (no NVIDIA GPU, no systemd user session).

## 7. Worker protocol

`<python> -m lscli.worker measure` reads one JSON request from stdin:

```json
{"M": 4096, "N": 4096, "K": 4096, "dtype": "fp16", "k": 3, "lt_n": 16, "confirm": 3, "router": true,
 "proposals": [{"family": "triton", "key": "bm128_bn128_bk32_w4_s3_g1_sk1", "setup_s": 0.0},
               {"family": "cublaslt_rank", "key": "r0_a21_t20_s8_sk1_r0_sw0_c0", "setup_s": 0.12}],
 "rank_s": 0.04, "cache": "/home/u/.local/share/lscli/winners.jsonl",
 "env_extra": {"clock_lock": "1755", "models": {"triton": "lgbm_rank_v2:0123456789ab", "source": "native"}},
 "order": "diverse", "ranker": "lgbm_rank_v2"}
```

and writes JSON lines to stdout:

* `{"op": "route", "t_lt_us": 41.2}` when `router` is true, after the cuBLASLt half is timed. The daemon answers
  on stdin with `{"skip": false, "p": 0.31, "threshold": 0.02}`.
* `{"op": "result", "choice": {...}, "candidates": [...], "router": {...}, "written": true}` as the last line.
* `{"op": "error", "error": "..."}` for any failure (the exit status is non-zero too).

The heuristic cuBLASLt candidates are not in `proposals`: the worker queries its own library's heuristic top `lt_n`.
Proposals that the worker's library cannot build are dropped (`backends_from_keys`).

## 8. Install

```bash
curl -fsSL https://raw.githubusercontent.com/Rayner3103/CS464/<tag>/install/install.sh | sh
```

1. **Preflight.** Linux; `systemctl --user` reachable; `nvidia-smi` works; Python ≥ 3.11 for the daemon venv
   (`tomllib`); one GPU (more: warn, use device 0). Exit 4 if any of these fails.
2. Download the repo at `<tag>` into `$LSCLI_HOME/src`. Create `venv/` and install the pinned `requirements.txt`
   (torch from the cu130 index) and `src` in editable mode.
3. Download the zero-shot models (GitHub release assets of `<tag>`; sha256 checked) into `models/zero_shot/`.
   Unpack `kernels/bundle.tar.xz`.
4. Write `~/.config/lscli/config.toml` if absent. Link `~/.local/bin/lscli`. Install and start `lscli.service`
   (`systemctl --user enable --now`). Run `loginctl enable-linger` when the user agrees (prompt; `--linger` /
   `--no-linger`).
5. Clock helper: ask "install the clock helper (needs sudo)? [y/N]" (`--clock-helper` / `--no-clock-helper`). On
   yes, install the helper, `/etc/lscli/clock.conf` and `/etc/sudoers.d/lscli` (validated with `visudo -c`). The lock
   MHz is probed at the first open gate.
6. Print the next step: `lscli enable-env $(command -v python3)`.

The installer is idempotent; re-running it upgrades `src/` and the venv and keeps the data.

## 9. Known limitations (v1)

* NN only: workloads whose GEMMs go through `nn.Linear` are counted but not sped up.
* The client package installs top-level `tune`, `kernels` and `baselines` packages into the user's env. A user
  package with one of these names conflicts; `enable-env` refuses if it finds one.
* Winners are timed in idle conditions and served under load; near-ties (within 2 % at ≥ 50 µs, 5 % below) are
  arbitrary.
* A user env whose Triton is not the bundle's version gets no prebuilt kernels: its compiles happen in the worker,
  in idle time.
* Zero-shot quality depends on the GPU: close to native on sm_86 (3090), Triton half useless on sm_75 (2060), where
  the cuBLASLt half carries the search.
