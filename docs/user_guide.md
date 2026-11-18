# lscli user guide

`lscli` (learned-selector CLI) picks a faster GPU kernel for the matrix multiplications your PyTorch programs run, on
your own machine, without changing your code beyond one opt-in switch.

How it works, in short:

1. You run your program with `LSCLI_PATCH=1`. lscli counts the `torch.matmul` shapes it runs and lets torch compute
   them as usual.
2. While the machine is idle (by default between 01:00 and 07:00, GPU unused, on AC power), a background service
   searches those shapes. It tries a few candidate kernels a trained model suggests (Triton kernels and cuBLASLt
   configurations), checks each one against a reference result, and times it.
3. The next time your program runs a searched shape, lscli runs the winning kernel instead of torch's default. It
   does this only when the winner is clearly faster: at least 4 % and 5 µs per call on kernels of at least 50 µs.
   Otherwise it keeps torch.

What to expect: gains are per shape and usually small (a few percent where they exist). Many shapes stay on torch,
and kernels under 50 µs always do. Nothing leaves your machine (no telemetry). The full contract is the specification,
[`lscli/spec.md`](lscli/spec.md). This guide is the how-to.

**Scope (v1).** Linux with systemd, one NVIDIA GPU (device 0), fp16 2-D row-major `torch.matmul` / `@` / `torch.mm`
with M, N, K up to 16384. bf16 is dispatched but was not validated. `nn.Linear` / `F.linear` shapes are only counted,
never sped up. Nothing inside `torch.compile` graphs or CUDA-graph capture is dispatched.

Contents: [Requirements](#requirements) · [1. Install](#1-install) · [2. Enable it for your programs](#2-enable-it-for-your-programs) ·
[3. Try it out](#3-try-it-out) · [4. Everyday commands](#4-everyday-commands) · [5. Upgrade](#5-upgrade) ·
[6. Uninstall](#6-uninstall) · [7. Troubleshooting](#7-troubleshooting) · [8. Limitations](#8-limitations)

## Requirements

| Need | Detail |
|---|---|
| OS | Linux with a systemd **user** session (`systemctl --user` works). macOS and Windows are not supported |
| GPU | One NVIDIA GPU with a working driver (`nvidia-smi`). The pinned wheels are CUDA 13.0, so driver ≥ 580. Developed on RTX 3080 / 3090 (sm_86) and RTX 2060 (sm_75) |
| Python | ≥ 3.11.4 with `venv`, `lzma` and the development headers (`Python.h`). Ubuntu: `sudo apt install python3-venv python3-dev` |
| Tools | `git`, `gcc` (Triton builds a small launcher with it). Ubuntu: `sudo apt install git build-essential` |
| Disk | About 8 GB: the daemon's venv (torch with its CUDA libraries) ≈ 5 GB, prebuilt kernels 1.5 GB. Background training adds a few GB more |
| Network | The first install downloads torch (cu130 wheel, ~3 GB) and the other pinned packages |
| Root | Not needed. Only the optional clock helper uses `sudo`, once at install |

The program you want to speed up runs in **its own** Python environment, which needs `torch` and `triton`
(any versions). lscli adds a hook file to it, plus `numpy` and `nvidia-ml-py` if they are missing.

Check the machine without installing anything (this only creates the empty state directory
`~/.local/share/lscli`):

```bash
curl -fsSL https://raw.githubusercontent.com/Rayner3103/CS464/master/install/install.sh | bash -s -- --check-only
```

## 1. Install

Pick one of four ways:

| Option | Use it when |
|---|---|
| [1.1 One-line install](#11-one-line-install) | You want lscli running with the defaults |
| [1.2 Flags and non-interactive install](#12-flags-and-non-interactive-install) | You want to choose training, the clock helper and linger up front, or script the install |
| [1.3 A pinned ref, or from a clone](#13-a-pinned-ref-or-from-a-clone) | You want a specific branch or tag, the newest development code, or your own fork |
| [1.4 Library only, no service](#14-library-only-no-service) | You want to call the selector from Python yourself, with no background service |

### 1.1 One-line install

```bash
curl -fsSL https://raw.githubusercontent.com/Rayner3103/CS464/master/install/install.sh | bash
```

The installer checks the machine and clones the repository (branch `master`) into `~/.local/share/lscli/src`. It then
builds a venv with the pinned versions, unpacks the starter models and the prebuilt kernels, writes the config, and
starts a systemd user service. It asks three questions (when there is no terminal, every question takes its default):

| Question | Default | What it means |
|---|---|---|
| Install the SM clock helper (needs sudo)? | **no** | Locks the GPU's SM clock around lscli's own measurements (more repeatable timings) and resets it afterwards. It installs `/usr/local/libexec/lscli-clock` and a one-line sudoers rule. If you say yes, it asks for the MHz (default ≈ 85 % of the card's maximum) |
| Train this GPU's models in idle time? | **yes** | The starter models come from an RTX 3080. Training builds this GPU's own models with `pipeline/train_best.sh` in idle time: about 35 GPU-hours on an RTX 3080, split into pieces. It stops as soon as anything else uses the GPU. Turn it off any time with `lscli config train.enabled false` |
| Keep the daemon running while you are logged out? | **no** | Runs `loginctl enable-linger`. Without it the service stops when you log out, and only idle time while you are logged in is used |

What it creates:

| Path | What |
|---|---|
| `~/.local/share/lscli/` (`$LSCLI_HOME`, mode 0700) | `src/` (the git clone), `venv/`, `models/`, the winner cache `winners.jsonl`, your usage record `profile.json`, logs, training runs |
| `~/.config/lscli/config.toml` | Settings (`lscli config` shows and edits them) |
| `~/.local/bin/lscli` | The command. Add `~/.local/bin` to your `PATH` if the installer says so |
| `~/.config/systemd/user/lscli.service` | The background service (`systemctl --user status lscli`) |
| `/usr/local/libexec/lscli-clock`, `/etc/lscli/clock.conf`, `/etc/sudoers.d/lscli` | Only with the clock helper |

Next: [enable it for your programs](#2-enable-it-for-your-programs).

### 1.2 Flags and non-interactive install

Pass options after `bash -s --`:

```bash
# no questions: no clock helper, training on, no linger
curl -fsSL https://raw.githubusercontent.com/Rayner3103/CS464/master/install/install.sh | bash -s -- --yes

# a quick evaluation: no background training, no sudo
curl -fsSL https://raw.githubusercontent.com/Rayner3103/CS464/master/install/install.sh | bash -s -- --yes --no-train --no-clock-helper

# a dedicated box: lock the clock at 1755 MHz for lscli's measurements, keep running when logged out
curl -fsSL https://raw.githubusercontent.com/Rayner3103/CS464/master/install/install.sh | bash -s -- --clock-helper --clock-mhz 1755 --linger
```

| Option | Effect |
|---|---|
| `--yes` | Answer every question with its default |
| `--train` / `--no-train` | Background training on / off. On a re-install the existing setting is kept unless one of these is given |
| `--clock-helper [--clock-mhz MHZ]` / `--no-clock-helper` | Install the clock helper without asking (MHz between 300 and the card's maximum) / never |
| `--linger` / `--no-linger` | `loginctl enable-linger` or not |
| `--no-start` | Install and enable the service but do not start it (start later with `lscli daemon start`) |
| `--home DIR` | State directory instead of `~/.local/share/lscli`. Must be new, empty, or an existing lscli home |
| `--python BIN` | Interpreter for the daemon's venv (default: the first of `python3.12`, `python3.11`, `python3` that is ≥ 3.11) |
| `--models-url URL --models-sha256 HEX` | Install another starter-model archive instead of the committed one |
| `--check-only` | Run the checks above and exit (exit code 4 = unsupported system) |
| `--ref REF`, `--repo URL`, `--reuse-site PYTHON` | See [1.3](#13-a-pinned-ref-or-from-a-clone) |

### 1.3 A pinned ref, or from a clone

**A branch or tag.** `--ref` chooses what is cloned into `src/`. Fetch `install.sh` from the same ref, so the
installer and the code it installs match. For the newest development code (branch `learned-selectors`):

```bash
curl -fsSL https://raw.githubusercontent.com/Rayner3103/CS464/learned-selectors/install/install.sh | bash -s -- --ref learned-selectors
```

`master` moves too. To stay on one version, pin a commit or tag with `--ref`.

**From a local clone or a fork.** Run the script from a checkout and point `--repo` at any URL or path `git clone`
accepts:

```bash
git clone https://github.com/Rayner3103/CS464.git && cd CS464
bash install/install.sh --repo https://github.com/<you>/CS464.git --ref my-branch
```

The installer always makes its **own** clone in `$LSCLI_HOME/src`. Your checkout is only the source. A re-install
resets `src/` to the ref and discards any edits made there.

`--reuse-site PYTHON` (for development) skips the 3 GB of pip installs: the daemon's venv borrows the site-packages
of `PYTHON`, which must already have the pinned `requirements.txt` versions.

There is one install per user: the command `~/.local/bin/lscli` and the service file are fixed paths, so a second
install with another `--home` replaces the first one's service.

### 1.4 Library only, no service

The selector is ordinary Python in this repository (`tune.online.select_kernel`). Without lscli there is no
background service, no usage record and no torch patch. You call it for the shapes you choose, and it searches the
first time each shape is seen (~0.2-0.9 s per shape), then reads the result from a per-shape cache.

```bash
git clone https://github.com/Rayner3103/CS464.git && cd CS464
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
mkdir -p models/zero_shot && tar -xJf install/zero_shot.tar.xz -C models/zero_shot   # the starter models
```

Then, from the repo root:

```python
from pathlib import Path
import torch
from analysis.learned import Ranker
from tune.online import select_kernel
from tune.winner_cache import backend_for

ranker = Ranker.load(Path("models/zero_shot/lgbm_rank_v2"))    # the RTX 3080's Triton ranker (rank v2)
M, N, K = 2048, 11008, 4096
sel = select_kernel(ranker, M, N, K)    # 3 Triton candidates + cuBLASLt top-16 + learned cuBLASLt picks, timed here
print(sel["reason"], sel["choice"]["family"], sel["choice"]["key"])   # "default" the first time, "cached" after

a = torch.randn(M, K, device="cuda", dtype=torch.float16)
b = torch.randn(K, N, device="cuda", dtype=torch.float16)
c = backend_for(sel["choice"], M, N, K).make_runner(a, b)()     # equals a @ b within fp16 tolerance
```

* `select_kernel(..., calls=n)` searches 5 Triton candidates instead of 3 for shapes called ≥ 1M times.
  `sel["choice"]` is `None` for a shape that does not fit in memory (`sel["skipped"]`).
* Winners are stored in `data/<hw_id>/winner_cache.jsonl`. `TUNE_WINNER_CACHE=<path>` moves the file and `=off`
  turns it off. The prebuilt Triton kernels (`kernels/bundle.tar.xz`) are unpacked on first use (~14 s, 1.5 GiB) and
  only match Triton 3.7.1 on sm_86 GPUs. Elsewhere the candidates compile on first use.
* The learned cuBLASLt ranker and the router were trained on an RTX 3080. On another GPU model they are skipped
  with a warning, and the search uses cuBLASLt's own top-16. More: [`kernel_bundle.md`](kernel_bundle.md), the
  `select_kernel` docstring in [`tune/online.py`](../tune/online.py).

To remove it, delete the clone (the winner cache lives in its `data/`) and the Triton cache (`~/.triton/cache`) if
you want that space back.

## 2. Enable it for your programs

**Register the Python environment your programs run in.** Run this once per environment, from a directory that is
**not** a CS464 checkout:

```bash
cd ~
lscli enable-env "$(command -v python)"        # e.g. ~/miniconda3/envs/llm/bin/python
```

It writes a small `lscli.pth` hook into that environment's `site-packages`. It refuses an environment where another
package named `tune`, `kernels`, `baselines` or `analysis` would shadow lscli's (for example, the interpreter you
develop CS464 itself with): use a separate environment for lscli. `lscli envs` lists the registered environments,
and `lscli disable-env PYTHON` removes one.

**Turn it on per run**, in one of two ways:

```bash
LSCLI_PATCH=1 python train.py
```

```python
import lscli
lscli.enable()      # patches torch.matmul / @ / torch.mm for the rest of the process; lscli.disable() undoes it
```

`LSCLI_DISABLE=1` is a kill switch that overrides both. A program that never imports torch pays nothing, and a call
lscli does not handle goes to torch untouched. If a dispatched kernel ever fails, that shape goes back to torch for
the rest of the process, with one warning.

## 3. Try it out

The repository ships a small demo, [`examples/lscli_demo.py`](../examples/lscli_demo.py). It runs four large fp16
GEMMs and prints, for each shape, torch's time, the time through lscli, and what lscli decided. Installed copy:
`~/.local/share/lscli/src/examples/lscli_demo.py`.

Instead of waiting for the night window, open the idle gate for the test. Turn training off while you test,
otherwise the daemon starts training as soon as the searches are done:

```bash
lscli config idle.window ""            # any time of day (default 01:00-07:00)
lscli config idle.gpu_idle_minutes 1   # GPU idle for 1 minute is enough (default 10)
lscli config train.enabled false       # only if you installed with training on
```

**1. First run: every shape is a miss.** Torch runs it, and lscli records the shape and queues it for a search:

```bash
cd ~ && LSCLI_PATCH=1 python ~/.local/share/lscli/src/examples/lscli_demo.py
```
```
GPU: NVIDIA GeForce RTX 3080 | torch 2.13.0+cu130 | lscli on
     shape (MxNxK)  torch us  lscli us  ratio  rel diff  decision
    4096x4096x4096    2359.5    2306.7  0.978   0.0e+00  miss (queued)
   2048x11008x4096    3188.9    3176.7  0.996   0.0e+00  miss (queued)
    512x4096x11008     750.6     751.1  1.001   0.0e+00  miss (queued)
     1000x3000x777     117.2     117.3  1.001   0.0e+00  miss (queued)
```

**2. Let the daemon search.** Leave the GPU alone (close notebooks or scripts that hold a CUDA context; the desktop
itself does not count). After the idle minute, the daemon searches the queued shapes one at a time, seconds each:

```bash
lscli status            # the gate and its reasons, the current job, the queue, cache counts
lscli logs -f           # the daemon's log; Ctrl-C to stop following
lscli cache ls          # the winners: winner us vs torch us per shape
```

**3. Second run: each shape now has a winner.**

```bash
cd ~ && LSCLI_PATCH=1 python ~/.local/share/lscli/src/examples/lscli_demo.py
```
```
     shape (MxNxK)  torch us  lscli us  ratio  rel diff  decision
    4096x4096x4096    2269.0    2269.9  1.000   0.0e+00  winner, kept torch
   2048x11008x4096    3096.5    3086.7  0.997   6.9e-04  dispatched cublaslt_rank r0_a5_t20_s0_sk1_r0_sw1_c0
    512x4096x11008     762.0     762.5  1.001   0.0e+00  winner, kept torch
     1000x3000x777     118.9     118.6  0.997   0.0e+00  winner, kept torch
```

This is a real run on an RTX 3080 at stock clocks. Read it as follows:

* `winner, kept torch`: the search found a winner, but it was not faster than torch by the dispatch margin (4 % and
  5 µs; 3 % under the clock helper's lock), so torch is kept. That is the selector working as intended: near-ties are
  not worth the risk.
* `dispatched <family> <key>`: the stored winner runs instead of torch. `rel diff` is its largest difference from
  torch's output, relative to the largest output value; differences of order 1e-3 are normal fp16 rounding.
* The search times each candidate briefly (10 samples), so a winner's margin can shrink when the demo re-times it,
  as for the dispatched shape above. `lscli cache ls` shows the margin the search measured.
* Your numbers depend on the GPU. Try your own shapes: `python lscli_demo.py 8192x8192x2048 3000x5000x7000`.

`lscli profile` lists every shape your programs ran, with call counts and the hit rate. A **hit** means the cache had
a winner, whether it was dispatched or torch was kept.

**4. Put the settings back** (or keep them, if the machine is free whenever it is idle):

```bash
lscli config idle.window 01:00-07:00
lscli config idle.gpu_idle_minutes 10
lscli config train.enabled true        # if you want this GPU's own models
```

To search shapes ahead of time without running the program, queue them with
`lscli prewarm 4096x4096x4096 2048x11008x4096`, or pass a shapes file with `--shapes FILE`.

## 4. Everyday commands

| Command | What it does |
|---|---|
| `lscli status` | Daemon state, idle gate and why it is closed, the running job, the queue, training progress, cache counts. `watch -n 5 lscli status` to follow it |
| `lscli doctor` | Checks the GPU, systemd, the service, linger, the clock helper, each registered environment, models, and processes that hold a CUDA context |
| `lscli profile [--top N]` | Your shapes, hottest first |
| `lscli cache ls [--stale]` / `verify` / `compact` / `clear --yes` | Inspect, re-check, tidy or delete the winners |
| `lscli prewarm MxNxK ... \| --shapes FILE` | Queue shapes for a search |
| `lscli pause [--for 2h]` / `lscli resume` | Stop background GPU work (a running job is stopped at once) / allow it again. Pause before a benchmark of your own: a job can overlap a process that starts during it by up to ~6 s |
| `lscli clock hold` / `release` | Tell lscli you are locking the SM clock yourself (`sudo nvidia-smi -lgc ...`): no GPU jobs and no clock changes until release |
| `lscli train status` / `start` / `retrain` | Training progress; train now (ignores the time window; also continues a failed run); start over |
| `lscli config [KEY [VALUE]]` | Show or change settings (the keys are listed in [spec 3.1](lscli/spec.md#31-configtoml-all-keys-optional-defaults-shown)). Setting a key rewrites `config.toml` and drops its comments |
| `lscli logs [-f] [--job ID]` | The daemon's log, or one job's |
| `lscli daemon start` / `stop` / `restart` | Control the service |

Every command takes `--json` for machine-readable output. Exit codes: 0 ok, 1 failure, 2 usage error, 3 daemon not
running, 4 unsupported system.

## 5. Upgrade

Re-run the installer, with the same `--ref` if you used one. It updates `src/` and the venv and keeps your data,
winners and config. Winners found with the old code are re-checked and re-searched in idle time, never served blindly.

## 6. Uninstall

```bash
lscli uninstall            # keeps your data in ~/.local/share/lscli
lscli uninstall --purge    # also deletes ~/.local/share/lscli and ~/.config/lscli/config.toml (asks you to type the path)
```

`uninstall` stops and removes the service. It also resets the GPU clock and removes the clock helper (asks for
sudo), removes the hook from every registered environment, and removes `~/.local/bin/lscli`. Add `--yes` to skip
the confirmation (with `--purge` too). `--purge` deletes only a directory the installer created (it carries an
`.lscli-home` marker).

It leaves these alone; remove them yourself if you want:

| Left behind | Remove with |
|---|---|
| Linger, if you enabled it | `loginctl disable-linger $USER` |
| A clock you locked yourself (`lscli clock hold`) | `sudo nvidia-smi -rgc` |
| Kernels compiled into your Triton cache (shared with your own Triton use) | `rm -rf ~/.triton/cache` (or your `TRITON_CACHE_DIR`) |
| `numpy` / `nvidia-ml-py`, if `enable-env` installed them into an environment | `pip uninstall` in that environment |

**If the `lscli` command itself is broken**, remove everything by hand:

```bash
systemctl --user disable --now lscli.service
rm -f ~/.config/systemd/user/lscli.service ~/.local/bin/lscli && systemctl --user daemon-reload
python -c 'import site; print(site.getsitepackages())'     # in each registered env: delete lscli.pth there
sudo /usr/local/libexec/lscli-clock reset                   # only if the clock helper is installed, then:
sudo rm -f /usr/local/libexec/lscli-clock /etc/sudoers.d/lscli /etc/lscli/clock.conf && sudo rmdir /etc/lscli
rm -rf ~/.local/share/lscli ~/.config/lscli
```

## 7. Troubleshooting

| Symptom | Check |
|---|---|
| Nothing is ever searched | `lscli status` lists why the gate is closed: outside the time window, a process holding a CUDA context (an idle Jupyter kernel counts), on battery, paused, the clock held. `lscli doctor` names the processes |
| `enable-env` refuses | Another `tune` / `kernels` / `baselines` / `analysis` package is importable there or from your current directory. Run it from `~`, or use another environment |
| `lscli: not patching torch (...)` warning | The same clash at run time (for example, a script started inside a CS464 checkout). lscli then leaves torch alone |
| Shapes are hits but nothing is dispatched | Expected when no winner beats torch by the margin; see `lscli cache ls`. Kernels under 50 µs are never dispatched |
| Nothing happens at all | The program must run in a registered environment with `LSCLI_PATCH=1` (or call `lscli.enable()`), on CUDA tensors that are fp16 or bf16, 2-D and contiguous. `lscli profile` shows whether any shape was counted |
| The service stops when I log out | Enable linger: `loginctl enable-linger $USER` |
| The models only work partly on my GPU | The starter models are an RTX 3080's. On sm_86 cards (3080, 3090) all of them apply. On others, the cuBLASLt half of the search carries it until this GPU's own models are trained (`lscli train status`) |

Logs: `lscli logs`, and per job `~/.local/share/lscli/logs/jobs/`.

## 8. Limitations

The full list is in [spec section 9](lscli/spec.md#9-known-limitations-v1). The main ones:

* Only NN `torch.matmul`; `nn.Linear` workloads are counted but not sped up.
* Winners are timed when the machine is idle and served under load. Near-ties (within ~4 %) are decided by noise,
  which is why the dispatch margin exists.
* A process that only holds a CUDA context keeps the idle gate closed.
* One GPU, one user. The starter models are tuned for sm_86. On other architectures the Triton half helps little
  until this GPU's own models are trained.
