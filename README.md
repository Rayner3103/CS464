# CS464: learned kernel selection for GPU GEMM

For each matrix-multiply shape (fp16 `C = A·B`) a program runs, this project picks the fastest GPU kernel, choosing
from a parameterized Triton kernel family and NVIDIA's cuBLASLt configurations. A LightGBM ranker proposes a short
list per shape, a short timing on the GPU picks the winner, and a per-shape cache remembers it. `lscli` runs all of
this as a background service on your own machine, with an opt-in `torch.matmul` patch.

> **New here? Start with [`results/`](results/README.md).** It is the project's write-up in seven short chapters:
> what a kernel selector is, how we measured, what we compared against, what we trained, how well it picks on sizes
> it never saw and on other GPUs, and how it ships. The short answer: the selector picks a kernel within a median
> 0.5 % of the best available in under a second, is faster than NVIDIA's default on most sizes and never slower
> beyond noise, but does not beat NVIDIA's offline-searched kernels on large GEMMs. Read its
> [summary page](results/README.md) first (key numbers, gates, glossary), then the chapters in order.

| I want to | Read |
|---|---|
| Understand the project and its results (start here) | [`results/`](results/README.md): summary, then chapters 1-7 |
| Install lscli, try it on my GPU, or uninstall it | [User guide](docs/user_guide.md) |
| Set up a development / training environment and find my way around | [Developer guide](docs/developer_guide.md) |
| Know every command, rule and result in detail | [`AGENTS.md`](AGENTS.md) |
| Check a finding at full detail | [`reports/week1.md`](reports/week1.md) (dated lab notebook, newest first), [`reports/results_overview.md`](reports/results_overview.md) (figures) |

## Quick start: lscli

Linux with systemd, one NVIDIA GPU, Python ≥ 3.11. No root needed.

```bash
curl -fsSL https://raw.githubusercontent.com/Rayner3103/CS464/master/install/install.sh | bash
cd ~ && lscli enable-env "$(command -v python3)"     # the Python env your programs use (not a CS464 checkout)
LSCLI_PATCH=1 python train.py                       # or: import lscli; lscli.enable()
lscli status                                        # lscli doctor if nothing happens
```

Searches run while the machine is idle (by default 01:00-07:00). Options (no questions, no background training,
pinned ref, library-only use), a short walkthrough with [`examples/lscli_demo.py`](examples/lscli_demo.py), and
uninstalling (`lscli uninstall [--purge]`) are in the [user guide](docs/user_guide.md).

## Quick start: development

```bash
git clone https://github.com/Rayner3103/CS464.git && cd CS464  # master holds the newest code
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
python -m tune env && python -m pytest -q             # GPU idle; most tests need it
```

Before measuring anything, read the standard conditions (SM clock locked at 1755 MHz) and the GPU job rules in the
[developer guide](docs/developer_guide.md#15-standard-conditions-and-gpu-hygiene).

## Where it stands (RTX 3080)

* The Triton family reaches a median 98.8 % of a searched cuBLASLt (B3+), and choosing per shape matters: the best
  single config is a median ~70 % slower than the per-shape best. It does not beat B3+ on kernels ≥ 50 µs.
* The shipped selector (rank v2 top-3 + cuBLASLt top-16 + learned cuBLASLt picks) lands a median 0.5 % from the best
  candidate in its pool on 86 held-out shapes (p95 5.5 %), and 0.6 % on 60 fresh shapes. It ties B3+ at ≥ 50 µs, at
  ~0.17 s first-use cost per shape with the prebuilt kernels.
* It transfers to an RTX 3090 (same architecture). On an RTX 2060, Triton has no tensor-core path and cuBLASLt
  carries the search.

Gates, numbers and caveats: [`results/`](results/README.md) and [developer guide section 5](docs/developer_guide.md#5-results). Open work:
[section 8](docs/developer_guide.md#8-open-items).

## Layout

```
tune/        measurement harness + CLI (python -m tune ...), the shipped selector (online.py), winner cache, router
kernels/     parameterized Triton GEMM, config space / pruning rules, prebuilt kernel bundle
baselines/   B0 roofline, B1 torch.matmul, B2/B3 cuBLASLt heuristic, B3+ cuBLASLt search, B5 Triton tutorial,
             cuBLASLt k-NN proposals and learned ranker
suites/      shape suites (versioned JSON; python -m suites.generate) and the tune / held-out split
analysis/    tables, model fits (learned rankers, router), cross-GPU scoring, figures (CPU only)
lscli/       the selector as a per-user service: client patch, worker, daemon, CLI (docs/lscli/spec.md)
install/     lscli installer, systemd unit, clock helper, the committed zero-shot models and their packer
pipeline/    one-command runs: run.sh (the study), train_best.sh (train a new GPU), zero_shot.sh
remoteGPU/   SLURM wrappers of the pipeline scripts
examples/    lscli_demo.py
tests/       pytest suite (python -m pytest -q)
reports/     write-ups, generated tables, figures/, snapshot/ (small committed copies of data)
data/<hw>/   raw results, screens, models, logs (gitignored, per machine)
```
