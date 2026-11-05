# Prebuilt kernel bundle and winner cache

Two things make the shipped selector cheap on a machine that has never seen a shape:

1. **The kernel bundle** (`kernels/bundle.py`, `kernels/bundle_build.py`): every Triton kernel the saved rankers can pick,
   already compiled, so the online search loads it instead of JIT-compiling it.
2. **The winner cache** (`tune/winner_cache.py`): the kernel the search picked for a shape, so a shape is searched once per machine.

Both are on by default in `tune.online.select_kernel(ranker, M, N, K, ...)` and off everywhere that measures the search itself
(`tune online`, `analysis.zero_shot`, the benchmark commands).

## LIMITATIONS (read first)

The bundle is a **speed-up with a fallback, not a guarantee**: any kernel it does not hold compiles on first use, exactly as
before it existed, and a result is never wrong because of a miss. But it holds only this:

| It is valid for | What happens otherwise |
|---|---|
| **Triton 3.7.1** (the compiler files are hashed into every key) | every lookup misses; everything compiles on first use |
| **one GPU architecture: sm_86** (RTX 3080, RTX 3090). A cubin is SASS for one architecture; the `ptxas` version is also in the key | same (the RTX 2060 is sm_75: no hit) |
| **the current `kernels/triton_matmul.py`** (kernel source is in the key) | same; rebuild after any edit |
| **the kernels the v1 and v2 rankers can propose** (union of their pools: 1,174 configs) **+ the 5 hand-picked configs** | a config outside the pool (`triton:<any key>`) or a retrained ranker's new picks compile on first use; rebuild after retraining |
| **fp16** | bf16 compiles on first use (`bundle build --dtypes fp16,bf16` adds it) |
| **16-byte-aligned base pointers and the contiguous row-major strides `TritonMatmul` passes** (torch allocations qualify) | a sliced or transposed view compiles a new variant |

Also not covered:

- Triton builds a small **C launcher** per argument signature with gcc on first launch, cached in the *user* cache. It depends on the
  Python ABI and the host compiler, so it is not bundled; the first launches in a fresh user cache pay it.
- A bundle hit skips **compilation, not validation**: the correctness gate still runs on every candidate.
- The checks are a fingerprint (Triton version + compiler-file hash, GPU target, kernel-source hash), recorded in `manifest.json`.
  `install()` warns once when it differs and still installs (a miss is only a compile); `python -m tune bundle status` shows the reason.
  A stale bundle that still matches the fingerprint (e.g. a ranker retrained with a new pool) is **not** detected: rebuild and repack.

The winner cache is per machine and per environment (GPU, Triton, cuBLASLt version, driver), never expires by itself, and stores the
fastest of one short online timing (10 cold samples, one placement). See the module docstring for the rest.

## What was built and measured (RTX 3080, Triton 3.7.1, 2026-11-04)

- **Enumeration.** 1,176 configs (the 1,174-config v2 pool contains v1's 868; plus the 2 hand-picked ones it lacks) x every reachable
  argument specialization = **21,175 `_matmul_kernel` + 12 `_splitk_reduce_kernel`** binaries. Triton specializes each integer argument as
  "equal to 1", "divisible by 16" or "other": M, N, K (stride_am, stride_bk, stride_cm share their classes), `stride_cs = M*N` for split-K
  (its class is *not* a function of M's and N's classes), and EVEN_K. A variant is built only if some shape in [1, 16384]^3 reaches it
  past the config's shape pruning rules. Checked on 202,525 random reachable shapes: none outside the enumerated variants.
- **Build.** `python -m tune bundle build`: 688 s of compiling on 10 workers (~1 GiB host RAM and ~360 MiB VRAM each), 0 compile errors,
  21,187 kernels, **799 MiB** of cubin + json (the IR dumps and group files, ~5 GiB in the scratch cache, are not kept).
- **Coverage gate.** `python -m tune bundle verify --per-shape 0`: all 777 suite + edge shapes x every runnable pool config = **456,550
  pairs, 14,229 distinct kernels loaded from the bundle, 0 compiles**; 20 kernels launched for real and checked against torch.matmul.
  (The other ~7k kernels are specializations no suite shape produces.)
- **First-use cost.** `select_kernel` on 10 held-out shapes, rank v2, k = 3 + cuBLASLt top-16, empty user cache, one process each:

  | | wall per shape | Triton compile + validate |
  |---|---|---|
  | without the bundle | 0.618 s | 0.480 s |
  | with the bundle | **0.150 s** | 0.005 s |

  32 bundle hits, 0 misses. 9 of 10 shapes picked the same kernel; I did not look into the tenth (selection is by a short timing,
  so near-ties can flip between any two runs). What remains per shape is ranking (0.036 s here), the short timings (~0.1 s) and the
  correctness gate. The earlier `rank_s` figures (0.095-0.133 s) come from `tune online` runs, which pin the process to one core; this
  comparison script did not pin, so do not mix the two. The first call in a process also pays one-time CUDA, cuBLASLt and launcher setup,
  which the comparison excluded with a warm-up call (a fresh-process first selection took 0.62 s with the bundle already installed).
- **Fallback.** A pool-external config compiled on first use (1 miss, 0.26 s), reused the shared split-K reduce kernel from the bundle, and
  matched torch.matmul.
- **Shipping size.** `kernels/bundle.tar.xz` is **19.2 MiB** (standard-library xz, 128 MiB window, kernels ordered by size so
  near-identical variants of a config sit together; zstd -19 on the same tree is 25 MiB, plain `tar | xz` 51 MiB). The first
  `install()` on a fresh clone unpacks it in ~11 s to `kernels/bundle/` (986 MiB on disk, gitignored); the unpacked tree was byte-identical
  to the one that was packed. Unpacking is atomic and checks the kernel count against the manifest, and an archive whose manifest does not
  match the machine is not unpacked at all.

## Commands

```bash
python -m tune bundle build                  # GPU idle. compile every kernel the pools can launch (resumable), write kernels/bundle/
python -m tune bundle verify --per-shape 0   # the gate: 0 compiles over every suite shape x runnable pool config
python -m tune bundle pack                   # kernels/bundle -> kernels/bundle.tar.xz (the file to commit, 19 MiB)
python -m tune bundle status                 # does the bundle match this Triton / GPU? lookup counters
python -m tune bundle unpack                 # what install() does on a fresh clone
```

Build options: `--pool <stems>` (default `data/<hw>/lgbm_rank` and `lgbm_rank_v2`), `--dtypes fp16[,bf16]`, `--workers N` (0 = auto, capped by
free RAM and VRAM), `--limit N` (smoke test; recorded in the manifest), `--build-cache DIR`. An interrupted build resumes from the scratch cache.

```python
from tune.online import select_kernel
from tune.winner_cache import backend_for

sel = select_kernel(ranker, M, N, K, calls=expected_calls)   # bundle + winner cache on
run = backend_for(sel["choice"], M, N, K).make_runner(a, b)  # a runnable backend for the pick
c = run()
```

`select_kernel(..., cache=None, use_bundle=False)` searches from scratch and compiles for real (what the experiments need).
`TUNE_WINNER_CACHE=<path>` moves the winner file (default `data/<hw>/winner_cache.jsonl`), `=off` disables it;
`TUNE_KERNEL_BUNDLE_DIR` moves the bundle.

## How the bundle plugs into Triton

- Triton's cache is `<TRITON_CACHE_DIR>/<key>/` holding the cubin, the metadata json, IR dumps and a `__grp__*.json` that lists those
  files by **absolute path**; `get_group` drops any path that does not exist. A copied cache therefore misses every entry on another
  directory or machine (the 3090 caches in `data/` still carry `/common/home/...` paths). The key itself is path independent.
- `BundleCacheManager` (selected through Triton's `TRITON_CACHE_MANAGER`) answers `get_group` from this machine's cache, then from
  `kernels/bundle/<key>/` by rebuilding the group from the two files that exist (`<stem>.json`, `<stem>.cubin`), and otherwise returns
  nothing, so Triton compiles and stores the result in the user cache. The bundle is never written to.
- The metadata json records the builder's path to `libdevice` (compile-time only); the build replaces the Triton install prefix with `<triton>`
  so no builder-specific path is shipped.

## When to rebuild

After a Triton upgrade, a change to `kernels/triton_matmul.py`, retraining a ranker (new pool), or moving to a new architecture:
`bundle build && bundle verify --per-shape 0 && bundle pack`. Every committed archive adds ~19 MiB to the git history; if it will be rebuilt
often, track it with Git LFS instead.
