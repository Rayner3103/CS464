"""Build and verify the prebuilt kernel bundle (runtime side: `kernels/bundle.py`; read its docstring for the limitations).

What gets built: every kernel the selector can launch. Triton compiles one binary per (config, argument specialization),
so "all kernels" is the saved rankers' pool (union of their `pool` lists, plus the hand-picked configs) times every
specialization a shape can produce. For `_matmul_kernel` the specialization is, per integer argument, "equal to 1",
"divisible by 16" or "other" (Triton's rule; pointers are assumed 16-byte aligned, as torch allocations are):

  * M, N, K themselves; stride_am = K, stride_bk = N and stride_cm = N share their class;
  * stride_cs = 0 for SPLIT_K == 1, and M * N for SPLIT_K > 1, whose class is NOT a function of M's and N's classes
    (2 * 8 = 16 is divisible, 3 * 5 is not), so it is enumerated from sample values;
  * the EVEN_K constexpr, K % (BLOCK_K * SPLIT_K) == 0, which only exists inside the "K divisible by 16" class.

A variant is built only if some shape in [1, 16384]^3 can reach it past the config's shape pruning rules
(`kernels.config_space.shape_violations`, vectorised below and tested equal), so a config that is pruned for every shape
of a class is not compiled for it. The pruning rules factor: BLOCK_M / GROUP_M look at M, BLOCK_N at N, BLOCK_K and the
split-K starvation rule at K, and only the split-K workspace bound couples M and N. So (M, N) variants and K variants
are enumerated separately and combined, which keeps the enumeration a few hundred thousand array elements per config.

`SPLIT_K > 1` also launches `_splitk_reduce_kernel`: specialized on M * N (1 / divisible / other), SPLIT_K, BLOCK, dtype.

Build flow (`python -m tune bundle build`, GPU idle, ~1 GiB host RAM and ~360 MiB VRAM per worker):
  1. enumerate tasks; 2. compile them in spawn workers into a persistent scratch cache (`data/<hw>/bundle_build_cache`,
  so an interrupted build resumes: Triton hits its own cache); 3. copy only `<stem>.cubin` + `<stem>.json` of each entry
  into the bundle (the IR dumps, `.source` and group files are 90% of the bytes and are not needed to load a kernel);
  4. write `manifest.json` with the fingerprint the bundle is valid for.

`verify` is the gate: with an empty user cache and the bundle installed it warms every (pool config, shape) pair of a
shape corpus and requires zero compiles, then runs a few kernels for real against torch.matmul.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from kernels import bundle as B
from kernels.config_space import HANDPICKED, shape_violations
from kernels.triton_matmul import MatmulConfig

DIM_MAX = 16384                  # plan section 2: M, N, K in [1, 16384]
MAX_WORKSPACE_BYTES = 512 * 2**20  # shape_violations default
CHUNK_TASKS = 200                # kernels per worker process (a worker's RSS grows with what it compiled; see tune/sweep.py)
REDUCE_BLOCK = 1024              # kernels/triton_matmul.py RBLOCK
STEMS = ("_matmul_kernel", "_splitk_reduce_kernel")
DTYPES = ("fp16", "bf16")


def cls(x) -> int:
    """Triton's integer-argument specialization: 0 = equal to 1, 1 = divisible by 16, 2 = other."""
    return 0 if x == 1 else (1 if x % 16 == 0 else 2)


def pool_configs(stems: list[Path], *, handpicked: bool = True) -> list[MatmulConfig]:
    """Union of the rankers' candidate pools (and the hand-picked configs), sorted by key."""
    from tune.sweep import parse_key
    keys: set[str] = set()
    for s in stems:
        keys |= set(json.loads(Path(s).with_suffix(".json").read_text())["pool"])
    cfgs = {parse_key(k).key(): parse_key(k) for k in keys}
    if handpicked:
        cfgs.update({c.key(): c for c in HANDPICKED.values()})
    return [cfgs[k] for k in sorted(cfgs)]


# ----------------------------------------------------------------------------------------------- enumeration
def _cover(x: int) -> int:
    return max(16, 1 << (x - 1).bit_length())


def sample_values() -> np.ndarray:
    """Dimension values that hit every specialization class at every power-of-two magnitude (where the pruning rules
    switch): 1, small odd multiples of each power of two, 2^p +- 1, and 2^v * (2^q +- 1) for valuations v = 1..4."""
    vals = {1, DIM_MAX}
    for p in range(0, 15):
        base = 1 << p
        for mult in range(1, 32, 2):
            vals.add(base * mult)
        vals.update((base - 1, base + 1, 3 * base // 2 - 1, 3 * base // 2 + 1))
    for v in range(1, 5):
        for q in range(1, 15):
            vals.update((2**v * (2**q - 1), 2**v * (2**q + 1)))
    return np.array(sorted(x for x in vals if 1 <= x <= DIM_MAX), dtype=np.int64)


_VALS = None


def _values() -> np.ndarray:
    global _VALS
    if _VALS is None:
        _VALS = sample_values()
    return _VALS


def _classes(a: np.ndarray) -> np.ndarray:
    return np.where(a == 1, 0, np.where(a % 16 == 0, 1, 2))


def mn_variants(cfg: MatmulConfig) -> dict[tuple, tuple[int, int]]:
    """(class of M, class of N, class of stride_cs) -> one representative (M, N) a shape can have for this config.
    Vectorised `shape_violations` for the M, N rules (BLOCK_M / BLOCK_N oversized, GROUP_M no-op, split-K workspace)."""
    v = _values()
    cov = np.array([_cover(int(x)) for x in v])
    okm = (cfg.BLOCK_M <= 2 * cov) & ((cfg.GROUP_M == 1) | (v > cfg.BLOCK_M))   # ceil(M / bm) > 1  <=>  M > bm
    okn = cfg.BLOCK_N <= 2 * cov
    m, n = v[okm], v[okn]
    if not len(m) or not len(n):
        return {}
    mm, nn = np.meshgrid(m, n, indexing="ij")
    ok = np.ones(mm.shape, bool)
    mn = mm * nn
    if cfg.SPLIT_K > 1:
        ws = cfg.SPLIT_K * mn
        ok &= (ws * 4 <= MAX_WORKSPACE_BYTES) & (ws < 2**31)
        ccs = _classes(mn)
    else:
        ccs = np.zeros(mm.shape, np.int64)
    code = (_classes(mm) * 3 + _classes(nn)) * 3 + ccs
    out: dict[tuple, tuple[int, int]] = {}
    for c in np.unique(code[ok]):
        i, j = np.argwhere(ok & (code == c))[0]
        out[(int(c) // 9, int(c) // 3 % 3, int(c) % 3)] = (int(mm[i, j]), int(nn[i, j]))
    return out


def k_variants(cfg: MatmulConfig) -> dict[tuple, int]:
    """(class of K, EVEN_K) -> one representative K. Vectorised BLOCK_K oversized and split-K starved rules."""
    bks = cfg.BLOCK_K * cfg.SPLIT_K
    v = np.unique(np.concatenate([_values(), [bks * j for j in (1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 64, 128, 256)]]))
    v = v[v <= DIM_MAX]
    cov = np.array([_cover(int(x)) for x in v])
    ok = cfg.BLOCK_K <= 2 * cov
    if cfg.SPLIT_K > 1:
        ok &= v >= bks
    v = v[ok]
    code = _classes(v) * 2 + (v % bks == 0)
    return {(int(c) // 2, bool(c % 2)): int(v[np.argmax(code == c)]) for c in np.unique(code)}


def matmul_tasks(cfgs: list[MatmulConfig]) -> list[tuple]:
    """(config dict, M, N, K) for every reachable specialization of `_matmul_kernel`."""
    tasks = []
    for cfg in cfgs:
        ks = list(k_variants(cfg).values())
        for m, n in mn_variants(cfg).values():
            for k in ks:
                tasks.append((cfg.to_dict(), m, n, k))
    return tasks


def reduce_tasks(cfgs: list[MatmulConfig]) -> list[tuple]:
    """("reduce", SPLIT_K, M * N) for `_splitk_reduce_kernel`: one per split factor and class of M * N."""
    return [("reduce", sk, mn) for sk in sorted({c.SPLIT_K for c in cfgs if c.SPLIT_K > 1}) for mn in (1, 256, 17)]


# ----------------------------------------------------------------------------------------------- compile
def _build_worker(args):
    """Compile (no launch) a chunk of kernels in a spawned process. Mirrors `tune.sweep._warm_worker`'s call exactly."""
    tasks, dtype = args
    os.sched_setaffinity(0, range(os.cpu_count()))
    import torch
    from kernels.triton_matmul import _matmul_kernel, _splitk_reduce_kernel, even_k
    td = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    t = torch.empty(16, 16, device="cuda", dtype=td)
    ws = torch.empty(16, 16, device="cuda", dtype=torch.float32)  # split-K writes fp32 partials
    errs = []
    for task in tasks:
        try:
            if task[0] == "reduce":
                _, sk, mn = task
                _splitk_reduce_kernel.warmup(ws, t, mn, SPLIT_K=sk, BLOCK=REDUCE_BLOCK, num_warps=4, grid=(1,))
                continue
            cd, M, N, K = task
            cfg = MatmulConfig(**cd)
            s_cs, s_cm = (M * N, N) if cfg.SPLIT_K > 1 else (0, N)
            _matmul_kernel.warmup(t, t, ws if cfg.SPLIT_K > 1 else t, M, N, K, K, 1, N, 1, s_cs, s_cm, 1,
                                  BLOCK_M=cfg.BLOCK_M, BLOCK_N=cfg.BLOCK_N, BLOCK_K=cfg.BLOCK_K, GROUP_M=cfg.GROUP_M,
                                  SPLIT_K=cfg.SPLIT_K, EVEN_K=even_k(K, cfg), num_warps=cfg.num_warps,
                                  num_stages=cfg.num_stages, grid=(1,))
        except Exception as e:                                   # a config that does not compile is recorded, not fatal
            errs.append((task, f"{type(e).__name__}: {str(e)[:150]}"))
    return errs


def compile_tasks(tasks: list[tuple], dtype: str, workers: int, log=print) -> list:
    """Compile `tasks` in a spawn pool of fresh-per-chunk processes. If a worker dies (host OOM kills the pool), halve the
    pool and redo only the unfinished chunks; returns the compile errors."""
    import multiprocessing as mp
    from concurrent.futures import BrokenExecutor, ProcessPoolExecutor, as_completed
    n_chunks = max(workers * 4, -(-len(tasks) // CHUNK_TASKS))
    pending = {i: tasks[i::n_chunks] for i in range(n_chunks)}
    errors, t0, done_n = [], time.perf_counter(), 0
    while pending:
        try:
            with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), max_tasks_per_child=1) as ex:
                futs = {ex.submit(_build_worker, (c, dtype)): i for i, c in pending.items()}
                for f in as_completed(futs):
                    try:
                        errors += f.result()
                    except BrokenExecutor:
                        break
                    except Exception as e:
                        log(f"  chunk {futs[f]} raised {type(e).__name__}: {str(e)[:120]}")
                        continue
                    done_n += len(pending.pop(futs[f]))
                    el = time.perf_counter() - t0
                    log(f"  compiled {done_n}/{len(tasks)} | {el:.0f}s elapsed, ETA {el / done_n * (len(tasks) - done_n):.0f}s")
        except BrokenExecutor:
            pass
        if not pending:
            break
        if workers == 1:
            raise RuntimeError(f"compile workers keep dying; {len(pending)} chunks unfinished (rerun to resume)")
        workers = max(1, workers // 2)
        log(f"  a compile worker died (out of memory?); retrying {len(pending)} chunks with {workers} workers")
    return errors


# ----------------------------------------------------------------------------------------------- collect / manifest
def _normalise_json(text: str) -> str:
    """Triton's metadata records the builder's absolute path to libdevice (`extern_libs`), used only at compile time. Replace
    the site-packages prefix so the bundle carries no builder-specific path (checked by `verify`'s real launches)."""
    import triton
    return text.replace(str(Path(triton.__file__).resolve().parent), "<triton>")


def collect(cache_dir: Path, bundle_dir: Path, log=print) -> dict:
    """Copy the loadable files of every kernel in a Triton cache dir into the bundle. Returns counts per kernel."""
    counts = {s: 0 for s in STEMS}
    nbytes = 0
    for d in sorted(p for p in cache_dir.iterdir() if p.is_dir()):
        for stem in STEMS:
            src_bin, src_json = d / f"{stem}.cubin", d / f"{stem}.json"
            if not (src_bin.exists() and src_json.exists()):
                continue
            dst = bundle_dir / d.name
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_bin, dst / src_bin.name)
            (dst / src_json.name).write_text(_normalise_json(src_json.read_text()))
            counts[stem] += 1
            nbytes += src_bin.stat().st_size + (dst / src_json.name).stat().st_size
    log(f"  collected {counts} ({nbytes / 2**20:.0f} MiB)")
    return {"counts": counts, "bytes": nbytes}


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"], cwd=B.REPO, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def build(stems: list[Path], bundle_dir: Path, build_cache: Path, *, dtypes=("fp16",), workers: int = 0, limit: int = 0,
          log=print) -> dict:
    """Enumerate, compile and collect. `limit` > 0 builds only the first N configs (a smoke test; the manifest says so)."""
    from tune import sweep
    B.uninstall()                                   # compile for real: a hit would leave nothing in the scratch cache
    os.environ.pop("TRITON_CACHE_DIR", None)
    build_cache.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(build_cache)
    cfgs = pool_configs(stems)
    if limit:
        cfgs = cfgs[:limit]
    t0 = time.perf_counter()
    mm, rd = matmul_tasks(cfgs), reduce_tasks(cfgs)
    log(f"{len(cfgs)} configs -> {len(mm)} _matmul_kernel + {len(rd)} _splitk_reduce_kernel specializations per dtype "
        f"(enumerated in {time.perf_counter() - t0:.1f}s)")
    w, why = sweep.workers_for(workers, os.cpu_count() or 1, sweep.host_available_gib(), sweep.gpu_free_mib())
    log(f"compile workers: {w}" + (f" ({why})" if why else ""))
    errors = []
    for dt in dtypes:
        log(f"compiling {dt}")
        errors += compile_tasks(mm + rd, dt, w, log)
    if errors:
        log(f"{len(errors)} kernels failed to compile (not bundled): {errors[:3]}")
    got = collect(build_cache, bundle_dir, log)
    sample = next(iter(bundle_dir.glob("*/_matmul_kernel.json")), None)
    manifest = {"schema": 1, "fingerprint": B.fingerprint(), "dtypes": list(dtypes), "n_configs": len(cfgs),
                "n_entries": sum(got["counts"].values()), "counts": got["counts"], "bytes": got["bytes"],
                "pools": [Path(s).name for s in stems], "partial_limit": limit or None, "compile_errors": len(errors),
                "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "harness_commit": _git_commit(),
                "sample_entry": sample.parent.name if sample else None}
    (bundle_dir / B.MANIFEST).write_text(json.dumps(manifest, indent=1))
    log(f"bundle: {manifest['n_entries']} kernels, {got['bytes'] / 2**20:.0f} MiB -> {bundle_dir}")
    return manifest


# ----------------------------------------------------------------------------------------------- verify
def verify_bundle(bundle_dir: Path, shapes: list[dict], cfgs: list[MatmulConfig], *, per_shape: int = 0, seed: int = 464,
                  dtype: str = "fp16", smoke: int = 6, log=print) -> dict:
    """The coverage gate. With the bundle installed and an EMPTY user cache, warm (no launch) every runnable (config, shape)
    pair, `per_shape` random configs per shape (0 = all), and count the lookups Triton had to compile. Then run `smoke` kernels
    for real, from the bundle, against torch.matmul. Returns a result dict; "ok" needs 0 misses, 0 failures."""
    import tempfile
    import torch
    from kernels.triton_matmul import TritonMatmul, _matmul_kernel, _splitk_reduce_kernel, even_k
    B.uninstall()
    scratch = tempfile.mkdtemp(prefix="bundle_verify_cache_")
    os.environ["TRITON_CACHE_DIR"] = scratch
    inst = B.install(bundle_dir, warn=False)
    if not inst.get("installed"):
        return {"ok": False, "reason": inst.get("reason")}
    B.reset_stats()
    td = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    t = torch.empty(16, 16, device="cuda", dtype=td)
    ws = torch.empty(16, 16, device="cuda", dtype=torch.float32)
    rng = np.random.default_rng(seed)
    missed, pairs, t0 = [], 0, time.perf_counter()
    for si, s in enumerate(shapes):
        M, N, K = s["M"], s["N"], s["K"]
        ok = [c for c in cfgs if not shape_violations(c, M, N, K)]
        if per_shape and len(ok) > per_shape:
            ok = [ok[i] for i in rng.choice(len(ok), per_shape, replace=False)]
        for cfg in ok:
            before = B.stats()["misses"]
            s_cs, s_cm = (M * N, N) if cfg.SPLIT_K > 1 else (0, N)
            _matmul_kernel.warmup(t, t, ws if cfg.SPLIT_K > 1 else t, M, N, K, K, 1, N, 1, s_cs, s_cm, 1,
                                  BLOCK_M=cfg.BLOCK_M, BLOCK_N=cfg.BLOCK_N, BLOCK_K=cfg.BLOCK_K, GROUP_M=cfg.GROUP_M,
                                  SPLIT_K=cfg.SPLIT_K, EVEN_K=even_k(K, cfg), num_warps=cfg.num_warps,
                                  num_stages=cfg.num_stages, grid=(1,))
            if cfg.SPLIT_K > 1:
                _splitk_reduce_kernel.warmup(ws, t, M * N, SPLIT_K=cfg.SPLIT_K, BLOCK=REDUCE_BLOCK, num_warps=4, grid=(1,))
            pairs += 1
            if B.stats()["misses"] > before:
                missed.append({"shape": [M, N, K], "config": cfg.key()})
        if (si + 1) % 50 == 0:
            log(f"  {si + 1}/{len(shapes)} shapes, {pairs} pairs, {len(missed)} misses, {time.perf_counter() - t0:.0f}s")
    cover = B.stats()
    # Real launches from the bundle (this process compiled nothing for them if the lookups above were all hits).
    bad = []
    for i in range(smoke):
        s = shapes[int(rng.integers(len(shapes)))]
        M, N, K = s["M"], s["N"], s["K"]
        ok = [c for c in cfgs if not shape_violations(c, M, N, K)]
        if not ok:
            continue
        cfg = ok[int(rng.integers(len(ok)))]
        a = torch.randn(M, K, device="cuda", dtype=td)
        b = torch.randn(K, N, device="cuda", dtype=td)
        out = TritonMatmul(cfg).make_runner(a, b)()
        ref = (a.float() @ b.float())
        err = (out.float() - ref).abs().max().item()
        if not torch.isfinite(out.float()).all() or err > 1e-2 * ref.abs().max().item() + 0.05 * K ** 0.5:
            bad.append({"shape": [M, N, K], "config": cfg.key(), "max_abs_err": err})
    shutil.rmtree(scratch, ignore_errors=True)
    res = {"pairs": pairs, "shapes": len(shapes), "stats": cover, "misses": len(missed), "missed": missed[:20],
           "smoke_run": smoke, "smoke_bad": bad, "seconds": round(time.perf_counter() - t0, 1)}
    res["ok"] = not missed and not bad and pairs > 0
    return res
