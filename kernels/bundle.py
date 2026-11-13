"""Prebuilt Triton kernel bundle: ship the compiled binaries, fall back to compiling on any miss.

Why: the online search (`tune.online`) JIT-compiles its top-k Triton configs on a shape's first use, ~0.13 s each, about
two thirds of the first-use cost (`reports/week1.md`, AGENTS.md section 6). A user cloning this repo should not pay it
for a kernel the rankers can pick. `python -m tune bundle build` compiles every such kernel once and keeps only the
files Triton needs to load it (`<stem>.cubin`, `<stem>.json`) under `kernels/bundle/<cache key>/`. `install()` makes
Triton look there before compiling.

How it plugs in: Triton picks its cache class from `TRITON_CACHE_MANAGER` (`knobs.cache.manager_class`).
`BundleCacheManager` is a `FileCacheManager` whose `get_group` also answers from the bundle. We cannot simply copy a
Triton cache directory: its `__grp__*.json` files store absolute paths and `get_group` drops any path that does not
exist, so a cache built in another directory (or on another machine) misses every entry. The bundle keeps no group
files; the manager rebuilds the group from the two files that exist. The bundle is read only. Everything compiled on a
miss goes to the normal user cache (`TRITON_CACHE_DIR`, default ~/.triton/cache), so a miss costs exactly what it cost
before and never writes into the repo.

Triton's cache key hashes the compiler (version, source files, libtriton), the kernel source, the argument
specialization (dtype, int 1 / divisible by 16, pointer alignment), constexprs, num_warps / num_stages and the target
(backend, compute capability). The key does not depend on the path or the machine, so the bundle is valid exactly when
all of those match.

LIMITATIONS (also in AGENTS.md section 10 and docs/kernel_bundle.md):
  * One Triton build and one GPU architecture. The bundle holds SASS for the architecture it was built on (RTX 3080,
    sm_86; an RTX 3090 is also sm_86). Another Triton version, another compute capability, a CUDA toolkit that changes
    ptxas, or an edit to `kernels/triton_matmul.py` changes the keys: every lookup misses and every kernel compiles on
    first use, as without a bundle. `install()` / `status()` report this; they do not fix it. Rebuild with `tune bundle build`.
  * Only kernels the saved rankers can propose (the union of their pools) plus the hand-picked configs, fp16 only (bf16
    compiles on first use), and only for tensors whose base pointer is 16-byte aligned (torch allocations are; a sliced
    view that is not compiles a new variant). A config outside the pool, such as `triton:<any key>` from the benchmark
    CLI, compiles on first use.
  * Triton also builds a small C launcher per argument signature with gcc on first launch (cached in the user cache). It is
    not bundled: it depends on the Python ABI and the host compiler, and the first launch in a fresh user cache pays it.
  * Bundle hits skip compilation, not validation: the correctness gate still runs on every candidate.
  * What is committed is the packed archive `kernels/bundle.tar.xz` (see `pack`); the unpacked `kernels/bundle/` is ~800 MiB,
    gitignored, and created by the first `install()`. A stale archive (older than the kernel source, Triton or the rankers'
    pools) is not detected beyond the fingerprint: after changing any of them, rebuild AND repack.

This module is import-light on purpose (Triton imports it when the env var is set). The build side is
`kernels/bundle_build.py`.
"""
from __future__ import annotations

import hashlib
import io
import json
import lzma
import os
import shutil
import tarfile
import tempfile
import time
import warnings
from pathlib import Path

from triton.runtime.cache import FileCacheManager

REPO = Path(__file__).resolve().parent.parent
BUNDLE_ENV = "TUNE_KERNEL_BUNDLE_DIR"        # where the bundle lives (default: kernels/bundle in this repo)
MANAGER_ENV = "TRITON_CACHE_MANAGER"          # Triton's own knob
MANAGER_SPEC = "kernels.bundle:BundleCacheManager"
MANIFEST = "manifest.json"
KERNEL_FILES = ("cubin", "json")              # what `CompiledKernel` needs: the metadata json and the binary
KERNEL_SOURCE = REPO / "kernels" / "triton_matmul.py"

_STATS = {"bundle_hits": 0, "local_hits": 0, "misses": 0}
_WARNED: set[str] = set()


def bundle_root() -> Path:
    return Path(os.environ.get(BUNDLE_ENV) or REPO / "kernels" / "bundle")


def archive_for(root: Path | None = None) -> Path:
    """The packed bundle that ships in the repo: `kernels/bundle` <-> `kernels/bundle.tar.xz` (a sibling of the directory)."""
    root = root or bundle_root()
    return root.with_name(root.name + ".tar.xz")


def stats() -> dict:
    """Lookups so far in this process: kernels served from the bundle, from this machine's own cache, and compiled (miss)."""
    return dict(_STATS)


def reset_stats() -> None:
    for k in _STATS:
        _STATS[k] = 0


class BundleCacheManager(FileCacheManager):
    """FileCacheManager that also serves kernels from the read-only bundle. Lookup order: this machine's cache, the
    bundle, then Triton compiles (and stores the result in the machine's cache)."""

    def __init__(self, key, override=False, dump=False):
        super().__init__(key, override=override, dump=dump)
        root = os.environ.get(BUNDLE_ENV)
        self._bundle = os.path.join(root, key) if root and not (override or dump) else None

    def _bundle_group(self, filename: str) -> dict[str, str] | None:
        if self._bundle is None or not filename.endswith(".json"):
            return None
        stem = filename[:-len(".json")]
        files = {f"{stem}.{ext}": os.path.join(self._bundle, f"{stem}.{ext}") for ext in KERNEL_FILES}
        return files if all(os.path.exists(p) for p in files.values()) else None

    def get_group(self, filename: str):
        group = super().get_group(filename)
        if group:                                   # compiled earlier on this machine (a previous miss)
            _STATS["local_hits"] += 1
            return group
        files = self._bundle_group(filename)
        if files:
            _STATS["bundle_hits"] += 1
            return files
        _STATS["misses"] += 1
        return group                                # None / {}: Triton compiles


def kernel_source_sha() -> str:
    return hashlib.sha256(KERNEL_SOURCE.read_bytes()).hexdigest()[:16]


def fingerprint() -> dict:
    """What a bundle is valid for: the Triton build, the target GPU architecture and the kernel source."""
    import triton
    from triton.runtime import driver
    from triton.runtime.cache import triton_key
    try:
        t = driver.active.get_current_target()
        target = {"backend": t.backend, "arch": t.arch, "warp_size": t.warp_size}
    except Exception:                                # no GPU / driver: a bundle cannot match
        target = None
    return {"triton": triton.__version__, "triton_key_sha": hashlib.sha256(triton_key().encode()).hexdigest()[:16],
            "target": target, "kernel_source_sha": kernel_source_sha()}


def read_manifest(root: Path | None = None) -> dict | None:
    p = (root or bundle_root()) / MANIFEST
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def mismatches(built: dict, now: dict) -> list[str]:
    """Human-readable reasons a bundle built for `built` will miss on `now` (empty = it matches)."""
    out = []
    if built.get("triton") != now.get("triton"):
        out.append(f"Triton {built.get('triton')} (bundle) vs {now.get('triton')}")
    elif built.get("triton_key_sha") != now.get("triton_key_sha"):
        out.append("Triton compiler files differ from the build the bundle was made with")
    if built.get("target") != now.get("target"):
        out.append(f"GPU target {built.get('target')} (bundle) vs {now.get('target')}")
    if built.get("kernel_source_sha") != now.get("kernel_source_sha"):
        out.append("kernels/triton_matmul.py changed since the bundle was built")
    return out


def status() -> dict:
    """The installed bundle's validity on this machine and this process's lookup counters."""
    root = bundle_root()
    m = read_manifest(root)
    if m is None:
        return {"present": False, "dir": str(root), "stats": stats()}
    bad = mismatches(m["fingerprint"], fingerprint())
    return {"present": True, "dir": str(root), "matches": not bad, "mismatch": bad, "entries": m.get("n_entries"),
            "configs": m.get("n_configs"), "dtypes": m.get("dtypes"), "built": m.get("built_utc"),
            "stats": stats()}


# ---------------------------------------------------------------------------------------------- pack / unpack
# 21k kernels are 800 MiB as files (cubin + json per kernel) but the variants of one config are near-identical, so a solid
# xz archive with a large window is ~50 MiB: small enough to commit, and standard library only (no new dependency). The
# manifest is the first member so `read_archive_manifest` can decide whether the archive is worth unpacking without
# decompressing the rest. Members are ordered by cubin size so similar kernels sit close together for the compressor.
PACK_DICT_BYTES = 128 * 2**20


def pack(bundle_dir: Path | None = None, archive: Path | None = None, log=print) -> Path:
    root = Path(bundle_dir or bundle_root())
    archive = Path(archive or archive_for(root))
    if read_manifest(root) is None:
        raise FileNotFoundError(f"no bundle at {root} (run `python -m tune bundle build`)")
    entries = sorted((d for d in root.iterdir() if d.is_dir()),
                     key=lambda d: (sum(f.stat().st_size for f in d.glob("*.cubin")), d.name))
    filters = [{"id": lzma.FILTER_LZMA2, "preset": 9, "dict_size": PACK_DICT_BYTES}]
    tmp = archive.with_name(archive.name + ".tmp")
    t0 = time.perf_counter()
    with lzma.LZMAFile(tmp, "wb", format=lzma.FORMAT_XZ, filters=filters) as xz, tarfile.open(fileobj=xz, mode="w|") as tf:
        tf.add(root / MANIFEST, arcname=MANIFEST)
        for d in entries:
            for f in sorted(d.iterdir()):
                tf.add(f, arcname=f"{d.name}/{f.name}")
    os.replace(tmp, archive)
    log(f"packed {len(entries)} kernels -> {archive} ({archive.stat().st_size / 2**20:.1f} MiB, {time.perf_counter() - t0:.0f}s)")
    return archive


def read_archive_manifest(archive: Path) -> dict | None:
    """The manifest of a packed bundle without decompressing the rest (it is the first member)."""
    try:
        with tarfile.open(archive, "r|xz") as tf:
            m = tf.next()
            if m is None or m.name != MANIFEST:
                return None
            return json.loads(tf.extractfile(m).read())
    except (OSError, ValueError, tarfile.TarError, lzma.LZMAError):
        return None


def unpack(archive: Path, dest: Path | None = None, log=print) -> Path:
    """Extract a packed bundle into `dest` (default: the bundle directory). Atomic: it is extracted next to `dest` and renamed,
    so an interrupted extraction never leaves a half bundle that `install` would trust."""
    dest = Path(dest or bundle_root())
    tmp = Path(tempfile.mkdtemp(prefix=dest.name + ".unpack-", dir=dest.parent))
    t0 = time.perf_counter()
    try:
        with lzma.LZMAFile(archive, "rb") as xz:
            with tarfile.open(fileobj=xz, mode="r|") as tf:
                tf.extractall(tmp, filter="data")
            while xz.read(1 << 20):                 # tar stops at its own end marker: read on to the xz trailer, which raises
                pass                                # EOFError if the archive was cut short
        m = read_manifest(tmp)
        n = len(list(tmp.glob("*/*.cubin")))
        if m is None or n != m.get("n_entries"):
            raise ValueError(f"{archive.name} is incomplete: {n} kernels extracted, manifest says {m and m.get('n_entries')}")
        if dest.exists():
            shutil.rmtree(dest)
        os.replace(tmp, dest)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    log(f"unpacked {archive.name} -> {dest} ({time.perf_counter() - t0:.0f}s)")
    return dest


def _unpack_if_useful(root: Path, warn: bool) -> str | None:
    """No bundle directory but a packed one next to it: unpack it, unless its manifest says it cannot match this machine (then
    it would only cost 800 MiB of disk and a few seconds). Returns why nothing was unpacked, or None when it was."""
    archive = archive_for(root)
    if not archive.exists():
        return f"no bundle at {root}"
    m = read_archive_manifest(archive)
    if m is None:
        return f"{archive.name} is unreadable"
    bad = mismatches(m["fingerprint"], fingerprint())
    if bad:
        if warn and "mismatch" not in _WARNED:
            _WARNED.add("mismatch")
            warnings.warn(f"{archive.name} was built for a different environment, so it is not unpacked and every Triton kernel "
                          "will compile on first use: " + "; ".join(bad) + ". Rebuild it with `python -m tune bundle build`.",
                          stacklevel=3)
        return "archive does not match this machine: " + "; ".join(bad)
    unpack(archive, root, log=lambda *_: None)
    return None


def install(bundle_dir: str | os.PathLike | None = None, *, warn: bool = True) -> dict:
    """Point Triton at the bundle (idempotent). Call once before the first Triton compile in the process; spawned
    workers inherit it through the environment. If only the packed `kernels/bundle.tar.xz` exists (a fresh clone), it is
    unpacked first (~10-30 s, once), unless its manifest does not match this machine. Never raises for a missing or
    mismatched bundle: no bundle, or a different cache manager already configured, leaves Triton exactly as it was. A bundle that does not match this machine is installed anyway (every
    lookup then misses and Triton compiles, the documented fallback) and warned about once. Returns `status()` plus
    `installed`."""
    if bundle_dir is not None:
        os.environ[BUNDLE_ENV] = str(bundle_dir)
    root = bundle_root()
    if read_manifest(root) is None:
        why = _unpack_if_useful(root, warn)
        if why is not None or read_manifest(root) is None:
            return {"installed": False, "reason": why or f"no bundle at {root}", **status()}
    cur = os.environ.get(MANAGER_ENV)
    if cur not in (None, MANAGER_SPEC):
        return {"installed": False, "reason": f"{MANAGER_ENV}={cur} is already set; not overriding it", **status()}
    os.environ[BUNDLE_ENV] = str(root)
    os.environ[MANAGER_ENV] = MANAGER_SPEC
    st = status()
    if warn and st.get("mismatch") and "mismatch" not in _WARNED:
        _WARNED.add("mismatch")
        warnings.warn("kernel bundle does not match this machine, so every Triton kernel will compile on first use: "
                      + "; ".join(st["mismatch"]) + ". Rebuild it with `python -m tune bundle build`.", stacklevel=2)
    return {"installed": True, **st}


def uninstall() -> None:
    """Stop using the bundle in this process (and its children)."""
    if os.environ.get(MANAGER_ENV) == MANAGER_SPEC:
        del os.environ[MANAGER_ENV]
    os.environ.pop(BUNDLE_ENV, None)
