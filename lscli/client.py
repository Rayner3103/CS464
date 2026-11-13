"""The lscli client (spec section 5): an opt-in patch of torch's NN matmul that serves the winner cache.

`enable()` replaces `torch.matmul`, `torch.mm` and `Tensor.matmul` / `Tensor.mm` / `Tensor.__matmul__` with a
dispatcher, and wraps `torch.nn.functional.linear` to count its (NT) shapes. `disable()` restores the originals (and a
wrapper captured earlier, e.g. `from torch import matmul`, passes straight through once disabled). Anything not in
scope goes to the original function untouched, with the caller's exact arguments.

Per call:
  1. route: only two positional 2-D CUDA operands of the same fp16 / bf16 dtype on the daemon's GPU, which must also be
     the current device; contiguous, 16-byte aligned, M, N, K <= 16384; plain tensors or nn.Parameter; not under
     torch.compile tracing, CUDA-graph capture, autocast, torch.func transforms or forward-mode AD. Everything in this
     step is inside a try: a failure here means torch, never an exception in the user's program;
  2. count the shape (in memory; flushed to `profile/<pid>-<t0>.jsonl` every 30 s, at exit, and never twice after fork);
  3. decide once per shape (and again after the winner file changed): current entry -> the dispatch rule; soft-stale
     entry -> one correctness check per environment (`mark_verified`; skipped for shapes whose check would allocate more
     than VERIFY_MAX_BYTES in the user's process: torch is used and a re-search queued), dispatch, queue `stale` once;
     miss -> an inline `select_kernel` for shapes the usage record counts as frequent (once per process, needs lightgbm
     and the models in client.json; the ranker is loaded once per process), otherwise torch + queue `miss` once.
     Decisions run outside the global lock with a per-shape guard: another thread meanwhile gets torch;
  4. dispatch rule: the entry's `torch_us` >= `min_kernel_us` and `torch_us - winner_us >= max(min_saving_us, f)`,
     f = 4 % of torch_us for an entry timed at stock clocks, 3 % under a declared clock lock (`saving_frac`).
A failure of a dispatched kernel falls back to torch, disables that shape for the process and warns once. Under
autograd the kernel runs inside an `autograd.Function` whose backward calls the ORIGINAL torch.matmul.

CUDA is not touched by `enable()`: the daemon's GPU (by UUID) is matched to a torch device index at the first CUDA
matmul, when the program has initialized CUDA itself. `lscli.boot` (the LSCLI_PATCH hook) enables only once the
program imports torch, so a script that never does pays nothing.
"""
from __future__ import annotations

import atexit
import os
import sys
import threading
import time
import warnings

import torch

_ORIG: dict = {}
_STATE: "_Client | None" = None
_LOCK = threading.Lock()
_DTYPES = {torch.float16: "fp16", torch.bfloat16: "bf16"}
_PLAIN = (torch.Tensor, torch.nn.Parameter)
MAX_DIM = 16384
FLUSH_S = 30.0
VERIFY_MAX_BYTES = 512 << 20
# Smallest relative saving the dispatch rule trusts. torch_us and winner_us both come from the short online timing
# (10 cold samples, one placement); its kernel-vs-kernel error at >= 50 us is p95 3.5 % at stock clocks and 2.7 %
# under a 1755 MHz lock on the 3080, rounded up (reports/week1.md 2026-11-13, analysis/noise_compare.py). The study's
# 2 % tie margin is for full-protocol medians and is too small for these timings.
SAVING_FRAC_UNLOCKED = 0.04
SAVING_FRAC_LOCKED = 0.03
_UNSET = object()


def _functorch_wrapped(t) -> bool:
    try:
        return torch._C._functorch.is_functorch_wrapped_tensor(t)
    except Exception:
        return False


def _forward_ad_active() -> bool:
    try:
        return torch.autograd.forward_ad._current_level >= 0
    except Exception:
        return False


def _autocast_on() -> bool:
    try:
        return torch.is_autocast_enabled("cuda")
    except TypeError:
        return torch.is_autocast_enabled()


class _Mm(torch.autograd.Function):
    """Forward: the selected kernel. Backward: the original (unpatched) torch.matmul, NT / TN as torch does it."""

    @staticmethod
    def forward(ctx, a, b, run):
        ctx.save_for_backward(a, b)
        return run(a, b)

    @staticmethod
    def backward(ctx, g):
        a, b = ctx.saved_tensors
        mm = _ORIG["matmul"]
        ga = mm(g, b.t()) if ctx.needs_input_grad[0] else None
        gb = mm(a.t(), g) if ctx.needs_input_grad[1] else None
        return ga, gb, None


def saving_frac(entry: dict) -> float:
    """The relative saving an entry must show, from the clock policy it was timed under (`env.clock_lock`): a declared
    lock (a MHz value) gets SAVING_FRAC_LOCKED; "unlocked", or no declaration, gets the larger SAVING_FRAC_UNLOCKED."""
    lock = str((entry.get("env") or {}).get("clock_lock") or "unlocked").strip().lower()
    return SAVING_FRAC_UNLOCKED if lock in ("unlocked", "none", "off", "0") else SAVING_FRAC_LOCKED


def dispatch_rule(entry: dict, settings: dict) -> bool:
    """Spec 5.3 step 6: dispatch only kernels where torch takes >= min_kernel_us (below that the Python-side launch costs
    as much as the kernel) and the winner saves >= max(min_saving_us, saving_frac(entry) * torch_us) (the noise of the
    short timing both were measured with). An entry without `torch_us` (written outside lscli) is never dispatched."""
    t, w = entry.get("torch_us"), entry.get("winner_us")
    if t is None or w is None or t < settings["min_kernel_us"]:
        return False
    return t - w >= max(settings["min_saving_us"], saving_frac(entry) * t)


def _env_id(paths) -> str:
    """This interpreter's registered env id: the record whose prefix is this sys.prefix (python vs python3, or a symlink
    to the interpreter, must not look like another env); else the id of sys.executable."""
    from lscli import envs
    for eid, rec in envs.load(paths).items():
        if rec.get("prefix") and os.path.realpath(rec["prefix"]) == os.path.realpath(sys.prefix):
            return eid
    return envs.env_id(sys.executable)


class _Client:
    def __init__(self, paths=None):
        from lscli.config import DEFAULTS, client_document
        from lscli.files import read_json
        from lscli.paths import Paths
        from lscli.usage import UsageLog
        self.paths = paths or Paths()
        doc = {**client_document(DEFAULTS), **(read_json(self.paths.client, None) or {})}
        self.settings, self.search, self.doc = doc["client"], doc["search"], doc
        self.env_id = _env_id(self.paths)
        self.env_extra = {**(doc.get("env_extra") or {}), "env_id": self.env_id}
        self.log = UsageLog(self.paths, self.env_id)
        self.device = _UNSET                                 # torch index of the daemon's GPU, resolved at first use
        self.wc = None
        self._wc_mtime = None
        self._wc_checked = 0.0
        self._last_flush = time.monotonic()
        self.runs: dict[tuple, tuple] = {}                   # shape -> (backend | None for torch, table has an entry)
        self.deciding: set = set()
        self.queued: set = set()
        self.disabled: set = set()
        self.inline_tried: set = set()
        self.inline_ok = True                                # False after the first failure (no lightgbm / models here)
        self.ranker = None
        self.profile = None

    # ------------------------------------------------------------------ device
    def _resolve_device(self):
        """The torch device index whose UUID is the daemon's GPU (None: not this process's GPU / no daemon yet with
        several GPUs). Called at the first in-scope CUDA call, when CUDA is initialized already."""
        uuid = self.doc.get("gpu_uuid")
        n = torch.cuda.device_count()
        if not uuid:
            return 0 if n == 1 else None
        want = uuid.removeprefix("GPU-")
        for i in range(n):
            if str(getattr(torch.cuda.get_device_properties(i), "uuid", "")).removeprefix("GPU-") == want:
                return i
        return None

    # ------------------------------------------------------------------ winner table
    def _cache(self):
        """The table, reloaded (under the lock, swapped atomically) when the file changed; checked at most once a second."""
        now = time.monotonic()
        if self.wc is not None and now - self._wc_checked < 1.0:
            return self.wc
        with _LOCK:
            if self.wc is not None and now - self._wc_checked < 1.0:
                return self.wc
            self._wc_checked = now
            try:
                m = self.paths.winners.stat().st_mtime_ns
            except OSError:
                m = None
            if self.wc is None or m != self._wc_mtime:
                if self.wc is None:
                    from lscli.worker import environment
                    from tune.winner_cache import WinnerCache
                    self.wc = WinnerCache(self.paths.winners, env=environment(self.env_extra))
                else:
                    self.wc.reload()
                self._wc_mtime = m
                self.runs = {k: v for k, v in dict(self.runs).items() if k in self.disabled}  # re-decide the rest
        return self.wc

    def _queue(self, key, reason: str) -> None:
        if (key, reason) in self.queued:
            return
        self.queued.add((key, reason))
        try:
            from lscli import search_queue
            search_queue.request(self.paths, key[0], key[1], key[2], key[3], self.env_id, reason)
        except OSError as e:
            warnings.warn(f"lscli: could not queue {key}: {e}")

    def _frequent_calls(self, key) -> float:
        from lscli import usage
        if self.profile is None:
            prof = usage.load(self.paths)
            self.profile = (prof, usage.lookup(prof))
        prof, look = self.profile
        s = look.get((*key, "NN"))
        return usage.calls_now(prof, s) if s else 0.0

    def _inline(self, key, dtype):
        """select_kernel for a frequent shape, once per process per shape; None when it cannot run here."""
        self.inline_tried.add(key)
        if not self.inline_ok or not self.settings.get("inline_search"):
            return None
        try:
            calls = self._frequent_calls(key)
            if calls < self.search["frequent_calls"]:
                return None
            stem = (self.doc.get("models") or {}).get("triton")
            if stem and self.ranker is None:
                from analysis.learned import Ranker
                self.ranker = Ranker.load(stem)
            # no Triton ranker yet: select_kernel(LT_ONLY, ...) is a cuBLASLt-only search (k = 0)
            from tune.online import LT_ONLY, KPolicy, select_kernel
            pol = KPolicy(self.search["k"], self.search["k_frequent"], self.search["frequent_calls"])
            sel = select_kernel(self.ranker if stem else LT_ONLY, *key[:3], calls=int(calls), policy=pol,
                                lt_n=self.search["lt_n"], dtype=dtype, cache=self.wc, use_bundle=False, serve_stale=True)
            return self.wc.get(*key, k=0, lt_n=0, allow_stale=True) if sel.get("choice") else None
        except Exception as e:                       # lightgbm / models missing in this env: behave as a miss
            self.inline_ok = False
            warnings.warn(f"lscli: inline search unavailable ({type(e).__name__}: {e}); misses use torch")
            return None

    def _decide(self, key, dtype) -> tuple:
        """(backend to dispatch | None for torch, whether the table had a usable entry)."""
        wc = self._cache()
        e = wc.get(*key, k=0, lt_n=0, allow_stale=True)                 # any entry, cuBLASLt-only (k = 0) included
        if e is None and key not in self.inline_tried:
            e = self._inline(key, dtype)
        if e is None:
            self._queue(key, "miss")
            return None, False
        from tune.winner_cache import backend_for
        b = backend_for(e["choice"], key[0], key[1], key[2], dtype, e["lt_n"])
        if b is None:
            self._queue(key, "miss")
            return None, False
        if e["stale"]:
            if not e["verified"]:
                M, N, K = key[:3]
                need = 2 * (M * K + K * N) + 6 * M * N + 4 * (M * K + K * N)   # inputs, output, fp32 reference
                if need > VERIFY_MAX_BYTES:
                    self._queue(key, "stale")         # too big to check inside the user's job: torch until re-searched
                    return None, True
                from tune.correctness import validate
                if not validate(b, M, N, K, dtype).valid:
                    self._queue(key, "miss")
                    return None, False
                wc.mark_verified(e)
            self._queue(key, "stale")
        return (b if dispatch_rule(e, self.settings) else None), True

    # ------------------------------------------------------------------ per call
    def flush(self) -> None:
        self._last_flush = time.monotonic()
        try:
            self.log.flush()
        except OSError:
            pass

    def _route(self, a, b):
        """The backend for this call, or None for torch. Raises only on unexpected errors (the caller uses torch)."""
        if (torch.compiler.is_compiling() or type(a) not in _PLAIN or type(b) not in _PLAIN or a.ndim != 2
                or b.ndim != 2 or not a.is_cuda or not b.is_cuda or a.dtype not in _DTYPES or b.dtype is not a.dtype):
            return None
        M, K = a.shape
        K2, N = b.shape
        if K != K2 or not (0 < M <= MAX_DIM and 0 < N <= MAX_DIM and 0 < K <= MAX_DIM):
            return None
        if not a.is_contiguous() or not b.is_contiguous() or _functorch_wrapped(a) or _functorch_wrapped(b):
            return None                                      # (functorch first: data_ptr raises on wrapped tensors)
        if a.data_ptr() % 16 or b.data_ptr() % 16:
            return None
        if self.device is _UNSET:
            self.device = self._resolve_device()
        dev = a.device.index
        if (self.device is None or dev != self.device or b.device.index != dev or torch._C._cuda_getDevice() != dev
                or torch._C._cuda_isCurrentStreamCapturing() or _autocast_on() or _forward_ad_active()):
            return None
        key = (M, N, K, _DTYPES[a.dtype])
        now = time.monotonic()
        if now - self._wc_checked >= 1.0:
            self._cache()                                    # picks up winners the daemon wrote since
        hit_be = self.runs.get(key)
        if hit_be is None:
            hit_be = self._decide_once(key, a.dtype)
        be, hit = hit_be
        self.log.add(M, N, K, key[3], "NN", hit=hit)
        if now - self._last_flush > FLUSH_S:
            self.flush()
        return be

    def _decide_once(self, key, dtype) -> tuple:
        with _LOCK:
            if key in self.runs:
                return self.runs[key]
            if key in self.deciding:                         # another thread is deciding: torch for this call
                return None, False
            if key in self.disabled or not self.settings.get("dispatch"):
                self.runs[key] = (None, False)
                return self.runs[key]
            self.deciding.add(key)
        try:
            out = self._decide(key, dtype)
        except Exception as e:
            warnings.warn(f"lscli: selection for {key} failed ({type(e).__name__}: {e}); using torch")
            self.disabled.add(key)
            out = (None, False)
        with _LOCK:
            self.deciding.discard(key)
            self.runs[key] = out
        return out

    def matmul(self, orig, a, b):
        """The dispatcher; `orig` is the original torch function for this entry point."""
        try:
            be = self._route(a, b)
        except Exception:
            be = None
        if be is None:
            return orig(a, b)
        try:
            if (a.requires_grad or b.requires_grad) and torch.is_grad_enabled():
                return _Mm.apply(a, b, lambda x, y: be.make_runner(x, y)())
            return be.make_runner(a, b)()
        except Exception as e:
            key = (a.shape[0], b.shape[1], a.shape[1], _DTYPES[a.dtype])
            warnings.warn(f"lscli: kernel for {key} failed ({type(e).__name__}: {e}); using torch for it")
            self.disabled.add(key)
            self.runs[key] = (None, False)
            return orig(a, b)

    def count_linear(self, args, kw) -> None:
        try:
            x = args[0] if args else kw.get("input")
            w = args[1] if len(args) > 1 else kw.get("weight")
            if isinstance(x, torch.Tensor) and isinstance(w, torch.Tensor) and x.is_cuda and x.dtype in _DTYPES \
                    and w.ndim == 2 and x.ndim >= 1 and not torch.compiler.is_compiling():
                self.log.add(x.numel() // max(x.shape[-1], 1), w.shape[0], w.shape[1], _DTYPES[x.dtype], "NT")
                if time.monotonic() - self._last_flush > FLUSH_S:
                    self.flush()
        except Exception:
            pass


def _reset_after_fork() -> None:
    """A forked child (DataLoader worker) must not re-count its parent's calls, nor inherit a lock another thread held."""
    global _LOCK
    _LOCK = threading.Lock()
    c = _STATE
    if c is not None:
        from lscli.usage import UsageLog
        c.log = UsageLog(c.paths, c.env_id)
        c.deciding = set()


def enabled() -> bool:
    return _STATE is not None


def enable(paths=None) -> bool:
    """Patch torch for the rest of the process. Returns False (and patches nothing) when LSCLI_DISABLE=1."""
    global _STATE
    if os.environ.get("LSCLI_DISABLE") == "1":
        return False
    if _STATE is not None:
        return True
    import torch.nn.functional as F
    c = _Client(paths)
    _ORIG.update(matmul=torch.matmul, mm=torch.mm, t_matmul=torch.Tensor.matmul, t_mm=torch.Tensor.mm,
                 t_rmatmul=torch.Tensor.__matmul__, linear=F.linear)

    def _wrap(orig):
        def f(*args, **kw):
            if kw or len(args) != 2 or _STATE is not c:     # keywords (input=, out=, ...) or disabled: not ours
                return orig(*args, **kw)
            return c.matmul(orig, args[0], args[1])
        f.__wrapped__ = orig
        f.__name__, f.__qualname__, f.__doc__ = orig.__name__, getattr(orig, "__qualname__", orig.__name__), orig.__doc__
        return f

    def linear(*args, **kw):
        if _STATE is c:
            c.count_linear(args, kw)
        return _ORIG["linear"](*args, **kw)
    linear.__wrapped__ = _ORIG["linear"]
    linear.__name__, linear.__doc__ = "linear", _ORIG["linear"].__doc__
    torch.matmul = _wrap(_ORIG["matmul"])
    torch.mm = _wrap(_ORIG["mm"])
    torch.Tensor.matmul = _wrap(_ORIG["t_matmul"])
    torch.Tensor.mm = _wrap(_ORIG["t_mm"])
    torch.Tensor.__matmul__ = _wrap(_ORIG["t_rmatmul"])
    F.linear = linear
    _STATE = c
    atexit.register(_atexit)
    if not getattr(enable, "_fork_hooked", False):
        os.register_at_fork(after_in_child=_reset_after_fork)
        enable._fork_hooked = True
    return True


def _atexit() -> None:
    if _STATE is not None:
        _STATE.flush()


def disable() -> None:
    """Restore torch and flush the usage counts."""
    global _STATE
    if _STATE is None:
        return
    import torch.nn.functional as F
    _STATE.flush()
    torch.matmul, torch.mm = _ORIG["matmul"], _ORIG["mm"]
    torch.Tensor.matmul, torch.Tensor.mm, torch.Tensor.__matmul__ = _ORIG["t_matmul"], _ORIG["t_mm"], _ORIG["t_rmatmul"]
    F.linear = _ORIG["linear"]
    _STATE = None
    try:
        atexit.unregister(_atexit)
    except Exception:
        pass
