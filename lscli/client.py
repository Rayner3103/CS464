"""The lscli client (spec section 5): an opt-in patch of torch's NN matmul that serves the winner cache.

`enable()` replaces `torch.matmul`, `torch.mm` and `Tensor.__matmul__` / `Tensor.matmul` / `Tensor.mm` with a
dispatcher, and wraps `torch.nn.functional.linear` to count its (NT) shapes. `disable()` restores the originals.
Everything not in scope goes to the original function untouched.

Per call:
  1. count the shape (in memory; flushed to `profile/<pid>-<t0>.jsonl` every 30 s, at exit, and never twice after a fork);
  2. look it up in the winner table (`tune.winner_cache.WinnerCache` with client.json's env_extra + this env's id;
     any search width; mtime re-checked at most once a second);
  3. current entry -> dispatch if the rule allows it; soft-stale entry -> one correctness check per environment
     (`mark_verified`), dispatch, queue `stale` once; miss -> an inline `select_kernel` for shapes the usage record
     counts as frequent (once per process, needs lightgbm + the models named in client.json), otherwise torch +
     queue `miss` once;
  4. dispatch rule: the entry's `torch_us` >= `min_kernel_us` and `torch_us - winner_us >= max(min_saving_us, 2 %)`;
     else torch (still a hit: the selector chose torch).
A failure of a dispatched kernel falls back to torch, disables that shape for the process and warns once. Under
autograd the dispatched kernel runs inside an `autograd.Function` whose backward calls the ORIGINAL torch.matmul.

The fast path is a few attribute checks and two dict lookups; `tests/test_lscli_client.py::test_overhead` measures it.
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
MAX_DIM = 16384
FLUSH_S = 30.0


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


class _Client:
    def __init__(self, paths=None):
        from lscli import envs
        from lscli.files import read_json
        from lscli.paths import Paths
        from lscli.usage import UsageLog
        self.paths = paths or Paths()
        doc = read_json(self.paths.client, None) or {}
        from lscli.config import DEFAULTS, client_document
        doc = {**client_document(DEFAULTS), **doc}
        self.settings, self.search, self.doc = doc["client"], doc["search"], doc
        self.env_id = envs.env_id(sys.executable)
        self.env_extra = {**(doc.get("env_extra") or {}), "env_id": self.env_id}
        self.log = UsageLog(self.paths, self.env_id)
        self.device = None                                   # torch device index of the daemon's GPU (None: unknown)
        uuid = doc.get("gpu_uuid")
        if uuid and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                if str(getattr(torch.cuda.get_device_properties(i), "uuid", "")) in (uuid, uuid.removeprefix("GPU-")):
                    self.device = i
        elif torch.cuda.is_available() and torch.cuda.device_count() == 1 and not uuid:
            self.device = 0                                  # no daemon has written the GPU yet: the only one
        self.wc = None
        self._wc_mtime = None
        self._wc_checked = 0.0
        self._last_flush = time.monotonic()
        self.runs: dict[tuple, tuple] = {}                   # shape -> (backend | None for torch, table has an entry)
        self.queued: set = set()
        self.disabled: set = set()
        self.inline_tried: set = set()
        self.inline_ok = True                                # False after the first failure (no lightgbm / models here)
        self.profile = None

    # ------------------------------------------------------------------ winner table
    def _cache(self):
        now = time.monotonic()
        if now - self._wc_checked < 1.0 and self.wc is not None:
            return self.wc
        self._wc_checked = now
        try:
            m = self.paths.winners.stat().st_mtime_ns
        except FileNotFoundError:
            m = None
        if self.wc is None or m != self._wc_mtime:
            from tune.winner_cache import WinnerCache
            from lscli.worker import environment
            if self.wc is None:
                self.wc = WinnerCache(self.paths.winners, env=environment(self.env_extra))
            else:
                self.wc.reload()
            self._wc_mtime = m
            self.runs = {k: v for k, v in self.runs.items() if k in self.disabled}  # re-decide with the new table
        return self.wc

    def _queue(self, key, reason: str) -> None:
        if (key, reason) in self.queued:
            return
        self.queued.add((key, reason))
        try:
            from lscli import search_queue as queue
            queue.request(self.paths, key[0], key[1], key[2], key[3], self.env_id, reason)
        except OSError as e:
            warnings.warn(f"lscli: could not queue {key}: {e}")

    def _frequent_calls(self, key) -> float:
        if self.profile is None:
            from lscli import usage
            self.profile = (usage.load(self.paths), usage.lookup(usage.load(self.paths)))
        prof, look = self.profile
        s = look.get((*key, "NN"))
        from lscli.usage import calls_now
        return calls_now(prof, s) if s else 0.0

    def _inline(self, key, dtype):
        """select_kernel for a frequent shape, once per process; None when it cannot run here."""
        self.inline_tried.add(key)
        if not self.inline_ok:
            return None
        try:
            calls = self._frequent_calls(key)
            if not self.settings.get("inline_search") or calls < self.search["frequent_calls"]:
                return None
            stem = (self.doc.get("models") or {}).get("triton")
            if not stem:
                return None
            from analysis.learned import Ranker
            from tune.online import KPolicy, select_kernel
            pol = KPolicy(self.search["k"], self.search["k_frequent"], self.search["frequent_calls"])
            sel = select_kernel(Ranker.load(stem), *key[:3], calls=int(calls), policy=pol, lt_n=self.search["lt_n"],
                                dtype=dtype, cache=self._cache(), use_bundle=False, serve_stale=True)
            self.wc.reload()
            return self.wc.get(*key, k=1, lt_n=0, allow_stale=True) if sel.get("choice") else None
        except Exception as e:                       # lightgbm / models missing in this env: behave as a miss
            self.inline_ok = False
            warnings.warn(f"lscli: inline search unavailable ({type(e).__name__}: {e}); misses use torch")
            return None

    def _decide(self, key, dtype) -> tuple:
        """(backend to dispatch | None for torch, whether the table had a usable entry). Called once per shape (and
        again after the table changed on disk)."""
        wc = self._cache()
        e = wc.get(*key, k=1, lt_n=0, allow_stale=True)
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
                from tune.correctness import validate
                if not validate(b, key[0], key[1], key[2], dtype).valid:
                    self._queue(key, "miss")
                    return None, False
                wc.mark_verified(e)
            self._queue(key, "stale")
        return (b if dispatch_rule(e, self.settings) else None), True

    # ------------------------------------------------------------------ per call
    def maybe_flush(self) -> None:
        if time.monotonic() - self._last_flush > FLUSH_S:
            self.flush()

    def flush(self) -> None:
        self._last_flush = time.monotonic()
        try:
            self.log.flush()
        except OSError:
            pass

    def matmul(self, orig, a, b):
        """The dispatcher; `orig` is the original torch function for this entry point."""
        if (type(a) is not torch.Tensor or type(b) is not torch.Tensor or a.ndim != 2 or b.ndim != 2
                or not a.is_cuda or a.dtype not in _DTYPES or b.dtype is not a.dtype or not b.is_cuda):
            return orig(a, b)
        M, K = a.shape
        K2, N = b.shape
        if K != K2 or not (0 < M <= MAX_DIM and 0 < N <= MAX_DIM and 0 < K <= MAX_DIM):
            return orig(a, b)
        if (self.device is None or a.device.index != self.device or b.device.index != self.device
                or not a.is_contiguous() or not b.is_contiguous() or a.data_ptr() % 16 or b.data_ptr() % 16
                or torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing()):
            return orig(a, b)
        key = (M, N, K, _DTYPES[a.dtype])
        if time.monotonic() - self._wc_checked >= 1.0:
            self._cache()                                    # picks up winners the daemon wrote since
        try:
            be, hit = self.runs[key]
        except KeyError:
            with _LOCK:
                try:
                    be, hit = self.runs[key] = (None, False) if key in self.disabled or not self.settings.get("dispatch") \
                        else self._decide(key, a.dtype)
                except Exception as e:
                    warnings.warn(f"lscli: selection for {key} failed ({type(e).__name__}: {e}); using torch")
                    self.disabled.add(key)
                    be, hit = self.runs[key] = (None, False)
        self.log.add(M, N, K, key[3], "NN", hit=hit)
        self.maybe_flush()
        if be is None:
            return orig(a, b)
        try:
            if (a.requires_grad or b.requires_grad) and torch.is_grad_enabled():
                return _Mm.apply(a, b, lambda x, y: be.make_runner(x, y)())
            return be.make_runner(a, b)()
        except Exception as e:
            warnings.warn(f"lscli: kernel for {key} failed ({type(e).__name__}: {e}); using torch for it")
            self.disabled.add(key)
            self.runs[key] = (None, False)
            return orig(a, b)

    def linear(self, x, w, bias=None):
        try:
            if type(x) is torch.Tensor and x.is_cuda and x.dtype in _DTYPES and w.ndim == 2:
                M = x.numel() // x.shape[-1] if x.ndim else 1
                self.log.add(M, w.shape[0], w.shape[1], _DTYPES[x.dtype], "NT")
                self.maybe_flush()
        except Exception:
            pass
        return _ORIG["linear"](x, w, bias)


def dispatch_rule(entry: dict, settings: dict) -> bool:
    """Spec 5.3 step 6: dispatch only kernels where torch takes >= min_kernel_us (below that the Python-side launch costs
    as much as the kernel) and the winner saves >= max(min_saving_us, 2 %) (the study's tie margin). An entry without
    `torch_us` (written outside lscli) is never dispatched."""
    t, w = entry.get("torch_us"), entry.get("winner_us")
    if t is None or w is None or t < settings["min_kernel_us"]:
        return False
    return t - w >= max(settings["min_saving_us"], 0.02 * t)


def _reset_after_fork() -> None:
    """A forked child (DataLoader worker) must not re-count or re-flush its parent's calls."""
    c = _STATE
    if c is not None:
        from lscli.usage import UsageLog
        c.log = UsageLog(c.paths, c.env_id)


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
        def f(a, b, *args, **kw):
            if args or kw:                                   # out= or anything unusual: not ours
                return orig(a, b, *args, **kw)
            return c.matmul(orig, a, b)
        f.__wrapped__ = orig
        return f
    torch.matmul = _wrap(_ORIG["matmul"])
    torch.mm = _wrap(_ORIG["mm"])
    torch.Tensor.matmul = _wrap(_ORIG["t_matmul"])
    torch.Tensor.mm = _wrap(_ORIG["t_mm"])
    torch.Tensor.__matmul__ = _wrap(_ORIG["t_rmatmul"])
    F.linear = lambda x, w, bias=None: c.linear(x, w, bias)
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
