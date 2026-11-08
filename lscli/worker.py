"""The measure worker (spec section 7): runs under the USER's interpreter, so a winner is timed with the torch / Triton /
cuBLASLt that will run it.

  <python> -m lscli.worker measure     one JSON request on stdin -> JSON lines on the protocol channel
  <python> -m lscli.worker env         this interpreter's winner-cache environment
  <python> -m lscli.worker verify --cache PATH [--env-extra JSON]   correctness gate over every entry served here

Protocol channel: the worker's original stdout. Before anything else is imported, fd 1 is duplicated to a private
descriptor and fd 1 / `sys.stdout` are pointed at stderr, so a print or warning from torch / Triton / a CUDA library
lands in the log instead of corrupting the protocol.

`measure` needs no ranking model: the daemon ranked the Triton candidates and the cuBLASLt ranker's proposals and sends
their keys. The worker queries its own library's heuristic top lt_n (`propose_lt` without proposals), rebuilds the
proposals with `backends_from_keys` (dropping keys its library rejects, and proposals that duplicate a heuristic
config), short-times torch.matmul for the dispatch rule (`torch_us`), runs `tune.online.measure` and writes the winner
to the cache LAST, so a worker killed by a preemption writes nothing. With `router: true` it asks the daemon whether to
skip the Triton half once the cuBLASLt half is timed (`{"op": "route"}`), and does not skip if no answer comes within
`reply_timeout_s`.
"""
from __future__ import annotations

import json
import os
import select
import sys
import time


def _channel():
    """Move the real stdout to a private fd and send fd 1 + sys.stdout to stderr. Returns the protocol writer."""
    fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return os.fdopen(fd, "w", buffering=1)


class _Proto:
    def __init__(self, out, inp=None):
        self.out, self.inp = out, inp if inp is not None else sys.stdin

    def send(self, obj: dict) -> None:
        self.out.write(json.dumps(obj, default=str) + "\n")
        self.out.flush()

    def ask(self, obj: dict, timeout_s: float) -> dict | None:
        self.send(obj)
        r, _, _ = select.select([self.inp], [], [], timeout_s)
        if not r:
            return None
        line = self.inp.readline()
        try:
            return json.loads(line) if line.strip() else None
        except ValueError:
            return None


class RouterProxy:
    """`tune.online.measure`'s router interface, answered by the daemon over the protocol channel."""

    def __init__(self, proto: _Proto, timeout_s: float, threshold: float | None = None):
        self.proto, self.timeout_s, self.threshold = proto, timeout_s, threshold

    def skip_triton(self, M: int, N: int, K: int, t_lt_us: float):
        ans = self.proto.ask({"op": "route", "t_lt_us": t_lt_us}, self.timeout_s)
        if not ans:
            return False, None
        if ans.get("threshold") is not None:
            self.threshold = ans["threshold"]
        return bool(ans.get("skip")), ans.get("p")


DTYPES = ("fp16", "bf16")


def _torch_dtype(name: str):
    import torch
    return {"fp16": torch.float16, "bf16": torch.bfloat16}[name]


def environment(env_extra: dict | None = None) -> dict:
    """`tune.winner_cache.environment()` of this interpreter plus lscli's fields (spec 3.6): clock policy, models, and
    `env_id`, the registered env the entry belongs to (it scopes `compact`, and makes another env's winner soft-stale
    here: served after the correctness check, then re-searched in this env)."""
    from tune.winner_cache import environment as wc_env
    extra = dict(env_extra or {})
    env = wc_env(models=extra.pop("models", None), clock_lock=extra.pop("clock_lock", None))
    return {**env, **extra}


def torch_time_us(M: int, N: int, K: int, dtype) -> float | None:
    """torch.matmul's short cold time for the shape (the same short protocol as the candidates)."""
    import torch
    from tune.online import short_configs
    from tune.timing import benchmark
    cold, _ = short_configs()
    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(K, N, device="cuda", dtype=dtype)

    class _Torch:
        def make_runner(self, a, b):
            return lambda: torch.matmul(a, b)
    try:
        return benchmark(_Torch().make_runner(a, b), cold).median_us
    finally:
        del a, b
        torch.cuda.empty_cache()


def measure_request(req: dict, proto: _Proto | None = None) -> dict:
    """Run one request; returns the result document (without "op")."""
    from baselines.cublaslt_knn import cfg_tuple
    from tune.online import backends_from_keys, choose, measure, propose_lt
    M, N, K = int(req["M"]), int(req["N"]), int(req["K"])
    if req.get("dtype", "fp16") not in DTYPES:
        raise ValueError(f"dtype must be one of {DTYPES}")
    dtype = _torch_dtype(req.get("dtype", "fp16"))
    k, lt_n = int(req["k"]), int(req.get("lt_n", 0))
    entries = [e for e in req.get("proposals", []) if e.get("family", "triton") != "cublaslt"]  # heuristic: our own
    back = backends_from_keys(entries, M, N, K, dtype, lt_n)
    triton = [b for b in back if b[0] == "triton"]
    heur, _ = propose_lt(M, N, K, dtype, lt_n)
    known = {cfg_tuple(b[2].cfg) for b in heur}
    lt = heur + [b for b in back if b[0] != "triton" and cfg_tuple(b[2].cfg) not in known]
    router = RouterProxy(proto, float(req.get("reply_timeout_s", 30))) if req.get("router") and proto else None
    t_us = torch_time_us(M, N, K, dtype)
    r = measure(M, N, K, dtype, lt, triton, lt_n=lt_n, router=router, confirm=int(req.get("confirm", 0)))
    env = environment(req.get("env_extra"))
    if r.get("skipped"):
        return {"skipped": r["skipped"], "choice": None, "candidates": [], "written": False, "env": env}
    choice = choose(r["candidates"], k)
    written = False
    if req.get("cache") and choice.get("key") is not None:
        from tune.winner_cache import WinnerCache
        n_lt_rank = sum(1 for b in lt if b[0] == "cublaslt_rank")
        WinnerCache(req["cache"], env=env).put(
            M, N, K, req.get("dtype", "fp16"), k=k, lt_n=lt_n, lt_rank=int(req.get("lt_rank", n_lt_rank)),
            lt_knn=int(req.get("lt_knn", 0)), choice=choice, candidates=r["candidates"], order=req.get("order"),
            ranker=req.get("ranker"), extra={"torch_us": t_us, "lscli": True})
        written = True
    return {"choice": choice, "candidates": r["candidates"], "router": r.get("router"), "torch_us": t_us,
            "written": written, "env": env, "pool": [b[1] for b in triton + lt]}


def verify(cache: str, env_extra: dict | None = None) -> dict:
    """Correctness gate over every entry this env would serve (current and soft-stale); soft-stale ones that pass are
    recorded with mark_verified, as the client would."""
    from tune.correctness import validate
    from tune.winner_cache import WinnerCache, backend_for
    wc = WinnerCache(cache, env=environment(env_extra))
    out = []
    for e in wc.entries(allow_stale=True):
        dtype = _torch_dtype(e["dtype"])
        b = backend_for(e["choice"], e["M"], e["N"], e["K"], dtype, e["lt_n"])
        ok = b is not None and validate(b, e["M"], e["N"], e["K"], dtype).valid
        if ok and e["stale"]:
            wc.mark_verified(e)
        out.append({"M": e["M"], "N": e["N"], "K": e["K"], "dtype": e["dtype"], "key": e["choice"]["key"],
                    "stale": e["stale"], "ok": bool(ok), "missing": b is None})
    return {"entries": out, "failed": sum(1 for x in out if not x["ok"])}


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m lscli.worker")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("measure")
    e = sub.add_parser("env")
    e.add_argument("--env-extra", default="{}")
    v = sub.add_parser("verify")
    v.add_argument("--cache", required=True)
    v.add_argument("--env-extra", default="{}")
    args = ap.parse_args(argv)
    proto = _Proto(_channel())
    t0 = time.perf_counter()
    try:
        if args.cmd == "measure":
            req = json.loads(sys.stdin.readline())
            res = measure_request(req, proto)
        elif args.cmd == "env":
            res = {"env": environment(json.loads(args.env_extra)), "python": sys.executable}
        else:
            res = verify(args.cache, json.loads(args.env_extra))
        proto.send({"op": "result", **res, "wall_s": time.perf_counter() - t0})
        return 0
    except Exception as ex:  # report, never hang the daemon
        import traceback
        traceback.print_exc()
        proto.send({"op": "error", "error": f"{type(ex).__name__}: {ex}"})
        return 1


if __name__ == "__main__":
    sys.exit(main())
