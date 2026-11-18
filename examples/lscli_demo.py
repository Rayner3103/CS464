"""Try lscli on this machine: run a few fp16 GEMMs through `torch.matmul` and report what lscli did with each shape.

The user guide (docs/user_guide.md, "Try it out") walks through it. In short:

    LSCLI_PATCH=1 python examples/lscli_demo.py      # 1st run: every shape is a miss, torch runs it, lscli queues it
    lscli status                                     # wait until the daemon has searched the shapes (cache: N current)
    LSCLI_PATCH=1 python examples/lscli_demo.py      # 2nd run: each shape now has a winner; dispatched or kept on torch

Per shape it prints torch's own time (the unpatched function), the time through lscli's patch, the largest
difference between the two outputs (relative to the largest output value), and lscli's decision:

* `miss (queued)`: no winner stored yet; torch ran it and the shape went into the daemon's search queue;
* `winner, kept torch`: a winner is stored but the dispatch rule kept torch (torch's kernel is under 50 us, or the
  winner saves less than max(5 us, 4 %), 3 % under a clock lock; docs/lscli/spec.md section 5.3). That is a hit too:
  the selector chose torch;
* `dispatched <family> <key>`: the call ran the stored winner.

It also runs without `LSCLI_PATCH=1`: it then calls `lscli.enable()` itself. With `--no-enable` it only times torch.
The shapes default to a few large LLM-layer and odd-sized GEMMs (all well above the 50 us dispatch floor); pass your
own as MxNxK. The decision column reads lscli's in-process state (`lscli.client._STATE`), which is not a stable API:
it is for this demo only.
"""
from __future__ import annotations

import argparse
import re
import sys
import time

import torch

DEFAULT_SHAPES = ["4096x4096x4096", "2048x11008x4096", "512x4096x11008", "1000x3000x777"]


def parse_shape(s: str) -> tuple[int, int, int]:
    m = re.fullmatch(r"(\d+)[xX](\d+)[xX](\d+)", s)
    if not m:
        raise argparse.ArgumentTypeError(f"bad shape {s!r}; use MxNxK")
    M, N, K = (int(x) for x in m.groups())
    if not all(1 <= x <= 16384 for x in (M, N, K)):
        raise argparse.ArgumentTypeError(f"{s}: M, N, K must be in [1, 16384] (lscli leaves other shapes to torch)")
    return M, N, K


def time_us(fn, a, b, seconds: float) -> tuple[float, int]:
    """Median per-call time over ~`seconds` of back-to-back calls (CUDA events, 20 calls per sample), and the number
    of calls made. A demo timer, not the study's protocol (tune/timing.py)."""
    for _ in range(3):
        fn(a, b)
    torch.cuda.synchronize()
    samples, calls, t_end = [], 3, time.monotonic() + seconds
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    while time.monotonic() < t_end or len(samples) < 5:
        start.record()
        for _ in range(20):
            fn(a, b)
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop) * 1000 / 20)
        calls += 20
    samples.sort()
    return samples[len(samples) // 2], calls


def decision(M: int, N: int, K: int) -> str:
    """lscli's choice for this shape in this process (reads private state: demo only)."""
    try:
        from lscli import client
    except ImportError:
        return "lscli not importable"
    st = client._STATE
    if st is None:
        return "lscli not enabled"
    be, hit = st.runs.get((M, N, K, "fp16"), (None, False))
    if be is not None:
        key = getattr(be, "key", None)                           # cuBLASLt backends carry their key
        if key is None:                                          # TritonMatmul: the config builds it
            key = getattr(getattr(be, "cfg", None), "key", lambda: "?")()
        return f"dispatched {getattr(be, 'family', '?')} {key}"
    return "winner, kept torch" if hit else "miss (queued)"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("shapes", nargs="*", type=parse_shape, help=f"MxNxK (default: {' '.join(DEFAULT_SHAPES)})")
    ap.add_argument("--seconds", type=float, default=2.0, help="time per shape and per path (default 2)")
    ap.add_argument("--no-enable", action="store_true", help="do not enable lscli: time torch alone")
    args = ap.parse_args(argv)
    shapes = args.shapes or [parse_shape(s) for s in DEFAULT_SHAPES]
    if not torch.cuda.is_available():
        print("no CUDA device visible to torch", file=sys.stderr)
        return 1

    enabled = False
    if not args.no_enable:
        try:
            import lscli
            enabled = lscli.enable()                 # no-op (True) when LSCLI_PATCH=1 already enabled it
        except ImportError:
            print("lscli is not importable from this interpreter: run `lscli enable-env "
                  f"{sys.executable}` first, or pass --no-enable", file=sys.stderr)
            return 1
        if not enabled:
            print("lscli did not enable (LSCLI_DISABLE=1, or the runtime guard refused: see the warning above)",
                  file=sys.stderr)
    # the original function, whether or not the patch is on (the patch keeps it as __wrapped__)
    torch_mm = getattr(torch.matmul, "__wrapped__", torch.matmul)

    print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | lscli {'on' if enabled else 'off'}")
    print(f"{'shape (MxNxK)':>18} {'torch us':>9} {'lscli us':>9} {'ratio':>6} {'rel diff':>9}  decision")
    for M, N, K in shapes:
        g = torch.Generator(device="cuda").manual_seed(464)
        a = torch.randn(M, K, device="cuda", dtype=torch.float16, generator=g)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16, generator=g)
        t_torch, _ = time_us(torch_mm, a, b, args.seconds)
        t_lscli, _ = time_us(torch.matmul, a, b, args.seconds)
        ref = torch_mm(a, b).float()
        diff = ((torch.matmul(a, b).float() - ref).abs().max() / ref.abs().max().clamp_min(1e-6)).item()
        print(f"{f'{M}x{N}x{K}':>18} {t_torch:9.1f} {t_lscli:9.1f} {t_lscli / t_torch:6.3f} {diff:9.1e}  "
              f"{decision(M, N, K) if enabled else 'torch (lscli off)'}")
        del a, b
    if enabled:
        print("\nUsage counts are flushed at exit; `lscli profile` lists them, `lscli status` shows the queue and cache.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
