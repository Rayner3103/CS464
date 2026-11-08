"""`python -m lscli.propose` (daemon venv, run as part of a search job): the cuBLASLt ranker's proposals for one shape.

The learned cuBLASLt ranker enumerates every valid config of the shape through cuBLASLt descriptors and
`cublasLtMatmulAlgoCheck`, so it needs the GPU; running it here (a child the gate tracks and preemption kills) keeps the
daemon process free of a CUDA context. Request on stdin: {"M", "N", "K", "dtype", "lt_n", "lt_rank", "stem"}; one result
line on the original stdout: {"op": "result", "proposals": [{family, key, setup_s}], "wall_s"} or {"op": "error"}.
"""
from __future__ import annotations

import json
import sys
import time


def main() -> int:
    from lscli.worker import _channel
    out = _channel()
    t0 = time.perf_counter()
    try:
        req = json.loads(sys.stdin.readline())
        import os
        os.environ["TUNE_LT_RANK"] = req["stem"]
        os.environ["TUNE_LT_RANK_FOREIGN"] = "1"     # the daemon already applied the architecture rule (lscli.models)
        import torch
        from tune.online import proposal_keys, propose_lt
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[req.get("dtype", "fp16")]
        backs, _ = propose_lt(int(req["M"]), int(req["N"]), int(req["K"]), dtype, int(req.get("lt_n", 16)),
                              lt_rank=int(req["lt_rank"]))
        keys = [k for k in proposal_keys(backs) if k["family"] == "cublaslt_rank"]
        out.write(json.dumps({"op": "result", "proposals": keys, "wall_s": time.perf_counter() - t0}) + "\n")
        return 0
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        out.write(json.dumps({"op": "error", "error": f"{type(e).__name__}: {e}"}) + "\n")
        return 1
    finally:
        out.flush()


if __name__ == "__main__":
    sys.exit(main())
