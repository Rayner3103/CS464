"""Benchmark runner: validate then time every (shape, backend config) pair, writing rows to JSONL (plan 7).

Crash safety: before each candidate runs, its key is written to `inflight.json`. If the process dies
(e.g. an illegal memory access poisons the CUDA context), the next run finds the marker, records that
candidate as invalid ("crashed_previous_run") and moves on. Completed rows are skipped on resume.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch

from tune.correctness import make_inputs, reference, validate
from tune.env import capture_env, write_env
from tune.records import RecordWriter, alignment_flags, completed_keys, measurement_key
from tune.timing import TimingConfig, benchmark

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
LAYOUT = "NN"


@dataclass
class Candidate:
    family: str
    config_key: str
    config: dict
    backend: object  # has make_runner(a, b, out=None) -> Callable[[], Tensor]


def candidates_for(spec: str, M: int, N: int, K: int, prune_shapes: bool = False) -> list[Candidate]:
    """Backend spec strings: torch | triton_tutorial | triton:handpicked | triton:<config_key>."""
    from baselines.torch_mm import TorchMatmul
    from baselines.triton_tutorial import TritonTutorialMatmul
    from kernels.config_space import HANDPICKED, shape_violations
    from kernels.triton_matmul import TritonMatmul

    if spec == "torch":
        return [Candidate(TorchMatmul.family, "default", {}, TorchMatmul())]
    if spec == "triton_tutorial":
        b = TritonTutorialMatmul()
        return [Candidate(b.family, "default", b.cfg, b)]
    if spec.startswith("triton:"):
        which = spec.split(":", 1)[1]
        if which == "handpicked":
            cfgs = list(HANDPICKED.values())
        else:
            cfgs = [c for c in HANDPICKED.values() if c.key() == which]
            if not cfgs:
                raise ValueError(f"unknown triton config {which!r}")
        out = []
        for c in cfgs:
            v = shape_violations(c, M, N, K)
            hard = [r for r in v if r in ("split_k_workspace_too_large",)]
            if hard or (prune_shapes and v):
                continue
            out.append(Candidate(TritonMatmul.family, c.key(), c.to_dict(), TritonMatmul(c)))
        return out
    raise ValueError(f"unknown backend spec {spec!r}")


def load_suite(name: str, version: str = "v1") -> list[dict]:
    path = Path(__file__).resolve().parent.parent / "suites" / version / f"{name}.json"
    return json.loads(path.read_text())["shapes"]


def run(shapes: Iterable[dict], backend_specs: list[str], out_dir: Path, *, dtype: str = "fp16",
        cache_modes=("cold",), timing_overrides: dict | None = None, budget_s: float | None = None,
        session_id: str | None = None, prune_shapes: bool = False, skip_out_of_range: bool = True,
        out_name: str = "results.jsonl", log: Callable[[str], None] = print) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = capture_env()
    if not (out_dir / "env.json").exists():
        write_env(out_dir)
    session_id = session_id or time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    results = out_dir / out_name
    inflight = out_dir / "inflight.json"
    done = completed_keys(results)
    tdtype = DTYPES[dtype]
    shapes = [s for s in shapes if not (skip_out_of_range and s.get("out_of_range"))]

    base_row = {"hw_id": env["hw_id"], "driver": env["driver"], "cuda": env["versions"]["torch_cuda"],
                "torch": env["versions"]["torch"], "triton": env["versions"]["triton"],
                "dtype": dtype, "layout": LAYOUT, "session_id": session_id}

    t0 = time.perf_counter()
    last_tel: dict = {}
    with RecordWriter(results) as w:
        if inflight.exists():
            crashed = json.loads(inflight.read_text())
            log(f"[runner] previous run died during {crashed['key']}; recording as invalid")
            w.write({**crashed["row"], "valid": False, "failure_reason": "crashed_previous_run",
                     "timestamp": time.time()})
            done.add(tuple(crashed["key"]))
            inflight.unlink()

        for i, s in enumerate(shapes):
            if budget_s is not None and time.perf_counter() - t0 > budget_s:
                log(f"[runner] budget of {budget_s:.0f}s exhausted after {i}/{len(shapes)} shapes")
                break
            M, N, K = s["M"], s["N"], s["K"]
            cands = [c for spec in backend_specs for c in candidates_for(spec, M, N, K, prune_shapes)]
            todo = []
            for c in cands:
                for cm in cache_modes:
                    row = {**base_row, "M": M, "N": N, "K": K, "alignment_flags": alignment_flags(M, N, K),
                           "kernel_family": c.family, "config_key": c.config_key, "config": c.config,
                           "cache_mode": cm, "batch": 1, "use_graph": False, "suite": s.get("suite"),
                           "split": s.get("split")}
                    if measurement_key(row) not in done:
                        todo.append((c, cm, row))
            if not todo:
                continue

            try:
                a, b = make_inputs(M, N, K, tdtype, seed=0)
                ref = reference(a, b)
            except torch.OutOfMemoryError:
                for c, cm, row in todo:
                    w.write({**row, "valid": False, "failure_reason": "oom_inputs", "timestamp": time.time()})
                torch.cuda.empty_cache()
                continue

            validity: dict[str, tuple] = {}
            for c, cm, row in todo:
                key = measurement_key(row)
                inflight.write_text(json.dumps({"key": list(key), "row": row}, default=str))
                if c.config_key not in validity:
                    chk = validate(c.backend, M, N, K, tdtype, ref=ref, inputs=(a, b))
                    validity[c.config_key] = (chk.valid, chk.failure_reason, chk.max_abs_err)
                valid, reason, err = validity[c.config_key]
                row = {**row, "valid": valid, "failure_reason": reason, "max_abs_err": err}
                if not valid:
                    w.write({**row, "timestamp": time.time()})
                    inflight.unlink()
                    continue
                try:
                    fn = c.backend.make_runner(a, b)
                    cfg = TimingConfig(cache_mode=cm, **(timing_overrides or {}))
                    r = benchmark(fn, cfg)
                except torch.OutOfMemoryError:
                    w.write({**row, "valid": False, "failure_reason": "oom_timing", "timestamp": time.time()})
                    inflight.unlink()
                    torch.cuda.empty_cache()
                    continue
                tel = last_tel = r.telemetry_after
                w.write({**row, "time_median_us": r.median_us, "time_p10_us": r.p10_us, "time_p90_us": r.p90_us,
                         "time_min_us": r.min_us, "time_max_us": r.max_us, "time_iqr_us": r.iqr_us,
                         "rse": r.rse, "n_samples": r.n_samples, "converged": r.converged,
                         "tflops": 2 * M * N * K / (r.median_us * 1e-6) / 1e12,
                         "clock_mhz": tel.get("clock_mhz"), "temp_c": tel.get("temp_c"),
                         "power_w": tel.get("power_w"), "throttle_reasons": tel.get("throttle_reasons"),
                         "clock_mhz_before": r.telemetry_before.get("clock_mhz"),
                         "timestamp": time.time()}, samples_us=r.samples_us)
                inflight.unlink()

            del a, b, ref
            torch.cuda.empty_cache()
            elapsed = time.perf_counter() - t0
            eta = elapsed / (i + 1) * (len(shapes) - i - 1)
            log(f"[{i + 1}/{len(shapes)}] {M}x{N}x{K} {len(todo)} meas | {elapsed:.0f}s elapsed, ETA {eta:.0f}s"
                f" | {last_tel.get('clock_mhz')} MHz {last_tel.get('temp_c')}C {last_tel.get('power_w')}W")
    return results
