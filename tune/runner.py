"""Benchmark runner: validate then time every (shape, backend config) pair, writing rows to JSONL (plan 7).

Crash safety: before each candidate runs, its key is written to `inflight.json`. If the process dies
(e.g. an illegal memory access poisons the CUDA context), the next run finds the marker, records that
candidate as invalid ("crashed_previous_run") and moves on. Completed rows are skipped on resume.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch

from tune.correctness import placed_inputs, reference, validate
from tune.env import provenance, write_env
from tune.profile import cuda_kernel_names
from tune.records import RecordWriter, alignment_flags, completed_keys, measurement_key
from tune.timing import TimingConfig, benchmark, pool

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
# Vendor backends whose choice of device kernel is opaque: record the kernel names they actually launch.
KERNEL_NAME_FAMILIES = {"torch_matmul", "cublaslt", "cublaslt_deep"}
LAYOUT = "NN"


@dataclass
class Candidate:
    family: str
    config_key: str
    config: dict
    backend: object  # has make_runner(a, b, out=None) -> Callable[[], Tensor]


def candidates_for(spec: str, M: int, N: int, K: int, prune_shapes: bool = False,
                   dtype: torch.dtype = torch.float16) -> list[Candidate]:
    """Backend spec strings: torch | triton_tutorial | triton:handpicked | triton:sweep[:k] | triton:<config_key> |
    cublaslt[:N] | cublaslt_deep[:budget_s].

    triton:sweep[:k] expands to the k fastest configs (default 8) of the stage-1 screen for this shape.

    cublaslt[:N] expands to cuBLASLt's top-N heuristic candidates for this shape (default 16; it often returns fewer).
    Rank 0 is B2; the per-shape best over all of them is B3.
    cublaslt_deep[:budget_s] runs an enumerate-and-screen search over cuBLASLt's configuration space (in a subprocess,
    cached in data/<hw>/lt_search/) and expands to the 8 fastest configs; the best over cublaslt and cublaslt_deep is B3+.
    """
    from baselines.torch_mm import TorchMatmul
    from baselines.triton_tutorial import TritonTutorialMatmul
    from kernels.config_space import HANDPICKED, shape_violations
    from kernels.triton_matmul import TritonMatmul

    if spec == "torch":
        return [Candidate(TorchMatmul.family, "default", {}, TorchMatmul())]
    if spec == "cublaslt_deep" or spec.startswith("cublaslt_deep:"):
        from baselines.cublaslt import deep_candidates
        budget = float(spec.split(":", 1)[1]) if ":" in spec else 20.0
        return [Candidate(c.family, c.key, c.cfg, c) for c in deep_candidates(M, N, K, dtype, budget_s=budget)]
    if spec == "cublaslt" or spec.startswith("cublaslt:"):
        from baselines.cublaslt import candidates as lt_candidates
        n = int(spec.split(":", 1)[1]) if ":" in spec else 16
        cands = lt_candidates(M, N, K, dtype, n)
        return [Candidate(c.family, c.key, {**c.cfg, "n_requested": n, "n_returned": len(cands)}, c) for c in cands]
    if spec == "triton_tutorial":
        b = TritonTutorialMatmul()
        return [Candidate(b.family, "default", b.cfg, b)]
    if spec.startswith("triton:"):
        which = spec.split(":", 1)[1]
        extra: dict[str, dict] = {}
        if which == "handpicked":
            cfgs = list(HANDPICKED.values())
        elif which == "sweep" or which.startswith("sweep:"):
            # The k fastest configs from the stage-1 screen (tune/sweep.py) for this shape.
            from tune.sweep import top_configs
            k = int(which.split(":", 1)[1]) if ":" in which else 8
            sweep_dir = os.environ.get("TUNE_SWEEP_DIR")
            if not sweep_dir:
                raise RuntimeError("triton:sweep needs the stage-1 screen directory (TUNE_SWEEP_DIR)")
            tops = top_configs(Path(sweep_dir), M, N, K, k)
            cfgs = [t["cfg"] for t in tops]
            extra = {t["cfg"].key(): {"screen_us": t["screen_us"], "screen_rank": t["screen_rank"]} for t in tops}
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
            out.append(Candidate(TritonMatmul.family, c.key(), {**c.to_dict(), **extra.get(c.key(), {})},
                                 TritonMatmul(c)))
        return out
    raise ValueError(f"unknown backend spec {spec!r}")


def load_suite(name: str, version: str = "v1") -> list[dict]:
    path = Path(__file__).resolve().parent.parent / "suites" / version / f"{name}.json"
    return json.loads(path.read_text())["shapes"]


def run(shapes: Iterable[dict], backend_specs: list[str], out_dir: Path, *, dtype: str = "fp16",
        cache_modes=("cold",), timing_overrides: dict | None = None, budget_s: float | None = None,
        session_id: str | None = None, prune_shapes: bool = False, skip_out_of_range: bool = True,
        n_allocs: int = 1,
        out_name: str = "results.jsonl", log: Callable[[str], None] = print) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    session_id = session_id or time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    env = write_env(out_dir, session_id=session_id, kind="benchmark")
    os.environ["TUNE_LT_SEARCH_DIR"] = str(out_dir / "lt_search")  # deep cuBLASLt search cache lives with the run
    os.environ.setdefault("TUNE_SWEEP_DIR", str(out_dir / "sweep_screen"))  # stage-1 Triton screen results
    results = out_dir / out_name
    inflight = out_dir / "inflight.json"
    done = completed_keys(results)
    tdtype = DTYPES[dtype]
    shapes = [s for s in shapes if not (skip_out_of_range and s.get("out_of_range"))]

    base_row = {"hw_id": env["hw_id"], "driver": env["driver"], "cuda": env["versions"]["torch_cuda"],
                "torch": env["versions"]["torch"], "triton": env["versions"]["triton"],
                "dtype": dtype, "layout": LAYOUT, "session_id": session_id, **provenance(env)}

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
            cands, seen = [], set()
            for c in (c for spec in backend_specs for c in candidates_for(spec, M, N, K, prune_shapes, tdtype)):
                if (c.family, c.config_key) not in seen:  # e.g. a hand-picked config that is also in the sweep top-k
                    seen.add((c.family, c.config_key))
                    cands.append(c)
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
                a, b, pad = placed_inputs(M, N, K, tdtype, seed=0)
                ref = reference(a, b)
            except torch.OutOfMemoryError:
                for c, cm, row in todo:
                    w.write({**row, "valid": False, "failure_reason": "oom_inputs", "timestamp": time.time()})
                torch.cuda.empty_cache()
                continue

            # Validate every candidate once, then time the valid ones on n_allocs input placements.
            validity: dict[tuple, tuple] = {}  # keyed by (family, config_key): "default" is shared across families
            timed: dict[int, list] = {}
            rows: dict[int, dict] = {}
            for j, (c, cm, row) in enumerate(todo):
                vkey = (c.family, c.config_key)
                if vkey not in validity:
                    inflight.write_text(json.dumps({"key": list(measurement_key(row)), "row": row}, default=str))
                    chk = validate(c.backend, M, N, K, tdtype, ref=ref, inputs=(a, b))
                    kernels = None
                    if chk.valid and c.family in KERNEL_NAME_FAMILIES:
                        kernels = cuda_kernel_names(c.backend.make_runner(a, b))
                    validity[vkey] = (chk.valid, chk.failure_reason, chk.max_abs_err, kernels)
                    inflight.unlink()
                valid, reason, err, kernels = validity[vkey]
                rows[j] = {**row, "valid": valid, "failure_reason": reason, "max_abs_err": err, "n_allocs": n_allocs,
                           "cuda_kernels": kernels}
                if valid:
                    timed[j] = []
            del ref
            for alloc in range(n_allocs):
                if alloc:
                    del a, b, pad
                    a, b, pad = placed_inputs(M, N, K, tdtype, seed=0, placement=alloc)
                for j in list(timed):
                    c, cm, _ = todo[j]
                    inflight.write_text(json.dumps({"key": list(measurement_key(rows[j])), "row": rows[j]}, default=str))
                    try:
                        cfg = TimingConfig(cache_mode=cm, **(timing_overrides or {}))
                        timed[j].append(benchmark(c.backend.make_runner(a, b), cfg))
                    except torch.OutOfMemoryError:
                        rows[j].update(valid=False, failure_reason="oom_timing")
                        del timed[j]
                        torch.cuda.empty_cache()
                    inflight.unlink()

            for j, row in rows.items():
                if j not in timed:
                    w.write({**row, "timestamp": time.time()})
                    continue
                r = pool(timed[j])
                tel = last_tel = r.telemetry_after
                w.write({**row, "time_median_us": r.median_us, "time_p10_us": r.p10_us, "time_p90_us": r.p90_us,
                         "time_min_us": r.min_us, "time_max_us": r.max_us, "time_iqr_us": r.iqr_us,
                         "rse": r.rse, "n_samples": r.n_samples, "converged": r.converged,
                         "alloc_medians_us": [x.median_us for x in timed[j]],
                         "tflops": 2 * M * N * K / (r.median_us * 1e-6) / 1e12,
                         "clock_mhz": tel.get("clock_mhz"), "temp_c": tel.get("temp_c"),
                         "power_w": tel.get("power_w"), "throttle_reasons": tel.get("throttle_reasons"),
                         "clock_mhz_before": r.telemetry_before.get("clock_mhz"),
                         "timestamp": time.time()}, samples_us=r.samples_us)

            del a, b, pad
            torch.cuda.empty_cache()
            elapsed = time.perf_counter() - t0
            eta = elapsed / (i + 1) * (len(shapes) - i - 1)
            log(f"[{i + 1}/{len(shapes)}] {M}x{N}x{K} {len(todo)} meas | {elapsed:.0f}s elapsed, ETA {eta:.0f}s"
                f" | {last_tel.get('clock_mhz')} MHz {last_tel.get('temp_c')}C {last_tel.get('power_w')}W")
    return results
