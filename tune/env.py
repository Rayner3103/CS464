"""Environment capture: hardware fingerprint, library versions, clock state (plan 3.2, 7.5).

Every run writes an env.json next to its results. Results without one are invalid.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import re
import socket
import subprocess
from pathlib import Path

import torch

try:
    import pynvml
except ImportError:  # pragma: no cover - telemetry is optional, but logged as missing
    pynvml = None

REPO_ROOT = Path(__file__).resolve().parent.parent

# Spec boost clocks (MHz) for computing the spec roofline. Measured peaks live in baselines/roofline.py.
_SPEC_BOOST_MHZ = {"NVIDIA GeForce RTX 3080": 1710}
# Dense tensor-core FLOP/clk/SM for fp16 inputs with fp32 accumulate. GeForce Ampere runs fp32-accumulate
# HMMA at half the fp16-accumulate rate (512 vs 1024 FLOP/clk/SM); A100 is 1024 for both.
_FP16_F32ACC_FLOP_PER_CLK_PER_SM = {(8, 6): 512, (8, 9): 512, (8, 0): 1024, (9, 0): 2048}


def _nvml_handle(device: int = 0):
    if pynvml is None:
        return None
    try:
        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetHandleByIndex(device)
    except pynvml.NVMLError:
        return None


def _safe(fn, *args):
    try:
        v = fn(*args)
        return v.decode() if isinstance(v, bytes) else v
    except Exception:
        return None


def hw_id(device: int = 0) -> str:
    """Stable key for the tuning database: GPU model, compute capability, SM count, memory."""
    p = torch.cuda.get_device_properties(device)
    name = re.sub(r"[^a-z0-9]+", "", p.name.lower().replace("nvidia", "").replace("geforce", ""))
    mem_gb = round(p.total_memory / 2**30)
    return f"{name}_sm{p.major}{p.minor}_{p.multi_processor_count}sms_{mem_gb}gb"


def gpu_telemetry(device: int = 0) -> dict:
    """Point-in-time clock / temperature / power reading. Cheap enough to call per measurement."""
    h = _nvml_handle(device)
    if h is None:
        return {"clock_mhz": None, "mem_clock_mhz": None, "temp_c": None, "power_w": None, "throttle_reasons": None}
    power_mw = _safe(pynvml.nvmlDeviceGetPowerUsage, h)
    return {
        "clock_mhz": _safe(pynvml.nvmlDeviceGetClockInfo, h, pynvml.NVML_CLOCK_SM),
        "mem_clock_mhz": _safe(pynvml.nvmlDeviceGetClockInfo, h, pynvml.NVML_CLOCK_MEM),
        "temp_c": _safe(pynvml.nvmlDeviceGetTemperature, h, pynvml.NVML_TEMPERATURE_GPU),
        "power_w": power_mw / 1000 if power_mw is not None else None,
        "throttle_reasons": _safe(pynvml.nvmlDeviceGetCurrentClocksThrottleReasons, h),
    }


def gpu_processes(device: int = 0) -> list[dict]:
    h = _nvml_handle(device)
    if h is None:
        return []
    procs = []
    for getter in ("nvmlDeviceGetComputeRunningProcesses", "nvmlDeviceGetGraphicsRunningProcesses"):
        for p in _safe(getattr(pynvml, getter), h) or []:
            procs.append({"kind": getter.split("Get")[1].split("Running")[0].lower(), "pid": p.pid,
                          "used_mem_mb": (p.usedGpuMemory or 0) / 2**20})
    return procs


def _git_state() -> dict:
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return None
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status) if status is not None else None}


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def spec_peaks(device: int = 0) -> dict:
    """Datasheet-derived roofline (B0, spec half). Measured peaks are added by baselines/roofline.py."""
    p = torch.cuda.get_device_properties(device)
    boost = _SPEC_BOOST_MHZ.get(p.name)
    per_clk = _FP16_F32ACC_FLOP_PER_CLK_PER_SM.get((p.major, p.minor))
    h = _nvml_handle(device)
    max_sm_mhz = _safe(pynvml.nvmlDeviceGetMaxClockInfo, h, pynvml.NVML_CLOCK_SM) if h else None
    out = {"fp16_f32acc_flop_per_clk_per_sm": per_clk, "spec_boost_mhz": boost, "max_sm_mhz": max_sm_mhz}
    if per_clk and boost:
        out["fp16_f32acc_tflops_at_boost"] = p.multi_processor_count * boost * 1e6 * per_clk / 1e12
    if per_clk and max_sm_mhz:
        out["fp16_f32acc_tflops_at_max_clock"] = p.multi_processor_count * max_sm_mhz * 1e6 * per_clk / 1e12
    # Memory bandwidth = bus width (bits) * memory clock * 2 (DDR) ; GDDR6X reports the effective-rate/2 clock.
    if p.memory_bus_width and p.memory_clock_rate:
        out["dram_gbps_spec"] = p.memory_bus_width / 8 * p.memory_clock_rate * 1e3 * 2 / 1e9
    return out


def capture_env(device: int = 0) -> dict:
    p = torch.cuda.get_device_properties(device)
    h = _nvml_handle(device)
    import numpy
    import triton

    env = {
        "timestamp": _dt.datetime.now().astimezone().isoformat(),
        "hw_id": hw_id(device),
        "gpu": {
            "name": p.name, "uuid": str(p.uuid), "compute_capability": f"{p.major}.{p.minor}",
            "sm_count": p.multi_processor_count, "total_memory_bytes": p.total_memory,
            "l2_cache_bytes": p.L2_cache_size, "memory_bus_width_bits": p.memory_bus_width,
            "memory_clock_khz": p.memory_clock_rate, "regs_per_sm": p.regs_per_multiprocessor,
            "smem_per_block_optin_bytes": p.shared_memory_per_block_optin,
            "smem_per_sm_bytes": p.shared_memory_per_multiprocessor,
            "max_threads_per_sm": p.max_threads_per_multi_processor,
        },
        "driver": _safe(pynvml.nvmlSystemGetDriverVersion) if h else None,
        "cuda_driver_api": _safe(pynvml.nvmlSystemGetCudaDriverVersion) if h else None,
        "versions": {
            "python": platform.python_version(), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "triton": triton.__version__, "numpy": numpy.__version__,
        },
        "torch_flags": {
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "matmul_allow_fp16_reduced_precision_reduction":
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        },
        "clocks": {
            "max_sm_mhz": _safe(pynvml.nvmlDeviceGetMaxClockInfo, h, pynvml.NVML_CLOCK_SM) if h else None,
            "max_mem_mhz": _safe(pynvml.nvmlDeviceGetMaxClockInfo, h, pynvml.NVML_CLOCK_MEM) if h else None,
            # Clock locking (nvidia-smi -lgc) needs root; we record rather than lock. See docs/methodology.md.
            "locked": False,
        },
        "power_limit_w": (_safe(pynvml.nvmlDeviceGetPowerManagementLimit, h) or 0) / 1000 if h else None,
        "persistence_mode": _safe(pynvml.nvmlDeviceGetPersistenceMode, h) if h else None,
        "ecc_mode": _safe(pynvml.nvmlDeviceGetEccMode, h) if h else None,
        "display_active": _safe(pynvml.nvmlDeviceGetDisplayActive, h) if h else None,
        "telemetry_at_capture": gpu_telemetry(device),
        "gpu_processes": gpu_processes(device),
        "spec_peaks": spec_peaks(device),
        "host": {"hostname": socket.gethostname(), "cpu": _cpu_model(), "os": platform.platform(),
                 "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None},
        "harness_git": _git_state(),
    }
    return env


def write_env(out_dir: Path, device: int = 0, extra: dict | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = capture_env(device)
    if extra:
        env.update(extra)
    (out_dir / "env.json").write_text(json.dumps(env, indent=2, default=str))
    return env


def default_data_dir(device: int = 0) -> Path:
    return REPO_ROOT / "data" / hw_id(device)


def pin_cpu(cores: list[int] | None = None) -> list[int] | None:
    """Pin this process to a fixed core so the launch thread does not migrate mid-measurement (plan 3.2)."""
    if not hasattr(os, "sched_setaffinity"):
        return None
    cores = cores if cores is not None else [max(os.sched_getaffinity(0))]
    os.sched_setaffinity(0, cores)
    return cores
