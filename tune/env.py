"""Environment capture: hardware fingerprint, library versions, clock state (plan 3.2, 7.5).

Every run (session) writes data/<hw_id>/env/<session_id>.json and refreshes env.json (= latest session).
Every result row carries the session_id plus a few provenance fields (see `provenance`), so rows can be
traced back to the exact code and clock state they were measured under. Results without these are invalid.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
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


def probe_clock_lock(device: int = 0, declared: str | None = None) -> dict:
    """Infer whether SM clocks are locked (nvidia-smi -lgc), since NVML cannot report it directly.

    Signals:
      * idle: an unlocked GPU drops to a low idle clock (~210 MHz on the RTX 3080); a locked one stays near
        the locked frequency even with the GpuIdle reason set.
      * light load: a device-side spin draws little power, so an unlocked GPU boosts near its max clock.
        A clock that stays well under max with no power/thermal limit active means it is capped.
    `declared` (or env var TUNE_CLOCK_LOCK, e.g. "1755" or "none") is the user's statement and takes priority
    for the reported state; the probe is still recorded so the two can be cross-checked.
    """
    declared = declared if declared is not None else os.environ.get("TUNE_CLOCK_LOCK")
    h = _nvml_handle(device)
    out: dict = {"declared": declared}
    if h is None:
        out["state"] = f"declared:{declared}" if declared else "unknown"
        return out
    max_mhz = _safe(pynvml.nvmlDeviceGetMaxClockInfo, h, pynvml.NVML_CLOCK_SM)

    torch.cuda.synchronize(device)
    time.sleep(0.5)
    idle_clock = _safe(pynvml.nvmlDeviceGetClockInfo, h, pynvml.NVML_CLOCK_SM)
    idle_reasons = _safe(pynvml.nvmlDeviceGetCurrentClocksThrottleReasons, h) or 0
    is_idle = bool(idle_reasons & pynvml.nvmlClocksThrottleReasonGpuIdle)

    clocks, reasons = [], 0
    with torch.cuda.device(device):
        torch.cuda._sleep(int(1.5e9 * 0.8))  # ~0.6-0.8 s of spin at 1.5-2 GHz
        t_end = time.perf_counter() + 0.5
        time.sleep(0.1)
        while time.perf_counter() < t_end:
            clocks.append(_safe(pynvml.nvmlDeviceGetClockInfo, h, pynvml.NVML_CLOCK_SM))
            reasons |= _safe(pynvml.nvmlDeviceGetCurrentClocksThrottleReasons, h) or 0
            time.sleep(0.05)
        torch.cuda.synchronize(device)
    clocks = [c for c in clocks if c]
    limit_bits = (pynvml.nvmlClocksThrottleReasonSwPowerCap | pynvml.nvmlClocksThrottleReasonSwThermalSlowdown
                  | pynvml.nvmlClocksThrottleReasonHwSlowdown)
    load_max = max(clocks) if clocks else None
    idle_high = is_idle and idle_clock is not None and max_mhz and idle_clock >= 0.5 * max_mhz
    load_capped = (load_max is not None and max_mhz and load_max <= 0.88 * max_mhz and not (reasons & limit_bits))
    probed = "likely_locked" if (idle_high or load_capped) else "likely_unlocked"
    out.update({"probe": probed, "idle_clock_mhz": idle_clock, "idle_reasons": idle_reasons,
                "load_clock_mhz_min": min(clocks) if clocks else None, "load_clock_mhz_max": load_max,
                "load_reasons": reasons, "max_sm_mhz": max_mhz,
                "inferred_lock_mhz": load_max if probed == "likely_locked" else None})
    if declared:
        d = declared.strip().lower()
        out["state"] = "unlocked" if d in ("none", "off", "0", "unlocked") else "locked"
        out["lock_mhz"] = None if out["state"] == "unlocked" else d
        out["declared_matches_probe"] = (out["state"] == "locked") == (probed == "likely_locked")
    else:
        out["state"] = probed
        out["lock_mhz"] = out["inferred_lock_mhz"]
    return out


def capture_env(device: int = 0, probe_clocks: bool = True) -> dict:
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
        },
        # nvidia-smi -lgc needs root and NVML has no getter for locked clocks, so the state is probed
        # (or declared by the user). See probe_clock_lock and docs/methodology.md.
        "clock_lock": probe_clock_lock(device) if probe_clocks else {"state": "not_probed"},
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


def write_env(out_dir: Path, device: int = 0, extra: dict | None = None, session_id: str | None = None,
              kind: str = "env") -> dict:
    """Capture the environment for one session: writes env/<session_id>.json and refreshes env.json."""
    session_id = session_id or f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    env = capture_env(device)
    env.update({"session_id": session_id, "kind": kind, "argv": sys.argv})
    if extra:
        env.update(extra)
    text = json.dumps(env, indent=2, default=str)
    (out_dir / "env").mkdir(exist_ok=True)
    (out_dir / "env" / f"{session_id}.json").write_text(text)
    (out_dir / "env.json").write_text(text)
    if env["clock_lock"].get("declared_matches_probe") is False:
        print(f"[env] WARNING: declared clock lock {env['clock_lock']['declared']!r} disagrees with the probe "
              f"({env['clock_lock']['probe']}, idle {env['clock_lock']['idle_clock_mhz']} MHz, "
              f"load max {env['clock_lock']['load_clock_mhz_max']} MHz)", file=sys.stderr)
    return env


def provenance(env: dict) -> dict:
    """Small per-row subset of the session env, so a row is traceable without joining files."""
    git, lock = env.get("harness_git") or {}, env.get("clock_lock") or {}
    return {"harness_commit": (git.get("commit") or "")[:12], "harness_dirty": git.get("dirty"),
            "clock_lock_state": lock.get("state"), "clock_lock_mhz": lock.get("lock_mhz"),
            "env_file": f"env/{env.get('session_id')}.json"}


def default_data_dir(device: int = 0) -> Path:
    return REPO_ROOT / "data" / hw_id(device)


def pin_cpu(cores: list[int] | None = None) -> list[int] | None:
    """Pin this process to a fixed core so the launch thread does not migrate mid-measurement (plan 3.2)."""
    if not hasattr(os, "sched_setaffinity"):
        return None
    cores = cores if cores is not None else [max(os.sched_getaffinity(0))]
    os.sched_setaffinity(0, cores)
    return cores
