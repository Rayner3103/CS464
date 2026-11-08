"""`lscli doctor` (spec section 6): checks that say why lscli is not doing anything, each OK / WARN / FAIL with a hint.
Nothing here changes the system."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from lscli import config, control, envs, gate, models
from lscli.files import read_json

HELPER = Path("/usr/local/libexec/lscli-clock")


def _run(*argv: str, timeout: float = 30) -> tuple[int, str]:
    try:
        r = subprocess.run(list(argv), capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
        return r.returncode, (r.stdout + r.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return 127, str(e)


_HEADERS = ("import os, sys, sysconfig; print(os.path.exists(os.path.join(sysconfig.get_paths()['include'], 'Python.h')))")


def checks(paths) -> list[dict]:
    out = []

    def add(status, name, detail, hint=""):
        out.append({"status": status, "check": name, "detail": detail, "hint": hint})

    ident = gate.gpu_identity()
    if ident.get("uuid"):
        add("OK", "GPU", f"{ident['name']} ({ident['uuid']})" + (f"; {ident['count']} GPUs, device 0 used" if ident["count"] > 1 else ""))
    else:
        add("FAIL", "GPU", ident.get("error", "NVML unavailable"), "install the NVIDIA driver; check nvidia-smi")
    rc, txt = _run("systemctl", "--user", "show-environment")
    add("OK" if rc == 0 else "FAIL", "systemd user session", "reachable" if rc == 0 else txt[-200:],
        "" if rc == 0 else "lscli needs a systemd --user session")
    rc, txt = _run("systemctl", "--user", "is-active", "lscli.service")
    add("OK" if txt.strip() == "active" else "WARN", "daemon unit", txt.strip() or "unknown",
        "" if txt.strip() == "active" else "systemctl --user enable --now lscli.service (lscli daemon start)")
    rc, txt = _run("loginctl", "show-user", os.environ.get("USER", ""), "-p", "Linger")
    linger = txt.strip().endswith("=yes")
    add("OK" if linger else "WARN", "linger", "on" if linger else "off",
        "" if linger else "the daemon stops at logout: loginctl enable-linger $USER")
    st = control.read_state(paths)
    if control.daemon_alive(st):
        g = st.get("gate") or {}
        add("OK", "daemon", f"{st.get('state')}; gate " + ("open" if g.get("open") else "closed: " + "; ".join(g.get("reasons") or [])))
    else:
        add("WARN", "daemon", "not running", "lscli daemon start")
    snap = gate.gpu_snapshot()
    own = {int(st["pid"])} if st and st.get("pid") else set()
    holders = [f"{p} ({snap.names.get(p, '?')})" for p in snap.pids if p not in own]
    add("OK" if not holders else "WARN", "CUDA contexts", ", ".join(holders) if holders else "none",
        "these processes keep the idle gate closed while they hold a CUDA context (an idle Jupyter kernel counts)" if holders else "")
    cfg = config.load()
    mhz = cfg["clock"]["lock_mhz"]
    if HELPER.exists():
        rc, txt = _run("sudo", "-n", "-l", str(HELPER), "reset")
        conf = Path("/etc/lscli/clock.conf")
        conf_mhz = conf.read_text().strip().removeprefix("lock_mhz=") if conf.exists() else None
        ok = rc == 0 and mhz and str(mhz) == conf_mhz
        add("OK" if ok else "WARN", "clock helper", f"installed; sudo {'allowed' if rc == 0 else 'NOT allowed'}; "
            f"config {mhz or 'unlocked'} MHz, clock.conf {conf_mhz}",
            "" if ok else "re-run the installer with --clock-helper, or set clock.lock_mhz to clock.conf's value")
    else:
        add("OK" if not mhz else "FAIL", "clock helper", "not installed" + (f" but clock.lock_mhz = {mhz}" if mhz else ""),
            "" if not mhz else "lscli config clock.lock_mhz 0, or install the helper")
    mode = paths.root.stat().st_mode & 0o777 if paths.root.exists() else None
    add("OK" if mode == 0o700 else "WARN", "home permissions", f"{paths.root} {oct(mode) if mode is not None else 'missing'}",
        "" if mode == 0o700 else f"chmod 700 {paths.root} (it holds your usage record)")
    hw = (st or {}).get("hw_id")
    ms = models.active(paths, hw)
    if ms is None:
        add("WARN", "models", "none yet", "the daemon trains this GPU's models in idle time (lscli train status)")
    else:
        add("OK", "models", f"{ms.source}: " + ", ".join(f"{k} {'yes' if v else 'no'}" for k, v in ms.stems.items()))
    registered = envs.load(paths)
    if not registered:
        add("WARN", "envs", "none registered", "lscli enable-env $(command -v python3)")
    for eid, rec in registered.items():
        py = rec.get("python", "?")
        try:
            info = envs.probe(py)
        except RuntimeError as e:
            add("FAIL", f"env {eid}", f"{py}: {e}", "lscli disable-env, or fix the interpreter")
            continue
        rc, txt = _run(py, "-c", _HEADERS)
        problems = [x for x, bad in (("no torch", not info.get("torch")), ("no triton", not info.get("triton")),
                                     ("no Python.h (Triton launcher)", txt.strip() != "True"),
                                     ("no gcc", shutil.which("gcc") is None)) if bad]
        site_ok = False
        rc, site = _run(py, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])")
        if rc == 0:
            site_ok = (Path(site.strip().splitlines()[-1]) / "lscli.pth").exists()
        if not site_ok:
            problems.append("lscli.pth missing")
        add("OK" if not problems else "FAIL", f"env {eid}",
            f"{py}: torch {info.get('torch')}, triton {info.get('triton')}" + (f" ({'; '.join(problems)})" if problems else ""),
            "" if not problems else f"lscli enable-env {py}")
    return out


def report(rows: list[dict]) -> str:
    w = max(len(r["check"]) for r in rows)
    lines = []
    for r in rows:
        lines.append(f"{r['status']:4} {r['check']:<{w}}  {r['detail']}")
        if r["hint"]:
            lines.append(f"{'':4} {'':<{w}}  -> {r['hint']}")
    return "\n".join(lines)


def exit_code(rows: list[dict]) -> int:
    return 1 if any(r["status"] == "FAIL" for r in rows) else 0


if __name__ == "__main__":
    from lscli.paths import Paths
    rows = checks(Paths())
    print(json.dumps(rows, indent=1))
