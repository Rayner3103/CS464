"""The runtime guard (spec 5.1): lscli must never run with another tree's harness. A fake checkout that only holds a
`tune` package is put first on PYTHONPATH, so `lscli` still comes from this repo while `tune` would not."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from lscli import envs, gate, guard, jobs
from lscli.paths import Paths

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def checkout(tmp_path):
    fake = tmp_path / "checkout"
    (fake / "tune").mkdir(parents=True)
    (fake / "tune" / "__init__.py").write_text("raise ImportError('the fake checkout was imported')\n")
    return fake


def _run(code_or_args, env_extra, cwd):
    env = {**os.environ, **env_extra}
    argv = [sys.executable, *code_or_args] if isinstance(code_or_args, list) else [sys.executable, "-c", code_or_args]
    return subprocess.run(argv, env=env, cwd=cwd, capture_output=True, text=True, timeout=300, stdin=subprocess.DEVNULL)


def test_guard_finds_a_harness_package_outside_lscli_s_tree(checkout, tmp_path):
    code = "import json; from lscli import guard; print(json.dumps(guard.mismatches()))"
    ok = _run(code, {"PYTHONPATH": str(REPO)}, tmp_path)
    assert json.loads(ok.stdout) == [], ok.stderr
    bad = _run(code, {"PYTHONPATH": f"{checkout}{os.pathsep}{REPO}"}, tmp_path)
    (m,) = json.loads(bad.stdout)
    assert m["package"] == "tune" and m["found"] == str((checkout / "tune").resolve())
    assert m["expected"] == str(REPO.resolve() / "tune")
    assert guard.describe([m]).startswith(guard.PREFIX)


def test_guard_reads_an_already_imported_module(monkeypatch, tmp_path):
    import types
    mod = types.ModuleType("kernels")
    mod.__file__ = str(tmp_path / "elsewhere" / "kernels" / "__init__.py")
    monkeypatch.setitem(sys.modules, "kernels", mod)
    assert [m["package"] for m in guard.mismatches()] == ["kernels"]


def test_the_worker_refuses_before_touching_cuda(checkout, tmp_path):
    r = _run(["-m", "lscli.worker", "env"], {"PYTHONPATH": f"{checkout}{os.pathsep}{REPO}"}, tmp_path)
    assert r.returncode == 1 and "the fake checkout was imported" not in r.stderr
    last = json.loads(r.stdout.strip().splitlines()[-1])
    assert last["op"] == "error" and last["error"].startswith(guard.PREFIX) and last["guard"][0]["package"] == "tune"


def test_the_client_does_not_patch_and_warns_once(checkout, tmp_path):
    code = ("import warnings, torch, lscli\n"
            "with warnings.catch_warnings(record=True) as w:\n"
            "    warnings.simplefilter('always')\n"
            "    print(lscli.enable(), lscli.enable(), torch.matmul.__module__)\n"
            "    print(sum('harness mismatch' in str(x.message) for x in w))\n")
    env = {"PYTHONPATH": f"{checkout}{os.pathsep}{REPO}", "LSCLI_HOME": str(tmp_path / "home")}
    r = _run(code, env, tmp_path)
    first, count = r.stdout.strip().splitlines()[-2:]
    assert first.split()[:2] == ["False", "False"] and "lscli" not in first.split()[2], r.stderr
    assert count == "1"                                                   # warned once, not per call


def _daemon(home):
    from lscli.daemon import Daemon
    return Daemon(home, None, snapshot=lambda: gate.GpuSnapshot([], 0, True, {}), battery=lambda: False)


def _hw_known(home, monkeypatch):
    """This GPU's hw_id is cached, so the daemon's probes are the CUDA-free ones (lscli/hwcache.py)."""
    from lscli.files import write_json
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(gate, "gpu_identity", lambda index=0: {"uuid": "GPU-t", "name": "t", "count": 1})
    write_json(home.gpu_hw, {"GPU-t": "hwX"})


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("LSCLI_HOME", str(tmp_path / "home"))
    cfg = tmp_path / "c.toml"
    cfg.write_text('[idle]\nwindow = ""\ngpu_idle_minutes = 0\nrequire_ac = false\n')
    monkeypatch.setenv("LSCLI_CONFIG", str(cfg))
    return Paths().ensure()


def test_the_daemon_blocks_a_refused_env_until_it_is_registered_again(home, monkeypatch, capsys):
    """No retry loop (spec 5.1): a probe the guard refused is not repeated until the env's record changes or the daemon
    restarts; status and doctor name the env and the package."""
    from lscli import cli, control
    eid, _ = envs.register(home, sys.executable, {"torch": "x"})
    d = _daemon(home)
    d.hw_id = "hwX"                                                       # env probes need a known hw_id
    probes = []
    msg = f"{guard.PREFIX}: tune resolves to /x/tune, not lscli's copy /src/tune"

    def worker_env(python, extra, hw_id=None):
        probes.append(python)
        d.probe_error = msg
        return None
    monkeypatch.setattr(d, "_worker_env", worker_env)
    d.clock = lambda: 1e9
    assert d.caches() == {} and eid in d.unknown_envs and len(probes) == 1
    d.clock = lambda: 1e9 + 10 * 86400                                    # long after any retry interval
    assert d.caches() == {} and len(probes) == 1                          # blocked: never probed again
    d.write_state()
    assert json.loads(home.state.read_text())["env_errors"] == {eid: msg}
    monkeypatch.setattr(control, "daemon_alive", lambda st: True)
    assert cli.main(["status"]) == 0 and f"env {eid}: blocked: {guard.PREFIX}" in capsys.readouterr().out

    envs.register(home, sys.executable, {"torch": "x"})                   # `lscli enable-env` again
    d.caches()
    assert len(probes) == 2


def test_a_search_the_guard_refused_blocks_its_env(home):
    from lscli import scheduler
    eid, _ = envs.register(home, sys.executable, {"torch": "x"})
    d = _daemon(home)

    def refused(task):
        raise jobs.JobError(f"{guard.PREFIX}: tune resolves to /x/tune, not lscli's copy /src/tune")
    d.search = refused
    d.run_task(scheduler.Task(kind="search", M=64, N=64, K=64, env_id=eid))
    assert eid in d.env_blocked and d.env_blocked[eid][1].startswith(guard.PREFIX)


def test_a_probe_that_times_out_after_a_refused_one_does_not_inherit_its_error(home, monkeypatch):
    """The guard message belongs to the probe that printed it: a later probe failing another way is not blocked."""
    from lscli import daemon as dmod
    eid, _ = envs.register(home, sys.executable, {"torch": "x"})
    _hw_known(home, monkeypatch)                                          # resolve_hw probes (CUDA-free)
    d = _daemon(home)
    msg = f"{guard.PREFIX}: tune resolves to /x/tune, not lscli's copy /src/tune"
    runs = []

    def run(argv, **kw):
        runs.append(argv)
        if len(runs) == 1:                                                # the daemon venv's own probe: refused
            return subprocess.CompletedProcess(argv, 1, json.dumps({"op": "error", "error": msg}) + "\n", "")
        raise subprocess.TimeoutExpired(argv, 300)                        # then the env's probe hangs
    monkeypatch.setattr(dmod.subprocess, "run", run)
    assert d.resolve_hw() is False and d.env_blocked["daemon"][1] == msg
    assert d.resolve_hw() is False and len(runs) == 1                      # the daemon venv is not probed again
    d.hw_id = "hwX"                                                       # env probes need a known hw_id
    d.caches()
    assert len(runs) == 2 and eid not in d.env_blocked and eid in d.unknown_envs   # failed, retried later, not blocked


def test_a_gated_probe_the_guard_refused_is_not_retried(home, monkeypatch):
    """The first probe on a GPU needs CUDA and runs with the gate open; a guard refusal there blocks the daemon venv
    (or the env) like a CUDA-free one: no probe every tick."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(gate, "gpu_identity", lambda index=0: {"uuid": "GPU-t", "name": "t", "count": 1})
    d = _daemon(home)
    msg = f"{guard.PREFIX}: tune resolves to /x/tune, not lscli's copy /src/tune"
    probes = []

    def worker_env(python, extra, hw_id=None):
        probes.append(hw_id)
        d.probe_error = msg
        return None
    monkeypatch.setattr(d, "_worker_env", worker_env)
    d.start()
    for _ in range(3):
        d.tick()
    assert probes == [None] and "daemon" in d.env_blocked and not d.need_cuda and d.hw_id is None
