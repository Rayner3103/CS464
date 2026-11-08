"""lscli client (lscli/client.py): the torch patch, dispatch rule, stale verification, queueing, usage counts, autograd,
fallback on failure, the start-up hook and enable-env. GPU tests use a hand-written winner entry so the dispatch
decision is deterministic; the kernels themselves are real."""
import json
import os
import subprocess
import sys
import time
import venv
from pathlib import Path

import pytest
import torch

from lscli import client, envs
from lscli import search_queue as queue
from lscli.paths import Paths
from tests.conftest import requires_cuda

REPO = Path(__file__).resolve().parent.parent
KEY = "bm64_bn64_bk32_w4_s3_g1_sk1"
SETTINGS = {"min_kernel_us": 50, "min_saving_us": 5}


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("LSCLI_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LSCLI_CONFIG", str(tmp_path / "c.toml"))
    monkeypatch.delenv("LSCLI_DISABLE", raising=False)
    yield Paths().ensure()
    client.disable()


def _put(paths, M, N, K, *, key=KEY, family="triton", torch_us=1000.0, winner_us=500.0, env_extra=None):
    from lscli.worker import environment
    from tune.winner_cache import WinnerCache
    env = environment({"clock_lock": "unlocked", "env_id": envs.env_id(sys.executable), **(env_extra or {})})
    WinnerCache(paths.winners, env=env).put(M, N, K, "fp16", k=3, lt_n=16,
                                            choice={"key": key, "family": family, "mode": "cold"},
                                            candidates=[{"config_key": key, "cold_us": winner_us}],
                                            extra={"torch_us": torch_us})


def test_dispatch_rule():
    assert client.dispatch_rule({"torch_us": 100.0, "winner_us": 90.0}, SETTINGS)
    assert not client.dispatch_rule({"torch_us": 40.0, "winner_us": 10.0}, SETTINGS)        # kernel too short
    assert not client.dispatch_rule({"torch_us": 1000.0, "winner_us": 985.0}, SETTINGS)     # within 2 %
    assert not client.dispatch_rule({"torch_us": 100.0, "winner_us": 96.0}, SETTINGS)       # < 5 us
    assert not client.dispatch_rule({"winner_us": 1.0}, SETTINGS)                           # no torch time: never


@requires_cuda
def test_hit_dispatches_the_winner_and_counts(home, monkeypatch):
    _put(home, 1024, 1024, 1024)
    calls = []
    from kernels.triton_matmul import TritonMatmul
    orig = TritonMatmul.make_runner
    monkeypatch.setattr(TritonMatmul, "make_runner", lambda self, a, b, out=None: calls.append(1) or orig(self, a, b, out))
    assert client.enable()
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    ref = client._ORIG["matmul"](a, b)
    for c in (torch.matmul(a, b), a @ b, torch.mm(a, b), a.matmul(b), a.mm(b)):
        torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-1)
    assert len(calls) == 5
    client._STATE.flush()
    (row,) = [json.loads(x) for f in home.profile_dir.glob("*.jsonl") for x in f.read_text().splitlines()]
    assert row["calls"] == 5 and row["hits"] == 5 and row["layout"] == "NN"
    assert queue.pending(home)[0] == []                                             # nothing to search
    client.disable()
    assert torch.matmul is client._ORIG["matmul"] and torch.Tensor.__matmul__ is client._ORIG["t_rmatmul"]


@requires_cuda
def test_out_of_scope_calls_go_to_torch_untouched(home, monkeypatch):
    _put(home, 64, 64, 64)
    client.enable()
    monkeypatch.setattr(client._STATE, "_decide", lambda *a: pytest.fail("must not be consulted"))
    x32 = torch.randn(64, 64, device="cuda")
    h = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    for a, b in ((x32, x32), (h.t(), h), (h[None], h), (h.cpu(), h.cpu())):
        torch.matmul(a, b)
    with pytest.raises(RuntimeError, match="same dtype"):                         # torch's own error, unchanged
        torch.matmul(h, h.float())
    out = torch.empty(64, 64, device="cuda", dtype=torch.float16)
    torch.matmul(h, h, out=out)


@requires_cuda
def test_miss_is_queued_once_and_runs_torch(home):
    client.enable()
    a = torch.randn(96, 80, device="cuda", dtype=torch.float16)
    b = torch.randn(80, 72, device="cuda", dtype=torch.float16)
    for _ in range(3):
        torch.testing.assert_close(torch.matmul(a, b), client._ORIG["matmul"](a, b))
    pend, _ = queue.pending(home)
    assert [(r["M"], r["N"], r["K"], r["reason"]) for r in pend] == [(96, 72, 80, "miss")]
    assert len(home.queue.read_text().splitlines()) == 1


@requires_cuda
def test_short_kernels_and_small_savings_run_torch_but_count_as_hits(home, monkeypatch):
    _put(home, 128, 128, 128, torch_us=20.0, winner_us=5.0)
    client.enable()
    a = torch.randn(128, 128, device="cuda", dtype=torch.float16)
    from kernels.triton_matmul import TritonMatmul
    monkeypatch.setattr(TritonMatmul, "make_runner", lambda *a, **k: pytest.fail("must not dispatch"))
    torch.matmul(a, a)
    assert client._STATE.runs[(128, 128, 128, "fp16")] == (None, True)


@requires_cuda
def test_soft_stale_entry_is_verified_once_then_dispatched_and_requeued(home):
    _put(home, 1024, 1024, 512, env_extra={"clock_lock": "1755"})                  # timed under another clock policy
    client.enable()
    a = torch.randn(1024, 512, device="cuda", dtype=torch.float16)
    b = torch.randn(512, 1024, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(torch.matmul(a, b), client._ORIG["matmul"](a, b), rtol=1e-2, atol=1e-1)
    be, hit = client._STATE.runs[(1024, 1024, 512, "fp16")]
    assert be is not None and hit
    assert any(json.loads(x).get("kind") == "verified" for x in home.winners.read_text().splitlines())
    assert [r["reason"] for r in queue.pending(home)[0]] == ["stale"]


@requires_cuda
def test_autograd_backward_uses_torch(home):
    _put(home, 1024, 1024, 1024)
    client.enable()
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16, requires_grad=True)
    b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16, requires_grad=True)
    (torch.matmul(a, b).float().sum()).backward()
    ga, gb = a.grad.clone(), b.grad.clone()
    a.grad = b.grad = None
    (client._ORIG["matmul"](a, b).float().sum()).backward()
    torch.testing.assert_close(ga, a.grad)
    torch.testing.assert_close(gb, b.grad)


@requires_cuda
def test_a_failing_kernel_falls_back_and_is_disabled(home, monkeypatch):
    _put(home, 1024, 1024, 1024)
    client.enable()
    from kernels.triton_matmul import TritonMatmul

    def boom(*a, **k):
        raise RuntimeError("launch failed")
    monkeypatch.setattr(TritonMatmul, "make_runner", boom)
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    with pytest.warns(UserWarning, match="using torch"):
        torch.testing.assert_close(torch.matmul(a, a), client._ORIG["matmul"](a, a))
    assert (1024, 1024, 1024, "fp16") in client._STATE.disabled
    torch.matmul(a, a)                                                             # no second warning, still correct


@requires_cuda
def test_new_winners_are_picked_up_without_restart(home):
    client.enable()
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    torch.matmul(a, a)
    assert client._STATE.runs[(1024, 1024, 1024, "fp16")] == (None, False)
    _put(home, 1024, 1024, 1024)
    client._STATE._wc_checked = 0.0                                                # as if a second passed
    torch.matmul(a, a)
    assert client._STATE.runs[(1024, 1024, 1024, "fp16")][0] is not None


@requires_cuda
def test_linear_shapes_are_counted_as_nt_and_never_dispatched(home):
    client.enable()
    lin = torch.nn.Linear(64, 32, device="cuda", dtype=torch.float16)
    x = torch.randn(8, 4, 64, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(lin(x), torch.nn.functional.linear(x, lin.weight, lin.bias))
    client._STATE.flush()
    rows = [json.loads(x) for f in home.profile_dir.glob("*.jsonl") for x in f.read_text().splitlines()]
    assert {(r["M"], r["N"], r["K"], r["layout"]) for r in rows} == {(32, 32, 64, "NT")}
    assert queue.pending(home)[0] == []


def test_kill_switch(home, monkeypatch):
    monkeypatch.setenv("LSCLI_DISABLE", "1")
    assert client.enable() is False and not client.enabled()


@requires_cuda
def test_fork_child_starts_a_fresh_log(home):
    client.enable()
    a = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    torch.matmul(a, a)
    parent_log = client._STATE.log
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:                                       # child: no CUDA work, only check the counters were reset
        os.close(r)
        os.write(w, json.dumps({"calls": sum(client._STATE.log.calls.values()),
                                "same": client._STATE.log is parent_log}).encode())
        os._exit(0)
    os.close(w)
    os.waitpid(pid, 0)
    doc = json.loads(os.read(r, 1000))
    assert doc == {"calls": 0, "same": False} and sum(parent_log.calls.values()) == 1


@requires_cuda
def test_overhead(home, capsys):
    """Host time per call vs the unpatched function (median of 10,000; spec 5.4)."""
    import platform
    _put(home, 1024, 1024, 1024)
    a32 = torch.randn(64, 64, device="cuda")
    h = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    orig = torch.matmul

    def per_call(f, x, n=10000):
        ts = []
        for _ in range(n):
            t = time.perf_counter_ns()
            f(x, x)
            ts.append(time.perf_counter_ns() - t)
            if len(ts) % 200 == 0:
                torch.cuda.synchronize()
        ts.sort()
        return ts[len(ts) // 2] / 1e3
    base32 = per_call(orig, a32)
    client.enable()
    patched32 = per_call(torch.matmul, a32)
    be = client._STATE._decide((1024, 1024, 1024, "fp16"), torch.float16)[0]
    direct = per_call(lambda x, y: be.make_runner(x, y)(), h, 2000)
    patched_hit = per_call(torch.matmul, h, 2000)
    with capsys.disabled():
        print(f"\n[overhead] {platform.processor() or platform.machine()}: undispatched +{patched32 - base32:.2f} us "
              f"(torch {base32:.2f} us); dispatched wrapper +{patched_hit - direct:.2f} us over the kernel's own launch "
              f"({direct:.2f} us)")
    assert patched32 - base32 <= 5.0
    assert patched_hit - direct <= 15.0


# ------------------------------------------------------------------------------------------------------ hook + enable-env
def _venv_with_this_torch(root: Path) -> Path:
    """A fresh venv whose site-packages also sees this interpreter's (so torch imports without reinstalling)."""
    venv.EnvBuilder(with_pip=False).create(root)
    import site as _site
    site_dir = next((root / "lib").glob("python3*/site-packages"))
    (site_dir / "base.pth").write_text("\n".join(_site.getsitepackages()) + "\n")
    return root / "bin" / "python"


@requires_cuda
def test_enable_env_installs_the_hook(home, tmp_path, capsys):
    from lscli import cli
    py = _venv_with_this_torch(tmp_path / "venv")
    assert cli.main(["enable-env", str(py), "--no-deps"]) == 0
    site_dir = next((tmp_path / "venv" / "lib").glob("python3*/site-packages"))
    assert (site_dir / "lscli.pth").exists() and envs.env_id(py) in envs.load(home)
    code = "import torch, lscli.client as c; print(c.enabled())"
    env = {**os.environ, "LSCLI_HOME": str(home.root)}
    assert subprocess.run([str(py), "-c", code], env={**env, "LSCLI_PATCH": "1"}, capture_output=True, text=True,
                          cwd=tmp_path).stdout.strip() == "True"
    assert subprocess.run([str(py), "-c", code], env=env, capture_output=True, text=True,
                          cwd=tmp_path).stdout.strip() == "False"                 # not without LSCLI_PATCH
    assert subprocess.run([str(py), "-c", code], env={**env, "LSCLI_PATCH": "1", "LSCLI_DISABLE": "1"},
                          capture_output=True, text=True, cwd=tmp_path).stdout.strip() == "False"
    assert cli.main(["disable-env", str(py)]) == 0
    assert not (site_dir / "lscli.pth").exists() and envs.env_id(py) not in envs.load(home)


@requires_cuda
def test_enable_env_refuses_a_shadowing_package(home, tmp_path):
    from lscli import cli
    py = _venv_with_this_torch(tmp_path / "venv")
    site_dir = next((tmp_path / "venv" / "lib").glob("python3*/site-packages"))
    (site_dir / "tune").mkdir()
    (site_dir / "tune" / "__init__.py").write_text("")
    assert cli.main(["enable-env", str(py), "--no-deps"]) == 1 and not (site_dir / "lscli.pth").exists()
