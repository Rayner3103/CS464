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
@pytest.mark.parametrize("grads", [(True, True), (True, False), (False, True)])
def test_autograd_backward_uses_torch(home, monkeypatch, grads):
    _put(home, 1024, 512, 768)
    client.enable()
    ran = []
    orig_apply = client._Mm.apply
    monkeypatch.setattr(client._Mm, "apply", lambda *a: ran.append(1) or orig_apply(*a))
    a = torch.randn(1024, 768, device="cuda", dtype=torch.float16, requires_grad=grads[0])
    b = torch.randn(768, 512, device="cuda", dtype=torch.float16, requires_grad=grads[1])
    w = torch.randn(1024, 512, device="cuda")
    (torch.matmul(a, b).float() * w).sum().backward()
    assert ran == [1]                                                          # the selected kernel really ran
    got = [t.grad.clone() if t.requires_grad else None for t in (a, b)]
    for t in (a, b):
        t.grad = None
    (client._ORIG["matmul"](a, b).float() * w).sum().backward()
    for g, t in zip(got, (a, b)):
        if t.requires_grad:
            torch.testing.assert_close(g, t.grad)


@requires_cuda
def test_keyword_calls_keep_torch_semantics(home):
    _put(home, 1024, 1024, 1024)
    client.enable()
    import torch.nn.functional as F
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    ref = client._ORIG["matmul"](a, a)
    torch.testing.assert_close(torch.matmul(input=a, other=a), ref)
    torch.testing.assert_close(torch.mm(input=a, mat2=a), ref)
    torch.testing.assert_close(a.matmul(other=a), ref)
    torch.testing.assert_close(a.mm(mat2=a), ref)
    w = torch.randn(32, 1024, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(F.linear(input=a, weight=w), client._ORIG["linear"](a, w))
    assert torch.matmul.__name__ == "matmul" and torch.matmul.__wrapped__ is client._ORIG["matmul"]


@requires_cuda
def test_transforms_autocast_and_other_devices_use_torch(home, monkeypatch):
    _put(home, 1024, 1024, 1024)
    client.enable()
    from kernels.triton_matmul import TritonMatmul
    calls = []
    orig = TritonMatmul.make_runner
    monkeypatch.setattr(TritonMatmul, "make_runner", lambda self, a, b, out=None: calls.append(1) or orig(self, a, b, out))
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert torch.matmul(a, b).dtype == torch.bfloat16                       # torch's cast, not our fp16 kernel
    g = torch.func.grad(lambda x: torch.matmul(x, b).float().sum())(a)
    torch.testing.assert_close(g, torch.func.grad(lambda x: client._ORIG["matmul"](x, b).float().sum())(a))
    _, t = torch.func.jvp(lambda x: torch.matmul(x, b), (a,), (torch.ones_like(a),))
    torch.testing.assert_close(t, client._ORIG["matmul"](torch.ones_like(a), b))
    torch.func.vmap(lambda x: torch.matmul(x, b))(a[None].expand(2, -1, -1).contiguous())
    with monkeypatch.context() as m:
        m.setattr(torch._C, "_cuda_getDevice", lambda: 1)                      # a model-parallel program elsewhere
        torch.matmul(a, b)
    assert calls == []
    torch.matmul(a, b)
    assert calls == [1]


@requires_cuda
def test_parameters_are_dispatched(home):
    _put(home, 1024, 1024, 1024)
    client.enable()
    w = torch.nn.Parameter(torch.randn(1024, 1024, device="cuda", dtype=torch.float16), requires_grad=False)
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(x @ w, client._ORIG["matmul"](x, w), rtol=1e-2, atol=1e-1)
    assert client._STATE.runs[(1024, 1024, 1024, "fp16")][0] is not None


@requires_cuda
def test_threads_while_the_table_changes(home):
    import threading
    _put(home, 1024, 1024, 1024)
    client.enable()
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    ref = client._ORIG["matmul"](a, a)
    errors = []

    def work():
        try:
            for i in range(50):
                torch.testing.assert_close(torch.matmul(a, a), ref, rtol=1e-2, atol=1e-1)
                torch.matmul(a[: 64 + i].contiguous(), a)
        except Exception as e:                                                 # noqa: BLE001
            errors.append(e)
    ts = [threading.Thread(target=work) for _ in range(6)]
    for t in ts:
        t.start()
    for i in range(10):
        _put(home, 512 + i, 512, 512)                                         # the daemon writes meanwhile
        client._STATE._wc_checked = 0.0
        time.sleep(0.01)
    for t in ts:
        t.join()
    torch.cuda.synchronize()
    assert not errors and client._STATE.runs[(1024, 1024, 1024, "fp16")][0] is not None


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


def test_fork_replaces_a_held_lock(home):
    client.enable()
    with client._LOCK:                                   # another thread holds it at fork time
        pid = os.fork()
        if pid == 0:
            os._exit(0 if client._LOCK.acquire(timeout=1) else 1)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0


@requires_cuda
def test_overhead(home, capsys):
    """Host time per call vs the unpatched function (median of 10,000; spec 5.4)."""
    import platform
    _put(home, 128, 128, 128)                    # a small dispatched shape: the GPU never backs up, host time is isolated
    a32 = torch.randn(64, 64, device="cuda")
    h = torch.randn(128, 128, device="cuda", dtype=torch.float16)
    orig = torch.matmul

    def per_call(f, x, n=10000, sync_every=200):
        """Median host time per call. Syncs often enough that the launch queue never fills (a full queue makes the
        host wait for the GPU, which would be measured as overhead)."""
        ts = []
        for _ in range(n):
            t = time.perf_counter_ns()
            f(x, x)
            ts.append(time.perf_counter_ns() - t)
            if len(ts) % sync_every == 0:
                torch.cuda.synchronize()
        ts.sort()
        return ts[len(ts) // 2] / 1e3
    hn = torch.randn(64, 64, device="cuda", dtype=torch.float16).t()              # fp16, in scope until contiguity
    base32, base_nc = per_call(orig, a32), per_call(orig, hn)
    client.enable()
    patched32, patched_nc = per_call(torch.matmul, a32), per_call(torch.matmul, hn)
    be = client._STATE._decide((128, 128, 128, "fp16"), torch.float16)[0]
    assert be is not None
    direct = per_call(lambda x, y: be.make_runner(x, y)(), h)
    patched_hit = per_call(torch.matmul, h)
    with capsys.disabled():
        print(f"\n[overhead] {platform.processor() or platform.machine()}: undispatched +{patched32 - base32:.2f} us "
              f"(torch {base32:.2f} us), +{patched_nc - base_nc:.2f} us for a late exit (non-contiguous fp16); "
              f"dispatched wrapper +{patched_hit - direct:.2f} us over the kernel's own launch ({direct:.2f} us)")
    assert patched32 - base32 <= 5.0 and patched_nc - base_nc <= 5.0
    assert patched_hit - direct <= 20.0


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
    lazy = "import sys; print('torch' in sys.modules)"                          # the hook never imports torch itself
    assert subprocess.run([str(py), "-c", lazy], env={**env, "LSCLI_PATCH": "1"}, capture_output=True, text=True,
                          cwd=tmp_path).stdout.strip() == "False"
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
