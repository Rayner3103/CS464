"""lscli daemon (lscli/{gate,jobs,scheduler,models,daemon}.py): gate rules, process-tree kills, preemption, scheduling,
and one end-to-end search (GPU): a client miss is queued, the daemon searches it in the env's interpreter, the next
client run dispatches the winner."""
import datetime as dt
import fcntl
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from lscli import config, envs, gate, jobs, models, scheduler
from lscli import search_queue as queue
from lscli.files import write_json
from lscli.paths import Paths
from tests.conftest import requires_cuda

REPO = Path(__file__).resolve().parent.parent
RANKER_2060 = REPO / "pipeline/runs/quick-20261029-095746/data/lgbm_rank"
SNAP = REPO / "reports/snapshot/lock1755"


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("LSCLI_HOME", str(tmp_path / "home"))
    cfg = tmp_path / "c.toml"
    cfg.write_text('[idle]\nwindow = ""\ngpu_idle_minutes = 0\nrequire_ac = false\nbackoff_minutes = 1\n')
    monkeypatch.setenv("LSCLI_CONFIG", str(cfg))
    return Paths().ensure()


# ---------------------------------------------------------------------------------------------- gate inputs
def test_window():
    at = lambda h, m: dt.datetime(2026, 11, 8, h, m)  # noqa: E731
    assert gate.in_window("", at(12, 0)) and gate.in_window("01:00-01:00", at(12, 0))
    assert gate.in_window("01:00-07:00", at(1, 0)) and not gate.in_window("01:00-07:00", at(7, 0))
    assert gate.in_window("22:00-06:30", at(23, 59)) and gate.in_window("22:00-06:30", at(6, 29))
    assert not gate.in_window("22:00-06:30", at(12, 0))


def test_battery(tmp_path):
    bat = tmp_path / "BAT0"
    bat.mkdir()
    (bat / "type").write_text("Battery\n")
    (bat / "status").write_text("Charging\n")
    assert not gate.on_battery(tmp_path)
    (bat / "status").write_text("Discharging\n")
    assert gate.on_battery(tmp_path) and not gate.on_battery(tmp_path / "none")


def test_lock_held(tmp_path):
    p = tmp_path / ".lock"
    assert not gate.lock_held(p)
    with open(p, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        assert gate.lock_held(p)
    assert not gate.lock_held(p)


def _snap(pids=(), util=0):
    return gate.GpuSnapshot(list(pids), util, True, {p: "proc" for p in pids})


def test_gate_opening_and_staying_open_rules():
    cfg = config.validate({"idle": {"window": "", "gpu_idle_minutes": 10, "require_ac": True}})
    g = gate.Gate(cfg)
    kw = dict(local=dt.datetime(2026, 1, 1, 3), battery=False, paused=False, backoff_until=None, pipeline_locked=False)
    assert any("GPU idle" in r for r in g.reasons(snap=_snap(), now=0, **kw))      # the idle clock starts now
    assert g.reasons(snap=_snap(), now=599, **kw)
    assert g.reasons(snap=_snap(), now=600, **kw) == []                             # 10 min idle
    assert g.reasons(snap=_snap(util=40), now=601, **kw)                            # busy: clock restarts
    assert g.reasons(snap=_snap(), now=1201, **kw) == []
    # while our job runs: its own processes and its utilization do not close the gate, a foreign process does
    assert g.reasons(snap=_snap([11, 12], util=99), now=1202, own={11, 12}, **kw) == []
    assert "compute process 13" in g.reasons(snap=_snap([11, 13], util=99), now=1203, own={11, 12}, **kw)[0]
    g2 = gate.Gate(cfg)
    g2.reasons(snap=_snap(), now=0, **kw)
    g2.reasons(snap=_snap(), now=600, **kw)
    g2.reasons(snap=_snap([5], util=99), now=601, own={5}, **kw)
    g2.job_ended(602)
    assert g2.reasons(snap=_snap(util=90), now=603, **kw) == []                     # our job's tail utilization
    for extra, word in ((dict(battery=True), "battery"), (dict(paused=True), "paused"),
                        (dict(backoff_until=10_000), "backoff"), (dict(pipeline_locked=True), "pipeline")):
        assert any(word in r for r in g2.reasons(snap=_snap(), now=604, **{**kw, **extra})), word
    assert gate.Gate(cfg).reasons(snap=gate.GpuSnapshot([], None, ok=False), now=0, **kw)[0] == "NVML unavailable"


# ---------------------------------------------------------------------------------------------- jobs
TREE = textwrap.dedent("""
    import json, subprocess, sys, time
    kid = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    req = json.loads(sys.stdin.readline())
    print(json.dumps({"op": "route", "t_lt_us": 42.0}), flush=True)
    ans = json.loads(sys.stdin.readline())
    time.sleep(req.get("sleep", 0))
    print(json.dumps({"op": "result", "kid": kid.pid, "ans": ans}), flush=True)
""")


def test_converse_answers_route_questions(tmp_path):
    script = tmp_path / "w.py"
    script.write_text(TREE)
    job = jobs.Job([sys.executable, str(script)])
    res = jobs.converse(job, {"sleep": 0}, watch=lambda tree: True, answer=lambda m: {"skip": m["t_lt_us"] > 40})
    assert res["ans"] == {"skip": True}
    os.kill(res["kid"], 9)


def test_preemption_kills_the_whole_tree_including_new_sessions(tmp_path):
    script = tmp_path / "w.py"
    script.write_text(TREE)
    job = jobs.Job([sys.executable, str(script)])
    seen, t0 = [], time.monotonic()

    def watch(tree):
        seen.append(set(tree))
        return time.monotonic() - t0 < 2
    with pytest.raises(jobs.Preempted):
        jobs.converse(job, {"sleep": 30}, watch=watch, answer=lambda m: {"skip": False})
    tree = set().union(*seen)
    assert len(tree) >= 2                                   # the grandchild (own session) was tracked
    time.sleep(0.5)
    assert not [p for p in tree if jobs._alive(p)]


def test_a_job_that_dies_is_an_error(tmp_path):
    job = jobs.Job([sys.executable, "-c", "import sys; sys.stdin.readline(); sys.exit(3)"])
    with pytest.raises(jobs.JobError, match="exited"):
        jobs.converse(job, {}, watch=lambda t: True)


# ---------------------------------------------------------------------------------------------- scheduler / models
class FakeCache:
    def __init__(self, rows):
        self.rows = rows

    def get(self, M, N, K, dtype, **kw):
        return self.rows.get((M, N, K, dtype))


def test_scheduler_classes_and_order():
    cfg = config.DEFAULTS
    cur = {"stale": [], "k": 3}
    caches = {"e1": FakeCache({(64, 64, 64, "fp16"): cur, (128, 128, 128, "fp16"): {"stale": ["driver"], "k": 3},
                               (256, 256, 256, "fp16"): {"stale": ["models"], "k": 5},
                               (512, 512, 512, "fp16"): {"stale": [], "k": 3}})}
    now = time.time()
    prof = {"updated": now, "half_life_days": 30, "shapes": [
        {"M": 64, "N": 64, "K": 64, "dtype": "fp16", "layout": "NN", "calls": 10, "envs": ["e1"]},
        {"M": 4096, "N": 4096, "K": 4096, "dtype": "fp16", "layout": "NN", "calls": 10, "envs": ["e1"]},
        {"M": 2048, "N": 2048, "K": 2048, "dtype": "fp16", "layout": "NN", "calls": 10, "envs": ["e1"]},
        {"M": 512, "N": 512, "K": 512, "dtype": "fp16", "layout": "NN", "calls": 2e6, "envs": ["e1"]},
        {"M": 9, "N": 9, "K": 9, "dtype": "fp16", "layout": "NT", "calls": 1e9, "envs": ["e1"]},
        {"M": 7, "N": 7, "K": 7, "dtype": "fp16", "layout": "NN", "calls": 1e9, "envs": ["gone"]}]}
    pending = [{"M": 128, "N": 128, "K": 128, "dtype": "fp16", "env_id": "e1", "reason": "stale"},
               {"M": 256, "N": 256, "K": 256, "dtype": "fp16", "env_id": "e1", "reason": "superseded"},
               {"M": 32, "N": 32, "K": 32, "dtype": "fp16", "env_id": "e1", "reason": "manual"}]
    tasks = scheduler.plan(cfg, prof, pending, caches, {}, now)
    got = [(t.kind, t.M) for t in tasks]
    assert got == [("search", 4096), ("search", 2048), ("search", 32), ("widen", 512), ("stale", 128),
                   ("superseded", 256)]
    assert tasks[3].k == 5
    assert [t.M for t in scheduler.plan(cfg, prof, pending, caches, {(4096, 4096, 4096, "fp16", "e1"): 2}, now)][:1] == [2048]
    assert scheduler.satisfied({"M": 64, "N": 64, "K": 64, "dtype": "fp16", "env_id": "e1"}, caches, {})
    assert not scheduler.satisfied(pending[0], caches, {}) and scheduler.satisfied({**pending[0], "env_id": "x"}, caches, {})


def _model(d: Path, stem: str, content: str = "x"):
    (d / f"{stem}.txt").write_text(content)
    (d / f"{stem}.json").write_text("{}")


def test_model_sets_prefer_native_and_refuse_foreign_architectures(home):
    z = home.models / "zero_shot"
    z.mkdir(parents=True)
    for s in ("tri", "lt", "rt"):
        _model(z, s)
    write_json(z / "MODELS.json", {"hw_id": "rtx3080_sm86_68sms_10gb", "source": "zero_shot",
                                   "models": {"triton": {"stem": "tri"}, "lt_rank": {"stem": "lt"}, "router": {"stem": "rt"}}})
    ms = models.active(home, "rtx3090_sm86_82sms_24gb")
    assert ms.source == "zero_shot" and ms.complete and ms.ids()["triton"].startswith("tri:")
    ms = models.active(home, "rtx2060_sm75_30sms_6gb")
    assert ms.stems["triton"] is not None and ms.stems["lt_rank"] is None and ms.stems["router"] is None
    n = home.models / "rtx2060_sm75_30sms_6gb"
    n.mkdir()
    _model(n, "tri2", "y")
    write_json(n / "MODELS.json", {"hw_id": "rtx2060_sm75_30sms_6gb", "source": "native", "models": {"triton": {"stem": "tri2"}}})
    ms = models.active(home, "rtx2060_sm75_30sms_6gb")
    assert ms.source == "native" and ms.ids()["triton"].startswith("tri2:") and not ms.complete


# ---------------------------------------------------------------------------------------------- end to end (GPU)
def _setup_models(home, hw_id):
    d = home.models / hw_id
    d.mkdir(parents=True)
    for name, src in (("tri", RANKER_2060), ("lt", SNAP / "lt_rank"), ("rt", SNAP / "router")):
        for ext in (".txt", ".json"):
            os.symlink(str(src) + ext, d / f"{name}{ext}")
    # test-only: label the 3080 cuBLASLt ranker / router with this GPU so the proposer and router paths run
    write_json(d / "MODELS.json", {"hw_id": hw_id, "source": "native",
                                   "models": {"triton": {"stem": "tri"}, "lt_rank": {"stem": "lt"}, "router": {"stem": "rt"}}})


def _daemon(home, snapshot):
    from lscli.daemon import Daemon
    d = Daemon(home, None, snapshot=snapshot, battery=lambda: False)
    d.start()
    return d


@requires_cuda
@pytest.mark.skipif(not Path(str(RANKER_2060) + ".txt").exists(), reason="needs the 2060 quick-run ranker")
def test_miss_is_searched_by_the_daemon_and_then_dispatched(home):
    envs.register(home, sys.executable)
    eid = envs.env_id(sys.executable)
    hw = json.loads(subprocess.run([sys.executable, "-m", "lscli.worker", "env"], cwd=REPO, capture_output=True,
                                   text=True).stdout.splitlines()[-1])["env"]["hw_id"]
    _setup_models(home, hw)
    queue.request(home, 1024, 1024, 1024, "fp16", eid, "miss")
    d = _daemon(home, lambda: gate.GpuSnapshot([], 0, True, {}))
    task = d.tick()
    assert task is not None and task.kind == "search" and d.counters == {"searches": 1, "preempted": 0, "failed": 0}
    line = [json.loads(x) for x in home.winners.read_text().splitlines()][-1]
    assert line["env"]["env_id"] == eid and line["env"]["models"]["triton"].startswith("tri:") and line["torch_us"] > 0
    assert line["env"]["clock_lock"] == "unlocked"
    assert [x.name.split("-")[-1] for x in sorted((home.logs / "jobs").glob("*.log"))] == ["propose.log", "measure.log"]
    d.tick()                                                   # nothing left: the request is consumed
    assert queue.pending(home)[0] == [] and json.loads(home.state.read_text())["state"] == "waiting"
    # the client in this env now sees a current entry (dispatch depends on the measured times)
    code = ("import torch, lscli.client as c; c.enable(); a = torch.randn(1024, 1024, device='cuda', dtype=torch.float16); "
            "torch.matmul(a, a); print(c._STATE.runs[(1024, 1024, 1024, 'fp16')][1])")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True,
                         env={**os.environ, "LSCLI_HOME": str(home.root)})
    assert out.stdout.strip() == "True", out.stderr[-1500:]


@requires_cuda
@pytest.mark.skipif(not Path(str(RANKER_2060) + ".txt").exists(), reason="needs the 2060 quick-run ranker")
def test_a_foreign_process_preempts_the_search_and_nothing_is_written(home):
    envs.register(home, sys.executable)
    eid = envs.env_id(sys.executable)
    hw = json.loads(subprocess.run([sys.executable, "-m", "lscli.worker", "env"], cwd=REPO, capture_output=True,
                                   text=True).stdout.splitlines()[-1])["env"]["hw_id"]
    _setup_models(home, hw)
    queue.request(home, 2048, 2048, 2048, "fp16", eid, "manual")
    calls = []

    def snapshot():
        calls.append(1)
        return gate.GpuSnapshot([999999] if len(calls) > 3 else [], 0, True, {999999: "user training job"})
    d = _daemon(home, snapshot)
    d.tick()
    assert d.counters["preempted"] == 1 and d.backoff_until and not home.winners.exists()
    st = json.loads(home.state.read_text())
    assert st["state"] == "backoff" and st["job"] is None
    assert any("compute process 999999" in r for r in d.gate_reasons())
