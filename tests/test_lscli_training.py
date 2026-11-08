"""lscli training orchestration (lscli/training.py + the daemon's train tasks). A stand-in for pipeline/train_best.sh
(LSCLI_TRAIN_SCRIPT) keeps train_best's run-directory conventions (mode, hw_id, stamps/<stage>.done, models/) so the
whole first training, a retraining after a cuBLASLt change, CPU stages under a busy GPU and a preempted stage run in
seconds without a GPU. The real script's stages were not run here (hours of GPU)."""
import json
import os
import sys
import time
from pathlib import Path

import pytest

from lscli import gate, training
from lscli.files import write_json
from lscli.paths import Paths

REPO = Path(__file__).resolve().parent.parent
RANKER = REPO / "pipeline/runs/quick-20261029-095746/data/lgbm_rank"
HW = "testgpu_sm86_68sms_10gb"
ENV = {"hw_id": HW, "triton": "3.7.1", "cublaslt": 130101, "driver": "595.99.02", "kernel_src": "abc", "torch": "2.13.0"}

FAKE = r"""#!/usr/bin/env bash
RUN=""; STAGE=""; MODE=""
while [[ $# -gt 0 ]]; do
  case "$1" in --run-dir|--resume) RUN="$2"; shift ;; --only) STAGE="$2"; shift ;; --full) MODE=full ;; --quick) MODE=quick ;; esac
  shift
done
# like train_best.sh: hold pipeline/.lock for the whole run, refuse a busy GPU in preflight, print a lot through tee
exec 9>"$LSCLI_HOME/src/pipeline/.lock"
flock -n 9 || { echo "another pipeline run holds pipeline/.lock" >&2; exit 7; }
if [[ "$STAGE" == preflight && -f "$FAKE_BUSY" ]]; then echo "GPU busy" >&2; exit 8; fi
head -c 200000 /dev/zero | tr '\0' 'x' | fold -w 100 | tee /dev/null
if [[ "$STAGE" == "$FAKE_FAIL_STAGE" ]]; then exit 1; fi
mkdir -p "$RUN/stamps" "$RUN/data/$FAKE_HW" "$RUN/reports"
[[ -f "$RUN/mode" ]] || echo "${MODE:-full}" > "$RUN/mode"
echo "$FAKE_HW" > "$RUN/hw_id"
echo "$STAGE" >> "$RUN/ran.txt"
echo "$STAGE" > "$RUN/reports/last.txt"
if [[ "$STAGE" == "$FAKE_SLOW_STAGE" && ! -f "$RUN/slow-once" ]]; then
  touch "$RUN/slow-once"; echo '{"key": 1}' > "$RUN/data/$FAKE_HW/inflight.json"; sleep 30
fi
case "$STAGE" in
  tune_sweep) mkdir -p "$RUN/data/$FAKE_HW/sweep_screen"; echo shape > "$RUN/data/$FAKE_HW/sweep_screen/a.json" ;;
  lt_screen) mkdir -p "$RUN/data/$FAKE_HW/lt_screen"; date +%s%N > "$RUN/data/$FAKE_HW/lt_screen/a.json" ;;
  package) mkdir -p "$RUN/models"; cp "$FAKE_RANKER.txt" "$RUN/models/lgbm_rank_v2.txt"; cp "$FAKE_RANKER.json" "$RUN/models/lgbm_rank_v2.json" ;;
esac
touch "$RUN/stamps/$STAGE.done"
"""


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("LSCLI_HOME", str(tmp_path / "home"))
    cfg = tmp_path / "c.toml"
    cfg.write_text('[idle]\nwindow = ""\ngpu_idle_minutes = 0\nrequire_ac = false\nbackoff_minutes = 1\n')
    monkeypatch.setenv("LSCLI_CONFIG", str(cfg))
    script = tmp_path / "train_best.sh"
    script.write_text(FAKE)
    monkeypatch.setenv("LSCLI_TRAIN_SCRIPT", str(script))
    monkeypatch.setenv("FAKE_HW", HW)
    monkeypatch.setenv("FAKE_RANKER", str(RANKER))
    monkeypatch.setenv("FAKE_BUSY", str(tmp_path / "busy"))
    paths = Paths().ensure()
    (paths.src / "pipeline").mkdir(parents=True)
    (paths.src / "pipeline" / ".lock").touch()
    return paths


def _daemon(home, snapshot=None, env=None, local_now=None):
    from lscli.daemon import Daemon
    kw = {"local_now": local_now} if local_now else {}
    d = Daemon(home, None, snapshot=snapshot or (lambda: gate.GpuSnapshot([], 0, True, {})), battery=lambda: False, **kw)
    d.identity, d.hw_id, d.daemon_env = {"uuid": "GPU-x"}, HW, dict(env or ENV)
    d._refresh_daemon_env = lambda: None             # the test controls the daemon's environment
    d.load_models()
    return d


# ---------------------------------------------------------------------------------------------- pure functions
def test_stage_sets():
    first = training.stages_for({"first"}, with_bundle=False)
    assert first[0] == "preflight" and first[-2:] == ["package", "summary"] and "bundle" not in first
    assert "bundle" in training.stages_for({"first"}, with_bundle=True)
    lt = training.stages_for({"cublaslt"}, False)
    assert lt == ["env", "lt_screen", "lt_fit", "tune_stage2_cold", "tune_stage2_graph", "router_fit", "package", "summary"]
    tri = training.stages_for({"triton"}, False)
    assert {"env", "check", "heldout_sweep", "selectors"} <= set(tri) and "lt_screen" not in tri
    assert "preflight" not in training.CPU_STAGES                          # train_best's preflight wants an idle GPU
    assert set(training.stages_for({"triton", "cublaslt"}, False)) == set(tri) | set(lt)
    assert {"selectors", "train_v1"}.isdisjoint(training.CPU_STAGES)        # both run the online search on the GPU


def test_changes():
    assert training.changes(None, ENV) == {"first"}
    assert training.changes(ENV, ENV) == set()
    assert training.changes(ENV, {**ENV, "driver": "600"}) == set()           # driver only: winners re-searched instead
    assert training.changes(ENV, {**ENV, "cublaslt": 1}) == {"cublaslt"}
    assert training.changes(ENV, {**ENV, "kernel_src": "z", "cublaslt": 1}) == {"triton", "cublaslt"}


def test_seed_links_frozen_outputs_copies_the_rest_and_drops_rerun_stages(tmp_path):
    prev = tmp_path / "1"
    data = prev / "data" / HW
    (data / "sweep_screen").mkdir(parents=True)
    (data / "lt_screen").mkdir()
    (prev / "stamps").mkdir()
    (prev / "reports").mkdir()
    (prev / "logs").mkdir()
    (prev / "hw_id").write_text(HW)
    (data / "sweep_screen" / "s.json").write_text("tri")
    (data / "sweep_screen" / "big.json").write_bytes(b"x" * training.LINK_MIN_BYTES)
    (data / "lt_screen" / "l.json").write_text("old lt")
    (data / "lt_rank.txt").write_text("old model")
    (prev / "reports" / "r.md").write_text("report")
    (prev / "logs" / "x.log").write_text("log")
    for st in ("tune_sweep", "lt_screen", "lt_fit"):
        (prev / "stamps" / f"{st}.done").touch()
    new = tmp_path / "2"
    training.seed(prev, new, ["lt_screen", "lt_fit"])
    nd = new / "data" / HW
    assert (nd / "sweep_screen" / "big.json").stat().st_ino == (data / "sweep_screen" / "big.json").stat().st_ino
    assert (nd / "sweep_screen" / "s.json").stat().st_ino != (data / "sweep_screen" / "s.json").stat().st_ino  # small: copied
    assert not (nd / "lt_screen").exists() and not (nd / "lt_rank.txt").exists()
    assert sorted(p.name for p in (new / "stamps").iterdir()) == ["tune_sweep.done"]
    assert (new / "reports" / "r.md").stat().st_ino != (prev / "reports" / "r.md").stat().st_ino   # copied
    assert not (new / "logs").exists()
    (new / "reports" / "r.md").write_text("changed")
    assert (prev / "reports" / "r.md").read_text() == "report"                # the old run is untouched


def test_activate_swaps_in_a_model_set(home, tmp_path):
    run = tmp_path / "run"
    (run / "models").mkdir(parents=True)
    for stem in ("lgbm_rank_v2", "lt_rank"):
        for ext in (".txt", ".json"):
            (run / "models" / f"{stem}{ext}").write_text(stem)
    d = training.activate(home, HW, run, ENV)
    doc = json.loads((d / "MODELS.json").read_text())
    assert doc["source"] == "native" and doc["trained_env"] == ENV and set(doc["models"]) == {"triton", "lt_rank"}
    (run / "models" / "lt_rank.txt").unlink()
    training.activate(home, HW, run, {**ENV, "cublaslt": 2})
    doc = json.loads((d / "MODELS.json").read_text())
    assert set(doc["models"]) == {"triton"} and not (d / "lt_rank.txt").exists() and doc["trained_env"]["cublaslt"] == 2
    assert not list(home.models.glob(".*"))                                    # no temp / old directories left


# ---------------------------------------------------------------------------------------------- daemon-driven training
needs_ranker = pytest.mark.skipif(not Path(str(RANKER) + ".txt").exists(), reason="needs the 2060 quick-run ranker")


def _run_until(d, cond, n=40):
    for _ in range(n):
        d.tick()
        if cond():
            return True
    return False


@needs_ranker
def test_first_training_then_retraining_after_a_cublaslt_change(home):
    d = _daemon(home)
    assert d.models is None
    assert _run_until(d, lambda: d.models is not None and d.models.source == "native")
    run1 = home.train / HW / "1"
    assert run1.joinpath("ran.txt").read_text().split() == training.stages_for({"first"}, False)
    doc = json.loads((home.models / HW / "MODELS.json").read_text())
    assert doc["trained_env"]["cublaslt"] == ENV["cublaslt"] and doc["models"]["triton"]["stem"] == "lgbm_rank_v2"
    assert json.loads(home.client.read_text())["env_extra"]["models"]["source"] == "native"
    lt_before = (run1 / "data" / HW / "lt_screen" / "a.json").read_text()
    # nothing changed: no training
    d.last_train_check = 0
    d.tick()
    assert training.Training(home, HW).state["status"] == "done"
    # cuBLASLt changed: a seeded run 2 re-runs only the cuBLASLt stages, from empty outputs
    d.daemon_env["cublaslt"] = 999
    d.last_train_check = 0
    old_ids = d.models.ids()
    assert _run_until(d, lambda: training.Training(home, HW).state.get("status") == "done"
                      and training.Training(home, HW).state.get("run") == 2)
    run2 = home.train / HW / "2"
    ran2 = run2.joinpath("ran.txt").read_text().split()
    assert ran2[len(training.stages_for({"first"}, False)):] == training.stages_for({"cublaslt"}, False)
    assert run1.joinpath("ran.txt").read_text().split() == training.stages_for({"first"}, False)   # run 1 untouched
    assert (run2 / "data" / HW / "lt_screen" / "a.json").read_text() != lt_before
    assert (run1 / "data" / HW / "lt_screen" / "a.json").read_text() == lt_before
    assert json.loads((home.models / HW / "MODELS.json").read_text())["trained_env"]["cublaslt"] == 999
    assert d.models.ids()["source"] == "native"
    for _ in range(3):                                   # done: no further training starts by itself
        d.tick()
    assert training.Training(home, HW).state["run"] == 2


@needs_ranker
def test_cpu_stages_run_while_the_gpu_is_busy_and_gpu_stages_wait(home):
    training.Training(home, HW).begin({"first"}, ["train_v2", "lt_screen"], ENV, "test")
    d = _daemon(home, snapshot=lambda: gate.GpuSnapshot([4242], 90, True, {4242: "user"}))
    d.tick()                                             # train_v2 is CPU-only: runs despite the user's job
    assert training.Training(home, HW).done() == {"train_v2"}
    for _ in range(3):
        d.tick()
    assert training.Training(home, HW).done() == {"train_v2"}        # lt_screen is a GPU stage: waits
    assert json.loads(home.state.read_text())["state"] == "waiting"


@needs_ranker
def test_preflight_waits_for_an_idle_gpu(home, tmp_path):
    d = _daemon(home, snapshot=lambda: gate.GpuSnapshot([4242], 90, True, {4242: "user"}))
    for _ in range(3):
        d.tick()
    assert training.Training(home, HW).done() == set() and d.counters["failed"] == 0


@needs_ranker
def test_our_own_train_best_lock_does_not_preempt_it_but_a_manual_run_blocks_training(home):
    import subprocess
    lock = home.src / "pipeline" / ".lock"
    holder = subprocess.Popen(["flock", str(lock), "sleep", "4"])          # someone runs train_best.sh by hand
    time.sleep(0.3)
    d = _daemon(home)
    d.tick()
    assert any("pipeline/.lock" in r for r in d.reasons) and training.Training(home, HW).done() == set()
    holder.wait()
    d.tick()                                             # our own run holds the same lock all through the stage
    assert training.Training(home, HW).done() == {"preflight"} and d.counters["preempted"] == 0


@needs_ranker
def test_train_start_ignores_the_window_and_is_cleared_when_done(home):
    import datetime as dt
    from lscli import config as cfgmod, control
    (Path(os.environ["LSCLI_CONFIG"])).write_text('[idle]\nwindow = "01:00-02:00"\ngpu_idle_minutes = 0\nrequire_ac = false\n')
    d = _daemon(home, local_now=lambda: dt.datetime(2026, 11, 8, 12, 0))
    d.tick()
    assert training.Training(home, HW).done() == set() and any("outside window" in r for r in d.reasons)
    control.train_now(home)
    assert _run_until(d, lambda: d.models is not None and d.models.source == "native")
    assert not control.read_control(home).get("train_now")


@needs_ranker
def test_a_stage_that_fails_twice_stops_the_training(home, monkeypatch):
    from lscli import control
    monkeypatch.setenv("FAKE_FAIL_STAGE", "check")
    control.train_now(home)
    d = _daemon(home)
    for _ in range(6):
        d.tick()
    tr = training.Training(home, HW)
    assert tr.state["status"] == "failed" and tr.state["failures"] == {"check": 2} and "check" not in tr.done()
    assert d.counters["failed"] == 2 and not control.read_control(home).get("train_now")
    d.tick()
    assert training.Training(home, HW).state["run"] == 1             # no automatic restart after a failure


@needs_ranker
def test_a_preempted_stage_is_killed_resumed_and_its_inflight_marker_removed(home, monkeypatch):
    monkeypatch.setenv("FAKE_SLOW_STAGE", "tune_sweep")
    d = None

    def snapshot():
        slow = d is not None and (d.job_info or {}).get("stage") == "tune_sweep" and \
            time.monotonic() - started[0] > 1.0
        return gate.GpuSnapshot([4242] if slow else [], 0, True, {4242: "user"})
    started = [0.0]
    d = _daemon(home, snapshot=snapshot)
    orig = d.run_training_stage

    def timed(stage):
        started[0] = time.monotonic()
        return orig(stage)
    d.run_training_stage = timed
    assert _run_until(d, lambda: d.counters["preempted"] == 1, n=10)
    run = home.train / HW / "1"
    assert "tune_sweep" not in training.Training(home, HW).done()
    assert not (run / "data" / HW / "inflight.json").exists()        # our kill is not a crash of that candidate
    assert d.backoff_until is not None
    d.backoff_until = None
    assert _run_until(d, lambda: "tune_sweep" in training.Training(home, HW).done(), n=5)


def test_train_cli(home, capsys):
    from lscli import cli
    assert cli.main(["train", "status"]) == cli.EXIT_NO_DAEMON
    assert cli.main(["train", "retrain", "--stages", "lt_fit,bogus"]) == cli.EXIT_USAGE
    assert cli.main(["train", "retrain", "--stages", "lt_screen,lt_fit"]) == 0
    assert json.loads(home.control.read_text())["retrain"]["stages"] == ["lt_screen", "lt_fit"]
    assert cli.main(["train", "start"]) == 0 and json.loads(home.control.read_text())["train_now"] is True
    write_json(home.state, {"hw_id": HW, "pid": 1})
    capsys.readouterr()
    assert cli.main(["--json", "train", "status"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["status"] == "none" and doc["trained_env"] is None


@needs_ranker
def test_requested_retraining_of_some_stages(home):
    from lscli import control
    d = _daemon(home)
    assert _run_until(d, lambda: d.models is not None and d.models.source == "native")
    control.retrain(home, ["package", "lt_fit"])                       # any order: run in train_best order
    assert _run_until(d, lambda: training.Training(home, HW).state.get("run") == 2
                      and training.Training(home, HW).state.get("status") == "done", n=10)
    ran = (home.train / HW / "2" / "ran.txt").read_text().split()
    assert ran[-2:] == ["lt_fit", "package"] and "retrain" not in control.read_control(home)
    for _ in range(3):                                   # the request was consumed once: no retraining loop
        d.tick()
    assert training.Training(home, HW).state["run"] == 2
    doc = json.loads((home.models / HW / "MODELS.json").read_text())
    assert doc["models"]["triton"]["id"].startswith("lgbm_rank_v2:") and "lt_fit" in doc["stages"]


def test_a_lost_state_file_never_reuses_a_run_directory(home):
    (home.train / HW / "1").mkdir(parents=True)
    (home.train / HW / "1" / "precious").write_text("30 h of sweeps")
    training.Training(home, HW).begin({"first"}, ["preflight"], ENV, "test")
    assert (home.train / HW / "1" / "precious").exists() and training.Training(home, HW).state["run"] == 2
