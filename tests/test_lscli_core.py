"""lscli core (lscli/{config,files,envs,usage,queue,control,cli}.py): file formats and the commands that need no daemon.
CPU only: interpreters are faked where probing would run torch."""
import json
import os
import subprocess
import sys
import time

import pytest

from lscli import cli, config, control, envs, queue, usage
from lscli.files import append_jsonl, read_jsonl, write_json
from lscli.paths import Paths


@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.setenv("LSCLI_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LSCLI_CONFIG", str(tmp_path / "config.toml"))
    return Paths().ensure()


def run(*argv):
    return cli.main(list(argv))


# ---------------------------------------------------------------------------------------------- files
def test_home_is_private(paths):
    assert oct(paths.root.stat().st_mode & 0o777) == "0o700"


def test_jsonl_reads_complete_lines_only_and_resumes_by_offset(tmp_path):
    p = tmp_path / "x.jsonl"
    append_jsonl(p, [{"a": 1}])
    with open(p, "a") as f:
        f.write('{"a": 2')                          # being written (or torn)
    rows, end = read_jsonl(p)
    assert rows == [{"a": 1}] and end == len('{"a": 1}\n')
    append_jsonl(p, [{"a": 3}])                     # terminates the torn line first, so {"a": 3} is intact
    rows, end2 = read_jsonl(p, end)
    assert rows == [{"a": 3}] and end2 == p.stat().st_size


# ---------------------------------------------------------------------------------------------- config
def test_config_defaults_set_and_validation(paths, tmp_path):
    cfg = config.load()
    assert cfg == config.DEFAULTS and cfg["client"]["min_kernel_us"] == 50
    assert run("config", "idle.window", "22:00-06:30") == 0
    assert config.load()["idle"]["window"] == "22:00-06:30"
    assert run("config", "search.k_frequent", "7") == 0 and config.load()["search"]["k_frequent"] == 7
    assert run("config", "idle.require_ac", "false") == 0 and config.load()["idle"]["require_ac"] is False
    for key, bad in (("idle.window", "25:00-01:00"), ("search.k", "x"), ("search.k", "9"), ("nope.key", "1"),
                     ("train.mode", "fast"), ("idle.require_ac", "yes"), ("clock.lock_mhz", "-5")):
        assert run("config", key, bad) == 1, (key, bad)
    doc = json.loads(paths.client.read_text())                       # client.json follows the config
    assert doc["env_extra"] == {"clock_lock": "unlocked"} and doc["search"]["k_frequent"] == 7
    assert json.loads(paths.control.read_text())["reload"] > 0       # the daemon is asked to reload
    (tmp_path / "config.toml").write_text("[idle]\nbogus = 1\n")
    with pytest.raises(config.ConfigError):
        config.load()


def test_clock_policy_is_part_of_the_winner_environment():
    cfg = config.validate({"clock": {"lock_mhz": 1755}})
    assert config.env_extra(cfg) == {"clock_lock": "1755"}
    assert config.env_extra(config.DEFAULTS, models={"triton": "x"}) == {"clock_lock": "unlocked", "models": {"triton": "x"}}


# ---------------------------------------------------------------------------------------------- envs
def test_env_registry(paths, tmp_path):
    py = tmp_path / "bin" / "python"
    py.parent.mkdir()
    py.write_text("")
    eid, rec = envs.register(paths, py, {"torch": "2.13.0+cu130", "triton": "3.7.1", "cuda_runtime": "13.0"})
    assert eid == envs.env_id(py) and len(eid) == 12 and rec["python"] == str(py)
    link = tmp_path / "venv" / "bin" / "python"                      # a venv interpreter: a symlink to the base one
    link.parent.mkdir(parents=True)
    link.symlink_to(py)
    assert envs.env_id(link) != eid and envs.register(paths, link, {})[1]["python"] == str(link)
    assert envs.resolve(paths, str(py)) == envs.resolve(paths, eid) == {eid: envs.load(paths)[eid]}
    with pytest.raises(KeyError):
        envs.resolve(paths, "nope")
    assert envs.unregister(paths, str(py)) and list(envs.load(paths)) == [envs.env_id(link)]


def test_env_probe_runs_the_interpreter():
    info = envs.probe(sys.executable)
    assert info["python"] == sys.executable and info["torch"] and info["triton"]
    with pytest.raises(RuntimeError):
        envs.probe("/nonexistent/python")


# ---------------------------------------------------------------------------------------------- usage
def test_usage_merge_decays_reads_incrementally_and_keeps_live_logs(paths):
    log = usage.UsageLog(paths, "env1")
    for _ in range(3):
        log.add(64, 64, 64, "fp16", hit=True)
    log.add(64, 64, 64, "fp16", hit=False)
    log.add(10, 20, 30, "fp16", layout="NT")
    assert log.flush() == 2 and log.flush() == 0
    now = time.time()
    prof = usage.merge(paths, half_life_days=30, now=now)
    s = usage.lookup(prof)[(64, 64, 64, "fp16", "NN")]
    assert s["calls_total"] == 4 and s["hits"] == 3 and s["fallbacks"] == 1 and s["envs"] == ["env1"]
    assert log.path.exists()                                    # this process is alive: its log is kept
    log.add(64, 64, 64, "fp16")
    log.flush()
    prof = usage.merge(paths, half_life_days=30, now=now + 30 * 86400)    # one half-life later
    s = usage.lookup(prof)[(64, 64, 64, "fp16", "NN")]
    assert s["calls_total"] == 5                                # only the new line was read
    assert s["calls"] == pytest.approx(4 * 0.5 + 1 * 0.5, rel=0.01)   # both increments a half-life old
    assert usage.calls_now(prof, s, now + 60 * 86400) == pytest.approx(s["calls"] / 2)


def test_usage_logs_of_exited_processes_are_deleted_once_read(paths):
    dead = paths.profile_dir / "999999999-1.jsonl"
    append_jsonl(dead, [{"t": time.time(), "env_id": "e", "M": 8, "N": 8, "K": 8, "dtype": "fp16", "layout": "NN", "calls": 2}])
    prof = usage.merge(paths, half_life_days=30, write=False)    # a dry run touches nothing
    assert dead.exists() and not paths.profile.exists() and prof["shapes"][0]["calls_total"] == 2
    usage.merge(paths, half_life_days=30)
    assert not dead.exists() and dead.name not in json.loads(paths.profile.read_text())["offsets"]
    assert usage.load(paths)["shapes"][0]["calls_total"] == 2


# ---------------------------------------------------------------------------------------------- queue
def test_queue_dedupes_keeps_most_urgent_reason_and_truncates_in_place(paths):
    queue.request(paths, 8, 8, 8, "fp16", "e1", "stale")
    queue.request(paths, 8, 8, 8, "fp16", "e1", "miss")
    queue.request(paths, 8, 8, 8, "fp16", "e2", "superseded")
    with pytest.raises(ValueError):
        queue.request(paths, 8, 8, 8, "fp16", "e1", "bogus")
    pend, end = queue.pending(paths)
    assert [(r["env_id"], r["reason"]) for r in pend] == [("e1", "miss"), ("e2", "superseded")]
    ino = paths.queue.stat().st_ino
    queue.request(paths, 9, 9, 9, "fp16", "e1", "manual")          # arrives after the read
    queue.consume(paths, end)
    pend, end = queue.pending(paths)
    assert [r["M"] for r in pend] == [9]                           # not lost, not re-delivered
    queue.consume(paths, end)
    assert paths.queue.stat().st_size == 0 and paths.queue.stat().st_ino == ino
    assert queue.pending(paths)[0] == []


# ---------------------------------------------------------------------------------------------- control / status
def test_pause_resume(paths):
    assert not control.paused(control.read_control(paths))
    control.pause(paths, 60)
    assert control.paused(control.read_control(paths)) and not control.paused(control.read_control(paths), time.time() + 61)
    control.pause(paths)
    assert control.paused(control.read_control(paths), time.time() + 1e6)
    control.resume(paths)
    assert not control.paused(control.read_control(paths))


def test_status_without_and_with_a_daemon(paths, capsys):
    assert run("status") == cli.EXIT_NO_DAEMON and "not running" in capsys.readouterr().out
    write_json(paths.state, {"pid": os.getpid(), "state": "waiting", "since": time.time(),
                             "gate": {"open": False, "reasons": ["outside window 01:00-07:00"]}})
    assert run("--json", "status") == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["daemon"]["running"] and doc["daemon"]["gate"]["reasons"] == ["outside window 01:00-07:00"]


# ---------------------------------------------------------------------------------------------- prewarm / profile / cache
def test_prewarm_queues_for_every_env(paths, tmp_path, capsys):
    assert run("prewarm", "64x64x64") == cli.EXIT_FAIL                 # no env registered yet
    envs.register(paths, sys.executable, {"torch": "t"})
    envs.register(paths, tmp_path, {"torch": "u"})
    shapes = tmp_path / "s.json"
    shapes.write_text(json.dumps({"shapes": [{"M": 1, "N": 2, "K": 3}]}))
    assert run("prewarm", "64x64x64", "128X64x32", "--shapes", str(shapes)) == 0
    pend, _ = queue.pending(paths)
    assert len(pend) == 6 and {r["reason"] for r in pend} == {"manual"}
    assert run("prewarm", "64x64") == cli.EXIT_FAIL and run("prewarm", "99999x1x1") == cli.EXIT_FAIL
    assert run("prewarm", "1x1x1", "--env", "nope") == cli.EXIT_FAIL


def test_profile_command_shows_unmerged_logs_without_merging(paths, capsys):
    log = usage.UsageLog(paths, "e")
    log.add(4096, 4096, 4096, "fp16", hit=True)
    log.flush()
    assert run("--json", "profile") == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["shapes"][0]["M"] == 4096 and not paths.profile.exists()


def test_cache_ls_raw_and_clear(paths, capsys):
    from tune.winner_cache import WinnerCache
    env = {"hw_id": "g", "triton": "3.7.1", "env_id": "e1"}
    WinnerCache(paths.winners, env=env).put(8, 8, 8, "fp16", k=3, lt_n=0, choice={"key": "bm16_bn16_bk16_w1_s1_g1_sk1",
                                            "family": "triton", "mode": "cold"}, extra={"torch_us": 12.0, "k": 99})
    assert run("--json", "cache", "ls") == 0
    (row,) = json.loads(capsys.readouterr().out)
    assert row["torch_us"] == 12.0 and row["tier"] is None and row["env_id"] == "e1"
    line = json.loads(paths.winners.read_text())
    assert line["k"] == 3                                               # extra cannot overwrite entry fields
    assert run("cache", "clear") == cli.EXIT_USAGE and paths.winners.exists()
    assert run("cache", "clear", "--yes") == 0 and not paths.winners.exists()


def test_compact_is_scoped_by_env_id(tmp_path):
    from tune.winner_cache import WinnerCache
    p = tmp_path / "w.jsonl"
    a = {"hw_id": "g", "torch": "1", "env_id": "A"}
    b = {"hw_id": "g", "torch": "2", "env_id": "B"}
    WinnerCache(p, env=b).put(8, 8, 8, "fp16", k=3, lt_n=0, choice={"key": "kb", "family": "triton"})
    WinnerCache(p, env=a).put(8, 8, 8, "fp16", k=3, lt_n=0, choice={"key": "ka", "family": "triton"})
    WinnerCache(p, env=a).compact()
    assert WinnerCache(p, env=b).get(8, 8, 8, "fp16")["choice"]["key"] == "kb"     # B's line survived A's compaction
    hit = WinnerCache(p, env={**b, "env_id": "C"}).get(8, 8, 8, "fp16", allow_stale=True)
    assert hit["choice"]["key"] == "ka" and "env_id" in hit["stale"]      # another env's winner is soft-stale here


def test_entry_point_runs():
    r = subprocess.run([sys.executable, "-m", "lscli", "--version"], capture_output=True, text=True)
    assert r.returncode == 0 and "lscli" in r.stdout
