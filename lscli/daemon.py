"""`lscli daemon run`: the idle-time loop (spec section 4).

Every TICK_S: re-read the config when asked, merge the usage logs, evaluate the gate, write state.json, and if the gate
is open run the most urgent task (lscli/scheduler.py) to completion or preemption. One job at a time, holding
`gpu.lock`; while it runs, a watch callback re-evaluates the staying-open rule about once a second and kills the job's
process tree when it fails (spec 4.2), followed by `backoff_minutes` of backoff.

The daemon process never creates a CUDA context (NVML only, LightGBM on the CPU). Everything that touches the GPU runs
in a job: the cuBLASLt ranker's proposals (`python -m lscli.propose`, daemon venv) and the measurement
(`<env python> -m lscli.worker measure`, the user's interpreter). Its own environment (hw_id) comes from a worker run
under the daemon's interpreter at start.
"""
from __future__ import annotations

import datetime as _dt
import fcntl
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from lscli import __version__, config, control, envs, gate, jobs, models, scheduler, usage
from lscli import search_queue as queue
from lscli.files import locked, read_json, write_json
from lscli.paths import Paths

TICK_S = 5.0
MERGE_EVERY_S = 60.0
COMPACT_EVERY_S = 86400.0
ENV_REFRESH_S = 86400.0
PROPOSE_TIMEOUT_S = 300.0


class Daemon:
    def __init__(self, paths: Paths | None = None, config_path: Path | None = None, *, python: str | None = None,
                 snapshot=gate.gpu_snapshot, battery=gate.on_battery, clock=time.time, local_now=_dt.datetime.now):
        self.paths = (paths or Paths()).ensure()
        self.config_path = config_path
        self.python = python or sys.executable
        self.snapshot, self.battery, self.clock, self.local_now = snapshot, battery, clock, local_now
        self.cfg = config.load(config_path)
        self.gate = gate.Gate(self.cfg)
        self.stop = False
        self.state = "starting"
        self.since = clock()
        self.backoff_until: float | None = None
        self.failures: dict[tuple, int] = {}
        self.env_docs: dict[tuple, tuple[float, dict]] = {}
        self.last_merge = self.last_compact = 0.0
        self.reload_seen = control.read_control(self.paths).get("reload")
        self.job_info: dict | None = None
        self.reasons: list[str] = ["starting"]
        self.identity: dict = {}
        self.hw_id: str | None = None
        self.models: models.ModelSet | None = None
        self.log = logging.getLogger("lscli.daemon")
        self.counters = {"searches": 0, "preempted": 0, "failed": 0}

    # ------------------------------------------------------------------------------------------------ set-up
    def start(self) -> None:
        self.identity = gate.gpu_identity()
        doc = self._worker_env(self.python, {})
        self.hw_id = (doc or {}).get("hw_id")
        self.log.info("lscli %s daemon: GPU %s (%s), hw_id %s", __version__, self.identity.get("name"),
                      self.identity.get("uuid"), self.hw_id)
        if self.identity.get("count", 0) > 1:
            self.log.warning("%d GPUs: lscli v1 uses device 0 only", self.identity["count"])
        self.load_models()

    def load_models(self) -> None:
        self.models = models.active(self.paths, self.hw_id)
        ids = self.models.ids() if self.models else None
        with locked(self.paths.client):
            config.write_client_json(self.paths, self.cfg, models=ids, gpu_uuid=self.identity.get("uuid"),
                                     model_stems=self.models.stem_paths() if self.models else {})
        self.env_docs.clear()
        self.log.info("models: %s", ids or "none (cuBLASLt heuristic + no Triton ranker: searches skipped)")

    def env_extra(self, eid: str) -> dict:
        return {**config.env_extra(self.cfg, self.models.ids() if self.models else None), "env_id": eid}

    def _worker_env(self, python: str, extra: dict) -> dict | None:
        from lscli.cli import source_dir
        try:
            r = subprocess.run([python, "-m", "lscli.worker", "env", "--env-extra", json.dumps(extra)],
                               cwd=source_dir(self.paths), capture_output=True, text=True, timeout=300,
                               stdin=subprocess.DEVNULL)
            for line in reversed(r.stdout.splitlines()):
                doc = json.loads(line)
                if doc.get("op") == "result":
                    return doc["env"]
        except (OSError, ValueError, subprocess.TimeoutExpired) as e:
            self.log.warning("environment of %s unavailable: %s", python, e)
        return None

    def caches(self) -> dict:
        """{env_id: WinnerCache in that env's environment}. Environments are probed (by running the env's interpreter)
        once a day or when the env's record or the model set changed."""
        from tune.winner_cache import WinnerCache
        out = {}
        now = self.clock()
        for eid, rec in envs.load(self.paths).items():
            extra = self.env_extra(eid)
            key = (eid, json.dumps(extra, sort_keys=True), rec.get("updated"))
            t, doc = self.env_docs.get(key, (0.0, None))
            if doc is None or now - t > ENV_REFRESH_S:
                doc = self._worker_env(rec["python"], extra)
                if doc is None:
                    continue
                self.env_docs[key] = (now, doc)
            out[eid] = WinnerCache(self.paths.winners, env=doc)
        return out

    # ------------------------------------------------------------------------------------------------ the gate
    def gate_reasons(self, own: set[int] | None = None) -> list[str]:
        lock = self.paths.src / "pipeline" / ".lock"
        return self.gate.reasons(snap=self.snapshot(), now=self.clock(), local=self.local_now(), battery=self.battery(),
                                 paused=control.paused(control.read_control(self.paths), self.clock()),
                                 backoff_until=self.backoff_until, pipeline_locked=lock.exists() and gate.lock_held(lock),
                                 own=own)

    def write_state(self, **extra) -> None:
        pend, _ = queue.pending(self.paths)
        ctl = control.read_control(self.paths)
        write_json(self.paths.state, {
            "pid": os.getpid(), "version": __version__, "state": self.state, "since": self.since,
            "gate": {"open": not self.reasons, "reasons": self.reasons}, "job": self.job_info,
            "backoff_until": self.backoff_until, "paused_until": ctl.get("paused_until") if control.paused(ctl) else None,
            "gpu": self.identity, "hw_id": self.hw_id,
            "models": {"source": self.models.source if self.models else None,
                       "ids": self.models.ids() if self.models else None, **extra.get("models", {})},
            "queue": {"pending": len(pend)}, "counters": self.counters, "updated": self.clock()})

    def set_state(self, s: str) -> None:
        if s != self.state:
            self.state, self.since = s, self.clock()

    # ------------------------------------------------------------------------------------------------ one tick
    def maintenance(self, caches: dict | None = None) -> None:
        now = self.clock()
        ctl = control.read_control(self.paths)
        if ctl.get("reload") != self.reload_seen:
            self.reload_seen = ctl.get("reload")
            try:
                self.cfg = config.load(self.config_path)
                self.gate.cfg = self.cfg
                self.load_models()
                self.log.info("configuration reloaded")
            except config.ConfigError as e:
                self.log.error("config not reloaded: %s", e)
        if now - self.last_merge >= MERGE_EVERY_S:
            usage.merge(self.paths, self.cfg["search"]["half_life_days"], now=now)
            self.last_merge = now
        if caches is not None:
            pend, end = queue.pending(self.paths)
            if pend and all(scheduler.satisfied(r, caches, self.failures) for r in pend):
                queue.consume(self.paths, end)
            elif not pend and end:
                queue.consume(self.paths, end)
            if now - self.last_compact >= COMPACT_EVERY_S and caches:
                for wc in caches.values():
                    wc.compact()
                self.last_compact = now

    def next_task(self, caches: dict) -> scheduler.Task | None:
        pend, _ = queue.pending(self.paths)
        tasks = scheduler.plan(self.cfg, usage.load(self.paths), pend, caches, self.failures, self.clock())
        if self.models is None or self.models.triton() is None:
            return None
        return tasks[0] if tasks else None

    def tick(self) -> scheduler.Task | None:
        """One iteration; returns the task it ran (None if it ran nothing)."""
        caches = self.caches()
        self.maintenance(caches)
        self.reasons = self.gate_reasons()
        if self.backoff_until and self.clock() >= self.backoff_until:
            self.backoff_until = None
        if self.reasons:
            self.set_state("backoff" if self.backoff_until else "paused" if any(r == "paused" for r in self.reasons)
                           else "waiting")
            self.write_state()
            return None
        task = self.next_task(caches)
        if task is None:
            self.set_state("waiting")
            self.reasons = []
            self.write_state()
            return None
        self.run_task(task)
        return task

    # ------------------------------------------------------------------------------------------------ jobs
    def _watch(self, tree: set[int]) -> bool:
        r = self.gate_reasons(own=tree | {os.getpid()})
        if r:
            self.reasons = r
            self.log.info("preempting %s: %s", self.job_info, "; ".join(r))
        return not r

    def run_task(self, task: scheduler.Task) -> None:
        lockf = open(self.paths.gpu_lock, "a")
        fcntl.flock(lockf, fcntl.LOCK_EX)
        self.set_state("running")
        self.job_info = {"kind": task.kind, "shape": [task.M, task.N, task.K], "dtype": task.dtype,
                         "env_id": task.env_id, "k": task.k, "reason": task.reason, "started": self.clock()}
        self.write_state()
        try:
            self.clock_lock("lock")
            self.search(task)
            self.counters["searches"] += 1
            self.failures.pop(task.key, None)
        except jobs.Preempted:
            self.counters["preempted"] += 1
            self.backoff_until = self.clock() + self.cfg["idle"]["backoff_minutes"] * 60
            self.set_state("backoff")
        except jobs.JobError as e:
            self.counters["failed"] += 1
            self.failures[task.key] = self.failures.get(task.key, 0) + 1
            self.log.warning("task %s failed: %s", task, e)
        finally:
            self.clock_lock("reset")
            self.gate.job_ended(self.clock())
            self.job_info = None
            fcntl.flock(lockf, fcntl.LOCK_UN)
            lockf.close()
            self.write_state()

    def _job_log(self, name: str) -> Path:
        d = self.paths.logs / "jobs"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{time.strftime('%Y%m%d-%H%M%S')}-{name}.log"

    def search(self, task: scheduler.Task) -> dict:
        """Spec 4.4: Triton proposals here (CPU), cuBLASLt ranker proposals in `lscli.propose`, measurement in the
        env's worker. Raises Preempted / JobError."""
        from lscli.cli import source_dir
        from tune.online import proposal_keys, propose_triton
        rec = envs.load(self.paths).get(task.env_id)
        if rec is None:
            raise jobs.JobError(f"env {task.env_id} is no longer registered")
        s = self.cfg["search"]
        tri, rank_s = propose_triton(self.models.triton(), task.M, task.N, task.K, task.k, "diverse")
        proposals = proposal_keys(tri)
        lt_rank = s["lt_rank"] if (s["lt_n"] and self.models.stems["lt_rank"] is not None) else 0
        cwd = source_dir(self.paths)
        if lt_rank:
            job = jobs.Job([self.python, "-m", "lscli.propose"], cwd=cwd, log=self._job_log("propose"))
            res = jobs.converse(job, {"M": task.M, "N": task.N, "K": task.K, "dtype": task.dtype, "lt_n": s["lt_n"],
                                      "lt_rank": lt_rank, "stem": str(self.models.stems["lt_rank"])},
                                watch=self._watch, silence_s=PROPOSE_TIMEOUT_S)
            proposals += res["proposals"]
        router = self.models.router() if s["lt_n"] else None

        def answer(msg: dict) -> dict:
            skip, p = router.skip_triton(task.M, task.N, task.K, float(msg["t_lt_us"]))
            return {"skip": bool(skip), "p": p, "threshold": router.threshold}
        req = {"M": task.M, "N": task.N, "K": task.K, "dtype": task.dtype, "k": task.k, "lt_n": s["lt_n"],
               "lt_rank": lt_rank, "confirm": 0, "router": router is not None, "proposals": proposals,
               "cache": str(self.paths.winners), "env_extra": self.env_extra(task.env_id), "order": "diverse",
               "ranker": self.models.stems["triton"].name, "reply_timeout_s": 30, "rank_s": rank_s}
        job = jobs.Job([rec["python"], "-m", "lscli.worker", "measure"], cwd=cwd, log=self._job_log("measure"))
        res = jobs.converse(job, req, watch=self._watch, answer=answer if router is not None else None)
        if res.get("skipped"):
            raise jobs.JobError(f"skipped: {res['skipped']}")
        if not res.get("written"):
            raise jobs.JobError("no valid kernel")
        self._update_env(task.env_id, rec, res.get("env") or {})
        self.log.info("searched %sx%sx%s %s for %s: %s (%s), torch %.1f us", task.M, task.N, task.K, task.dtype,
                      task.env_id, res["choice"]["key"], res["choice"].get("family"), res.get("torch_us") or -1)
        return res

    def _update_env(self, eid: str, rec: dict, env: dict) -> None:
        fresh = {k: env.get(k) for k in ("torch", "triton", "cuda_runtime") if env.get(k) is not None}
        if any(rec.get(k) != v for k, v in fresh.items()):
            envs.register(self.paths, rec["python"], {**rec, **fresh})

    def clock_lock(self, action: str) -> None:
        """Lock / reset the SM clock through the optional helper (spec 4.6); a no-op without lock_mhz."""
        if not self.cfg["clock"]["lock_mhz"]:
            return
        helper = Path("/usr/local/libexec/lscli-clock")
        if not helper.exists():
            self.log.warning("clock.lock_mhz is set but %s is missing", helper)
            return
        r = subprocess.run(["sudo", "-n", str(helper), action], capture_output=True, text=True)
        if r.returncode != 0:
            self.log.warning("clock %s failed: %s", action, r.stderr.strip()[-200:])

    # ------------------------------------------------------------------------------------------------ main loop
    def run(self) -> int:
        signal.signal(signal.SIGTERM, lambda *a: setattr(self, "stop", True))
        signal.signal(signal.SIGINT, lambda *a: setattr(self, "stop", True))
        self.start()
        while not self.stop:
            try:
                ran = self.tick()
            except Exception:                         # one bad tick must not end the service
                self.log.exception("tick failed")
                self.set_state("error")
                self.write_state()
                ran = None
            if ran is None:
                end = time.monotonic() + TICK_S
                while not self.stop and time.monotonic() < end:
                    time.sleep(0.2)
        self.set_state("stopped")
        self.write_state()
        return 0


def setup_logging(paths: Paths) -> None:
    paths.logs.mkdir(parents=True, exist_ok=True)
    from logging.handlers import RotatingFileHandler
    h = RotatingFileHandler(paths.logs / "daemon.log", maxBytes=5 << 20, backupCount=3)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root = logging.getLogger("lscli")
    root.setLevel(logging.INFO)
    root.addHandler(h)
    if sys.stderr.isatty():
        root.addHandler(logging.StreamHandler())
