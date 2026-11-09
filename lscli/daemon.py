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

from lscli import __version__, config, control, envs, gate, jobs, models, scheduler, training, usage
from lscli import search_queue as queue
from lscli.files import locked, read_json, write_json
from lscli.paths import Paths

TICK_S = 5.0
GPU_REASONS = ("compute process", "GPU idle", "NVML unavailable")   # what a CPU-only training stage may ignore
TRAIN_CHECK_S = 86400.0
MERGE_EVERY_S = 60.0
COMPACT_EVERY_S = 86400.0
ENV_REFRESH_S = 86400.0
ENV_RETRY_S = 600.0                  # a registered env whose probe failed is retried this often (not every tick)
PROPOSE_TIMEOUT_S = 300.0
HEARTBEAT_S = 5.0


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
        self.env_failed: dict[str, float] = {}               # env id -> last failed probe
        self.unknown_envs: set[str] = set()                  # registered, but no environment yet (probe failed)
        self.wcs: dict[str, tuple] = {}                      # env id -> (env doc, file stamp, WinnerCache)
        self.last_heartbeat = 0.0
        self.daemon_env: dict = {}
        self.last_merge = self.last_compact = 0.0
        self.reload_seen = control.read_control(self.paths).get("reload")
        self.job_info: dict | None = None
        self.reasons: list[str] = ["starting"]
        self.identity: dict = {}
        self.hw_id: str | None = None
        self.last_train_check = 0.0
        self.models: models.ModelSet | None = None
        self.log = logging.getLogger("lscli.daemon")
        self.counters = {"searches": 0, "preempted": 0, "failed": 0}

    # ------------------------------------------------------------------------------------------------ set-up
    def start(self) -> None:
        self.identity = gate.gpu_identity()
        self.resolve_hw()
        self.log.info("lscli %s daemon: GPU %s (%s), hw_id %s", __version__, self.identity.get("name"),
                      self.identity.get("uuid"), self.hw_id)
        if self.identity.get("count", 0) > 1:
            self.log.warning("%d GPUs: lscli v1 uses device 0 only", self.identity["count"])
        self.load_models()

    def resolve_hw(self) -> bool:
        """Probe the daemon venv's environment (hw_id, versions) in a worker; retried on later ticks until it works
        (the driver may not be ready at login). Returns True when it just became known."""
        if self.hw_id:
            return False
        doc = self._worker_env(self.python, {})
        if not doc:
            return False
        self.daemon_env, self.hw_id = doc, doc.get("hw_id")
        if not self.identity.get("uuid"):
            self.identity = gate.gpu_identity()
        return bool(self.hw_id)

    def load_models(self) -> None:
        self.models = models.active(self.paths, self.hw_id)
        ids = self.models.ids() if self.models else None
        with locked(self.paths.client):
            config.write_client_json(self.paths, self.cfg, models=ids, gpu_uuid=self.identity.get("uuid"),
                                     model_stems=self.models.stem_paths() if self.models else {})
        self.env_docs.clear()
        self.log.info("models: %s", ids or "none: cuBLASLt-only searches until this GPU's models are trained")

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

    def _env_key(self, eid: str, rec: dict) -> tuple:
        return (eid, json.dumps(self.env_extra(eid), sort_keys=True), rec.get("updated"))

    def caches(self) -> dict:
        """{env_id: WinnerCache in that env's environment}. An environment is probed (by running the env's interpreter)
        once a day, when the env's record or the model set changed, or when a worker reported a different one; a failed
        probe is retried after ENV_RETRY_S and the env is listed in `unknown_envs` (its requests stay queued). A cache
        instance is kept per env and re-read only when winners.jsonl changed."""
        from tune.winner_cache import WinnerCache
        out, self.unknown_envs = {}, set()
        now = self.clock()
        try:
            st = self.paths.winners.stat()
            stamp = (st.st_mtime_ns, st.st_size, st.st_ino)
        except OSError:
            stamp = None
        for eid, rec in envs.load(self.paths).items():
            key = self._env_key(eid, rec)
            t, doc = self.env_docs.get(key, (0.0, None))
            if doc is None or now - t > ENV_REFRESH_S:
                if now - self.env_failed.get(eid, -1e18) < ENV_RETRY_S:
                    doc = None
                else:
                    doc = self._worker_env(rec["python"], self.env_extra(eid))
                    if doc is None:
                        self.env_failed[eid] = now
                    else:
                        self.env_docs[key] = (now, doc)
                        self.env_failed.pop(eid, None)
            if doc is None:
                self.unknown_envs.add(eid)
                continue
            have = self.wcs.get(eid)
            if have is None or have[0] != doc:
                have = (doc, stamp, WinnerCache(self.paths.winners, env=doc))
            elif have[1] != stamp:
                have[2].reload()
                have = (doc, stamp, have[2])
            self.wcs[eid] = have
            out[eid] = have[2]
        return out

    # ------------------------------------------------------------------------------------------------ the gate
    def gate_reasons(self, own=None) -> list[str]:
        """Why the gate is closed. `own` = the running job's process tree, or a function returning it: called AFTER the
        GPU snapshot, so a job process that created its CUDA context just before is in the tree it is compared with."""
        lock = self.paths.src / "pipeline" / ".lock"
        snap = self.snapshot()
        if callable(own):
            own = own()
        locked = lock.exists() and gate.lock_held(lock, exclude=own or frozenset())   # our own train_best holds it
        return self.gate.reasons(snap=snap, now=self.clock(), local=self.local_now(), battery=self.battery(),
                                 paused=control.paused(control.read_control(self.paths), self.clock()),
                                 backoff_until=self.backoff_until, pipeline_locked=locked, own=own)

    def write_state(self, **extra) -> None:
        pend, _ = queue.pending(self.paths)
        ctl = control.read_control(self.paths)
        write_json(self.paths.state, {
            "pid": os.getpid(), "version": __version__, "state": self.state, "since": self.since,
            "gate": {"open": not self.reasons, "reasons": self.reasons}, "job": self.job_info,
            "backoff_until": self.backoff_until, "paused_until": ctl.get("paused_until") if control.paused(ctl) else None,
            "gpu": self.identity, "hw_id": self.hw_id,
            "models": {"source": self.models.source if self.models else None,
                       "ids": self.models.ids() if self.models else None,
                       "training": training.Training(self.paths, self.hw_id).summary() if self.hw_id else None},
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
            pend, _ = queue.pending(self.paths)
            if pend:                                       # drop finished requests, keep the rest (any order)
                queue.retain(self.paths, lambda r: not scheduler.satisfied(r, caches, self.failures, self.unknown_envs))
            if now - self.last_compact >= COMPACT_EVERY_S and caches:
                for wc in caches.values():
                    wc.compact()
                self.last_compact = now

    def next_task(self, caches: dict) -> scheduler.Task | None:
        pend, _ = queue.pending(self.paths)
        triton = self.models is not None and self.models.triton() is not None
        tasks = scheduler.plan(self.cfg, usage.load(self.paths), pend, caches, self.failures, self.clock(), triton=triton)
        if not triton and not self.cfg["search"]["lt_n"]:
            tasks = []                                       # no Triton ranker and no cuBLASLt pool: only training runs
        stage = self.training_stage()
        if stage is not None:
            tasks = sorted(tasks + [scheduler.Task("train", stage=stage)])
        return tasks[0] if tasks else None

    # ------------------------------------------------------------------------------------------------ training
    def training_stage(self) -> str | None:
        """The next train_best stage to run, starting a training when one is needed (spec 4.5); None otherwise."""
        if not self.cfg["train"]["enabled"] or not self.hw_id:
            return None
        tr = training.Training(self.paths, self.hw_id)
        ctl = control.read_control(self.paths)
        req = ctl.get("retrain")
        if req:
            control.write_control(self.paths, {k: v for k, v in ctl.items() if k != "retrain"})   # consumed once
            self._refresh_daemon_env()
            if req.get("stages"):
                if tr.state.get("status") != "done" or tr.run_dir is None:
                    self.log.warning("retrain of %s refused: no finished training to start from", req["stages"])
                else:
                    tr.begin({"manual"}, list(req["stages"]), self.daemon_env, "requested: " + ", ".join(req["stages"]),
                             request_t=req.get("t"))
            else:
                tr.begin({"first"}, training.stages_for({"first"}, self._with_bundle()), self.daemon_env,
                         "requested (lscli train retrain)", request_t=req.get("t"))
        elif not tr.active() and tr.state.get("status") != "failed" and \
                self.clock() - self.last_train_check >= TRAIN_CHECK_S:
            self.last_train_check = self.clock()
            self._refresh_daemon_env()                      # a Triton / cuBLASLt upgrade of the daemon venv shows here
            ch = training.changes(training.trained_env(self.paths, self.hw_id), self.daemon_env)
            if ch:
                tr.begin(ch, training.stages_for(ch, self._with_bundle()), self.daemon_env,
                         "no models for this GPU" if "first" in ch else "changed: " + ", ".join(sorted(ch)))
        if not tr.active():
            self._clear_train_now()                         # nothing to train: `train start` does not linger
            return None
        stage = tr.next_stage()
        if stage is None:                                    # every stage stamped: activate the new models
            try:
                env = tr.state.get("env") or self.daemon_env  # the environment the stages ran under
                training.activate(self.paths, self.hw_id, tr.run_dir,
                                  {k: env.get(k) for k in ("triton", "cublaslt", "driver", "kernel_src", "torch")})
                tr.finish()
                self.log.info("training run %s done: models activated", tr.state["run"])
                self.load_models()
            except Exception as e:                          # never let a bad run break the loop
                self.log.error("activating run %s failed: %s: %s", tr.state["run"], type(e).__name__, e)
                tr.state["status"] = "failed"
                tr.save()
            self._clear_train_now()
            return None
        return stage

    def _refresh_daemon_env(self) -> None:
        doc = self._worker_env(self.python, {})
        if doc:
            self.daemon_env = doc

    def _clear_train_now(self) -> None:
        ctl = control.read_control(self.paths)
        if ctl.get("train_now"):
            control.write_control(self.paths, {**ctl, "train_now": False})

    def _with_bundle(self) -> bool:
        """Build the kernel bundle only when some registered env runs the daemon venv's Triton (else no hit)."""
        t = self.daemon_env.get("triton")
        return bool(t) and any(e.get("triton") == t for e in envs.load(self.paths).values())

    def run_training_stage(self, stage: str) -> None:
        """One train_best stage as a supervised job (spec 4.5). Raises Preempted / JobError."""
        from lscli.cli import source_dir
        tr = training.Training(self.paths, self.hw_id)
        run = tr.run_dir
        src = source_dir(self.paths)
        script = Path(os.environ.get("LSCLI_TRAIN_SCRIPT") or src / "pipeline" / "train_best.sh")
        argv = ["bash", str(script)]
        argv += ["--resume", str(run)] if (run / "mode").exists() else \
            ["--run-dir", str(run), "--full" if self.cfg["train"]["mode"] == "full" else "--quick"]
        mhz = self.cfg["clock"]["lock_mhz"]
        argv += ["--skip-install", "--skip-validation", "--only", stage]
        argv += ["--clock-lock", str(mhz)] if mhz else ["--clock-lock", "none", "--allow-unlocked"]
        if "bundle" in tr.state.get("stages", []):
            argv.append("--with-bundle")
        env = {**os.environ, "PATH": f"{Path(self.python).parent}{os.pathsep}{os.environ.get('PATH', '')}"}
        job = jobs.Job(argv, cwd=src, env=env, log=self._job_log(f"train-{stage}"), protocol=False)
        t0 = time.monotonic()
        try:
            code = jobs.supervise(job, watch=lambda tree_fn: self._watch(tree_fn, stage))
        except jobs.Preempted:
            hw_data = run / "data" / (self.hw_id or "")
            (hw_data / "inflight.json").unlink(missing_ok=True)   # our kill, not a crash of that candidate
            raise
        finally:
            job.close()
        self.gate.job_ended(self.clock())       # the check below must not count the stage's own utilization (as search)
        if code != 0 or stage not in tr.done():
            # train_best's own idle-GPU check or lock can lose a race with a user process that appeared after our
            # gate poll: a quick failure with the GPU now busy is a preemption, not a broken stage.
            if time.monotonic() - t0 < 60 and any(r.startswith(GPU_REASONS[0]) for r in self.gate_reasons()):
                raise jobs.Preempted()
            n = tr.record_failure(stage)
            if tr.state.get("status") == "failed":
                self._clear_train_now()
            raise jobs.JobError(f"train_best stage {stage} failed (exit {code}, attempt {n})")
        tr.record_success(stage)
        self.log.info("training stage %s done (run %s)", stage, tr.state["run"])

    def tick(self) -> scheduler.Task | None:
        """One iteration; returns the task it ran (None if it ran nothing)."""
        if self.resolve_hw():
            self.load_models()
        caches = self.caches()
        self.maintenance(caches)
        self.reasons = self.gate_reasons()
        if self.backoff_until and self.clock() >= self.backoff_until:
            self.backoff_until = None
        if self.reasons:                                   # a training stage may still run (spec 4.3 / `train start`)
            stage = self.training_stage()
            if stage is not None and not self._remaining(self.reasons, stage):
                task = scheduler.Task("train", stage=stage)
                self.run_task(task)
                return task
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
    def _remaining(self, reasons: list[str], stage: str | None) -> list[str]:
        """The reasons that still block a training stage: a CPU-only stage ignores GPU business, and after `lscli train
        start` training ignores the time window (the rest of the gate still applies)."""
        if stage is None:
            return reasons
        rest = list(reasons)
        if stage in training.CPU_STAGES:
            rest = [r for r in rest if not r.startswith(GPU_REASONS)]
        if control.read_control(self.paths).get("train_now"):
            rest = [r for r in rest if not r.startswith("outside window")]
        return rest

    def _watch(self, tree_fn, stage: str | None = None) -> bool:
        if self.stop:                                      # SIGTERM: end the job now, the clock reset still runs
            self.reasons = ["stopping"]
            return False
        r = self._remaining(self.gate_reasons(own=lambda: tree_fn() | {os.getpid()}), stage)
        if r:
            self.reasons = r
            self.log.info("preempting %s: %s", self.job_info, "; ".join(r))
        elif self.clock() - self.last_heartbeat >= HEARTBEAT_S:
            self.last_heartbeat = self.clock()
            self.write_state()
        return not r

    def run_task(self, task: scheduler.Task) -> None:
        lockf = open(self.paths.gpu_lock, "a")
        try:
            fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:                            # another lscli daemon (or a stale one) holds it: not now
            lockf.close()
            self.log.warning("gpu.lock is held by another process; skipping this tick")
            return
        self.set_state("running")
        self.job_info = ({"kind": "train", "stage": task.stage, "started": self.clock()} if task.kind == "train" else
                         {"kind": task.kind, "shape": [task.M, task.N, task.K], "dtype": task.dtype, "env_id": task.env_id,
                          "k": task.k, "reason": task.reason, "started": self.clock()})
        self.write_state()
        try:
            if task.kind == "train":
                if task.stage not in training.CPU_STAGES:
                    self.clock_lock("lock")
                self.run_training_stage(task.stage)
                self.counters["train_stages"] = self.counters.get("train_stages", 0) + 1
                return
            self.clock_lock("lock")
            self.search(task)
            self.counters["searches"] += 1
            self.failures.pop(task.key, None)
        except jobs.Preempted:
            self.counters["preempted"] += 1
            if not self.stop and not any(r == "paused" for r in self.reasons):   # a pause or stop is not a busy GPU
                self.backoff_until = self.clock() + self.cfg["idle"]["backoff_minutes"] * 60
        except Exception as e:                             # JobError or anything unexpected: count it, never loop on it
            self.counters["failed"] += 1
            if task.kind != "train":                       # a stage's failures are counted by the Training state
                self.failures[task.key] = self.failures.get(task.key, 0) + 1
            self.log.warning("task %s failed: %s: %s", task, type(e).__name__, e)
        finally:
            self.clock_lock("reset")
            self.gate.job_ended(self.clock())
            self.job_info = None
            fcntl.flock(lockf, fcntl.LOCK_UN)
            lockf.close()
            self.set_state("backoff" if self.backoff_until else "waiting")
            self.write_state()

    def _job_log(self, name: str) -> Path:
        d = self.paths.logs / "jobs"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{time.strftime('%Y%m%d-%H%M%S')}-{name}.log"

    def search(self, task: scheduler.Task) -> dict:
        """Spec 4.4: Triton proposals here (CPU), cuBLASLt ranker proposals in `lscli.propose`, measurement in the
        env's worker. With no model set the search is cuBLASLt's heuristic top lt_n alone (task.k = 0; spec 4.4).
        Raises Preempted / JobError."""
        from lscli.cli import source_dir
        from tune.online import proposal_keys, propose_triton
        rec = envs.load(self.paths).get(task.env_id)
        if rec is None:
            raise jobs.JobError(f"env {task.env_id} is no longer registered")
        s = self.cfg["search"]
        ms = self.models
        tri, rank_s = propose_triton(ms.triton() if ms else None, task.M, task.N, task.K, task.k, "diverse")
        proposals = proposal_keys(tri)
        lt_stem = ms.stems["lt_rank"] if ms else None
        lt_rank = s["lt_rank"] if (s["lt_n"] and lt_stem is not None) else 0
        cwd = source_dir(self.paths)
        if lt_rank:
            job = jobs.Job([self.python, "-m", "lscli.propose"], cwd=cwd, log=self._job_log("propose"))
            res = jobs.converse(job, {"M": task.M, "N": task.N, "K": task.K, "dtype": task.dtype, "lt_n": s["lt_n"],
                                      "lt_rank": lt_rank, "stem": str(lt_stem)},
                                watch=self._watch, silence_s=PROPOSE_TIMEOUT_S)
            proposals += res["proposals"]
        router = ms.router() if (ms and s["lt_n"] and tri) else None     # nothing to skip without Triton candidates

        def answer(msg: dict) -> dict:
            skip, p = router.skip_triton(task.M, task.N, task.K, float(msg["t_lt_us"]))
            return {"skip": bool(skip), "p": p, "threshold": router.threshold}
        req = {"M": task.M, "N": task.N, "K": task.K, "dtype": task.dtype, "k": task.k, "lt_n": s["lt_n"],
               "lt_rank": lt_rank, "confirm": 0, "router": router is not None, "proposals": proposals,
               "cache": None, "env_extra": self.env_extra(task.env_id), "order": "diverse",
               "ranker": ms.stems["triton"].name if ms else None, "reply_timeout_s": 30, "rank_s": rank_s}
        job = jobs.Job([rec["python"], "-m", "lscli.worker", "measure"], cwd=cwd, log=self._job_log("measure"))
        res = jobs.converse(job, req, watch=self._watch, answer=answer if router is not None else None)
        if res.get("skipped"):
            raise jobs.JobError(f"skipped: {res['skipped']}")
        if not (res.get("choice") or {}).get("key"):
            raise jobs.JobError("no valid kernel")
        # Final check, after the worker is gone: a foreign process that appeared since the last watch poll (<= 1 s)
        # may have overlapped the last samples. Then nothing is written (spec 4.2). The utilization NVML reports now is
        # still the worker's own: start the post-job grace first, or the check would count it as a busy GPU and restart
        # the idle clock (one search per gpu_idle_minutes instead of back to back). A foreign process still counts.
        self.gate.job_ended(self.clock())
        late = self.gate_reasons(own=set())
        if any(r.startswith("compute process") for r in late):
            self.reasons = late
            raise jobs.Preempted()
        from lscli.worker import write_entry
        write_entry(str(self.paths.winners), res["env"], res["put"])
        self._update_env(task.env_id, rec, res.get("env") or {})
        self.log.info("searched %sx%sx%s %s for %s: %s (%s), torch %.1f us", task.M, task.N, task.K, task.dtype,
                      task.env_id, res["choice"]["key"], res["choice"].get("family"), res.get("torch_us") or -1)
        return res

    def _update_env(self, eid: str, rec: dict, env: dict) -> None:
        """The worker's environment is authoritative: refresh the env's record and the cached environment doc (a driver
        or cuBLASLt change would otherwise make every fresh entry look stale to the scheduler)."""
        fresh = {k: env.get(k) for k in ("torch", "triton", "cuda_runtime") if env.get(k) is not None}
        if any(rec.get(k) != v for k, v in fresh.items()):
            rec = envs.register(self.paths, rec["python"], {**rec, **fresh})[1]
        if env:
            self.env_docs[self._env_key(eid, rec)] = (self.clock(), env)

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
