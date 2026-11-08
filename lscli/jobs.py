"""Running daemon jobs as killable process trees (spec 4.2, 4.4).

Every job process starts in its own session (process group). `Job.kill()` sends SIGTERM to the group, waits, then
SIGKILLs the group and every descendant seen by the last /proc walk (spawned compile workers can leave the group).
A `Watch` callback decides, every second while the job runs, whether the gate is still open; when it is not, the job is
killed and `Preempted` is raised. The search job talks to the worker over its line protocol (raw fd + own buffer,
never `communicate()`), answering router questions with the daemon's router model.
"""
from __future__ import annotations

import json
import os
import select
import signal
import subprocess
import time
from typing import Callable

from lscli.gate import descendants


class Preempted(Exception):
    """The gate closed while the job ran; the job was killed and wrote nothing (spec 4.2)."""


class JobError(Exception):
    pass


class Job:
    def __init__(self, argv: list[str], *, cwd=None, env=None, log=None):
        self.log = open(log, "ab") if log else subprocess.DEVNULL
        self.p = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=self.log, start_new_session=True)
        self.seen: set[int] = {self.p.pid}
        self.buf = b""

    def tree(self) -> set[int]:
        if self.p.poll() is None:
            self.seen |= descendants(self.p.pid)
        return {p for p in self.seen if _alive(p)} | ({self.p.pid} if self.p.poll() is None else set())

    def send(self, obj: dict) -> None:
        self.p.stdin.write((json.dumps(obj) + "\n").encode())
        self.p.stdin.flush()

    def readline(self, timeout_s: float) -> str | None:
        """Next protocol line, or None after timeout_s / at end of output."""
        fd = self.p.stdout.fileno()
        deadline = time.monotonic() + timeout_s
        while b"\n" not in self.buf:
            wait = deadline - time.monotonic()
            if wait <= 0:
                return None
            r, _, _ = select.select([fd], [], [], min(wait, 1.0))
            if not r:
                continue
            chunk = os.read(fd, 65536)
            if not chunk:
                return None
            self.buf += chunk
        line, self.buf = self.buf.split(b"\n", 1)
        return line.decode()

    def kill(self, grace_s: float = 5.0) -> None:
        pids = self.tree()
        try:
            os.killpg(self.p.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            self.p.wait(grace_s)
        except subprocess.TimeoutExpired:
            pass
        pids |= {p for p in self.seen if _alive(p)}
        try:
            os.killpg(self.p.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        try:
            self.p.wait(5)
        except subprocess.TimeoutExpired:
            pass
        self.close()

    def close(self) -> None:
        for f in (self.p.stdin, self.p.stdout):
            try:
                f.close()
            except Exception:
                pass
        if self.log is not subprocess.DEVNULL:
            self.log.close()


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:                                      # a zombie is dead for our purposes
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def converse(job: Job, request: dict, *, watch: Callable[[set], bool], answer: Callable[[dict], dict] | None = None,
             silence_s: float = 600.0) -> dict:
    """Send one request, then read protocol lines until the result, answering `route` questions; `watch(tree)` is
    called about once a second and returning False preempts. Returns the result document; raises Preempted or
    JobError."""
    try:
        job.send(request)
    except (BrokenPipeError, OSError) as e:
        job.kill()
        raise JobError(f"job exited before taking its request: {e}") from None
    last = time.monotonic()
    while True:
        if not watch(job.tree()):
            job.kill()
            raise Preempted()
        line = job.readline(1.0)
        if line is None:
            if job.p.poll() is not None and not job.buf:
                code = job.p.returncode
                job.close()
                raise JobError(f"job exited ({code}) without a result")
            if time.monotonic() - last > silence_s:
                job.kill()
                raise JobError(f"no output for {silence_s:.0f}s")
            continue
        last = time.monotonic()
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        op = msg.get("op")
        if op == "route":
            ans = answer(msg) if answer else {"skip": False}
            try:
                job.send(ans)
            except (BrokenPipeError, OSError):
                pass
        elif op == "result":
            try:
                job.p.wait(30)
            except subprocess.TimeoutExpired:
                job.kill()
            job.close()
            return msg
        elif op == "error":
            try:
                job.p.wait(30)
            except subprocess.TimeoutExpired:
                job.kill()
            job.close()
            raise JobError(msg.get("error", "unknown error"))
