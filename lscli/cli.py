"""`lscli` command line (spec section 6). This file holds the commands that need no daemon running; the daemon, client
installation, training and install commands are added by later PRs of the stack (docs/lscli/plan.md).

Exit codes: 0 success, 1 failure, 2 usage error (argparse), 3 the daemon is needed but not running, 4 unsupported system.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from lscli import __version__, config, control, envs, queue, usage
from lscli.files import read_json
from lscli.paths import Paths

EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_NO_DAEMON, EXIT_UNSUPPORTED = 0, 1, 2, 3, 4


def _out(args, doc, text: str) -> None:
    print(json.dumps(doc, indent=1, sort_keys=True, default=str) if args.json else text)


def _ago(t: float | None) -> str:
    if not t:
        return "never"
    d = time.time() - t
    return f"{d:.0f}s ago" if d < 120 else f"{d / 60:.0f}min ago" if d < 7200 else f"{d / 3600:.1f}h ago"


# ------------------------------------------------------------------------------------------------------------ status
def cmd_status(args, paths: Paths) -> int:
    st = control.read_state(paths)
    alive = control.daemon_alive(st)
    ctl = control.read_control(paths)
    pend, _ = queue.pending(paths)
    prof = usage.load(paths)
    doc = {"daemon": {"running": alive, **(st or {})}, "control": ctl, "queue": {"pending": len(pend)},
           "envs": len(envs.load(paths)), "profile": {"shapes": len(prof["shapes"]), "updated": prof.get("updated")}}
    lines = [f"lscli {__version__}  home {paths.root}"]
    if alive:
        gate = st.get("gate") or {}
        lines.append(f"daemon: {st.get('state')} since {_ago(st.get('since'))} (pid {st.get('pid')})")
        lines.append("gate: open" if gate.get("open") else "gate: closed - " + "; ".join(gate.get("reasons") or ["?"]))
        if st.get("job"):
            lines.append(f"job: {st['job']}")
        if (st.get("models") or {}).get("training"):
            tr = st["models"]["training"]
            lines.append(f"training: {tr.get('stage')} ({tr.get('done')}/{tr.get('total')} stages)")
    else:
        lines.append("daemon: not running" + (f" (last state {st.get('state')} {_ago(st.get('since'))})" if st else ""))
    if control.paused(ctl):
        lines.append("paused" + (f" until {time.strftime('%H:%M', time.localtime(ctl['paused_until']))}"
                                 if ctl.get("paused_until") else ""))
    merged = (read_json(paths.profile, None) or {}).get("updated")
    lines.append(f"queue: {len(pend)} pending   envs: {doc['envs']}   profile: {len(prof['shapes'])} shapes "
                 f"(merged {_ago(merged)})")
    _out(args, doc, "\n".join(lines))
    return EXIT_OK if alive else EXIT_NO_DAEMON


# ------------------------------------------------------------------------------------------------------------ config
def cmd_config(args, paths: Paths) -> int:
    try:
        cfg = config.load(args.config)
        if args.key is None:
            _out(args, cfg, config.dumps(cfg).rstrip())
            return EXIT_OK
        if args.value is None:
            v = config.get(cfg, args.key)
            _out(args, {args.key: v}, config.dumps({args.key: v}).rstrip() if isinstance(v, dict) else str(v))
            return EXIT_OK
        cfg = config.set_value(args.key, args.value, args.config)
    except config.ConfigError as e:
        print(f"lscli config: {e}", file=sys.stderr)
        return EXIT_FAIL
    cur = read_json(paths.client, {}) or {}
    config.write_client_json(paths.ensure(), cfg, models=(cur.get("env_extra") or {}).get("models"),
                             gpu_uuid=cur.get("gpu_uuid"), model_stems=cur.get("models"))
    control.request_reload(paths)
    _out(args, {args.key: config.get(cfg, args.key)}, f"{args.key} = {config.get(cfg, args.key)}")
    return EXIT_OK


# ------------------------------------------------------------------------------------------------------------ envs
def cmd_envs(args, paths: Paths) -> int:
    es = envs.load(paths)
    rows = [f"{eid}  torch {e.get('torch', '?'):<14} triton {e.get('triton', '?'):<7} cuda {e.get('cuda_runtime', '?'):<5} "
            f"{e.get('python')}" for eid, e in es.items()]
    _out(args, es, "\n".join(rows) if rows else "no registered envs (lscli enable-env PYTHON)")
    return EXIT_OK


# ------------------------------------------------------------------------------------------------------------ profile
def cmd_profile(args, paths: Paths) -> int:
    cfg = config.load(args.config)
    prof = usage.merge(paths, cfg["search"]["half_life_days"], write=False)
    shapes = [s for s in prof["shapes"] if args.layout is None or s.get("layout") == args.layout][:args.top]
    lines = [f"{'M':>6} {'N':>6} {'K':>6} {'dtype':5} {'lay':3} {'calls (decayed)':>16} {'total':>12} {'hit %':>6}  last seen"]
    for s in shapes:
        tot = s.get("hits", 0) + s.get("fallbacks", 0)
        hit = f"{100 * s.get('hits', 0) / tot:5.1f}" if tot else "    -"
        lines.append(f"{s['M']:>6} {s['N']:>6} {s['K']:>6} {s['dtype']:5} {s.get('layout', 'NN'):3} {s['calls']:>16.0f} "
                     f"{s.get('calls_total', 0):>12} {hit:>6}  {_ago(s.get('last_seen'))}")
    _out(args, {**prof, "shapes": shapes}, "\n".join(lines) if shapes else "no shapes recorded yet")
    return EXIT_OK


# ------------------------------------------------------------------------------------------------------------ prewarm
_SHAPE = re.compile(r"^(\d+)[xX](\d+)[xX](\d+)$")


def _parse_shapes(args) -> list[tuple[int, int, int]]:
    out = []
    for s in args.shapes:
        m = _SHAPE.match(s)
        if not m:
            raise ValueError(f"bad shape {s!r}; use MxNxK")
        out.append(tuple(int(x) for x in m.groups()))
    if args.shapes_file:
        doc = json.loads(Path(args.shapes_file).read_text())
        out += [(int(s["M"]), int(s["N"]), int(s["K"])) for s in (doc["shapes"] if isinstance(doc, dict) else doc)]
    bad = [s for s in out if not all(1 <= x <= 16384 for x in s)]
    if bad:
        raise ValueError(f"M, N, K must be in [1, 16384]: {bad[:3]}")
    return out


def cmd_prewarm(args, paths: Paths) -> int:
    try:
        shapes = _parse_shapes(args)
        targets = envs.resolve(paths, args.env)
    except (ValueError, KeyError, OSError) as e:
        print(f"lscli prewarm: {e}", file=sys.stderr)
        return EXIT_FAIL
    if not shapes:
        print("lscli prewarm: no shapes given", file=sys.stderr)
        return EXIT_USAGE
    if not targets:
        print("lscli prewarm: no registered env (lscli enable-env PYTHON)", file=sys.stderr)
        return EXIT_FAIL
    for eid in targets:
        for M, N, K in shapes:
            queue.request(paths.ensure(), M, N, K, args.dtype, eid, "manual")
    n = len(shapes) * len(targets)
    _out(args, {"queued": n}, f"queued {len(shapes)} shape(s) x {len(targets)} env(s)")
    return EXIT_OK


# ------------------------------------------------------------------------------------------------------------ cache
def _worker(python: str, *argv: str, timeout: float = 3600) -> dict:
    """Run `python -m lscli.worker ...` and return its result line (raises on an error line or no result)."""
    r = subprocess.run([python, "-m", "lscli.worker", *argv], capture_output=True, text=True, timeout=timeout,
                       stdin=subprocess.DEVNULL)
    for line in reversed(r.stdout.strip().splitlines()):
        try:
            doc = json.loads(line)
        except ValueError:
            continue
        if doc.get("op") == "result":
            return doc
        if doc.get("op") == "error":
            raise RuntimeError(doc["error"])
    raise RuntimeError(f"no result from the worker of {python} (exit {r.returncode}): {r.stderr.strip()[-300:]}")


def env_extra_for(paths: Paths, eid: str) -> dict:
    """The env_extra a worker / client of env `eid` declares: client.json's fields plus its env id."""
    cur = read_json(paths.client, None) or config.client_document(config.load())
    return {**(cur.get("env_extra") or {}), "env_id": eid}


def _raw_entries(paths: Paths) -> list[dict]:
    from tune.winner_cache import WinnerCache
    rows = {}
    for r, _ in WinnerCache._parse(paths.winners.read_text().splitlines() if paths.winners.exists() else []):
        if r.get("kind") == "verified":
            continue
        rows[(r["M"], r["N"], r["K"], r["dtype"], json.dumps(r.get("env"), sort_keys=True))] = r
    return list(rows.values())


def cmd_cache(args, paths: Paths) -> int:
    if args.action == "clear":
        if not args.yes:
            print("lscli cache clear: deletes every stored winner; pass --yes", file=sys.stderr)
            return EXIT_USAGE
        paths.winners.unlink(missing_ok=True)
        _out(args, {"cleared": True}, f"removed {paths.winners}")
        return EXIT_OK
    try:
        targets = envs.resolve(paths, args.env)
    except KeyError as e:
        print(f"lscli cache: {e}", file=sys.stderr)
        return EXIT_FAIL
    if args.action == "ls":
        if args.env is None and len(targets) != 1:
            rows = _raw_entries(paths)      # no single env to rate tiers against: list the lines as stored
            doc = [{**{k: r[k] for k in ("M", "N", "K", "dtype")}, "key": r["choice"]["key"], "family": r["choice"].get("family"),
                    "winner_us": r.get("winner_us"), "torch_us": r.get("torch_us"), "env_id": (r.get("env") or {}).get("env_id"),
                    "tier": None} for r in rows]
        else:
            from tune.winner_cache import tier
            (eid, e), = targets.items()
            env = _worker(e["python"], "env", "--env-extra", json.dumps(env_extra_for(paths, eid)))["env"]
            doc = []
            for r in _raw_entries(paths):
                t, fields = tier(r, env)
                if args.stale and t == "current":
                    continue
                doc.append({**{k: r[k] for k in ("M", "N", "K", "dtype")}, "key": r["choice"]["key"],
                            "family": r["choice"].get("family"), "winner_us": r.get("winner_us"),
                            "torch_us": r.get("torch_us"), "env_id": (r.get("env") or {}).get("env_id"),
                            "tier": t, "stale": fields})
        lines = [f"{'M':>6} {'N':>6} {'K':>6} {'dtype':5} {'tier':8} {'winner us':>10} {'torch us':>10}  key"]
        for d in sorted(doc, key=lambda d: (d["M"], d["N"], d["K"])):
            f = lambda v: f"{v:10.1f}" if isinstance(v, (int, float)) else f"{'-':>10}"  # noqa: E731
            lines.append(f"{d['M']:>6} {d['N']:>6} {d['K']:>6} {d['dtype']:5} {d['tier'] or '?':8} {f(d['winner_us'])} "
                         f"{f(d['torch_us'])}  {d['key']}" + (f"  ({', '.join(d['stale'])})" if d.get("stale") else ""))
        _out(args, doc, "\n".join(lines) if doc else "no entries")
        return EXIT_OK
    if not targets:
        print("lscli cache: no registered env (lscli enable-env PYTHON)", file=sys.stderr)
        return EXIT_FAIL
    results, failed = {}, 0
    for eid, e in targets.items():
        extra = json.dumps(env_extra_for(paths, eid))
        try:
            if args.action == "verify":
                res = _worker(e["python"], "verify", "--cache", str(paths.winners), "--env-extra", extra)
                failed += res["failed"]
                results[eid] = {"checked": len(res["entries"]), "failed": res["failed"]}
            else:                                       # compact: each env's scope with its own environment as current
                from tune.winner_cache import WinnerCache
                env = _worker(e["python"], "env", "--env-extra", extra)["env"]
                results[eid] = WinnerCache(paths.winners, env=env).compact()
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as ex:
            results[eid] = {"error": str(ex)}
            failed += 1
    _out(args, results, "\n".join(f"{eid}: {r}" for eid, r in results.items()))
    return EXIT_FAIL if failed else EXIT_OK


# ------------------------------------------------------------------------------------------------------------ main
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="lscli", description="learned-selector CLI (docs/lscli/spec.md)")
    ap.add_argument("--home", help="state directory (default $LSCLI_HOME or ~/.local/share/lscli)")
    ap.add_argument("--config", type=Path, default=None, help="config file (default $LSCLI_CONFIG or ~/.config/lscli/config.toml)")
    ap.add_argument("--json", action="store_true", help="print one JSON document")
    ap.add_argument("--version", action="version", version=f"lscli {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="daemon state, gate, queue, profile").set_defaults(fn=cmd_status)
    c = sub.add_parser("config", help="show / get / set a config key")
    c.add_argument("key", nargs="?")
    c.add_argument("value", nargs="?")
    c.set_defaults(fn=cmd_config)
    sub.add_parser("envs", help="registered interpreters").set_defaults(fn=cmd_envs)
    p = sub.add_parser("profile", help="the usage record, hottest first")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--layout", choices=("NN", "NT"))
    p.set_defaults(fn=cmd_profile)
    w = sub.add_parser("prewarm", help="queue shapes for an idle-time search")
    w.add_argument("shapes", nargs="*", help="MxNxK")
    w.add_argument("--shapes", dest="shapes_file", help="shapes JSON ({'shapes': [{M, N, K}]})")
    w.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    w.add_argument("--env", help="env id or interpreter path (default: every registered env)")
    w.set_defaults(fn=cmd_prewarm)
    k = sub.add_parser("cache", help="inspect / verify / compact / clear the winner cache")
    k.add_argument("action", choices=("ls", "verify", "compact", "clear"))
    k.add_argument("--env", help="env id or interpreter path")
    k.add_argument("--stale", action="store_true", help="ls: only entries that are not current")
    k.add_argument("--yes", action="store_true", help="clear: confirm")
    k.set_defaults(fn=cmd_cache)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.home:
        os.environ["LSCLI_HOME"] = str(Path(args.home).expanduser().resolve())
    return args.fn(args, Paths())


if __name__ == "__main__":
    sys.exit(main())
