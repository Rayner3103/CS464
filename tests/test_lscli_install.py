"""install/install.sh, lscli doctor / uninstall and the zero-shot packer, in a sandboxed HOME. systemctl / loginctl
are stand-ins that record their arguments; the venv reuses this interpreter's packages (--reuse-site) instead of
downloading torch, so the test needs no network. The source is a clone of this repository at the current commit."""
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
FAKE_SYSTEMCTL = """#!/bin/sh
echo "$@" >> "$FAKE_LOG"
case "$*" in *show-environment*) exit 0 ;; *is-active*) echo inactive; exit 3 ;; esac
exit 0
"""


def _sandbox(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    fake = tmp_path / "fakebin"
    fake.mkdir()
    for name in ("systemctl", "loginctl"):
        (fake / name).write_text(FAKE_SYSTEMCTL)
        (fake / name).chmod(0o755)
    env = {**{k: v for k, v in os.environ.items() if not k.startswith("LSCLI_")}, "HOME": str(home),
           "PATH": f"{fake}:{os.environ['PATH']}", "FAKE_LOG": str(tmp_path / "calls.log")}
    return home, env


def _clean_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()


@pytest.mark.skipif(subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode != 0, reason="needs nvidia-smi")
def test_install_then_uninstall(tmp_path):
    home, env = _sandbox(tmp_path)
    r = subprocess.run(["bash", str(REPO / "install/install.sh"), "--repo", str(REPO), "--ref", _clean_commit(),
                        "--python", sys.executable, "--reuse-site", sys.executable, "--no-clock-helper", "--no-linger",
                        "--yes"], env=env, capture_output=True, text=True, timeout=900, stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    lh = home / ".local/share/lscli"
    assert (lh / "src/.git").is_dir() and oct(lh.stat().st_mode & 0o777) == "0o700"
    assert (home / ".config/lscli/config.toml").exists()
    unit = (home / ".config/systemd/user/lscli.service").read_text()
    assert f"Environment=LSCLI_HOME={lh}" in unit and f"ExecStart={lh}/venv/bin/python -m lscli daemon run" in unit
    calls = (tmp_path / "calls.log").read_text()
    assert "--user daemon-reload" in calls and "--user enable --now lscli.service" in calls and "enable-linger" not in calls
    v = subprocess.run([str(home / ".local/bin/lscli"), "--version"], env=env, capture_output=True, text=True)
    assert v.returncode == 0 and "lscli" in v.stdout
    st = subprocess.run([str(home / ".local/bin/lscli"), "--json", "status"], env=env, capture_output=True, text=True)
    assert json.loads(st.stdout)["daemon"]["running"] is False
    # re-running upgrades in place and keeps the data
    (lh / "keep.txt").write_text("data")
    r = subprocess.run(["bash", str(REPO / "install/install.sh"), "--repo", str(REPO), "--ref", _clean_commit(),
                        "--python", sys.executable, "--reuse-site", sys.executable, "--no-clock-helper", "--no-linger",
                        "--yes", "--no-start"], env=env, capture_output=True, text=True, timeout=900, stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stderr[-2000:]
    assert (lh / "keep.txt").exists()
    u = subprocess.run([str(home / ".local/bin/lscli"), "uninstall", "--yes"], env=env, capture_output=True, text=True)
    assert u.returncode == 0, u.stderr
    assert not (home / ".config/systemd/user/lscli.service").exists() and not (home / ".local/bin/lscli").exists()
    assert (lh / "keep.txt").exists()                                       # data kept without --purge
    assert "--user disable --now lscli.service" in (tmp_path / "calls.log").read_text()
    u = subprocess.run([sys.executable, "-m", "lscli", "uninstall", "--yes", "--purge"], env={**env, "LSCLI_HOME": str(lh)},
                       capture_output=True, text=True, cwd=REPO)
    assert u.returncode == 0 and not lh.exists() and not (home / ".config/lscli").exists()


def test_install_preflight_rejects_unsupported_systems(tmp_path):
    home, env = _sandbox(tmp_path)
    (tmp_path / "fakebin" / "systemctl").write_text("#!/bin/sh\nexit 1\n")       # no user session
    r = subprocess.run(["bash", str(REPO / "install/install.sh"), "--check-only"], env=env, capture_output=True, text=True)
    assert r.returncode == 4 and "systemd user session" in r.stderr
    r = subprocess.run(["bash", str(REPO / "install/install.sh"), "--bogus"], env=env, capture_output=True, text=True)
    assert r.returncode == 2


def test_uninstall_removes_env_hooks(tmp_path, monkeypatch):
    from lscli import cli, envs
    from lscli.paths import Paths
    home, env = _sandbox(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", env["PATH"])
    monkeypatch.setenv("FAKE_LOG", env["FAKE_LOG"])
    monkeypatch.setenv("LSCLI_HOME", str(tmp_path / "lh"))
    monkeypatch.setenv("LSCLI_CONFIG", str(tmp_path / "c.toml"))
    import venv
    venv.EnvBuilder(with_pip=False).create(tmp_path / "v")
    py = tmp_path / "v" / "bin" / "python"
    site = next((tmp_path / "v" / "lib").glob("python3*/site-packages"))
    (site / "lscli.pth").write_text("x\n")
    envs.register(Paths().ensure(), py, {"torch": "t", "triton": "t", "prefix": str(tmp_path / "v")})
    assert cli.main(["uninstall", "--yes"]) == 0
    assert not (site / "lscli.pth").exists() and envs.load(Paths()) == {}


def test_doctor_reports_and_flags_a_lock_without_helper(tmp_path, monkeypatch, capsys):
    from lscli import cli
    home, env = _sandbox(tmp_path)
    monkeypatch.setenv("PATH", env["PATH"])
    monkeypatch.setenv("FAKE_LOG", env["FAKE_LOG"])
    monkeypatch.setenv("LSCLI_HOME", str(tmp_path / "lh"))
    cfg = tmp_path / "c.toml"
    monkeypatch.setenv("LSCLI_CONFIG", str(cfg))
    code = cli.main(["--json", "doctor"])
    rows = {r["check"]: r for r in json.loads(capsys.readouterr().out)}
    assert {"GPU", "systemd user session", "daemon unit", "clock helper", "models", "envs"} <= set(rows)
    if not Path("/usr/local/libexec/lscli-clock").exists():
        cfg.write_text("[clock]\nlock_mhz = 1500\n")
        assert cli.main(["--json", "doctor"]) == 1                               # a lock with no helper is a FAIL
        rows = {r["check"]: r for r in json.loads(capsys.readouterr().out)}
        assert rows["clock helper"]["status"] == "FAIL"
    assert code in (0, 1)


def test_clock_helper_refuses_anything_but_lock_and_reset():
    for argv in ([], ["lock", "1500"], ["set"], ["--help"]):
        r = subprocess.run(["bash", str(REPO / "install/lscli-clock"), *argv], capture_output=True, text=True)
        assert r.returncode in (2, 3), argv
    r = subprocess.run(["bash", str(REPO / "install/lscli-clock"), "lock"], capture_output=True, text=True)
    assert r.returncode == 3 or Path("/etc/lscli/clock.conf").exists()      # no root-owned clock.conf here


def test_pack_zero_shot(tmp_path):
    sys.path.insert(0, str(REPO / "install"))
    import pack_zero_shot
    for stem in ("tri", "lt"):
        for ext in (".txt", ".json"):
            (tmp_path / f"{stem}{ext}").write_text(stem)
    out = tmp_path / "z.tar.gz"
    digest = pack_zero_shot.pack({"triton": tmp_path / "tri", "lt_rank": tmp_path / "lt", "router": None},
                                 "rtx3080_sm86_68sms_10gb", out)
    assert len(digest) == 64
    with tarfile.open(out) as t:
        names = set(t.getnames())
        doc = json.loads(t.extractfile("MODELS.json").read())
    assert names == {"tri.txt", "tri.json", "lt.txt", "lt.json", "MODELS.json"}
    assert doc["source"] == "zero_shot" and doc["models"] == {"triton": {"stem": "tri"}, "lt_rank": {"stem": "lt"}}
    with pytest.raises(ValueError):
        pack_zero_shot.pack({"triton": None}, "x", tmp_path / "y.tar.gz")
