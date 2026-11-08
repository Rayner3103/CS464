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


def _archive(tmp_path, name, evil=False):
    import hashlib
    sys.path.insert(0, str(REPO / "install"))
    import pack_zero_shot
    d = tmp_path / f"m-{name}"
    d.mkdir()
    for ext in (".txt", ".json"):
        (d / f"tri{ext}").write_text("x")
    out = tmp_path / f"{name}.tar.gz"
    pack_zero_shot.pack({"triton": d / "tri"}, "rtx3080_sm86_68sms_10gb", out)
    if evil:                                                                 # add a symlink member pointing outside
        import io
        with tarfile.open(out, "r:gz") as t:
            members = [(m, t.extractfile(m).read()) for m in t.getmembers()]
        with tarfile.open(out, "w:gz") as t:
            for m, data in members:
                t.addfile(m, io.BytesIO(data))
            link = tarfile.TarInfo("escape")
            link.type, link.linkname = tarfile.SYMTYPE, "/etc"
            t.addfile(link)
    return f"file://{out}", hashlib.sha256(out.read_bytes()).hexdigest()


needs_gpu = pytest.mark.skipif(subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode != 0,
                               reason="needs nvidia-smi")


@needs_gpu
def test_install_refuses_an_archive_with_links(tmp_path):
    home, env = _sandbox(tmp_path)
    url, digest = _archive(tmp_path, "evil", evil=True)
    r = subprocess.run(["bash", str(REPO / "install/install.sh"), "--repo", str(REPO), "--ref", _clean_commit(),
                        "--python", sys.executable, "--reuse-site", sys.executable, "--no-clock-helper", "--no-linger",
                        "--yes", "--models-url", url, "--models-sha256", digest], env=env, capture_output=True, text=True,
                       timeout=900, stdin=subprocess.DEVNULL)
    assert r.returncode != 0 and "unsafe or broken model archive" in r.stderr
    r = subprocess.run(["bash", str(REPO / "install/install.sh"), "--repo", str(REPO), "--ref", _clean_commit(),
                        "--python", sys.executable, "--reuse-site", sys.executable, "--no-clock-helper", "--no-linger",
                        "--yes", "--models-url", url, "--models-sha256", "0" * 64], env=env, capture_output=True,
                       text=True, timeout=900, stdin=subprocess.DEVNULL)
    assert r.returncode != 0 and "sha256 mismatch" in r.stderr


@needs_gpu
def test_install_then_uninstall(tmp_path):
    home, env = _sandbox(tmp_path)
    url, digest = _archive(tmp_path, "good")
    r = subprocess.run(["bash", str(REPO / "install/install.sh"), "--repo", str(REPO), "--ref", _clean_commit(),
                        "--python", sys.executable, "--reuse-site", sys.executable, "--no-clock-helper", "--no-linger",
                        "--yes", "--models-url", url, "--models-sha256", digest], env=env, capture_output=True, text=True,
                       timeout=900, stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    lh = home / ".local/share/lscli"
    assert (lh / "src/.git").is_dir() and oct(lh.stat().st_mode & 0o777) == "0o700"
    assert (home / ".config/lscli/config.toml").exists()
    unit = (home / ".config/systemd/user/lscli.service").read_text()
    assert f"Environment=LSCLI_HOME={lh}" in unit and f"ExecStart={lh}/venv/bin/python -m lscli daemon run" in unit
    calls = (tmp_path / "calls.log").read_text()
    assert "--user daemon-reload" in calls and "--user enable --now lscli.service" in calls and "enable-linger" not in calls
    assert json.loads((lh / "models/zero_shot/MODELS.json").read_text())["source"] == "zero_shot"
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
    # no --models-url: the committed archive (install/zero_shot.tar.xz) replaces the downloaded set
    doc = json.loads((lh / "models/zero_shot/MODELS.json").read_text())
    assert doc["hw_id"] == "rtx3080_sm86_68sms_10gb" and set(doc["models"]) == {"triton", "lt_rank", "router"}
    assert (lh / "models/zero_shot/lgbm_rank_v2.txt").stat().st_size > 1_000_000 and "zero-shot models" in r.stdout
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
        assert r.returncode == 2, argv


def _helper_copy(tmp_path):
    """The helper with its two paths pointed at a stub nvidia-smi and a test clock.conf, and the root-owner check
    accepting this user (the installed helper never takes paths from its environment)."""
    stub = tmp_path / "nvidia-smi"
    stub.write_text('#!/bin/sh\ncase "$*" in *clocks.max.sm*) echo 1950 ;; *) echo "CALLED $*" ;; esac\n')
    stub.chmod(0o755)
    conf = tmp_path / "clock.conf"
    text = (REPO / "install/lscli-clock").read_text()
    text = text.replace("CONF=/etc/lscli/clock.conf", f"CONF={conf}").replace("NVSMI=/usr/bin/nvidia-smi", f"NVSMI={stub}")
    text = text.replace('"$(stat -c %u "$CONF")" == 0', f'"$(stat -c %u "$CONF")" == {os.getuid()}')
    helper = tmp_path / "lscli-clock"
    helper.write_text(text)
    return helper, conf


@pytest.mark.parametrize("content, ok", [("lock_mhz=1500\n", True), ("lock_mhz=0800\n", False), ("lock_mhz=2500\n", False),
                                        ("lock_mhz=200\n", False), ("lock_mhz=1500; rm -rf /\n", False), ("", False)])
def test_clock_helper_lock_validates_the_conf(tmp_path, content, ok):
    helper, conf = _helper_copy(tmp_path)
    conf.write_text(content)
    r = subprocess.run(["bash", str(helper), "lock"], capture_output=True, text=True)
    assert (r.returncode == 0 and "CALLED -i 0 -lgc 1500,1500" in r.stdout) if ok else r.returncode == 3, (content, r)


def test_clock_helper_rejects_a_symlinked_conf_and_resets(tmp_path):
    helper, conf = _helper_copy(tmp_path)
    (tmp_path / "real.conf").write_text("lock_mhz=1500\n")
    conf.symlink_to(tmp_path / "real.conf")
    assert subprocess.run(["bash", str(helper), "lock"], capture_output=True).returncode == 3
    r = subprocess.run(["bash", str(helper), "reset"], capture_output=True, text=True)
    assert r.returncode == 0 and "CALLED -i 0 -rgc" in r.stdout


def test_purge_refuses_home_and_foreign_dirs_and_removes_only_the_config_file(tmp_path, monkeypatch, capsys):
    from lscli import cli
    home, env = _sandbox(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", env["PATH"])
    monkeypatch.setenv("FAKE_LOG", env["FAKE_LOG"])
    shared = tmp_path / "projects"
    shared.mkdir()
    (shared / "work.txt").write_text("keep me")
    monkeypatch.setenv("LSCLI_CONFIG", str(shared / "c.toml"))
    (shared / "c.toml").write_text("")
    for root in (home, tmp_path / "empty"):
        root.mkdir(exist_ok=True)
        monkeypatch.setenv("LSCLI_HOME", str(root))
        assert cli.main(["uninstall", "--yes", "--purge"]) == cli.EXIT_FAIL and root.exists()
    lh = tmp_path / "lh"
    (lh / "venv").mkdir(parents=True)
    monkeypatch.setenv("LSCLI_HOME", str(lh))
    foreign = home / ".local" / "bin" / "lscli"
    foreign.parent.mkdir(parents=True)
    foreign.write_text("#!/bin/sh\necho my own lscli tool\n")
    assert cli.main(["uninstall", "--yes", "--purge"]) == 0
    assert not lh.exists() and not (shared / "c.toml").exists() and (shared / "work.txt").exists()
    assert foreign.exists()                                                 # not install.sh's wrapper: left alone


def test_install_refuses_a_home_that_is_not_its_own(tmp_path):
    home, env = _sandbox(tmp_path)
    for bad in (str(home), "/", "relative/dir", f"{home}/with space"):
        r = subprocess.run(["bash", str(REPO / "install/install.sh"), "--home", bad, "--check-only"], env=env,
                           capture_output=True, text=True)
        assert r.returncode != 0, bad


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
    # .tar.xz, and reproducible: the same models pack to the same bytes (a repack changes no committed byte)
    stems = {"triton": tmp_path / "tri", "lt_rank": tmp_path / "lt"}
    assert pack_zero_shot.pack(stems, "h", tmp_path / "a.tar.xz") == pack_zero_shot.pack(stems, "h", tmp_path / "b.tar.xz")
    first = pack_zero_shot.pack(stems, "h", tmp_path / "a.tar.gz")
    import time
    time.sleep(1.1)                                    # a gzip header with the time in it would differ now
    assert pack_zero_shot.pack(stems, "h", tmp_path / "other-name.tar.gz") == first
    (tmp_path / "sub").mkdir()
    for ext in (".txt", ".json"):
        (tmp_path / "sub" / f"tri{ext}").write_text("other")
    with pytest.raises(ValueError, match="two models"):
        pack_zero_shot.pack({"triton": tmp_path / "tri", "router": tmp_path / "sub/tri"}, "h", tmp_path / "d.tar.xz")
    with tarfile.open(tmp_path / "a.tar.xz", "r:xz") as t:
        assert all(m.mtime == 0 and m.uid == 0 and m.isfile() for m in t.getmembers())
    with pytest.raises(ValueError):
        pack_zero_shot.pack(stems, "h", tmp_path / "c.zip")


def test_the_committed_zero_shot_set(tmp_path):
    """install/zero_shot.tar.xz: the RTX 3080's rank v2 + cuBLASLt ranker + router, loadable by lscli.models; on another
    architecture only the Triton ranker is used (spec 3.7)."""
    from lscli import models
    from lscli.paths import Paths
    arc = REPO / "install/zero_shot.tar.xz"
    assert arc.stat().st_size < 20 * 2**20
    home = Paths(tmp_path / "home").ensure()
    with tarfile.open(arc, "r:xz") as t:
        assert all(m.isfile() and "/" not in m.name for m in t.getmembers())
        t.extractall(home.models / "zero_shot", filter="data")
    doc = json.loads((home.models / "zero_shot/MODELS.json").read_text())
    assert doc == {"hw_id": "rtx3080_sm86_68sms_10gb", "source": "zero_shot",
                   "models": {"triton": {"stem": "lgbm_rank_v2"}, "lt_rank": {"stem": "lt_rank"}, "router": {"stem": "router"}}}
    snap = REPO / "reports/snapshot/lock1755"
    for stem in ("lt_rank", "router"):                                   # the same models as the committed snapshot
        assert models.model_id(home.models / "zero_shot" / stem).split(":")[1] == models.model_id(snap / stem).split(":")[1]
    assert (home.models / "zero_shot/lgbm_rank_v2.json").read_bytes() == (snap / "lgbm_rank_v2.json").read_bytes()
    import hashlib                                     # the 3080's rank v2 booster (not committed elsewhere: 35 MB)
    assert hashlib.sha256((home.models / "zero_shot/lgbm_rank_v2.txt").read_bytes()).hexdigest() == \
        "7d0ce7610c62af97a2b2c5ac772627869fac25ae0e9b5501729f3d637fbcf06c"
    same = models.active(home, "rtx3090_sm86_82sms_24gb")
    assert same.source == "zero_shot" and same.complete
    other = models.active(home, "rtx2060_sm75_30sms_6gb")
    assert other.stems["triton"] is not None and other.stems["lt_rank"] is None and other.stems["router"] is None
    pytest.importorskip("lightgbm")
    r = other.triton()
    assert r.rank(1024, 1024, 1024, 3, order="diverse")
