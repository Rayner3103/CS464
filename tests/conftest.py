"""Shared test helpers: the CUDA skip marker and `shipped_ranker`, the Triton ranker every user install has, so the
lscli daemon / training tests that need a real model file run on any machine with a clone (not only where a particular
pipeline run directory happens to exist)."""
from __future__ import annotations

import json
import os
import tarfile
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
ZERO_SHOT_ARCHIVE = REPO / "install" / "zero_shot.tar.xz"
# lscli.paths.home(), read at import: the tests' own `home` fixtures point LSCLI_HOME at a temp dir later
USER_HOME = Path(os.environ.get("LSCLI_HOME") or Path.home() / ".local" / "share" / "lscli")

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def triton_stem(models_dir: Path) -> Path | None:
    """The Triton ranker stem a model-set directory's MODELS.json names (lscli.models layout), if both files exist."""
    try:
        doc = json.loads((models_dir / "MODELS.json").read_text())
        stem = models_dir / doc["models"]["triton"]["stem"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return stem if all(Path(f"{stem}{ext}").is_file() for ext in (".txt", ".json")) else None


@pytest.fixture(scope="session")
def shipped_ranker(tmp_path_factory) -> Path:
    """Stem (no .txt / .json) of the Triton ranker a user install has: the zero-shot set install.sh unpacks into
    $LSCLI_HOME/models/zero_shot, else the same archive (install/zero_shot.tar.xz, part of every clone) unpacked into a
    session temp dir with the installer's checks. Skips only if neither exists."""
    stem = triton_stem(USER_HOME / "models" / "zero_shot")
    if stem is not None:
        return stem
    if not ZERO_SHOT_ARCHIVE.is_file():
        pytest.skip(f"no installed zero-shot models in {USER_HOME} and no {ZERO_SHOT_ARCHIVE.relative_to(REPO)}")
    out = tmp_path_factory.mktemp("zero_shot")
    with tarfile.open(ZERO_SHOT_ARCHIVE, "r:*") as t:
        for m in t.getmembers():                                    # same member rule as install.sh
            if not m.isfile() or m.name.startswith("/") or ".." in m.name.split("/"):
                pytest.fail(f"{ZERO_SHOT_ARCHIVE.name}: refusing member {m.name!r}")
        t.extractall(out, filter="data")
    stem = triton_stem(out)
    if stem is None:
        pytest.fail(f"{ZERO_SHOT_ARCHIVE.name} has no Triton ranker in its MODELS.json")
    return stem
