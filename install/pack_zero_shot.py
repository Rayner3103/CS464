"""Pack the zero-shot model set that install.sh unpacks into $LSCLI_HOME/models/zero_shot: one GPU's Triton ranker,
cuBLASLt ranker and router, plus the MODELS.json that lscli.models reads. The committed set is install/zero_shot.tar.xz
(the RTX 3080's rank v2, lt_rank and router); rebuild it with

  python install/pack_zero_shot.py --triton <dir>/lgbm_rank_v2 \
      --lt-rank reports/snapshot/lock1755/lt_rank --router reports/snapshot/lock1755/router \
      --hw-id rtx3080_sm86_68sms_10gb --out install/zero_shot.tar.xz

Run from the repo root. The compression follows --out (.tar.xz or .tar.gz). The archive is reproducible (sorted members,
fixed owner and mtime, no pack time in MODELS.json), so repacking the same models changes no committed byte. Each
rebuild that does change it adds its size (~11 MB) to git history. Prints the sha256 (for --models-sha256 when an
archive is served from a URL instead).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path


def _mode(out: Path) -> str:
    name = out.name
    if name.endswith((".tar.xz", ".txz")):
        return "w:xz"
    if name.endswith((".tar.gz", ".tgz")):
        return "w:gz"
    raise ValueError(f"{out}: expected a .tar.xz or .tar.gz name")


def _info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size, info.mtime, info.mode = size, 0, 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


def pack(stems: dict[str, Path | None], hw_id: str, out: Path) -> str:
    doc = {"hw_id": hw_id, "source": "zero_shot", "models": {}}
    files: dict[str, bytes] = {}
    for kind, stem in stems.items():
        if stem is None:
            continue
        stem = Path(stem)
        for ext in (".txt", ".json"):
            f = Path(str(stem) + ext)
            if not f.exists():
                raise FileNotFoundError(f)
            files[f"{stem.name}{ext}"] = f.read_bytes()
        doc["models"][kind] = {"stem": stem.name}
    if "triton" not in doc["models"]:
        raise ValueError("a zero-shot set needs a Triton ranker")
    files["MODELS.json"] = (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode()
    kw = {"preset": 9} if _mode(out) == "w:xz" else {}
    with tarfile.open(out, _mode(out), format=tarfile.PAX_FORMAT, **kw) as tar:
        for name in sorted(files):
            tar.addfile(_info(name, len(files[name])), io.BytesIO(files[name]))
    return hashlib.sha256(out.read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--triton", type=Path, required=True)
    ap.add_argument("--lt-rank", type=Path)
    ap.add_argument("--router", type=Path)
    ap.add_argument("--hw-id", required=True, help="the GPU the models were trained on")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    digest = pack({"triton": a.triton, "lt_rank": a.lt_rank, "router": a.router}, a.hw_id, a.out)
    print(f"{a.out}  {a.out.stat().st_size / 2**20:.1f} MiB  sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
