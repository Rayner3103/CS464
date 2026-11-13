"""Pack the zero-shot model set (a release asset for install.sh): Triton ranker, cuBLASLt ranker and router of one GPU,
plus the MODELS.json that lscli.models reads. Prints the archive's sha256, which install.sh pins (MODELS_SHA256).

  python install/pack_zero_shot.py --triton data/rtx3080_sm86_68sms_10gb/lgbm_rank_v2 \
      --lt-rank reports/snapshot/lock1755/lt_rank --router reports/snapshot/lock1755/router \
      --hw-id rtx3080_sm86_68sms_10gb --out zero_shot-rtx3080.tar.gz

Run from the repo root. Publishing the archive (a GitHub release asset) is a separate, manual step.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
import time
from pathlib import Path


def pack(stems: dict[str, Path], hw_id: str, out: Path) -> str:
    doc = {"hw_id": hw_id, "source": "zero_shot", "models": {}, "packed": time.time()}
    with tarfile.open(out, "w:gz") as tar:
        for kind, stem in stems.items():
            if stem is None:
                continue
            for ext in (".txt", ".json"):
                f = Path(str(stem) + ext)
                if not f.exists():
                    raise FileNotFoundError(f)
                tar.add(f, arcname=f"{stem.name}{ext}")
            doc["models"][kind] = {"stem": stem.name}
        if "triton" not in doc["models"]:
            raise ValueError("a zero-shot set needs a Triton ranker")
        data = json.dumps(doc, indent=1).encode()
        info = tarfile.TarInfo("MODELS.json")
        info.size, info.mtime = len(data), int(time.time())
        tar.addfile(info, io.BytesIO(data))
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
    print(f"{a.out}  sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
