"""Prebuilt kernel bundle (`kernels/bundle.py`, `kernels/bundle_build.py`). Most tests are CPU only (fake bundle directories,
the variant enumeration against the scalar pruning rules); the round trip through Triton needs a GPU."""
import json
import random
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from kernels import bundle as B
from kernels import bundle_build as BB
from kernels.config_space import shape_violations
from kernels.triton_matmul import MatmulConfig
from tests.conftest import requires_cuda

FP = {"triton": "3.7.1", "triton_key_sha": "abc", "target": {"backend": "cuda", "arch": 86, "warp_size": 32},
      "kernel_source_sha": "def"}


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for k in (B.BUNDLE_ENV, B.MANAGER_ENV):
        monkeypatch.setenv(k, "x")
        monkeypatch.delenv(k)                       # registered, so the test cannot leak an installed bundle
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "user_cache"))
    B._WARNED.clear()
    B.reset_stats()
    yield
    B.reset_stats()


def _entry(root: Path, key: str, stem="_matmul_kernel", files=("json", "cubin")):
    d = root / key
    d.mkdir(parents=True, exist_ok=True)
    for ext in files:
        (d / f"{stem}.{ext}").write_bytes(b"{}")


def test_manager_serves_the_bundle_and_counts_misses(clean_env, tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    _entry(bundle, "KEY1")
    _entry(bundle, "KEY2", files=("json",))                     # half an entry is not an entry
    monkeypatch.setenv(B.BUNDLE_ENV, str(bundle))
    g = B.BundleCacheManager("KEY1").get_group("_matmul_kernel.json")
    assert set(g) == {"_matmul_kernel.json", "_matmul_kernel.cubin"} and all(p.startswith(str(bundle)) for p in g.values())
    assert B.BundleCacheManager("KEY2").get_group("_matmul_kernel.json") is None
    assert B.BundleCacheManager("KEY3").get_group("_matmul_kernel.json") is None
    assert B.stats() == {"bundle_hits": 1, "local_hits": 0, "misses": 2}


def test_manager_prefers_what_this_machine_compiled_and_never_writes_to_the_bundle(clean_env, tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    _entry(bundle, "KEY1")
    monkeypatch.setenv(B.BUNDLE_ENV, str(bundle))
    m = B.BundleCacheManager("KEY1")
    cubin = m.put(b"local", "_matmul_kernel.cubin")             # what Triton does after a miss
    js = m.put("{}", "_matmul_kernel.json", binary=False)
    m.put_group("_matmul_kernel.json", {"_matmul_kernel.cubin": cubin, "_matmul_kernel.json": js})
    g = B.BundleCacheManager("KEY1").get_group("_matmul_kernel.json")
    assert g["_matmul_kernel.cubin"] == cubin and not cubin.startswith(str(bundle))
    assert B.stats()["local_hits"] == 1 and (bundle / "KEY1" / "_matmul_kernel.cubin").read_bytes() == b"{}"


def test_manager_works_without_a_bundle(clean_env):
    assert B.BundleCacheManager("KEY1").get_group("_matmul_kernel.json") is None
    assert B.stats()["misses"] == 1


def test_mismatches_names_the_reason():
    assert B.mismatches(FP, dict(FP)) == []
    assert "Triton 3.7.1 (bundle) vs 3.8.0" in B.mismatches(FP, {**FP, "triton": "3.8.0"})[0]
    assert any("GPU target" in m for m in B.mismatches(FP, {**FP, "target": {"backend": "cuda", "arch": 75, "warp_size": 32}}))
    assert any("triton_matmul.py changed" in m for m in B.mismatches(FP, {**FP, "kernel_source_sha": "zzz"}))
    assert any("compiler files differ" in m for m in B.mismatches(FP, {**FP, "triton_key_sha": "zzz"}))


def _bundle_with_manifest(tmp_path, fp):
    d = tmp_path / "bundle"
    d.mkdir()
    (d / B.MANIFEST).write_text(json.dumps({"fingerprint": fp, "n_entries": 1, "n_configs": 1, "dtypes": ["fp16"]}))
    return d


def test_install_without_a_bundle_changes_nothing(clean_env, tmp_path):
    r = B.install(tmp_path / "nothing")
    import os
    assert not r["installed"] and B.MANAGER_ENV not in os.environ


def test_install_sets_the_manager_and_warns_once_on_mismatch(clean_env, tmp_path, monkeypatch):
    import os
    d = _bundle_with_manifest(tmp_path, FP)
    monkeypatch.setattr(B, "fingerprint", lambda: dict(FP))
    r = B.install(d)
    assert r["installed"] and r["matches"] and os.environ[B.MANAGER_ENV] == B.MANAGER_SPEC
    B.uninstall()
    assert B.MANAGER_ENV not in os.environ and B.BUNDLE_ENV not in os.environ
    monkeypatch.setattr(B, "fingerprint", lambda: {**FP, "triton": "9.9.9"})
    with pytest.warns(UserWarning, match="compile on first use"):
        r = B.install(d)
    assert r["installed"] and not r["matches"]                  # installed anyway: a miss is only a compile
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        B.install(d)                                            # second call: no second warning


def test_install_does_not_override_another_cache_manager(clean_env, tmp_path, monkeypatch):
    import os
    monkeypatch.setenv(B.MANAGER_ENV, "somewhere:Else")
    r = B.install(_bundle_with_manifest(tmp_path, FP))
    assert not r["installed"] and os.environ[B.MANAGER_ENV] == "somewhere:Else"


# ------------------------------------------------------------------------------------ variant enumeration (CPU)
CFGS = [MatmulConfig(128, 128, 32, 4, 3, 8, 1), MatmulConfig(16, 64, 128, 4, 4, 1, 1), MatmulConfig(64, 64, 64, 4, 3, 1, 8),
        MatmulConfig(32, 32, 16, 2, 2, 4, 16), MatmulConfig(256, 128, 16, 8, 3, 8, 2), MatmulConfig(16, 32, 32, 2, 2, 1, 1)]


def test_class_function_matches_tritons_rule():
    assert [BB.cls(x) for x in (1, 2, 15, 16, 17, 32, 1000, 1024)] == [0, 2, 2, 1, 2, 1, 2, 1]


@pytest.mark.parametrize("cfg", CFGS, ids=lambda c: c.key())
def test_variant_representatives_are_reachable(cfg):
    ks = BB.k_variants(cfg)
    for (cm, cn, ccs), (M, N) in BB.mn_variants(cfg).items():
        assert (BB.cls(M), BB.cls(N)) == (cm, cn) and (ccs == BB.cls(M * N) if cfg.SPLIT_K > 1 else ccs == 0)
        for (ck, even), K in ks.items():
            assert BB.cls(K) == ck and even == (K % (cfg.BLOCK_K * cfg.SPLIT_K) == 0)
            assert shape_violations(cfg, M, N, K) == [], (M, N, K)


@pytest.mark.parametrize("cfg", CFGS, ids=lambda c: c.key())
def test_every_reachable_shape_falls_in_an_enumerated_variant(cfg):
    mv, kv = BB.mn_variants(cfg), BB.k_variants(cfg)
    rng = random.Random(7)
    pick = lambda extra: rng.choice([1, rng.randint(1, 16384), 16 * rng.randint(1, 1024), rng.randint(1, 64) * rng.choice([1, 2, 4, 8]),
                                     extra * rng.randint(1, 16384 // extra)])
    n = 0
    for _ in range(4000):
        M, N, K = pick(1), pick(1), min(pick(cfg.BLOCK_K * cfg.SPLIT_K), 16384)
        if shape_violations(cfg, M, N, K):
            continue
        n += 1
        assert (BB.cls(M), BB.cls(N), BB.cls(M * N) if cfg.SPLIT_K > 1 else 0) in mv, (M, N, K)
        assert (BB.cls(K), K % (cfg.BLOCK_K * cfg.SPLIT_K) == 0) in kv, (M, N, K)
    assert n > 100


def test_unreachable_variants_are_not_built():
    # M == 1 cannot take BLOCK_M = 256 (more than twice the covering tile), so that class is absent for it
    v = BB.mn_variants(MatmulConfig(256, 128, 16, 8, 3, 8, 1))
    assert all(cm != 0 for cm, _, _ in v)
    # a split-K config cannot have K == 1 (each split needs a full BLOCK_K step)
    assert all(ck != 0 for ck, _ in BB.k_variants(MatmulConfig(64, 64, 64, 4, 3, 1, 8)))


def test_reduce_tasks_cover_each_split_factor_and_mn_class():
    cfgs = [MatmulConfig(64, 64, 32, 4, 3, 1, sk) for sk in (1, 2, 8)]
    t = BB.reduce_tasks(cfgs)
    assert {x[1] for x in t} == {2, 8} and {BB.cls(x[2]) for x in t} == {0, 1, 2}


# ------------------------------------------------------------------------------------ round trip through Triton (GPU)
@requires_cuda
def test_bundle_round_trip_loads_without_compiling(tmp_path):
    """Compile one kernel into cache A, pack it with `collect`, then in a NEW process with an empty cache B and the bundle
    installed: Triton must take it from the bundle (no compile) and the kernel must run correctly."""
    root = Path(__file__).resolve().parent.parent
    cache_a, cache_b, bundle = tmp_path / "a", tmp_path / "b", tmp_path / "bundle"
    script = textwrap.dedent(f"""
        import sys, json
        sys.path.insert(0, {str(root)!r})
        import torch
        mode = sys.argv[1]
        from kernels import bundle as B
        if mode == "load":
            assert B.install({str(bundle)!r}, warn=False)["installed"]
        from kernels.triton_matmul import MatmulConfig, TritonMatmul
        cfg = MatmulConfig(64, 64, 32, 4, 3, 8, 2)                  # split-K: both kernels
        a = torch.randn(200, 256, device="cuda", dtype=torch.float16)
        b = torch.randn(256, 96, device="cuda", dtype=torch.float16)
        out = TritonMatmul(cfg).make_runner(a, b)()
        ref = a.float() @ b.float()
        print(json.dumps({{"stats": B.stats(), "err": (out.float() - ref).abs().max().item(), "scale": ref.abs().max().item()}}))
    """)
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}
    import os
    base = {**os.environ, "PYTHONPATH": str(root)}
    base.pop(B.MANAGER_ENV, None)
    base.pop(B.BUNDLE_ENV, None)
    subprocess.run([sys.executable, "-c", script, "build"], check=True, capture_output=True,
                   env={**base, "TRITON_CACHE_DIR": str(cache_a)})
    got = BB.collect(cache_a, bundle, log=lambda *_: None)
    assert got["counts"]["_matmul_kernel"] == 1 and got["counts"]["_splitk_reduce_kernel"] == 1
    assert not any("/home" in p.read_text() for p in bundle.glob("*/*.json")), "builder path left in the bundle"
    (bundle / B.MANIFEST).write_text(json.dumps({"fingerprint": B.fingerprint(), "n_entries": 2}))
    r = subprocess.run([sys.executable, "-c", script, "load"], check=True, capture_output=True, text=True,
                       env={**base, "TRITON_CACHE_DIR": str(cache_b)})
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["stats"] == {"bundle_hits": 2, "local_hits": 0, "misses": 0}, out
    assert out["err"] < 1e-2 * out["scale"] + 1.0


# ------------------------------------------------------------------------------------ pack / unpack (CPU)
def _fake_bundle(root: Path, fp=FP, n=4):
    for i in range(n):
        _entry(root, f"KEY{i}")
        (root / f"KEY{i}" / "_matmul_kernel.cubin").write_bytes(bytes([i]) * (1000 + 10 * i))
    (root / B.MANIFEST).write_text(json.dumps({"fingerprint": fp, "n_entries": n, "n_configs": 1, "dtypes": ["fp16"]}))
    return root


@pytest.fixture
def small_dict(monkeypatch):
    monkeypatch.setattr(B, "PACK_DICT_BYTES", 1 << 20)      # the shipped 128 MiB window is for the real 800 MiB bundle


def _tree(root: Path):
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_pack_unpack_round_trip(tmp_path, small_dict):
    src = _fake_bundle(tmp_path / "bundle")
    archive = B.pack(src, log=lambda *_: None)
    assert archive == tmp_path / "bundle.tar.xz" == B.archive_for(src)
    assert B.read_archive_manifest(archive)["fingerprint"] == FP        # readable without unpacking the rest
    out = B.unpack(archive, tmp_path / "out", log=lambda *_: None)
    assert _tree(out) == _tree(src)
    assert not [p for p in tmp_path.iterdir() if ".unpack-" in p.name]


def test_unpack_is_atomic(tmp_path, small_dict):
    archive = B.pack(_fake_bundle(tmp_path / "bundle"), log=lambda *_: None)
    archive.write_bytes(archive.read_bytes()[:-40])                      # a download cut short
    with pytest.raises(Exception):
        B.unpack(archive, tmp_path / "out", log=lambda *_: None)
    assert not (tmp_path / "out").exists() and not [p for p in tmp_path.iterdir() if ".unpack-" in p.name]


def test_pack_refuses_without_a_bundle(tmp_path):
    with pytest.raises(FileNotFoundError):
        B.pack(tmp_path / "nothing", log=lambda *_: None)


def test_install_unpacks_a_matching_archive_on_first_use(clean_env, tmp_path, monkeypatch, small_dict):
    import os
    B.pack(_fake_bundle(tmp_path / "made"), tmp_path / "bundle.tar.xz", log=lambda *_: None)
    monkeypatch.setenv(B.BUNDLE_ENV, str(tmp_path / "bundle"))           # a fresh clone: only the archive exists
    monkeypatch.setattr(B, "fingerprint", lambda: dict(FP))
    r = B.install()
    assert r["installed"] and r["matches"] and os.environ[B.MANAGER_ENV] == B.MANAGER_SPEC
    assert (tmp_path / "bundle" / "KEY0" / "_matmul_kernel.cubin").exists()
    assert B.install()["installed"]                                      # second call: already unpacked, nothing to do


def test_install_does_not_unpack_an_archive_for_another_environment(clean_env, tmp_path, monkeypatch, small_dict):
    import os
    B.pack(_fake_bundle(tmp_path / "made"), tmp_path / "bundle.tar.xz", log=lambda *_: None)
    monkeypatch.setenv(B.BUNDLE_ENV, str(tmp_path / "bundle"))
    monkeypatch.setattr(B, "fingerprint", lambda: {**FP, "target": {"backend": "cuda", "arch": 75, "warp_size": 32}})
    with pytest.warns(UserWarning, match="compile on first use"):
        r = B.install()
    assert not r["installed"] and "does not match" in r["reason"] and not (tmp_path / "bundle").exists()
    assert B.MANAGER_ENV not in os.environ
