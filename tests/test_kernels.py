import pytest
import torch

from tests.conftest import requires_cuda
from kernels.config_space import HANDPICKED, HwLimits, full_space, prune, shape_violations, smem_bytes
from kernels.triton_matmul import MatmulConfig, TritonMatmul
from baselines.triton_tutorial import TritonTutorialMatmul
from tune.correctness import validate

SHAPES = [(1, 1, 1), (1, 4096, 4096), (17, 33, 65), (128, 128, 128), (129, 257, 1031), (513, 255, 16),
          (2048, 2048, 8), (64, 64, 16384)]


@requires_cuda
@pytest.mark.parametrize("name", list(HANDPICKED))
@pytest.mark.parametrize("shape", SHAPES)
def test_handpicked_correct(name, shape):
    cfg = HANDPICKED[name]
    M, N, K = shape
    if "split_k_workspace_too_large" in shape_violations(cfg, M, N, K):
        pytest.skip("workspace limit")
    r = validate(TritonMatmul(cfg), M, N, K)
    assert r.valid, r


@requires_cuda
@pytest.mark.parametrize("split_k", [2, 4, 16])
def test_split_k_uneven_k(split_k):
    cfg = MatmulConfig(32, 32, 16, num_warps=2, num_stages=2, GROUP_M=1, SPLIT_K=split_k)
    assert validate(TritonMatmul(cfg), 100, 70, 1000).valid


@requires_cuda
def test_split_k_deterministic():
    cfg = MatmulConfig(64, 64, 32, SPLIT_K=8)
    a = torch.randn(256, 4096, device="cuda", dtype=torch.float16)
    b = torch.randn(4096, 256, device="cuda", dtype=torch.float16)
    run = TritonMatmul(cfg).make_runner(a, b)
    c1 = run().clone()
    c2 = run().clone()
    assert torch.equal(c1, c2)


@requires_cuda
def test_bf16():
    assert validate(TritonMatmul(HANDPICKED["medium"]), 300, 200, 500, dtype=torch.bfloat16).valid


@requires_cuda
def test_tutorial_baseline_correct():
    for M, N, K in [(1, 1, 1), (129, 257, 1031), (1024, 1024, 1024)]:
        assert validate(TritonTutorialMatmul(), M, N, K).valid


def test_pruning_rules():
    hw = HwLimits(smem_per_block_bytes=101376, regs_per_sm=65536, sm_count=68)
    space = full_space()
    assert len(space) == 5 * 4 * 4 * 3 * 4 * 4 * 5
    kept, dropped = prune(space, hw)
    assert 0 < len(kept) < len(space)
    assert all(smem_bytes(c) <= hw.smem_per_block_bytes for c in kept)
    assert "smem_exceeds_limit" in dropped
    # Every handpicked config must survive static pruning.
    assert set(HANDPICKED.values()) <= set(kept)


def test_shape_rules():
    big = MatmulConfig(256, 128, 64)
    assert "block_m_oversized" in shape_violations(big, 1, 4096, 4096)
    assert "block_m_oversized" not in shape_violations(big, 4096, 4096, 4096)
    sk = MatmulConfig(64, 64, 64, SPLIT_K=16)
    assert "split_k_starved" in shape_violations(sk, 64, 64, 512)
    assert "split_k_workspace_too_large" in shape_violations(sk, 16384, 16384, 16384)


@requires_cuda
def test_cublaslt_candidates_correct():
    from baselines.cublaslt import candidates
    for M, N, K in [(1, 1, 1), (129, 257, 1031), (64, 4096, 4096)]:
        cs = candidates(M, N, K, n=8)
        assert cs, (M, N, K)
        assert cs[0].cfg["heuristic_rank"] == 0
        for c in cs:
            assert validate(c, M, N, K).valid, c.key


@requires_cuda
def test_runner_validates_each_family_separately(tmp_path, monkeypatch):
    """torch and triton_tutorial both use config_key 'default'; each must get its own validation and kernel names."""
    import json
    from tune.runner import run
    monkeypatch.setenv("TUNE_CLOCK_LOCK", "none")
    path = run([{"M": 64, "N": 64, "K": 64, "suite": "t"}], ["torch", "triton_tutorial", "cublaslt:4"], tmp_path,
               cache_modes=("hot",), timing_overrides={"max_time_s": 0.05}, log=lambda *_: None)
    rows = [json.loads(l) for l in open(path)]
    fams = {r["kernel_family"] for r in rows}
    assert {"torch_matmul", "triton_tutorial_default", "cublaslt"} <= fams
    assert all(r["valid"] for r in rows)
    assert next(r for r in rows if r["kernel_family"] == "torch_matmul")["cuda_kernels"]
    assert next(r for r in rows if r["kernel_family"] == "triton_tutorial_default")["cuda_kernels"] is None
