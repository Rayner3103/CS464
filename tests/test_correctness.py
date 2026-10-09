import pytest
import torch

from tests.conftest import requires_cuda
from tune.correctness import check_output, make_inputs, reference, validate

pytestmark = requires_cuda


class _Backend:
    def __init__(self, fn):
        self.fn = fn

    def make_runner(self, a, b, out=None):
        def run():
            return self.fn(a, b, out)
        return run


def _good(a, b, out):
    return torch.matmul(a, b, out=out)


def test_reference_passes():
    assert validate(_Backend(_good), 129, 257, 1031).valid


def test_detects_wrong_tile():
    def bad(a, b, out):
        torch.matmul(a, b, out=out)
        out[64:80, 32:48] += 1.0  # one corrupted 16x16 tile
        return out
    r = validate(_Backend(bad), 128, 128, 4096)
    # Error 1.0 vs atol 0.01*sqrt(4096)=0.64 -> must fail.
    assert not r.valid and r.failure_reason == "tolerance_exceeded"


def test_detects_unwritten_elements():
    def partial(a, b, out):
        out[:-1] = a[:-1] @ b  # forgets the last row; buffer was NaN-filled
        return out
    r = validate(_Backend(partial), 65, 64, 64)
    assert not r.valid and r.failure_reason.startswith("non_finite")


def test_detects_fp16_accumulation():
    # Accumulating in fp16 over a long K drifts well beyond the fp32-accumulate tolerance.
    def fp16_acc(a, b, out):
        acc = torch.zeros(a.shape[0], b.shape[1], device=a.device, dtype=torch.float16)
        for k in range(0, a.shape[1], 16):
            acc += a[:, k:k + 16] @ b[k:k + 16]
        out.copy_(acc)
        return out
    r = validate(_Backend(fp16_acc), 64, 64, 8192)
    assert not r.valid


def test_exceptions_become_failures():
    def boom(a, b, out):
        raise RuntimeError("out of resource: shared memory")
    r = validate(_Backend(boom), 16, 16, 16)
    assert not r.valid and "RuntimeError" in r.failure_reason


def test_reference_is_fp32_not_tf32():
    a, b = make_inputs(256, 256, 256)
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        ref = reference(a, b)
        exact = (a.double() @ b.double()).float()
        assert (ref - exact).abs().max() < 1e-3
    finally:
        torch.backends.cuda.matmul.allow_tf32 = False
