# CS464: learned kernel selection for GPU GEMM

Phase 1 sets the baselines before any ML. The full plan is in [`plan.md`](plan.md), the measurement protocol in
[`docs/methodology.md`](docs/methodology.md), and the latest findings in [`reports/week1.md`](reports/week1.md).

## Layout

```
tune/        harness: env capture, timing, correctness gate, records, runner, noise/drift, CLI
kernels/     parameterized Triton GEMM family + config space / pruning rules
baselines/   B0 roofline, B1 torch.matmul (cuBLAS), B5 Triton tutorial kernel
suites/      shape suites (versioned JSON; regenerate with `python -m suites.generate`)
analysis/    report generation (plots + tables)
tests/       unit tests (`python -m pytest`)
data/<hw>/   results + env.json (gitignored)
reports/     write-ups and figures
```

## Usage

The project uses the existing Python environment, pinned in `requirements.txt`. Run everything from the repo root:

```bash
python -m pytest -q                                  # unit tests (needs a CUDA GPU for most)
python -m tune env                                   # data/<hw_id>/env.json
python -m tune check --suite edge                    # correctness gate over the edge-shape suite
python -m tune roofline                              # B0: measured peak TFLOP/s and DRAM GB/s
python -m tune noise --rounds 3                      # noise-floor study (run again on other days)
python -m tune drift --minutes 10                    # thermal drift
python -m tune baseline --suite S1,S2 --which torch,triton_tutorial --cache-modes cold,hot
python -m tune benchmark --suite S1,S2 --backends triton:handpicked --cache-modes cold,hot --budget 30m
python -m tune analyze                               # reports/week1_results.md + reports/figures/
```

`benchmark` and `baseline` append to `data/<hw_id>/results.jsonl` and can be resumed: re-running skips
measurements that are already done.
