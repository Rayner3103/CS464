"""Is across-round noise explained by input allocation? Same shape, 8 fresh allocations x 3 timings each.

    python analysis/alloc_variance.py data/<hw_id>/alloc_variance.json
"""
import sys, json, numpy as np, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from tune.timing import benchmark, TimingConfig
from tune.env import gpu_telemetry
from tune.correctness import make_inputs
out = {}
for (M, N, K) in [(8192, 64, 8192), (32, 8192, 8192), (4096, 4096, 4096)]:
    allocs = []
    keep = []
    for alloc in range(8):
        # shift the allocator so each trial lands at a different physical address
        keep.append(torch.empty(int(np.random.default_rng(alloc).integers(1, 64)) * 2**20, dtype=torch.uint8, device="cuda"))
        a, b = make_inputs(M, N, K)
        c = torch.empty(M, N, device="cuda", dtype=torch.float16)
        meds = []
        for rep in range(3):
            r = benchmark(lambda: torch.matmul(a, b, out=c), TimingConfig("cold", min_reps=100, max_reps=100, target_rse=0))
            meds.append(r.median_us)
        allocs.append({"a_ptr_mod_2MB": a.data_ptr() % (2 << 20), "meds": meds, "mem_clock": gpu_telemetry()["mem_clock_mhz"], "clock": gpu_telemetry()["clock_mhz"]})
        del a, b, c
    m = np.array([x["meds"] for x in allocs])
    within = np.median((m.max(1) - m.min(1)) / np.median(m, 1))
    across = (np.median(m, 1).max() - np.median(m, 1).min()) / np.median(m)
    out[f"{M}x{N}x{K}"] = {"within_alloc_range": within, "across_alloc_range": across, "allocs": allocs}
    print(f"{M}x{N}x{K}: within-alloc range {within:.4f}  across-alloc range {across:.4f}")
    print("   per-alloc medians:", np.round(np.median(m, 1), 1).tolist())
json.dump(out, open(sys.argv[1], "w"), indent=1, default=float)
