# GPU Programming → FlashAttention: 4-Week Study Plan

**Goal:** Go from zero CUDA/Triton experience to a working, benchmarked FlashAttention forward pass implemented in Triton.
**Budget:** Week 1 already done. Weeks 2–4 at ~10–16 hrs/week (~30–48 hrs remaining).
**Scope note:** Optimization depth lives in Triton, not raw CUDA — full hand-tuned CUDA kernels to near-cuBLAS performance is a stretch goal *after* this project, not inside it.
**Compression note:** Original weeks 2–3 (memory hierarchy + profiling) and weeks 5–6 (implementation + benchmarking) have been merged. The riskiest cut is profiling practice time in Week 2 — if you're short on time anywhere, protect that over adding extra matmul-tiling variants, since it directly feeds your ability to tune the Week 4 kernel.

---

## Progress at a Glance

- [x] Week 1 — CUDA Fundamentals & First Kernels
- [ ] Week 2 — Memory Hierarchy, Naive Matmul & Profiling
- [ ] Week 3 — Triton Fundamentals, FlashAttention Algorithm & Start Implementation
- [ ] Week 4 — Finish Implementation, Optimize, Benchmark & Wrap Up

---

## Week 1 — CUDA Fundamentals & First Kernels ✅ Complete
**Key learning points**
- [x] The thread / block / grid hierarchy — how CUDA maps parallel work onto hardware
- [x] Kernel launch configuration (grid dim, block dim) and why the choice matters
- [x] Host ↔ device memory management (`cudaMalloc`, `cudaMemcpy`)
- [x] Thread indexing (`threadIdx`, `blockIdx`, `blockDim`) and bounds checking
- [x] SIMT execution — how it differs from CPU threading

---

## Week 2 — Memory Hierarchy, Naive Matmul & Profiling
**Time allocation:** ~10–16 hrs (~3 hrs theory / ~7–9 hrs matmul coding / ~2–3 hrs profiling)

**Key learning points**
- [x] Global memory coalescing — why contiguous access patterns matter
- [x] Shared memory as a manually-managed, on-chip cache
- [x] Occupancy — why more threads-in-flight doesn't always mean faster
- [x] The tiling concept for matrix multiplication
- [x] 1D block tiling — reducing global memory loads per thread
- [x] Reading Nsight Compute output: memory throughput vs. compute throughput
- [x] Diagnosing memory-bound vs. compute-bound kernels

**Tasks**
- [x] Read: PMPP ch. 5–6, or Hajj lectures 4–5
- [x] Code: implement Boehm's naive matmul kernel
- [x] Code: implement the memory-coalescing optimization pass
- [x] Code: implement shared-memory tiling
- [x] Code: implement 1D block tiling
- [x] Install Nsight Compute and profile each version — confirm you can see the improvement (and why) at each stage

**Resources**
- https://siboehm.com/articles/22/CUDA-MMM

---

## Week 3 — Triton Fundamentals, FlashAttention Algorithm & Start Implementation
**Time allocation:** ~10–16 hrs (~3 hrs Triton tutorials / ~3–4 hrs paper + notes / ~5–8 hrs starting implementation)

**Key learning points**
- [ ] Triton's block-level programming model (vs. CUDA's thread-level model)
- [ ] Core Triton primitives: `tl.program_id`, `tl.load`/`tl.store`, masking
- [ ] FlashAttention's core insight: never materialize the full N×N attention matrix
- [ ] The online softmax rescaling trick (running max/sum update)
- [ ] Why fewer HBM reads/writes translates directly to higher throughput

**Tasks**
- [ ] Work through Triton's official tutorials: vector-add, then fused-softmax
- [ ] Read the FlashAttention paper, focusing on Algorithm 1 (tiled forward pass)
- [ ] Write your own notes on the algorithm *before* touching code
- [ ] Begin the Triton kernel: implement the tiled QK^T computation and the skeleton of the online softmax loop

**Resources**
- https://github.com/openai/triton/blob/master/python/tutorials/06-fused-attention.py
- https://arxiv.org/abs/2205.14135

---

## Week 4 — Finish Implementation, Optimize, Benchmark & Wrap Up
**Time allocation:** ~10–16 hrs (~5–6 hrs finishing the kernel / ~4–5 hrs tuning & profiling / ~3–4 hrs benchmarking & writeup)

**Key learning points**
- [ ] Numerical stability considerations in the online softmax
- [ ] Validating correctness against a reference implementation (rtol/atol), not just eyeballing output
- [ ] Effect of `BLOCK_SIZE` and `num_warps` on occupancy and performance
- [ ] Sound benchmarking methodology (warm-up runs, multiple trials, varying sequence length)
- [ ] Documenting speedup/memory results as a portfolio-ready artifact

**Tasks**
- [ ] Finish the PV accumulation step
- [ ] Write a correctness test against PyTorch's `scaled_dot_product_attention`
- [ ] Tune `BLOCK_SIZE` and `num_warps`
- [ ] Benchmark throughput and memory usage vs. PyTorch's native SDPA across a few sequence lengths
- [ ] Write a short README documenting your methodology and results
- [ ] *(Stretch, only if time remains)* Add the backward pass via recomputation

**Resources**
- https://alexdremov.me/understanding-flash-attention-writing-the-algorithm-from-scratch-in-triton/

---

## Stretch Goals (post-project)
- [ ] Raw CUDA register tiling / warp tiling for matmul (continue Boehm's worklog to the end)
- [ ] Tensor core usage via `wmma` or CUTLASS
- [ ] FlashAttention backward pass in Triton (if not already done in Week 4)
- [ ] Explore FlashAttention-4 (CuTeDSL, Hopper/Blackwell) once FA2-era fundamentals are solid

## Resource Index
| Resource | Link |
|---|---|
| An Even Easier Introduction to CUDA | https://developer.nvidia.com/blog/even-easier-introduction-cuda/ |
| Izzat Hajj CUDA lectures | https://www.youtube.com/playlist?list=PLRRuQYjFhpmubuwx-w8X964ofVkW1T8O4 |
| CUDA Matmul Worklog (Simon Boehm) | https://siboehm.com/articles/22/CUDA-MMM |
| GPU MODE resource-stream | https://github.com/gpu-mode/resource-stream |
| Triton fused-attention tutorial | https://github.com/openai/triton/blob/master/python/tutorials/06-fused-attention.py |
| FlashAttention paper | https://arxiv.org/abs/2205.14135 |
| FlashAttention from scratch in Triton | https://alexdremov.me/understanding-flash-attention-writing-the-algorithm-from-scratch-in-triton/ |
