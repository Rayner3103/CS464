# GPU Programming → FlashAttention: 6-Week Study Plan

**Goal:** Go from zero CUDA/Triton experience to a working, benchmarked FlashAttention forward pass implemented in Triton.
**Budget:** 5–8 hrs/week × 6 weeks (~35–45 hrs total)
**Scope note:** Optimization depth lives in Triton, not raw CUDA — full hand-tuned CUDA kernels to near-cuBLAS performance is a stretch goal *after* this project, not inside it.

---

## Progress at a Glance

- [ ] Week 1 — CUDA Fundamentals & First Kernels
- [ ] Week 2 — Memory Hierarchy & Naive Matmul
- [ ] Week 3 — Kernel Optimization & Profiling
- [ ] Week 4 — Triton & the FlashAttention Algorithm
- [ ] Week 5 — Implement FlashAttention Forward Pass
- [ ] Week 6 — Optimize, Benchmark & Wrap Up

---

## Week 1 — CUDA Fundamentals & First Kernels
**Time allocation:** ~5–7 hrs total (2–3 hrs theory / 3–4 hrs hands-on)

**Key learning points**
- [ ] The thread / block / grid hierarchy — how CUDA maps parallel work onto hardware
- [ ] Kernel launch configuration (grid dim, block dim) and why the choice matters
- [ ] Host ↔ device memory management (`cudaMalloc`, `cudaMemcpy`)
- [ ] Thread indexing (`threadIdx`, `blockIdx`, `blockDim`) and bounds checking
- [ ] SIMT execution — how it differs from CPU threading

**Tasks**
- [ ] Read/watch: NVIDIA "An Even Easier Introduction to CUDA"
- [ ] Watch: Izzat Hajj lecture series, lectures 1–3
- [ ] Code: implement a vector-add kernel
- [ ] Code: implement a simple elementwise kernel (e.g. ReLU)

**Resources**
- https://developer.nvidia.com/blog/even-easier-introduction-cuda/
- https://www.youtube.com/playlist?list=PLRRuQYjFhpmubuwx-w8X964ofVkW1T8O4

---

## Week 2 — Memory Hierarchy & Naive Matmul
**Time allocation:** ~6–8 hrs total (~2 hrs theory / 4–5 hrs hands-on)

**Key learning points**
- [ ] Global memory coalescing — why contiguous access patterns matter
- [ ] Shared memory as a manually-managed, on-chip cache
- [ ] Occupancy — why more threads-in-flight doesn't always mean faster
- [ ] The tiling concept for matrix multiplication

**Tasks**
- [ ] Read: PMPP ch. 5–6, or Hajj lectures 4–5
- [ ] Code: implement Simon Boehm's naive matmul kernel yourself, alongside his writeup
- [ ] Code: implement the memory-coalescing optimization pass

**Resources**
- https://siboehm.com/articles/22/CUDA-MMM

---

## Week 3 — Kernel Optimization & Profiling
**Time allocation:** ~5–6 hrs, mostly hands-on

**Key learning points**
- [ ] Shared-memory tiling and 1D block tiling — reducing global memory loads per thread
- [ ] Reading Nsight Compute output: memory throughput vs. compute throughput
- [ ] Diagnosing memory-bound vs. compute-bound kernels
- [ ] The qualitative idea behind a roofline model

**Tasks**
- [ ] Continue Boehm's worklog through shared-memory tiling and 1D block tiling
- [ ] Install and run Nsight Compute on your own kernels
- [ ] Profile your naive vs. tiled matmul and confirm you can read the bottleneck signal
- [ ] *(Buffer)* If ahead of schedule, start the Triton official tutorials early

**Resources**
- https://siboehm.com/articles/22/CUDA-MMM
- https://github.com/gpu-mode/resource-stream

---

## Week 4 — Triton & the FlashAttention Algorithm
**Time allocation:** ~6–7 hrs (3 hrs Triton tutorials / 3–4 hrs paper + notes)

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

**Resources**
- https://github.com/openai/triton/blob/master/python/tutorials/06-fused-attention.py
- https://arxiv.org/abs/2205.14135

---

## Week 5 — Implement FlashAttention Forward Pass
**Time allocation:** ~6–7 hrs (heaviest week — borrow time from Week 4 if needed)

**Key learning points**
- [ ] Translating the tiled algorithm into a Triton kernel loop over K/V blocks
- [ ] Rescaling the accumulated output as new blocks are processed
- [ ] Numerical stability considerations in the online softmax
- [ ] Validating correctness against a reference implementation (rtol/atol), not just eyeballing output

**Tasks**
- [ ] Implement tiled QK^T computation
- [ ] Implement the online softmax + rescaling step
- [ ] Implement the PV accumulation
- [ ] Write a correctness test against PyTorch's `scaled_dot_product_attention`

**Resources**
- https://alexdremov.me/understanding-flash-attention-writing-the-algorithm-from-scratch-in-triton/

---

## Week 6 — Optimize, Benchmark & Wrap Up
**Time allocation:** ~5–6 hrs

**Key learning points**
- [ ] Reading Nsight/Triton profiler output for a fused kernel
- [ ] Effect of `BLOCK_SIZE` and `num_warps` on occupancy and performance
- [ ] Sound benchmarking methodology (warm-up runs, multiple trials, varying sequence length)
- [ ] Documenting speedup/memory results as a portfolio-ready artifact

**Tasks**
- [ ] Tune `BLOCK_SIZE` and `num_warps`
- [ ] Benchmark throughput and memory usage vs. PyTorch's native SDPA across a few sequence lengths
- [ ] *(Stretch)* Add the backward pass via recomputation
- [ ] Write a short README documenting your methodology and results

---

## Stretch Goals (post-project)
- [ ] Raw CUDA register tiling / warp tiling for matmul (continue Boehm's worklog to the end)
- [ ] Tensor core usage via `wmma` or CUTLASS
- [ ] FlashAttention backward pass in Triton
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
