# Week 1: Phase 0 + first baselines (RTX 3080)

**Date:** 2026-10-09 · **hw_id:** `rtx3080_sm86_68sms_10gb` · driver 595.91.07 · torch 2.13.0+cu130 · triton 3.7.1
**Scope:** milestone M0 and the plan §13 first-week checklist. All tables are generated in
[`week1_results.md`](week1_results.md) (`python -m tune analyze`). The protocol is in [`../docs/methodology.md`](../docs/methodology.md).

## Update 2026-10-27: fp16-accumulation test was seed-dependent (found on an RTX 2060 test run)

**What ran.** A `pytest` run on an RTX 2060 (sm_75) gave 94 passed, 2 failed. No benchmark rows were produced, so no session ids apply.
- `test_detects_fp16_accumulation` failed: the fake fp16-accumulating kernel (K = 8192) passed the correctness gate. Not a 2060 bug. Re-run on the RTX 3080 over 20 seeds,
  the gate rejects that kernel on only **9 of 20** seeds (a correct fp32-accumulate kernel is accepted 20 of 20). The test used seed 0, which happens to be a rejecting seed on the 3080.
  A different GPU can draw different inputs from the same seed (PyTorch's CUDA `randn` layout depends on the SM count; not verified on the 2060).
- `test_handpicked_correct[shape0-large_square]` failed with `ptxas failed with error code -1` and an empty stderr at shape (1, 1, 1), config 128x256x32 / 8 warps for `sm_75`.
  The PTX in the log compiles cleanly with Triton's ptxas for `sm_75` on the 3080 machine (exit 0), and the same config passed on other shapes, so this looks like a killed process
  (memory or limits) on the 2060 machine. **Unresolved:** not reproduced, not fixed.

**What the gate does and does not catch.** RMS error against the fp32 reference, fp16-accumulate over fp32-accumulate, on the same inputs: about 11x at K = 8192, 15-16x at K = 16384,
about 32x at K = 65536. The gate's relative tolerance (1e-2 of |ref|) is roughly 20x above a correct kernel's error, so a kernel that accumulates in fp16 can slip through at short K.
Nonzero-mean inputs did not help (mean 1: rejected 5 of 20 at K = 8192, 0 of 20 at K = 2048).

**Change.** `tests/test_correctness.py`: the gate test now uses K = 16384 (rejected 20 of 20 seeds on the 3080), and a new test compares the RMS errors directly (`bad > 5 x good`,
seed- and architecture-independent). Gate tolerances are unchanged, so no gate in the plan moved and no earlier result is affected. Full suite on the 3080: 97 passed.

**Next steps.** Re-run `pytest tests/test_correctness.py` on the 2060 to confirm both new tests pass there. Repeat the `large_square` (1, 1, 1) test several times and check `dmesg` for an OOM kill.
Consider pruning or skipping the 128x256 tile on sm_75, where it compiles to a very large unrolled kernel.

Reproduce: `python -m pytest -q tests/test_correctness.py`.

## Update 2026-10-19 (evening): inner variety pass (warps / stages within the top tile) — fixes one shape, otherwise a tie

**What changed.** `diverse_order` gained an optional inner pass. Within the top pick's (BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K) group, it adds up to `inner` configs with a new
num_warps (or warps × stages, or warps × stages × GROUP_M) value, best-scored first and within `inner_delta` of the top pick. The pass runs before or after the outer tile pass.
**Fit on tune shapes only**, jointly with the outer rule, on the cached out-of-fold scores: 90 inner rules plus the 57 earlier ones, with "no inner pass" as a candidate.
- Placed after the outer pass, it is a no-op at k = 3, because the outer picks fill the slots. Those rules tie outer-only exactly.
- The best rule that spends a top-3 slot on it, `{"inner_group": "warps", "inner_delta": 0.02, "inner_pos": "before"}` on the same outer rule, is slightly *worse* at k = 2–3:
  tune best-of-3 mean 2.84% vs 2.77%, best-of-2 3.49% vs 3.43%. It is marginally better at k = 4–5 (2.42% vs 2.45%, 2.12% vs 2.14%).

So the CV-chosen rule stays **outer-only** (`Ranker.rank(order="diverse")`). The inner rule is saved as `meta["inner_rule"]` and measured as a diagnostic order (`--order inner`).
It changes the held-out top 3 on 10 of 86 shapes, and the online top-3 choice on 5.

**Re-measured** on commit `9e6e4be` (clean). Each session holds plain, outer-only and inner top-5 together with knnf / knn5:
- `bench-20261019-173739-lock1755-topki`: cold, 2,686 rows, 0 invalid, 4.8% power-capped.
- `-topki-graph`: 1,766 rows, 0 invalid, 9.4% power-capped.
- `online-20261019-173739-lock1755-topki`: inner, empty Triton cache, 86 × 5, 0 invalid.

The plain and outer online choices from the earlier runs are re-priced here. Tables: [`learned_topk.md`](learned_topk.md).

**Online top-3 choice re-timed with the full protocol, regret vs the per-shape Triton oracle, median / p95 (mean):**

| selector | cold, all 86 | cold, ≥ 50 µs (30) | graph, < 50 µs (56) |
|---|---|---|---|
| knnf | 2.5% / 22.3% (6.5%) | 2.2% / 36.5% (8.7%) | 4.9% / 25.8% (7.7%) |
| plain online top-3 | 0.9% / 12.2% (3.1%) | 0.9% / 16.7% (3.8%) | 0.4% / 12.3% (2.7%) |
| outer online top-3 (bar) | 1.1% / 9.1% (2.6%) | 1.0% / 10.0% (2.4%) | 0.2% / 9.7% (2.1%) |
| **inner online top-3** | 1.1% / 9.1% (2.5%) | 1.0% / 10.0% (2.4%) | **0.2% / 8.2% (1.7%)** |
| inner online top-5 | 0.8% / 8.5% (2.3%) | 0.3% / 8.1% (1.9%) | 0.1% / 7.8% (1.5%) |

- **It does what it was built for, on one shape.** 381×1401×250 goes from +14.2% to +0.2% (graph), and cold the inner pick is 7.3% faster than the outer-only pick. The second pick is `bm64_bn64_bk32_w8_s2_g4_sk1`, the
  top tile with 8 warps, which is that shape's oracle. 380×6978×216 also gains 5.6% (graph).
- **Elsewhere it is a tie.** Beyond ε, inner beats outer-only on 1 of 86 cold shapes and 2 of 56 graph shapes, and is never slower (geomean 1.001× cold, 1.004× graph). The graph p95 drop (9.7% → 8.2%)
  is that one shape leaving the tail. Cost is the same: k = 3 median 0.39 s, p90 0.81 s.
- **It cannot reach 4065×8842×28** (+10.6%). There the oracle `bm64_bn128_bk16_w4_s3_g1` differs from the top pick in stages and GROUP_M at the same warp count. The CV-best inner grouping is warps-only,
  and the stage / group variants the model ranks above it (g4, g16, g8 at s2) fill the slots. The other large misses are unchanged, and they are ranking errors no variety pass reaches:
  320×1×79 (+22% cold / +16% graph, oracle at rank 134) and 5398×196×774 (+20%, oracle tile at rank 68).
- **Outer-only reproduces** for its second session: online top-3 cold 1.1% / 9.1% / 2.6% vs 1.0% / 9.9% / 2.7%, graph 0.2% / 9.7% / 2.1% vs 0.2% / 9.9% / 2.1%.

**Decision: the bar stays outer-only diverse top-3.** Tune CV ranks it at or above every inner rule at k = 3. Promoting the inner rule because it fixed a held-out shape would fit the rule to held-out
data. It is safe to use, since it was never slower here, and `--order inner` stays available. A fair test of it needs new shapes.

**Next:** the remaining tail is ranking error, so it needs a better model, not more variety:
- Training data or features for N = 1 / tiny-K shapes (320×1×79).
- A ranking objective that weights near-optimal tiles (the 5398×196×774 tile sits at rank 68 of 868).
- Joint {Triton ∪ cuBLASLt} candidates.

Reproduce: the 2026-10-19 (late night) commands with `--out-name results_lock1755_topki[_graph].jsonl`, then
`python -m tune --clock-lock 1755 online --shapes-file $D/sweep_screen_heldout/subset.json --k 5 --order inner --fresh-cache` and
`python -m tune topk --results results_lock1755_topki.jsonl --graph-results results_lock1755_topki_graph.jsonl`.

## Update 2026-10-19 (late night): diversified top-3 — tail p95 12.8% → 9.9% at the same cost, never worse per shape

**What changed.** Plain top-k re-timed near-duplicates: the held-out top 3 covered 1.55 distinct (BLOCK_M, BLOCK_N, SPLIT_K) choices on average. `diverse_order` in `analysis/learned.py`
keeps the top-1 pick. It then takes the best-scored config of each new group, as long as its predicted log regret is within δ of the top pick, and fills any remaining slots in score order.
The groups are (BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K), and the result covers 2.23 distinct tile × split choices in the top 3. **Fit on tune shapes only.** I computed out-of-fold lgbm_rank scores
for all 477 tune shapes on the search's 20 grouped folds (cached in `data/<hw>/lgbm_rank_cv_scores.npz`). Then I scored 57 rules (4 groupings × δ ∈ {0.02 … ∞} × 1–2 plain picks, plus plain) by best-of-3 screen regret.

The winner is `{"group": "tile_split_bk", "delta": 0.05, "plain": 1}`, with an interior δ. Tune CV best-of-3 goes from 1.5% / 12.0% / 3.3% to 1.2% / 10.1% / 2.8% (median / p95 / mean). The rule is saved in the
ranker's metadata and used by `Ranker.rank(order="diverse")` and `tune online --order diverse`. The plain picks are unchanged and identical to the previous sessions'.

**Re-measured** in fresh sessions on a clean harness (commit `b56c906`). Each session times the plain and diverse top-5 together with knnf / knn5, so they share one oracle and B3+:
- `bench-20261019-112027-lock1755-topkd`: cold, 86 shapes, 2,683 rows, 0 invalid, 5.3% power-capped.
- `-topkd-graph`: 56 shapes, 1,765 rows, 0 invalid, 8.3% power-capped.
- `online-20261019-112027-lock1755-topkd`: diverse, empty Triton cache, 86 × 5 candidates, 0 invalid.

The plain online choices from `online-20261019-083637` are re-priced in these sessions. Tables: [`learned_topk.md`](learned_topk.md).

**Online top-k choice re-timed with the full protocol, regret vs the per-shape Triton oracle, median / p95 (mean):**

| selector | cold, all 86 | cold, ≥ 50 µs (30) | graph, < 50 µs (56) | vs B3+, cold all / graph |
|---|---|---|---|---|
| knnf | 2.7% / 22.8% (6.5%) | 2.5% / 36.4% (8.6%) | 4.2% / 26.5% (7.7%) | 0.960 / 0.949 |
| top-1 (no search) | 1.9% / 18.1% (4.1%) | 1.9% / 18.4% (5.0%) | 2.6% / 13.6% (4.4%) | 0.964 / 0.994 |
| plain online top-3 | 1.1% / 12.8% (3.1%) | 1.2% / 15.8% (3.7%) | 0.5% / 12.2% (2.7%) | 0.973 / 1.005 |
| **diverse online top-3** | **1.0% / 9.9% (2.7%)** | **0.9% / 11.1% (2.5%)** | **0.2% / 9.9% (2.1%)** | 0.982 / 1.011 |
| diverse online top-5 | 0.8% / 9.3% (2.5%) | 0.5% / 8.3% (1.9%) | 0.2% / 8.1% (1.9%) | 0.987 / 1.011 |
| diverse best-of-3, full-protocol timing | 0.9% / 9.3% (2.4%) | 0.9% / 11.1% (2.4%) | 0.2% / 9.8% (2.0%) | 0.983 / 1.011 |

- **Diversity strictly helps the tail.**
  - Against plain top-3, p95 drops from 12.8% to 9.9% cold, from 15.8% to 11.1% at ≥ 50 µs and from 12.2% to 9.9% under 50 µs (graph). The ≥ 50 µs mean drops from 3.7% to 2.5%.
  - Shape by shape, beyond ε, diverse is faster on 5 of 86 cold shapes and 2 of 56 graph shapes, and **never slower**. The rest tie (geomean 1.003× / 1.006×).
  - The cold < 50 µs median goes from 1.1% to 1.3%, which is within the event tick. The graph protocol for that regime improves.
- **Unlike plain, a larger k still pays.** Diverse top-5 reaches an ≥ 50 µs p95 of 8.3%, against plain top-5's 15.8%. That is now below the 48-config portfolio oracle's held-out p95 (~8.9%).
- **The cost is the same.** k = 3 takes a median 0.39 s per shape (p90 0.86 s), against 0.40 s for plain. Diverse picks are no harder to compile.
- **Fixed** (plain → diverse top-3):
  - 14244×16×1257: 18.2% → 4.6%.
  - 277×3×161: 12.8% → 0% (graph).
  - 1316×272×100: 11.7% → 0% (graph).
  - 417×512×6739: 12.9% → 9.5%.
  - 51×324×5709: 10.2% → 7.4% (graph).
- **Not fixed, and why.** These are ranker errors that no diversification within δ reaches:
  - **320×1×79** (+21% cold, +16% graph): the oracle `bm16_bn32_bk128_w2_s5` sits at rank 134, predicted 0.66 log regret behind the top. The model is simply wrong on this tiny-K, N = 1 decode-like shape.
  - **5398×196×774** (+20%): the oracle's tile (64×128, BLOCK_K 16) sits at rank 68, 0.10 behind, outside δ = 0.05.
  - **4065×8842×28** (+12%) and **381×1401×250** (+13–14%): the oracle shares the top pick's tile × BLOCK_K × split and differs only in num_stages / GROUP_M or num_warps (ranks 7–8). The grouping deliberately collapses those.
- **Against B3+:** cold all moves from 0.973 to 0.982. Under 50 µs graph it moves from 1.005 to 1.011, so the searched Triton pick is now marginally ahead of cuBLASLt's searched best at the median. G4 at ≥ 50 µs still fails (Triton oracle 0.96).

**New bar: diverse online top-3.** Cold all: 1.0% / 9.9% (2.7%). ≥ 50 µs: 0.9% / 11.1% (2.5%). Graph < 50 µs: 0.2% / 9.9% (2.1%). Cost: ~0.4 s first use per shape.

**Next:** the remaining tail needs a better ranker, not a wider search:
- An inner diversity pass over stages / warps for the top group, which would catch 4065×8842×28 and 381×1401×250.
- Training targets or features for N = 1 / tiny-K shapes (320×1×79).
- Joint {Triton ∪ cuBLASLt} candidates, now that Triton matches B3+ under 50 µs.

Reproduce:
```bash
python -m tune topk               # train, fit the diversification rule (tune CV), write plain + diverse picks
D=data/rtx3080_sm86_68sms_10gb
export TUNE_SWEEP_DIR=$D/sweep_screen_heldout TUNE_PICKS=$D/learned_topk_picks.json
python -m tune --clock-lock 1755 benchmark --shapes-file $D/sweep_screen_heldout/subset.json \
  --backends torch,triton:sweep:8,triton:picks,cublaslt:16,cublaslt_deep:20 --cache-modes cold --out-name results_lock1755_topkd.jsonl
python -m tune --clock-lock 1755 benchmark --shapes-file $D/sweep_screen_heldout/subset_lt50us.json \
  --backends torch,triton:sweep:8,triton:picks,cublaslt:16,cublaslt_deep:20 --cache-modes hot --graph-batch 50 \
  --out-name results_lock1755_topkd_graph.jsonl
python -m tune --clock-lock 1755 online --shapes-file $D/sweep_screen_heldout/subset.json --k 5 --order diverse --fresh-cache
python -m tune topk --results results_lock1755_topkd.jsonl --graph-results results_lock1755_topkd_graph.jsonl
```

## Update 2026-10-19 (night): lgbm_rank top-k + short online search — top-3 cuts p95 regret by a third for ~0.4 s per shape

**What ran.** `python -m tune topk` refits lgbm_rank on all 477 tune shapes, using the hyperparameters already chosen by tune CV. It saves the model as a standalone `analysis.learned.Ranker` in
`data/<hw>/lgbm_rank.txt/.json`: a LightGBM booster, an 868-config pool and the GPU numbers the features need. The `Ranker` ranks any (M, N, K) in a median 92 ms. On all 86 held-out shapes its top 5
match the fitted model exactly, and the build fails otherwise. The top 5 per held-out shape went into `learned_topk_picks.json` for `triton:picks`.

The new `python -m tune online` (`tune/online.py`) is the first-use search. It ranks the shape, then compiles and validates the top k from an **empty Triton cache** (`--fresh-cache`). It times
each candidate with 10 cold samples on one placement, re-times them hot with CUDA graphs (20 × 5) when the fastest is under 50 µs, and keeps the fastest.
One k = 5 run stores every candidate's short times and costs, so the choice and cost for every k ≤ 5 are recomputed offline. Each choice is then priced at its full-protocol stage-2 time.

All runs used commit `a24d834`, `harness_dirty: false`, clock locked at 1755 MHz:
- `bench-20261019-083637-lock1755-topk`: cold, 86 shapes, 2,561 rows, 0 invalid, 4.7% power-capped.
- `-topk-graph`: graph-amortized, 56 shapes under 50 µs, 1,685 rows, 0 invalid, 7.3% power-capped.
- `online-20261019-083637-lock1755-topk`: 86 shapes × 5 candidates, 0 invalid.

Tables: [`learned_topk.md`](learned_topk.md).

**Regret vs the per-shape Triton oracle, median / p95 (mean).** The online rows are what the short search actually picked, re-timed with the full protocol:

| selector | cold, all 86 | cold, ≥ 50 µs (30) | graph, < 50 µs (56) | vs B3+, cold all / graph |
|---|---|---|---|---|
| knnf | 2.5% / 22.1% (6.6%) | 2.4% / 36.7% | 4.7% / 26.2% (7.6%) | 0.955 / 0.953 |
| knn5 | 2.5% / 23.9% (9.4%) | 4.2% / 65.1% | 3.8% / 21.8% (6.9%) | 0.964 / 0.973 |
| lgbm top-1 (no search) | 2.0% / 18.3% (4.1%) | 2.2% / 19.1% | 2.3% / 13.5% (4.4%) | 0.968 / 0.993 |
| lgbm online top-2 | 1.2% / 12.5% (3.3%) | 2.0% / 18.3% | 0.4% / 12.2% (2.9%) | 0.971 / 1.002 |
| **lgbm online top-3** | **0.8% / 12.2% (2.9%)** | **1.1% / 16.5% (3.8%)** | **0.3% / 12.2% (2.7%)** | 0.973 / 1.002 |
| lgbm online top-5 | 0.7% / 11.7% (2.8%) | 1.1% / 16.5% (3.3%) | 0.1% / 12.1% (2.2%) | 0.990 / 1.004 |
| lgbm best-of-5, full-protocol timing | 0.1% / 11.2% (2.3%) | 0.7% / 16.4% (3.1%) | 0.0% / 11.7% (2.2%) | 0.990 / 1.004 |

**First-use cost per shape (wall clock, empty Triton cache):**

| k | compile + validate s, median / p90 | short timing s, median | total s, median / p90 |
|---|---|---|---|
| 1 | 0.14 / 0.29 | 0.01 | 0.16 / 0.33 |
| 3 | 0.37 / 0.70 | 0.02 | 0.40 / 0.76 |
| 5 | 0.61 / 1.18 | 0.04 | 0.66 / 1.28 |

- **Top-3 is the sweet spot.** Going from k = 1 to k = 3:
  - Cold mean regret drops from 4.1% to 2.9% and p95 from 18.3% to 12.2%. At ≥ 50 µs the median drops from 2.2% to 1.1%. Under 50 µs (graph) the median drops from 2.3% to 0.3% and p95 from 13.5% to 12.2%.
  - Top-3 beats top-1 by more than 2% on 17 of 86 cold shapes and 12 of 56 graph shapes.
  - k = 4–5 leaves p95 almost unchanged (11.7–12.1%) while costing more.
- **The short protocol is good enough.** Online top-k lands within 0.1–0.7 points (median and mean) of choosing with full-protocol times, so 10 samples per candidate are enough to rank the top k.
  Graph mode triggered on exactly the 56 shapes under 50 µs.
- **Cost is dominated by compilation, and it is small.** A config never compiled before costs a median 0.16 s (p90 0.33 s, max 0.81 s including the ~0.8 s one-time JIT startup). Timing costs
  hundredths of a second. The table's medians are measured in one process, where 186 of 430 candidates reused a kernel compiled for an earlier shape (~0 s). A cold process pays about 3 × 0.16 ≈ 0.5 s
  (median) for k = 3. That is once per new shape, against kernels of 2 µs to 12 ms.
- **The remaining tail is ranker error, not search width.** The worst shapes are the same at k = 1 and k = 5: 14244×16×1257 at +23%, 5398×196×774 at +20%, 320×1×79 at +16–18%. On average the top 5 cover
  only 1.9 distinct (BLOCK_M, BLOCK_N, SPLIT_K) tiles, and on 30 of 86 shapes all five share one tile and differ only in stages, group or BLOCK_K. A wider k mostly re-times near-duplicates.
- **Against B3+:** online top-3 moves from 0.968 to 0.973 cold (all shapes) and from 0.993 to 1.002 under 50 µs graph. Below 50 µs the searched Triton pick now ties cuBLASLt's searched best at the median.
  At ≥ 50 µs G4 still fails.
- knnf / knn5 reproduce again (knnf cold 2.5% / 22.1% / 6.6%; knn5 graph 3.8% / 21.8% / 6.9%).

**New bar: lgbm online top-3.** Cold all: 0.8% / 12.2% (2.9%). ≥ 50 µs: 1.1% / 16.5%. Graph < 50 µs: 0.3% / 12.2% (2.7%). Cost: ~0.4–0.5 s first use per shape.

**Next:** a **diversified top-k** (best-predicted config per distinct tile × split, then by score) to attack the tail that a wider plain top-k cannot reach. Then joint
{Triton ∪ cuBLASLt} candidates in the same search, since under 50 µs the margin over B3+ is now about zero at the median.

Reproduce:
```bash
python -m tune topk               # train + save data/<hw>/lgbm_rank.*, write learned_topk_picks.json
D=data/rtx3080_sm86_68sms_10gb
export TUNE_SWEEP_DIR=$D/sweep_screen_heldout TUNE_PICKS=$D/learned_topk_picks.json
python -m tune --clock-lock 1755 benchmark --shapes-file $D/sweep_screen_heldout/subset.json \
  --backends torch,triton:sweep:8,triton:picks,cublaslt:16,cublaslt_deep:20 --cache-modes cold --out-name results_lock1755_topk.jsonl
python -m tune --clock-lock 1755 benchmark --shapes-file $D/sweep_screen_heldout/subset_lt50us.json \
  --backends torch,triton:sweep:8,triton:picks,cublaslt:16,cublaslt_deep:20 --cache-modes hot --graph-batch 50 \
  --out-name results_lock1755_topk_graph.jsonl
python -m tune --clock-lock 1755 online --shapes-file $D/sweep_screen_heldout/subset.json --k 5 --fresh-cache
python -m tune topk               # re-score with the stage-2 and online files
```

## Update 2026-10-19 (late): learned selectors v0 — per-candidate LightGBM sets a new bar

**What ran.** `analysis/learned.py` (`python -m tune learned`) fits five model families in two forms, on the tune screen only:
- **cls:** shape → config classification over the first P members of a greedy portfolio, refit per fold. A shape's label is the first member within 2% of its best member.
  Inputs are the 22 shape features (dims, fill, align). Models: decision tree, LinearSVC, LightGBM, random forest, logistic regression.
- **rank:** per-candidate scoring over the union of each training shape's top-5 configs (868 candidates). Each row has the 22 shape features, the 7 config parameters and the 6
  `candidate_features`. The trees, the forest and LightGBM regress capped log regret (LightGBM could also pick lambdarank). LinearSVR regresses log regret and logistic regression
  classifies "within 5% of the oracle", both on standardized inputs plus every shape × candidate product.

Hyperparameters come from grouped 20-fold CV on the tune shapes, lowest capped mean regret. Each fit sees 290–353 shapes, and drops every tune shape that shares two of M, N, K with a
fold shape. The share-two-dims graph has one component of 258 shapes, so GroupKFold was not an option. 198 grid points, about 2 h of CPU, cached per model in
`data/<hw>/learned_search.json`. The grids were widened once, on tune CV only, wherever the first pass's winner sat on an edge. No held-out shape informed any fit or choice.

**Re-measured** in fresh sessions on a clean harness (commit `3bbc349`, `harness_dirty: false`). Each session also re-times b6 / knn5 / knnf, so everything shares one oracle and B3+:
- `bench-20261019-032812-lock1755-learned`: cold, 86 shapes, 2,760 rows, 0 invalid, 5.8% power-capped.
- `-learned-graph`: graph-amortized, 56 shapes under 50 µs, 1,815 rows, 0 invalid, 5.5% power-capped.

Tables: [`learned_selectors.md`](learned_selectors.md).

**Regret vs the per-shape Triton oracle, median / p95 (mean):**

| selector | tune CV mean | held-out screen | stage 2 cold, all 86 | cold, ≥ 50 µs (30) | graph, < 50 µs (56) |
|---|---|---|---|---|---|
| knn5 (bar < 50 µs) | 8.4% | 4.7% / 25.4% (10.8%) | 3.0% / 24.2% (9.7%) | 3.6% / 65.3% | 3.7% / 22.1% (6.8%) |
| knnf (bar ≥ 50 µs) | 7.5% | 4.1% / 27.3% (7.8%) | 2.3% / 23.0% (6.6%) | 1.9% / 36.4% | 4.7% / 25.4% (7.5%) |
| dt_cls | 14.6% | 7.8% / 43.9% (15.1%) | 6.4% / 41.4% (14.2%) | 5.3% / 38.3% | 10.8% / 51.9% (17.0%) |
| svm_cls | 13.5% | 8.0% / 53.6% (15.3%) | 6.5% / 51.8% (14.1%) | 5.2% / 57.9% | 11.1% / 57.6% (18.2%) |
| lgbm_cls | 10.1% | 4.1% / 37.7% (11.0%) | 3.0% / 39.0% (9.9%) | 2.3% / 70.1% | 5.1% / 31.8% (8.9%) |
| rf_cls | 9.0% | 4.3% / 32.7% (11.4%) | 3.2% / 32.3% (10.3%) | 1.7% / 28.4% | 5.5% / 52.1% (13.4%) |
| logreg_cls | 11.1% | 7.4% / 52.5% (13.2%) | 5.9% / 51.8% (11.7%) | 3.3% / 38.7% | 10.5% / 71.4% (17.5%) |
| dt_rank | 11.6% | 4.6% / 65.7% (14.3%) | 4.4% / 67.7% (13.2%) | 1.1% / 43.5% | 9.4% / 80.0% (17.4%) |
| svm_rank | 22.4% | 13.7% / 73.5% (21.9%) | 10.5% / 70.4% (19.7%) | 15.9% / 83.7% | 15.1% / 85.5% (21.6%) |
| **lgbm_rank** | **4.8%** | **3.6% / 18.1% (5.5%)** | **1.4% / 18.3% (4.1%)** | 1.8% / 19.1% | **2.4% / 13.5% (4.3%)** |
| rf_rank | 5.3% | 4.4% / 18.5% (6.4%) | 2.9% / **17.3%** (5.1%) | **1.2% / 18.6%** | 3.0% / 13.5% (5.2%) |
| logreg_rank | 22.1% | 8.5% / 63.7% (17.3%) | 5.4% / 64.6% (15.3%) | 6.8% / 81.7% | 8.4% / 53.2% (15.1%) |

lgbm_rank: regression objective, 255 leaves, 1,500 trees, learning rate 0.05. rf_rank: 100 trees, max_features 0.66, min leaf 5. Tune CV for knn5 / knnf uses the same folds,
with their weights fixed as fit on all tune shapes, which slightly favours them.

- **Per-candidate tree ensembles are the first selectors to beat the k-NN bar, and they do it in every regime.** Against knnf in cold stage 2, lgbm_rank's mean drops from 6.6% to 4.1% and its p95 from 23% to 18%.
  At ≥ 50 µs its p95 is 19% against knnf's 36%, about half. Under 50 µs (graph) its mean drops from 6.8% to 4.3% and its p95 from 22% to 13.5% against knn5. The ranking also holds shape by shape:
  - Cold, beyond ε (2% ≥ 50 µs, 5% below): lgbm_rank is faster than knnf on 21 shapes, slower on 10 and tied on 55 (geomean 1.022×). Against knn5 it is 23 / 7 / 57 (1.036×).
  - Graph, ε 5%: 17 / 6 / 33 against knnf and 16 / 10 / 32 against knn5.
  
  Its worst stage-2 shape is 226×2141×1139 at +29%, where knnf's worst is +45%.
- **CV picks the same winner as held-out does.** lgbm_rank has the lowest tune-CV mean (4.8%), rf_rank is second (5.3%). The held-out screen mean (5.5%, 6.4%) and stage-2 cold mean (4.1%, 5.1%) rank them the same, with every other learned selector at ≥ 9.9%. So naming
  lgbm_rank the bar is not a choice made on held-out data.
- **The form matters more than the model.** The same features fed to classifiers over a static label set (cls) stay at a 32–54% held-out p95. That is despite label-set ceilings of 12.5% p95 for P = 32,
  because P has to be small for the labels to be learnable, and a misclassification costs the full regret. Per-candidate scoring lets the model reuse what it learned about one config on others.
- **Linear models fail in both forms**, even with every shape × candidate product (svm_rank / logreg_rank CV about 22%). Regret is not close to linear in these features. Single decision trees overfit (rank p95 66–68%).
- **What the trees use.** In both lgbm_rank and rf_rank, the candidate's own K steps per split and its waves (from `candidate_features`) carry 36–38% of the importance. Next come
  BLOCK_M, wave efficiency and BLOCK_N. The per-candidate features that k-NN voting could not exploit (knnc) do work when a model can weigh them against the shape.
- **Against B3+ nothing changes** (median t_B3+/t_pick, cold ≥ 50 µs): lgbm_rank 0.933, knnf 0.913, Triton oracle 0.964. G4 still fails at ≥ 50 µs. Under 50 µs (graph), lgbm_rank reaches
  0.992, against 0.951 for knnf and 1.017 for the oracle, so a joint {Triton ∪ cuBLASLt} selector is where the remaining margin is.
- **knnf / knn5 reproduce for the fourth session in a row** (knnf cold 2.3% / 23.0% / 6.6% vs 1.8% / 22.9% / 6.5% in `selectors3`; knn5 graph 3.7% / 22.1% vs 3.7% / 22.3%).

**Caveats.** lgbm_rank's winner is again on the edge of its grid (255 leaves, 1,500 trees), but CV gains have flattened (4.9% → 4.8% from 127 to 255 leaves). Lambdarank has the best
CV median (1.6–1.9%) but a worse mean, so it lost under the mean rule. The cold ≥ 50 µs column has only 30 shapes, so its p95 is about the second-worst shape.

**New bar:** lgbm_rank, cold all median 1.4% / p95 18.3% / mean 4.1%, ≥ 50 µs 1.8% / 19.1%, graph < 50 µs 2.4% / 13.5% / 4.3%.

**Next:** top-k from lgbm_rank plus a short online search (time its top 3–5, AGENTS next step 2), which now has a calibrated ranker to draw from. Add cuBLASLt algorithms as
candidates in the same ranker (joint selection). Check the ranker on a lambdarank / regression blend, or try a pool beyond the top 5.

Reproduce:
```bash
python -m tune learned            # CV search (cached in data/<hw>/learned_search.json), held-out picks -> data/<hw>/learned_picks.json
D=data/rtx3080_sm86_68sms_10gb
export TUNE_SWEEP_DIR=$D/sweep_screen_heldout TUNE_PICKS=$D/learned_picks.json
python -m tune --clock-lock 1755 benchmark --shapes-file $D/sweep_screen_heldout/subset.json \
  --backends torch,triton:sweep:8,triton:picks,cublaslt:16,cublaslt_deep:20 --cache-modes cold --out-name results_lock1755_learned.jsonl
python -m tune --clock-lock 1755 benchmark --shapes-file $D/sweep_screen_heldout/subset_lt50us.json \
  --backends torch,triton:sweep:8,triton:picks,cublaslt:16,cublaslt_deep:20 --cache-modes hot --graph-batch 50 \
  --out-name results_lock1755_learned_graph.jsonl
python -m tune learned            # re-score with both stage-2 files
```

## Update 2026-10-19 (later): per-candidate fill and split features (knnc) do not beat knnf

**What changed.** `candidate_features` computes the fill and split features for **each candidate's own tile and split** rather than for fixed
reference tiles. For every (shape, config) it has:
- **waves:** CTAs × SPLIT_K / (68 SMs × resident CTAs per SM), with residency set by shared memory, threads and the 16-CTA limit.
- **last-wave efficiency**, **K steps per split**, **M × N padding** to whole tiles, **K padding** to whole BLOCK_K × SPLIT_K steps, and **EVEN_K**.
- **log2 M, N, K** as shape context.

The new selector **knnc** scores every runnable config on the target. A config's score is its own mean log regret on the k tune shapes where that config saw
the closest conditions, so it can pick a config that was no neighbour's winner. The dims weight and k were fit on tune shapes only, by grouped leave-one-out over
16 settings: dims weight 0.5, k = 10. In-sample that gives mean 6.6%, median 2.5% and p95 25.6%, against knnf's 5.9%, 2.1% and 22.5%. The fit takes about 6 min and is cached in
`data/<hw>/selector_knnc_fit.json` (snapshot without picks in `reports/snapshot/lock1755/`).

**Re-measured** with all seven selectors in fresh sessions on a clean harness (commit `5424930`, `harness_dirty: false`):
- `bench-20261019-003241-lock1755-selectors3`: cold, 86 shapes, 2,526 rows, 0 invalid, 3.4% power-capped.
- `-selectors3-graph`: 56 shapes under 50 µs, 1,670 rows, 0 invalid, 6.1% power-capped.

Tables: [`selector_baselines.md`](selector_baselines.md).

**Regret vs the per-shape Triton oracle, median / p95 (mean):**

| selector | held-out screen | stage 2 cold, all 86 | cold, ≥ 50 µs (30) | graph, < 50 µs (56) |
|---|---|---|---|---|
| knn5 | 4.7% / 25.4% (10.8%) | 2.7% / 22.6% (9.5%) | 3.9% / 65.4% | **3.7% / 22.3%** (**6.7%**) |
| **knnf** | **4.1% / 27.3% (7.8%)** | **1.8% / 22.9% (6.5%)** | **1.9%** / 36.6% | 4.7% / 26.2% (7.3%) |
| **knnc** | 5.3% / 29.3% (10.7%) | 4.4% / 26.7% (9.2%) | 3.8% / **31.9%** | 7.1% / 36.0% (10.5%) |

Against B3+ (median t_B3+ / t_pick): ≥ 50 µs cold, knnc 0.899, knnf 0.916, knn5 0.894 and oracle 0.964. Under 50 µs graph, knnc 0.940, knnf 0.955 and knn5 0.972.

- **knnc is worse than knnf almost everywhere.** Its median is higher in every column, from 1.3× on the screen to 2.4× in cold stage 2, and below 50 µs it is the worst of the three. The one gain is the ≥ 50 µs tail (p95 31.9%
  vs 36.6%), which is about one shape out of 30, and it is still well short of the 48-config portfolio oracle (8.9%).
- **It regresses on the shape that motivated the features.** 2×8901×9643 goes to +120% on the screen. knnc picks `bm16_bn256_bk32_w8_s4_g1_sk1`, whose record on decode shapes with
  similar waves and K steps is good, but here it is 2.2× slower than the oracle.
- **The cause is not the size of the candidate set.** I checked two explanations as diagnostics on held-out data; neither was used for fitting:
  - Restricting knnc to the 48-config portfolio (25 candidates per shape) or to the 284 tune-oracle configs (144 per shape) does not help: screen p95 31% / 38%.
  - Its picks are not over-optimistic: predicted median regret 6.2% vs actual 5.3%. Only the tail is under-predicted (mean 7.6% predicted vs 10.7% actual).
  
  So it is the transfer itself that fails. A config's regret on tune shapes with similar waves, K steps and padding does not predict its regret on a new shape well enough.
  The ranking among near-equal configs depends on things these features omit, such as memory- vs compute-boundness, L2 reuse and per-warp work.
- **knnf and knn5 reproduce across sessions.** knnf cold is 1.8% / 22.9% (6.5%) here vs 2.7% / 22.9% (6.4%) on `selectors2`, and knn5 2.7% / 22.6% vs 2.9% / 23.7%. Medians differ by under 1 point
  and p95 by about 1 point.

**The bar is unchanged:** knnf (cold ≥ 50 µs median 1.9%, p95 37%) and knn5 (graph < 50 µs median 3.7%, p95 22%). The per-candidate features stay in the code as inputs
for a trained ranking model (next step 1 in `AGENTS.md`), which can weigh them against shape features rather than averaging neighbours. Nearest-neighbour voting over them is not enough.

Reproduce: the 2026-10-18 commands with `--out-name results_lock1755_selectors3[_graph].jsonl`, then `python -m tune selectors`, which scores those files by default.

## Update 2026-10-19: selector baselines rebuilt with GPU-fill and alignment features

**What changed.** The k-NN distance now uses derived shape features beyond (log2 M, log2 N, log2 K). The groups are built in `shape_features` and weighted in `embed`:
- **fill:** for 6 reference tiles (16×64 … 128×128), log2(CTAs / 68 SMs) and wave efficiency, which is CTAs / (waves × SMs) and is low when the last wave is nearly empty.
  It also includes log2 of the K steps per split if a 64×64 grid were split to fill the SMs.
- **align:** whether M, N and K are divisible by 16 and by 8.

The group weights and k come from a grid of 120 settings, scored only on the tune shapes with **grouped leave-one-out**: a tune shape may not use any neighbour that shares two
of M, N, K with it. Plain leave-one-out let S2's M sweeps over one layer act as near-copies of each other. The best setting gives **knnf** (fill 1.5, align 0.5,
k = 3), and the best k = 1 setting gives **nn1f** (fill 0.5, align 0.5). Both were fixed before any held-out shape was scored. The four earlier selectors and their picks are unchanged.
I chose the feature groups after seeing the 2026-10-18 held-out misses, which were split-K and wave choices, but no weight or k was fit on held-out data.

**Re-measured** in two new sessions with the same protocol as 2026-10-18, so all six selectors share each session's oracle and B3+:
- `bench-20261018-235334-lock1755-selectors2`: cold, 86 shapes, 2,459 rows, 0 invalid.
- `-selectors2-graph`: graph-amortized, 56 shapes under 50 µs, 1,625 rows, 0 invalid.

5.0% and 4.4% of rows hit the power cap. Rows record `harness_dirty: true` because the feature code was uncommitted.
Tables: [`selector_baselines.md`](selector_baselines.md).

**Regret vs the per-shape Triton oracle, median / p95 (mean):**

| selector | held-out screen | stage 2 cold, all 86 | cold, ≥ 50 µs (30) | graph, < 50 µs (56) | tune, grouped LOO |
|---|---|---|---|---|---|
| nn1 | 5.4% / 132% (30.7%) | 3.8% / 129% (29.0%) | 1.5% / 194% | 9.3% / 113% (26.5%) | 4.9% / 53% |
| **nn1f** | 4.5% / 40% (10.2%) | 3.5% / **40%** (8.8%) | 1.5% / 40% | 8.2% / 41% (13.0%) | 2.9% / 30% |
| knn5 | 4.7% / 25.4% (10.8%) | 2.9% / 23.7% (9.4%) | 3.7% / 65% | **3.5% / 20.9%** (6.5%) | 2.8% / 29% |
| **knnf** | **4.1%** / 27.3% (**7.8%**) | **2.7% / 22.9% (6.4%)** | **1.9% / 37%** | 3.8% / 26.7% (7.3%) | 2.1% / 22.5% |
| rule | 10.3% / 40.5% | 7.4% / 37.1% (13.2%) | 4.7% / 37% | 12.9% / 58.7% (18.4%) | 7.5% / 44% |

**Against B3+, median t_B3+ / t_pick:** at ≥ 50 µs (cold), the Triton oracle is 0.964, knnf **0.922**, nn1f 0.927, knn5 0.896 and rule 0.879. Below 50 µs (graph), the oracle is 1.018,
knn5 0.975 and knnf 0.952.

- **The features fix 1-NN's catastrophic misses.** Its p95 drops from 129% to 40% (cold) and from 113% to 41% (graph). 14244×16×1257 goes from +466% to +46%, and 2×8901×9643 from +95%
  to +9%.
- **For k-NN they help on the larger kernels and not below 50 µs.** At ≥ 50 µs, knnf halves the median (3.7% → 1.9%) and the p95 tail (65% → 37%), and moves from 0.896 to
  0.922 of B3+. The two motivating shapes are fixed: 2×8901×9643 goes from +95% to +9% and 28×3030×4012 from +272% to +17%, both now with split-K picks. Below 50 µs knnf is
  slightly worse than knn5 (graph p95 26.7% vs 20.9%, mean 7.3% vs 6.5%). Across all 86 shapes it is a modest gain: mean 9.4% → 6.4%, p95 23.7% → 22.9%.
- **The tail is not closed.** p95 is still about 23% overall. The worst ≥ 50 µs shapes for knnf are now different ones: 68×6420×10359 +46% (knn5 +29%), 79×15624×2135 +40% and
  443×4723×393 +32%. These features stop the selector from transferring a config that is wrong for the target's grid, but they do not make it find the winner. With only
  shape-level features, every neighbour vote still ranks whole configs.
- **No selector differentiates from B3+ at ≥ 50 µs**, and the oracle itself cannot (0.964). That is unchanged.

**The bar for a learned selector** is now knnf on large kernels (cold ≥ 50 µs: median 1.9%, p95 37%) and knn5 on small ones (graph: median 3.5%, p95 21%), with an
overall p95 of about 23% and mean of about 6%. The next step is unchanged: per-candidate features (the fill and K-step features computed for each candidate's own tile and split,
not for fixed reference tiles), so a model can score configs rather than vote between neighbours' winners.

Reproduce: the commands at the end of the 2026-10-18 section with `--out-name results_lock1755_selectors2[_graph].jsonl`; `python -m tune selectors`
now scores those files by default (pass `--results results_lock1755_selectors.jsonl --graph-results results_lock1755_selectors_graph.jsonl` for the first run).

## Update 2026-10-18: trivial selector baselines on held-out shapes (plan 8.1 item 7)

**What ran.** Three selectors, each mapping (M, N, K) to one Triton config. All were fit only on the 477-shape tune screen and scored on the 86 held-out shapes
(`analysis/selectors.py`, `python -m tune selectors`):
- **nn1**: the nearest tune shape in (log2 M, log2 N, log2 K), Euclidean distance; use that shape's fastest config.
- **knn5**: the 5 nearest tune shapes; use the config with the lowest mean log regret over them.
- **rule**: a hand-written rule (`rule_config`). BLOCK_M follows M and BLOCK_N follows N. Split-K doubles while the grid has fewer CTAs than the 68 SMs.
  Warps follow the tile. I wrote it while looking only at the tune-split oracle configs (5 iterations), then froze it.
- **b6** (best single tune config) is included as the floor.

Every pick passes the pruning rules for its target shape, so a neighbour's best config is skipped when it is pruned on the target. The new backend
`triton:picks` times the picks (`data/<hw>/selector_picks.json`; snapshot in `reports/snapshot/lock1755/selector_picks.json`).
- **Stage 2, cold:** session `bench-20261018-230423-lock1755-selectors` (10 min, 2,375 rows, 0 invalid). It covered all 86 held-out shapes, each timed over
  its 8 best screen tiles, every selector pick, B1, B3 (`cublaslt:16`) and B3+ (`cublaslt_deep:20`), all in one session.
- **Stage 2, graph:** session `bench-20261018-230423-lock1755-selectors-graph` (5 min, 1,567 rows, 0 invalid). It ran the same backends, hot and CUDA-graph
  amortized (`--graph-batch 50`), on the 56 held-out shapes whose screen oracle is under 50 µs.
- Both rows record `harness_dirty: true`, because the `triton:picks` backend was not yet committed when they ran; it is committed with this update.
  3.3% (cold) and 5.1% (graph) of rows hit the power cap, all on the largest shapes in each set.

Tables: [`selector_baselines.md`](selector_baselines.md).

**Regret vs the per-shape Triton oracle, median / p95:**

| selector | held-out screen (full-space oracle) | stage 2 cold, all 86 | stage 2 cold, ≥ 50 µs (29) | stage 2 graph, < 50 µs (56) | in-sample (tune, leave-one-out), screen |
|---|---|---|---|---|---|
| b6 | 48.1% / 487% | 48.8% / 484% | 74.8% / 327% | 24.7% / 455% | 68.4% / 343% |
| nn1 | 5.4% / 132% | 3.6% / 129% | 1.1% / 123% | 9.2% / 116% | 4.1% / 60% |
| **knn5** | **4.7% / 25.4%** | **2.1% / 23.7%** | 3.1% / 69.0% | **3.6% / 21.7%** | 2.3% / 23.2% |
| rule | 10.3% / 40.5% | 7.4% / 37.4% | 4.7% / 37.9% | 13.0% / 59.2% | 7.5% / 44.3% |

On the screen, the k-NN neighbourhood size matters for the tail: k = 3 gives p95 43%, k = 5 25%, k = 10 25% (median 5.0–5.4% for all three).

**Against B3+ (t_B3+ / t_pick, > 1 = the pick is faster), median [p10, p90], and shapes ahead by > 5% / behind by > 5%:**

| | cold, ≥ 50 µs (29) | cold, < 50 µs (57) | graph, < 50 µs (56) |
|---|---|---|---|
| Triton oracle (top-8 screen tiles + picks) | 0.954 [0.83, 1.02] · 2 / 14 | 1.033 [0.83, 1.16] · 24 / 19 | 1.017 [0.77, 1.18] · 24 / 20 |
| knn5 | 0.897 [0.64, 1.01] · 2 / 18 | 1.000 [0.74, 1.14] · 20 / 24 | 0.977 [0.65, 1.17] · 20 / 25 |
| nn1 | 0.920 [0.51, 1.01] · 2 / 18 | 0.952 [0.52, 1.12] · 15 / 29 | 0.898 [0.47, 1.13] · 17 / 33 |
| rule | 0.877 [0.67, 0.98] · 2 / 18 | 0.940 [0.68, 1.11] · 12 / 32 | 0.880 [0.53, 1.11] · 13 / 35 |
| b6 | 0.566 [0.27, 0.77] · 0 / 28 | 0.821 [0.22, 1.03] · 3 / 39 | 0.800 [0.22, 1.02] · 0 / 39 |

- **k-NN-5 is the baseline to beat.** It is the best on almost every column, with a held-out median regret of 2–5% and p95 of 22–25%. That is far better than any fixed
  config (b6 median 48%), but the tail is roughly 3× worse than the 48-config portfolio oracle (8.9% p95, stage 2). A learned selector has to close the gap
  between "pick 1 of 10,840 from the neighbours" (p95 ≈ 24%) and "best of the 48-portfolio" (≈ 9%).
- **1-NN has a good median and a bad tail.** It is often exactly right (stage-2 median 1.1% at ≥ 50 µs), but when the neighbour's winner does not transfer it fails badly
  (14244×16×1257: +469%, 37×922×10110: +135%). Averaging over 5 neighbours fixes most of those. nn1 and knn5 agree on only 7% of shapes.
- **The hand rule is robust but imprecise.** Its median is 2–4× knn5's. At ≥ 50 µs its tail is better than knn5's (38% vs 69% p95; with 29 shapes, p95 is the
  2nd-worst shape, so this is one shape's difference), and below 50 µs it is worse. Out-of-sample regret matches in-sample regret (7.5% → 7.4%), so it did not overfit the tune shapes. It also never wins by luck.
- **The worst misses are split-K and wave-quantization decisions that (log M, log N, log K) cannot express.** 2×8901×9643 (255 µs, +95% for both knn5 and rule)
  wants `bn64 … sk2`. A BLOCK_N = 128 tile gives 70 CTAs on 68 SMs, which is 2 waves with the second one nearly empty. 28×3030×4012 (+277% for knn5) wants split-K 4,
  but its neighbours picked split-K 1. 68×6420×10359 wants split-K 4 as well. The obvious next features are the tile and wave counts per candidate tile
  (ceil(M/BM)·ceil(N/BN) / SMs) and K steps per split.
- **Against B3+, no trivial selector differentiates at ≥ 50 µs.** Even the Triton oracle is behind (median 0.954, ahead of B3+ by > 5% on 2 of 29), consistent
  with G4 at ≥ 50 µs on the tune shapes. knn5 loses another ~6% on top (0.897). Below 50 µs, knn5 is at parity with B3+ (cold 1.000, graph 0.977) and ahead on
  20 of 56 shapes, but B3+ is ahead on just as many.

**Gate implications.** G3 holds out of sample for selection: even the simplest selector recovers most of the per-shape value (knn5 median 2–5% vs b6 48%).
The selector problem is the tail. The bar for a learned model is set: on held-out shapes it must beat **knn5 at p95 ≈ 24% / median ≈ 2–5% regret**, and to be
useful against cuBLAS it must pick from {Triton candidates ∪ cuBLASLt configs}, since neither family dominates.

**Next steps.**
1. Learned selector v0: rank candidate configs per shape, using shape features plus per-tile derived features (CTA count, waves, K steps per split, padding waste).
   Evaluate it with exactly this harness (`triton:picks` + `python -m tune selectors`) against knn5.
2. Top-k + short online search: time the selector's top 3–5 at first use. This tests whether the tail closes once the model only has to get the winner into a short list.
3. Joint {Triton, cuBLASLt} selection at < 50 µs, where the two families split the wins.

```bash
python -m tune selectors        # writes data/<hw>/selector_picks.json and scores the screen (and stage 2 if present)
D=data/rtx3080_sm86_68sms_10gb
TUNE_SWEEP_DIR=$D/sweep_screen_heldout python -m tune --clock-lock 1755 benchmark --shapes-file $D/sweep_screen_heldout/subset.json \
  --backends torch,triton:sweep:8,triton:picks,cublaslt:16,cublaslt_deep:20 --cache-modes cold --out-name results_lock1755_selectors.jsonl
TUNE_SWEEP_DIR=$D/sweep_screen_heldout python -m tune --clock-lock 1755 benchmark --shapes-file $D/sweep_screen_heldout/subset_lt50us.json \
  --backends torch,triton:sweep:8,triton:picks,cublaslt:16,cublaslt_deep:20 --cache-modes hot --graph-batch 50 --out-name results_lock1755_selectors_graph.jsonl
python -m tune selectors        # re-score with both stage-2 files
```

## Update 2026-10-16: 3× more tune shapes does not fix held-out G5; the tail is not a coverage problem

**What ran.** Option 1 from the previous update: build the portfolio from more tune shapes.
- **Tune screen, all eligible shapes:** session `sweep-tune2-20261015-235431-lock1755` (`python -m tune sweep --per-suite 1000`, 4.5 h).
  It screened every tune-split shape up to 1.2 TFLOP: **477 shapes**, the original 160 (reused from cache) plus 317 new
  (S7 103, S2 84, S6 66, S3 36, S5 22, S4 6). It used the same space and protocol as before: 2,558,593 cells, 0 invalid, 0 crashes. The subset is in
  `reports/snapshot/lock1755/sweep_subset_477.json`. The 160-shape summaries are kept as `*_160.json` in that directory.
- **Stage-2 confirmation:** session `bench-20261016-042636-lock1755-heldout477` (20 min, 5,730 rows, all valid). It re-timed the 86 held-out shapes over
  each shape's 8 best screen tiles plus the **first 60** members of the new portfolio (`triton:sweep:8,triton:portfolio:60`). The portfolio it ran with
  is `data/<hw>/results_lock1755_heldout477.portfolio.json`. 6.5% of rows hit the power cap, on 10 shapes, mostly the largest. One of them, 4065×8842×28, is among the worst.

Tables: [`sweep_portfolio.md`](sweep_portfolio.md), [`sweep_portfolio_heldout.md`](sweep_portfolio_heldout.md)
(`python -m tune portfolio --heldout --results results_lock1755_heldout477.jsonl`).

| p95 regret (mean) | 160-shape portfolio | **477-shape portfolio** |
|---|---|---|
| in-sample size for p95 < 5% | 29 | **48** |
| held-out screen, 29 configs | 17.0% (4.2%) | 13.0% (3.9%) |
| held-out screen, 60 configs | 10.3% (3.1%) | 11.2% (3.1%) |
| held-out screen, 100 configs | 9.3% (2.4%) | 11.1% (2.5%) |
| held-out stage 2, 29 configs | 11.0% (2.2%) | 11.4% (2.4%) |
| held-out stage 2, 48 configs | – | **8.9% (1.8%)** |
| held-out stage 2, 60 configs | – | 8.3% (1.5%) |

Stage-2 oracles differ between the two runs: each is the fastest of that run's candidates, and the new run has 60 portfolio members instead of 29. Compare the 29-config rows only roughly.

- **G5 still fails out of sample.** The best stage-2 result is 8.3% p95 with 60 configs. In-sample, 3× the shapes needs a larger portfolio for the same bar (29 → 48 configs).
  The set cover at ε = 5% grows from 32 to 59 configs.
- **More tune shapes barely help.** A learning curve on screen data (greedy fit on random tune subsets, mean of 5 seeds; held-out p95 at 29 / 48 / 60 configs)
  gives 80 shapes: 15.7 / 12.5 / 12.3%, 160: 14.8 / 13.9 / 12.7%, 320: 13.9 / 12.9 / 11.3%, 477: 13.0 / 12.2 / 11.2%. A portfolio fit only on
  the 127 S7 tune shapes (the same distribution as `heldout_random`) is no better (14.0 / 13.3 / 13.3%).
- **This rejects the coverage explanation from the previous update.** Allowing *every* config that is the screen oracle on some tune shape (284 distinct configs)
  still leaves held-out screen p95 at **7.7%**. The configs that win on the tail held-out shapes are not the best config on any tune shape. Only the
  top-5 union (896 configs) gets to 4.0%. The held-out tail is idiosyncratic: many configs sit within a few % of each other, and which one wins on a given shape
  does not repeat across neighbouring shapes.
- **The tail is a handful of shapes.** p95 over 86 shapes is about the 5th-worst shape, and over the 30 shapes ≥ 50 µs it is the 2nd-worst. At 60 configs (stage 2) the
  worst shapes are 93×2101×22 at +16% (5.3 µs, about one event tick), 417×512×6739 at +15% (`heldout_region`, outside the tuned K range), 37×922×10110 at +10% (region),
  4065×8842×28 at +9% (power-capped) and 51×324×5709 at +9%. Several sit at the limit of what cold per-launch timing can resolve.
- **The typical shape is fine.** Mean stage-2 regret is 1.5–1.8% at 48–60 configs.

**Implication for the selector.** Static portfolio coverage plateaus at about 8–9% p95 out of sample whatever the number of tune shapes, so option 1 is closed.
The remaining levers are at selection time: option 3, a short online search over the top-k predicted configs, or a larger candidate set (hundreds of configs)
that the selector ranks rather than a fixed portfolio. The top-5 union result says such a candidate set exists; whether a model can rank it well enough is the
next experiment. G5 as written (a static portfolio of ≤ 50 with p95 < 5%) should be reported as failed
out of sample, with mean regret under 2%.

## Update 2026-10-15 (night): held-out G5 check; the 29-config portfolio does not generalize to p95 < 5%

**What ran.** The portfolio, B6 and the set covers were fit only on the 160 tune-split screen shapes. They were then scored on shapes they never saw:
- **Held-out screen:** session `sweep-heldout-20261015-210132-lock1755` (`python -m tune sweep --split heldout`, 1.3 h). It covered all
  **86 held-out shapes** up to 1.2 TFLOP: 55 `heldout_random` (S7) and 31 `heldout_region` (K between 9000 and 12000, or M between 300 and 450; from S2, S6 and S7).
  It used the same full pruned space and screen protocol as the tune screen: 307,648 cells, 0 invalid, 0 crashes. The data is in `data/<hw>/sweep_screen_heldout/`,
  kept apart from the tune screen; the subset is in `reports/snapshot/lock1755/sweep_subset_heldout.json`.
- **Stage-2 confirmation:** session `bench-20261015-222149-lock1755-heldout` (10 min, 3,102 rows, all valid). Each held-out shape was re-timed with
  the full cold protocol over its 8 best screen tiles plus the 29 portfolio members (`triton:sweep:8,triton:portfolio:29`). The oracle is the fastest
  of those. 7.7% of rows hit the power cap (throttle 0x4); all of them are on the 11 largest shapes, and none is among the worst-regret shapes.

Tables: [`sweep_portfolio_heldout.md`](sweep_portfolio_heldout.md) (`python -m tune portfolio --heldout`).

### G5 out of sample

| 29-config portfolio | p95 regret | mean regret | shapes within 5% of oracle |
|---|---|---|---|
| tune shapes (in-sample, screen) | 5.0% | 1.4% | 95% |
| held-out, screen vs full-space oracle | **17.0%** | 4.2% | 76% |
| held-out, stage 2 (re-measured) | **11.0%** | 2.2% | 84% |
| held-out ≥ 50 µs, stage 2 | 9.7% | 2.3% | 83% |
| held-out `heldout_region`, stage 2 | 8.9% | 2.6% | 77% |
| held-out `heldout_random`, stage 2 | 11.9% | 1.9% | 87% |

- **G5 fails out of sample.** Following the tune-fitted greedy order, held-out p95 regret stays above 5% even at 100 configs on screen data (9.3%).
  In stage 2, 29 configs give 11.0%. The two numbers bracket the truth: the screen oracle is the minimum over about 10,000 noisy few-sample times,
  so it is biased low and regret is overstated. The stage-2 oracle only sees 37 candidates, so regret is understated.
- **The typical shape is fine; the tail is the problem.** Median regret is 0–1%, mean 2.2%. About 16% of held-out shapes (14 of 86) are more than 5% off.
- **This is overfitting, not a harder shape set.** A greedy portfolio re-fit on the held-out shapes themselves needs 31 configs for p95 < 5%,
  almost the same as the in-sample 29. Held-out shapes are not intrinsically harder. Their best configs are just not in a portfolio chosen
  from 160 shapes: the tail of the greedy list mostly memorizes individual tune shapes.
- **The worst shapes** (stage 2) are skinny shapes (one of M or N small) with K over 1000, where the right split-K and tile combination is missing.
  Examples: 226×2141×1139 at +19.5%, 15×3363×2632 at +13.2%, 20×1×11869 at +12.5%, 14244×16×1257 at +11.4%, 37×922×10110 at +9.6%.
  The largest regret is 124×28×26 at +21.9%, but that is only 0.9 µs on a 4.1 µs kernel, inside the cold event tick.
- **G3 holds out of sample.** B6 (`bm32_bn32_bk32_w4_s5_g1_sk1`) has median regret 48% and p95 487% on held-out shapes, so per-shape selection is still worth a lot.

**Implication for the selector.** A fixed small portfolio with a perfect picker is not enough to stay within 5% at the tail on unseen shapes.
Options, cheapest first:
1. Build the portfolio from more tune shapes. The screen is about 55 s per shape, so a few hundred more tune shapes is an overnight run.
2. Allow a larger portfolio (about 60–100), since the selector only needs to rank members, not cover the space.
3. Fall back to a short online search over the top-k predicted configs for shapes outside the training distribution.

Option 2 can be checked now: on screen data, held-out p95 is 10.3% at 60 configs and 9.3% at 100. So size alone does not close
the gap; coverage of shape space does.

## Update 2026-10-15 (evening): full Triton config sweep (M3/M4); G2, G3 and G5 pass, G4 only for tiny GEMMs

**Stage 1 (screen):** session `sweep-20261015-075746-lock1755` (`python -m tune sweep`, 4.1 h). It covered a coarse subset of
**160 shapes**: about 23 per suite from S1–S7, tune split only, at most 1.2 TFLOP per shape, stratified by FLOPs
(`reports/snapshot/lock1755/sweep_subset.json`). Every config in the pruned space was tried on each shape: up to 10,840 per shape,
**912,913 shape × config cells** in total. Each cell was compiled, validated against the fp32 reference and screen-timed with a cold L2
(1 sample, cut at 1.5× the best so far, otherwise 5 samples; 15 samples with a trimmed mean under 50 µs). Results:
0 invalid configs, 0 crashes. Per-shape files are in `data/<hw>/sweep_screen/`.

**Stage 2 (full protocol):** session `bench-20261015-075746-lock1755-sweep` (38 min, 9,226 rows, all valid). For each shape, the 8 best
distinct tiles from the screen were timed alongside the 5 hand-picked configs, B1, B5, B2/B3 and B3+, in one session.
Tables: [`week1_results_lock1755_sweep.md`](week1_results_lock1755_sweep.md).

**Stage 2b (µs kernels):** the 74 subset shapes whose best kernel runs under 50 µs were re-timed with **CUDA-graph amortized timing** (hot, 50 launches
per graph, `--graph-batch 50`), as the methodology prescribes for kernels below the event tick. Session `bench-20261015-125953-lock1755-sweep-graph`, 2,259 rows, all valid.
Tables: [`week1_results_lock1755_sweep_graph.md`](week1_results_lock1755_sweep_graph.md).

### Headline (cold L2, per-launch, 160 shapes)

| | median | p10 | p90 | min | max |
|---|---|---|---|---|---|
| ours (sweep oracle) vs **B3+** | **0.988** | 0.85 | 1.07 | 0.69 | 1.27 |
| sweep oracle vs the 5 hand-picked configs | **1.075** | 1.00 | 2.20 | 1.00 | 4.72 |
| ours vs B1 | 1.075 | 0.95 | 1.36 | 0.78 | 1.94 |
| B3+ vs B1 | 1.078 | 1.01 | 1.36 | 0.99 | 1.95 |

The sweep's gain over the hand-picked configs is largest in S4 large-K (median 2.16×, from split-K variants), S3 skinny (1.26×) and S7 random (1.16×).

### Where ours stands against B3+

| kernel size | timing | n | median ours/B3+ | ours ahead >5% | within 2% | ours behind >5% |
|---|---|---|---|---|---|---|
| ≥ 50 µs | cold, per-launch | 86 | 0.981 | **0** | 39 | 28 |
| ≥ 50 µs, all of M, N, K divisible by 16 | cold, per-launch | 50 | 0.988 | 0 | | |
| ≥ 50 µs, some dimension unaligned | cold, per-launch | 36 | **0.904** | 0 | | |
| < 50 µs | hot, graph-amortized | 74 | 0.979 | 22 | 13 | 25 |
| B3+ < 5 µs | hot, graph-amortized | 20 | 1.087 | **12** | | 4 |
| B3+ 15–50 µs | hot, graph-amortized | 17 | 0.954 | 0 | | 6 |

- **For kernels at or above 50 µs (the range G1 covers), ours never beats B3+ by more than 5%.** On aligned shapes we are at parity (0.988). All 10 worst
  losses are unaligned shapes (3000³ 0.75×, 4088³ 0.76×, 4094³ 0.78×, 8191³ 0.84×, and S5/S7 odd shapes). This is a kernel-level limitation, not a
  selection one: with row strides that are not 16-byte aligned, Triton cannot issue vectorized 16-byte loads. A fix needs a different kernel
  (padding or a copy-to-aligned prologue), not better config choice.
- **Ours does win on tiny GEMMs (B3+ under about 5 µs).** With graph-amortized timing (the event tick does not limit it), ours is 1.08–1.23× faster
  on 12 of 20 such shapes: 64³, 128³, 256³, 127×2039×7, 4093×7×7 and seven S7 shapes with K ≤ 449. The absolute savings are only
  0.23–0.59 µs per call. Just above the band (5–15 µs) a few more wins persist: 8–16 × 1024 × 1024 at 1.11–1.12×, 1024×64×1024 at 1.14×, 16 × {128, 256} × 4096
  with split-K at 1.12–1.14×, and 518×813×87 at 1.20×. Losses there are just as common. The likely cause is cuBLAS's fixed per-kernel cost on problems
  that use only a handful of CTAs.
- **Losses under 50 µs** cluster in large-K split-K with tiny M and N (14×2×5131 0.63×, 7×7×2039 0.80×, 16×64×16384 0.85×). Our split-K writes fp32 partials
  and runs a separate reduction kernel, while cuBLAS uses a fused or atomic reduction. That is also a kernel-level fix.

### Portfolio study (G3, G5) from the screen data

From [`sweep_portfolio.md`](sweep_portfolio.md) (`python -m tune portfolio`). Regret is measured against the per-shape screen oracle.
- **G3, best single config over the full space:** B6 = `bm32_bn32_bk32_w4_s5_g1_sk1`. Its regret vs the per-shape oracle is median **74%**, p90 307%.
  In stage 2, the hand-picked B6 vs the sweep oracle has median 1.23×, p90 6.2×. Per-shape selection is clearly worth it.
- **G5, portfolio size:** a greedy portfolio of **29 configs** reaches p95 regret under 5% (mean 1.3%). 16 configs give a p95 of 8.7%, 50 configs 2.0%.
  Set cover needs 32 configs at ε = 5%, 43 at 3% and 54 at 2%.
- **Which parameters matter**, ranked by the regret from forcing each one to its worst value: SPLIT_K (121%) ≫ BLOCK_M (48%) > BLOCK_K (26%) >
  BLOCK_N (19%) ≫ num_warps (8%), num_stages (5%), GROUP_M (2%). The selector's features should capture the split-K and tile-size
  decisions. Stages and GROUP_M barely matter.
- **Caveats:** this is in-sample: the portfolio is chosen and evaluated on the same 160 tune-split shapes. Screen times are few-sample, so the regrets
  are slightly pessimistic. The held-out check is still to do: screen the held-out shapes, then evaluate the tune-shape portfolio on them.

### Gate status (supersedes earlier tables)

| Gate | Status |
|---|---|
| G1 measurement validity | **Pass** for kernels ≥ 50 µs, cold per-launch. The graph-amortized mode used for < 50 µs has no across-session noise study yet |
| G2 family ≥ 90% of B3+ | **Pass**: median 98.8% over 160 shapes from all 7 suites (aligned shapes at or above 50 µs: 98.8%; unaligned: 90.4%) |
| G3 selection value | **Pass**: full-space B6 regret median 74%, p90 307% (the gate asks for > 10% median or > 30% p90) |
| G4 differentiation | **Fails at or above 50 µs** (0 of 86 shapes beat B3+ by > 5%). **Narrow pass for tiny GEMMs** (B3+ < ~5 µs: 12 of 20 shapes, 8–23%, 0.23–0.59 µs absolute), pending a repeat session in graph mode |
| G5 tractability | **Pass (in-sample)**: 29 configs reach p95 regret < 5%. Held-out validation pending |

### Conclusion (plan §8: written down explicitly, because G4 is mostly failing)

On the RTX 3080, a well-searched cuBLASLt (B3+) is at least as fast as the best of our Triton family for every GEMM large enough
to matter (at least 50 µs). The family reaches parity on aligned shapes and loses about 10% on unaligned ones. The only regime where we beat B3+ is tiny
GEMMs under about 5 µs, by fractions of a microsecond. That matters only where launches are amortized (CUDA graphs) and such GEMMs dominate.
So **the project is not viable as a cuBLAS competitor on this GPU. It is viable as a selection project:** G3 and G5 both pass strongly
(selection is worth 74% at the median over any single config, and about 30 configs suffice). The practical framing that follows is a selector that
picks from {our portfolio ∪ cuBLASLt configs} (B3+ is itself a selection over a config space), or a non-NVIDIA backend target where no B3+ exists.

### Next steps
1. **Held-out G5 check:** screen the held-out S1/S2/S6/S7 shapes and evaluate the 29-config tune-split portfolio on them.
2. **Trivial selector baselines (plan 8.1.7):** nearest-neighbour on (log M, log N, log K) and a hand-written rule, evaluated on held-out shapes
   against the oracle and against B3+.
3. **Repeat the graph-mode µs measurement in a second session** (and after a reboot) to establish its noise floor before relying on the tiny-GEMM wins.
4. Kernel-level fixes, if differentiation still matters: an aligned-copy or padded path for unaligned strides, and a fused or atomic split-K reduction.

## Update 2026-10-15 (later): deeper B3 (B3+) closes the K = 4096 gap; G4 does not hold with 5 configs

Run: `bench-20261015-023430-lock1755-b3plus` (standard conditions, commit `08e2a2c`, 5624 rows, all valid), in
`results_lock1755_b3plus.jsonl`. It re-measures B1, B5, ours, B2/B3 and B3+ in the same session. Full tables:
[`week1_results_lock1755_b3plus.md`](week1_results_lock1755_b3plus.md).
**B3+** = the best of B3 and the 8 fastest configs from an enumerate-and-screen search over cuBLASLt's algorithm space
(algo × tile × stages × split-K × reduction × swizzle × custom; `baselines/cublaslt_search.py`). Each search ran in its own process with a 20 s budget per shape.
Every shape was screened exhaustively except 33 large ones (median 27% of valid configs screened, minimum 1% on 16384³; the heuristic's seeds were always included).

### Headline (cold L2, 143 shapes)

| | median | p10 | p90 | min | max |
|---|---|---|---|---|---|
| ours (5-config oracle) vs **B3+** | **0.959** | 0.83 | 1.01 | 0.66 | 1.15 |
| ours vs B3 (same session) | 0.991 | 0.88 | 1.08 | 0.66 | 1.36 |
| **B3+ vs B3** (what the deeper search buys) | 1.028 | 1.00 | 1.15 | 1.00 | **1.67** |
| B3+ vs B1 | 1.055 | 1.00 | 1.44 | 1.00 | 1.67 |

| regime (cold) | ours/B3+ median | B3+/B3 median | ours > B3+ by >5% | within 2% | ours < B3+ by >5% |
|---|---|---|---|---|---|
| S1 square (22) | 0.921 | 1.011 | 3 (64³, 128³, 256³; all < 7 µs) | 1 | 14 |
| S2 decode, M ≤ 16 (55) | 0.989 | 1.028 | **0** | 33 | 13 |
| S2 small batch, 32–128 (33) | 0.901 | 1.060 | **0** | 5 | 25 |
| S2 prefill, M ≥ 512 (33) | 0.957 | 1.026 | **0** | 0 | 10 |

Hot L2 gives the same picture: overall 0.947, with a single win over B3+ (128³, under 5 µs).

- **The K = 4096 small-M wins were cuBLAS heuristic misses.** On 2–16 × {11008, 12288} × 4096, ours/B3 was 1.07–1.17. Against
  B3+ it is **1.000–1.007**, i.e. a tie. The deep search finds split-K / tile configs that the heuristic never offers (e.g.
  8×11008×4096: B3 159 µs → B3+ 136 µs, ours 136 µs). The 128×1024×4096 split-K win (1.34× over B3) turns into a 0.80× loss: B3+ is
  1.67× faster than B3 there.
- **No regime survives G4 with the 5 hand-picked configs.** The only shapes where we beat B3+ by more than 5% are µs-scale squares, which G1 does not cover
  (per-launch timing has ~1 µs granularity) and which do not count.
- **G2 still passes, with less margin.** Median 95.9% of B3+. The weakest regime is small batch (median 0.901, right at the bar).
  Prefill sits at a uniform ~4% gap (0.93–0.97), and B3+ was budget-limited on most of those shapes, so the true gap may be slightly larger.
- **Largest remaining losses:** non-multiple-of-16 squares (1000³, 1023³, 1025³ at 0.66–0.71; 3000³ at 0.67), M = 128 small batch
  (0.80–0.87), and M = 1 GEMV (0.76 on 1×1024×4096).
- **The vendor oracle is a moving target.** Timing cuBLAS's own heuristic list gives +0.1% at the median (B3/B1). A 20 s enumerate-and-screen
  search adds a further 2.8% at the median and up to 1.67×. Any claim of "beats cuBLAS" has to be made against B3+.

### What G4 failing means here

G4 is defined on **our family's oracle**, and so far that oracle covers 5 configs. This result shows that 5 configs do not differentiate
from a well-searched cuBLAS. It does not yet show that the full Triton space cannot. That is settled by the full sweep (M3/M4), compared against B3+ under
the same budget treatment.

If G4 still fails after the sweep, then per plan §8, **the project remains viable as a research/learning project on selection (G3, G5) or as a
non-NVIDIA backend project, not as a cuBLAS competitor on the RTX 3080.** In that case the selector's target becomes "pick the best of
{our configs ∪ cuBLASLt configs}", and B3+ is the oracle to approach.

### Gate status (supersedes earlier tables)

| Gate | Status |
|---|---|
| G1 measurement validity | **Pass** for kernels ≥ 50 µs under standard conditions |
| G2 family ≥ 90% of B3+ | **Pass**: median 95.9% with 5 hand-picked configs (small batch 90.1%, at the bar) |
| G3 selection value | Positive: oracle/B6 median 1.04, p90 2.19; decode median 1.60 |
| G4 differentiation | **Fails with 5 configs** against B3+ (0 regimes; earlier K = 4096 wins are ties). Final verdict after the full sweep |
| G5 tractability | Not started |

### Next steps
1. Full pruned Triton config sweep (M3/M4) on a coarse subset of all suites, judged against B3+. This is the deciding input for G4 and
   the input to the portfolio study (M5, G5).
2. Targeted configs for the loss clusters: a GEMV-style config for M = 1, split-K variants for M = 128 × small N, and the
   unaligned-stride / masking path for non-multiple-of-16 squares.
3. If the sweep is run on large shapes, raise the B3+ budget there (e.g. 60 s) or note the coverage, since B3+ was budget-limited on 33 shapes.

## Update 2026-10-15: B2/B3 (cuBLASLt) added; G2 passes, G4 narrows sharply

Run: `bench-20261015-015054-lock1755-b3` (standard conditions, commit `8c896ab`, 3336 rows, all valid), in
`results_lock1755_b3.jsonl`. Full tables: [`week1_results_lock1755_b3.md`](week1_results_lock1755_b3.md).
B2 = cuBLASLt heuristic top-1. B3 = best of the heuristic's top-16 after timing each, which in practice is **a median of 4 candidates (1–9)**.

### Headline (cold L2, 143 shapes)

| | median | p10 | p90 | min | max |
|---|---|---|---|---|---|
| ours (5-config oracle) vs **B3** | **0.993** | 0.88 | 1.09 | 0.66 | 1.36 |
| ours vs B1 (for reference) | 1.008 | 0.92 | 1.31 | 0.66 | 1.51 |
| **B3 vs B1** (what timing cuBLAS's own candidates buys) | 1.001 | 1.00 | 1.18 | 0.99 | **1.57** |
| B2 vs B1 | 1.000 | 1.00 | 1.00 | 0.96 | 1.05 |

- **B1 is B2.** `torch.matmul` launches exactly the cuBLASLt top-1 kernel on 143/143 shapes (recorded kernel names).
- **G2 passes.** Median ours/B3 = 0.993, well above the 0.90 bar. Five hand-picked Triton configs reach vendor-oracle parity
  on the median shape.
- **Most of our earlier "wins over cuBLAS" were cuBLAS heuristic misses, not kernel advantages.** B3 recovers them by timing ≤ 8
  of cuBLAS's own candidates:
  - The 5120/13824 family (512×5120×5120: B1 751 µs → B3 483 µs): the heuristic ranks split-K algorithms 5th–6th.
  - M=1 decode (1×1024×4096): a dedicated `gemvx` kernel, ranked 4th, beats the GEMM that B1 uses.
  - 32×13824×5120 (the 38% "regression"): the heuristic picks a 256×64-tile kernel for 32 rows; its 2nd candidate is 11% faster.

  Against B3, all of these sit at 0.93–1.03×.

### G4: where ours still beats B3 by more than 5%

| regime | n | ours > B3 by >5% | within 2% | ours < B3 by >5% |
|---|---|---|---|---|
| S2 decode (M ≤ 16) | 55 | **11 (20%)** | 20 | 3 |
| S2 small batch (32–128) | 33 | 5 (15%) | 7 | 16 |
| S1 square | 22 | 5 (23%) | 3 | 11 |
| S2 prefill (M ≥ 512) | 33 | 1 (3%) | 16 | 1 |

The surviving wins cluster in **one identifiable sub-regime: K = 4096 with N ∈ {1024, 11008, 12288} at M ≤ 128**
(LLaMA-7B gate/up, the QKV projection, 1024-wide KV). There we are 1.06–1.17× faster than B3 with `bm16_bn64_bk128` (decode) and
`bm64_bn64` (M = 32–64). Two shapes also show split-K wins: 128×1024×4096 at 1.34×, and 512×11008×4096 at 1.15×.

**Caveats that keep G4 open:**
- **B3 is shallowest exactly where we win.** On the 11008 shapes cuBLASLt returns 1–2 near-identical candidates (both the
  older `s1688` 256×64 kernel). A full algorithm enumeration (`cublasLtMatmulAlgoGetIds`/`AlgoInit`/`AlgoCheck` over
  tiles × stages × split-K) may close these gaps. Until it is tried, these wins mean "beats cuBLAS's best heuristic candidate", not
  "beats cuBLAS".
- The square-shape wins (512³ at 1.36×, 1024³ at 1.22×, 128³, 256³, 64³) are all under 50 µs except 1024³, which is outside G1's coverage.
- Losses to B3 are concentrated in unaligned squares (4097³ 0.79×, 2049³ 0.84×, 4095³ 0.84×, 1000³ 0.66×), M = 128 small batch, and
  M = 1 GEMV (0.79×; we have no GEMV-style config).

### Gate status (supersedes earlier tables)

| Gate | Status |
|---|---|
| G1 measurement validity | **Pass** for kernels ≥ 50 µs under standard conditions |
| G2 family ≥ 90% of B3 | **Pass**: median 99.3% with only 5 hand-picked configs |
| G3 selection value | Positive: oracle/B6 median 1.04, p90 2.19; decode median 1.53 |
| G4 differentiation | **Narrow and provisional**: a K = 4096, M ≤ 128 sub-regime, 6–17% over a shallow B3. Needs a deeper B3 |
| G5 tractability | Not started |

**Implications for the project:**
1. The honest bar is B3, not B1. A trivial "time cuBLAS's own top-8" selector already captures up to 1.57× over `torch.matmul`.
   Any learned selector must be compared against that, not against B1.
2. Per-shape selection is valuable for both families. Whether *our* family adds anything beyond the vendor's candidates
   depends on the deeper-B3 check and on the full config sweep (M3/M4), which may widen the K = 4096 / small-M advantage.

### Next steps
1. **Deeper B3:** enumerate cuBLASLt algorithms beyond the heuristic (AlgoGetIds → AlgoInit → set tile/stages/split-K →
   AlgoCheck), capped per shape, at least on the G4 shapes. This decides whether the K = 4096 wins are real.
2. Full pruned Triton config sweep (M3/M4) on a coarse subset of all suites, then the portfolio study (M5) for G5.
3. Add a GEMV-style config (M = 1) and fix unaligned strides; both are among our main losses to B3.

## Update 2026-10-14: G1 closed under standard conditions; baselines re-run locked

> The sections below this one are the original 2026-10-09 results, measured **unlocked on driver 595.91.07**. For any
> new comparison, use the locked numbers in this section. Full tables: [`week1_results_lock1755.md`](week1_results_lock1755.md).

**Standard conditions from now on:** SM clock locked at 1755 MHz (`sudo nvidia-smi -lgc 1755,1755`, then
`python -m tune --clock-lock 1755 ...`), driver 595.99.02, 3 allocations per measurement, cold L2 primary. Every session now
writes `data/<hw>/env/<session_id>.json`, and every row carries its commit, clock-lock state and driver.

### G1: passes for kernels ≥ 50 µs

There were three complete locked sessions, the last one 2 minutes after a reboot and about 15 h after the first:

| | p50 | p95 | max |
|---|---|---|---|
| range of medians across the 3 sessions, cold | 0.24% | **1.75%** | 2.5% |
| range of medians across the 3 sessions, hot | 0.09% | **1.78%** | 3.0% |
| range of the kernel-vs-kernel ratio across the 3 sessions | 0.38% | **1.94%** | 2.2% |

Within each session, the p95 range across rounds is 1.3–1.9%. The median shift between any two sessions is ≤ 0.05%.
**The tie margin is now 2%** for kernels ≥ 50 µs (`docs/methodology.md`). Kernels under 50 µs are still about 6% p95 in cold
per-launch mode; rank them with graph-amortized timing. The lock is a ceiling only: under heavy load the power cap still pulls the clock
to about 1455–1590 MHz on some rows, and it does so the same way every session.

### B0 under the lock

60.0 TFLOP/s (59.7 on the clean-commit re-run) vs 62.9 unlocked, and 687 GB/s (unchanged). The unlocked file is kept as
`data/<hw>/roofline-20261009-unlocked.json`.

### S1 + S2 re-run under the lock (`results_lock1755.jsonl`, 1960 rows, all valid)

| (cold L2) | unlocked, 10-09 | **locked, 10-14** |
|---|---|---|
| ours vs B1, median [p10, p90] | 1.006 [0.92, 1.27] | **1.008 [0.92, 1.34]** |
| ours vs B5, median | 2.12 | **2.21** |
| oracle vs B6, median [p90] | 1.08 [2.13] | **1.04 [2.20]** |
| decode (M ≤ 16): ours vs B1 / oracle vs B6, median | 1.03 / 1.53 | **1.05 / 1.58** |
| shapes: win > 5% / within 2% / loss > 5% vs B1 | 42 / – / 26 | **44 / 45 / 28** |
| shapes beating B5 by > 5% (losing by > 5%) | 107 (0) | **111 (0)** |

**The conclusions hold.** Comparing unlocked to locked per shape, no shape flipped between win and loss. 23 of 143 moved between
neighbouring classes (tie ↔ win, tie ↔ loss), almost all of them already near the 5% boundary. Clocking down 5% slowed B1 by a median 3.1% and our
oracle by 1.5%.

The largest wins and losses are unchanged:
- Wins: 512×5120×5120 at 1.51×, 512×5120×13824 at 1.48×, M = 64 on the 13824/14336 layers at about 1.44×, decode M = 8 at 1.40×.
- Losses: 1000³ at 0.66×, 3000³ at 0.70×, M = 128 on large N/K at 0.84–0.86×.

**One unexplained cuBLAS change:** 32×13824×5120 went from parity to 1.36× because **B1** slowed from 229 µs to 316 µs (+38%,
the same on all 3 allocations). Our kernels moved less than 2%. A 5% lower clock cannot explain that on a memory-bound shape, so cuBLAS most likely
picked a different algorithm under the new driver or clock. torch and its bundled cuBLAS are unchanged. B1 rows don't record which cuBLAS
kernel ran, so this can't be confirmed from the data. That is one more reason to build B3, and to log the chosen cuBLAS kernel per row.

### Gate status (supersedes the table below)

| Gate | Status |
|---|---|
| G1 measurement validity | **Pass** for kernels ≥ 50 µs under standard conditions (p95 < 2% across 3 sessions and a reboot). Not covered: kernels under 50 µs |
| G2 family ≥ 90% of B3 | Not testable yet (no B3). Against B1: median 100.8% |
| G3 selection value | Positive early signal. Oracle/B6 median 1.04, p90 2.20; decode median 1.58 |
| G4 differentiation | Candidates: decode, the 5120/13824 family, M = 64. All need re-checking against B3 |
| G5 tractability | Not started |

### Next steps (in order)
1. **B3 (cuBLASLt exhaustive) and B2 (heuristic top-1)**, plus logging the chosen cuBLAS kernel name on B1 rows.
2. Kernel fix for unaligned strides (1000³, 3000³), then re-check S1 and S5.
3. M3/M4: full pruned config sweep on a coarse subset of all suites under standard conditions, then the portfolio study (M5).
4. Re-run `python -m tune --clock-lock 1755 noise --rounds 3 --repeats 50 --n-allocs 3` after any driver update.

## TL;DR

1. **The harness works and is validated.** It has env capture, adaptive CUDA-event timing (cold/hot L2, backlogged launches,
   graph-amortized mode), a correctness gate, resumable JSONL records, and a CLI. 67 unit tests pass. All 7 kernels
   (B1, B5, 5 hand-picked configs) pass correctness on all 169 edge shapes: 1183/1183.
2. **Noise floor ≈ 3% (p95), 0.3% (median), after a methodology fix.** The dominant noise source was **input allocation
   placement**, not heat or clocks. The same data at a different address shifts memory-bound kernels by up to 3.7%.
   Pooling every measurement over 3 placements halves the p95. **Tie margin ε = 3%.**
3. **Even 5 hand-picked Triton configs are competitive with cuBLAS on this card.** The per-shape oracle over them
   beats `torch.matmul` (B1) by more than 5% on 42/143 S1+S2 shapes, ties within 3% on 61, and loses by more than 5% on 26. The median is 1.006×,
   the range 0.67×–1.42×. It beats the Triton tutorial kernel (B5) by more than 5% on 107/143 and never loses to it by more than 5%.
4. **Per-shape selection already matters.** The best single hand-picked config (B6) gives up a median 7.7% vs the
   5-config oracle, and **53% (median), up to 3×, on decode shapes**. That is a first, small-scale signal for gate G3.

## Checklist status (plan §13)

| Item | Status |
|---|---|
| Pin environment; `env.json` capture | ✅ `requirements.txt`; `python -m tune env` (GPU and NVML state, versions, git commit, display/processes) |
| Timing utility + unit tests | ✅ `tune/timing.py`, `tests/test_timing.py` |
| Correctness checker + edge-shape suite | ✅ `tune/correctness.py`, `suites/v1/edge.json` (169 shapes); 1183/1183 pass |
| Noise-floor study on ~30 shapes (multi-session) | ⚠️ **2 sessions, same day** (the second one tests the allocation fix). Needs re-runs on other days for the across-session component |
| B0, B1 on S1 + S2, first sanity plot | ✅ `figures/roofline.png`, `figures/speedup_{cold,hot}.png` |
| Minimal Triton matmul with 3–5 hand-picked configs; beats B5 somewhere | ✅ 5 configs; beats B5 on 107/143 shapes by more than 5% |
| Decide primary cache mode | ✅ **cold L2** (reasons in methodology.md; the data shows median cold/hot = 1.003) |
| Shape suites S1–S7 as versioned JSON with held-out splits | ✅ `suites/v1/` with 608 shapes; region holdout (9000 ≤ K ≤ 12000, 300 ≤ M ≤ 450) plus 30% of S7 |

## B0: roofline

| | spec | measured |
|---|---|---|
| fp16 → fp32-acc tensor TFLOP/s | 59.5 (68 SMs × 1710 MHz × 512 FLOP/clk) | **62.9** (cuBLAS 8192³ at about 1875 MHz) |
| DRAM GB/s | 760 | **686** (1 GiB D2D copy, 90% of spec) |

The card boosts above its spec clock (up to 1965 MHz seen), which is why measured compute exceeds spec. Under sustained load
it is **power-capped** (NVML throttle reason 0x4 on about 90% of rows) and settles at about 1800 MHz. See `figures/roofline.png`:
decode GEMMs from both B1 and ours sit on the bandwidth roof, and large prefill or square GEMMs reach 55–62 TFLOP/s.

## Noise floor (Q4 / gate G1)

| | 1 allocation (session 1) | 3 allocations pooled (session 2) |
|---|---|---|
| across-round range of medians, cold, p50 / p95 / max | 0.67% / 5.4% / 20% | **0.29% / 2.6%** / 9.6% |
| same, kernels ≥ 50 µs, p95 | 5.2% | **2.4%** |
| paired ratio (B1 / ours) range, cold, p50 / p95 | 2.1% / 6.2% | **0.33% / 3.5%** |
| within-run IQR / median, p50 | 0.7% | 0.9% |

- **Thermal drift is negligible.** Continuous 8192³ for 10 min: +0.01% first minute → last minute, std 0.1%
  (`figures/thermal_drift.png`).
- **Allocation placement is the dominant term.** 8 fresh allocations × 3 timings each, for 8192×64×8192: 0.0% within an
  allocation, 3.7% across allocations. It does not cancel in kernel ratios. The fix is now the default
  (`--n-allocs 3`). Session 1 was run before the fix and is kept in `noise.jsonl` for comparison.
- **µs-scale kernels** (under about 20 µs) are quantized to a ~1.02 µs grid in per-launch mode, even though event resolution is 32 ns.
  The worst remaining outlier (9.6%) is one of these. Rank them with graph-amortized timing.
- **G1 verdict: provisionally passes at the median (0.3% ≪ 2%) and is borderline at p95 (2.6–3.5%).** Not yet shown
  reproducible across sessions. Clock locking (`sudo nvidia-smi -lgc`) was not available to the harness. It is the
  obvious next lever if p95 must come below 2%.

## Baselines on S1 + S2 (cold L2, 3 placements; 143 shapes, the 22 shapes with dims > 16384 excluded)

Speedups are time_baseline / time_ours (>1 means ours is faster). "ours" = per-shape best of the 5 hand-picked configs.

| regime | n | ours vs B1, median [p10, p90] | ours vs B5, median | oracle vs B6, median [p90] |
|---|---|---|---|---|
| S1 square | 22 | 0.94 [0.90, 1.16] | 1.87 | 1.08 [1.27] |
| S2 decode (M ≤ 16) | 55 | 1.03 [0.99, 1.33] | 4.14 | **1.53 [2.54]** |
| S2 small batch (32 ≤ M ≤ 128) | 33 | 0.98 [0.88, 1.11] | 2.94 | 1.00 [1.13] |
| S2 prefill (M ≥ 512) | 33 | 1.00 [0.95, 1.10] | 1.00 | 1.05 [1.08] |
| **all** | 143 | **1.006 [0.92, 1.27]** | **2.12** | 1.08 [2.13] |

The hot-L2 numbers are in `week1_results.md` and tell the same story.

**Where we win (more than 5% outside the noise floor):**
- Skinny decode GEMMs (`bm16_bn64_bk128`): up to 1.39× (M = 4–16, N = K = 5120).
- M = 64 and M = 512 on LLaMA-13B/8B-style dims: up to 1.42× (512×5120×5120: cuBLAS reaches 38 TFLOP/s, ours 54).

  The 5120/13824 family looks like a **cuBLAS heuristic miss**. B3 (cuBLASLt exhaustive) may close it, so these
  wins are not yet evidence for G4.

**Where we lose:**
- Unaligned squares: 1000³ at 0.67×, 3000³ at 0.71×, and 2^k ± 1 at about 0.89×. The likely cause (not yet profiled): Triton
  specializes integer args on divisibility by 16. Strides of 1000 or 3000 (≡ 8 mod 16) and odd strides miss that
  specialization, so it cannot prove alignment for wide vectorized loads. That is a kernel-level fix, not a selection
  problem (plan §8.1 item 5).
- M = 128 with large N and K: 0.84–0.87×. No hand-picked config targets 128×(large) tiles with deep K pipelining. The full sweep
  will tell whether the family covers it.

**B5 is a weak baseline.** It is 2× slower than cuBLAS at the median, and up to 14× slower on decode, because a 128×256 tile is
mostly padding for M ≤ 16. Beating B5 says little about the family's quality. B3 is the real bar.

## Gate status after week 1

| Gate | Status |
|---|---|
| G1 measurement validity | **Provisional pass** at the median; p95 borderline (≈3%); across-session not yet measured |
| G2 family ≥ 90% of B3 | Not testable yet (no B3). Against B1, 5 configs already reach a median of 100.6% |
| G3 selection value | **Early positive signal.** Oracle/B6 median 1.08, p90 2.13 with only 5 configs. Needs the full config space |
| G4 differentiation | Candidate regimes: decode (M ≤ 16) and the 5120 family. Must be re-checked against B3 |
| G5 tractability | Not started (needs the full sweep, M4–M5) |

## Recommended next steps

1. **Repeat the noise study on 2+ other days**, including one after a cold boot:
   `python -m tune noise --rounds 3 --repeats 50 --n-allocs 3`. That closes G1.
2. If you are OK with root, test clock locking: `sudo nvidia-smi -lgc 1755,1755`, re-run the noise study, then
   `sudo nvidia-smi -rgc`. Expect p95 to drop if clock variation is a meaningful part of the residual.
3. **Build B3** (cuBLASLt heuristic top-N enumeration via a ctypes/C++ shim) before trusting any "beats cuBLAS" claim.
   Add B2 (top-1 heuristic) from the same shim.
4. Kernel fix for unaligned K/N strides (e.g. a padded-K or an EVEN_N/aligned-pointer specialization), then re-check
   S1 and S5.
5. M3/M4: run the full pruned config space (`kernels/config_space.py`: static pruning keeps 10,840 of 19,200, and per-shape rules cut more; the GROUP_M × SPLIT_K axes alone are a 20× multiplier, so a coarse first pass should fix those) on a
   coarse subset of all suites, compute the real B6 and oracle, then do the portfolio study (M5).

## Reproduce

```bash
python -m tune env && python -m tune check --suite edge && python -m tune roofline
python -m tune noise --rounds 3 --repeats 50 --n-allocs 3
python -m tune drift --minutes 10
python -m tune benchmark --suite S1,S2 --backends torch,triton_tutorial,triton:handpicked --cache-modes cold,hot
python -m tune analyze
```
