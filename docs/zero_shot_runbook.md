# Runbook: zero-shot transfer of the 3080 selectors to another GPU

Runs the saved RTX 3080 rankers (`lgbm_rank` v1, `lgbm_rank_v2`) on a different NVIDIA GPU with **no refit** and scores the picks against that
GPU's own held-out screen. Code: `analysis/zero_shot.py` (module docstring has the long form); comparison across GPUs: `analysis/cross_gpu.py`.
First worked example (RTX 2060): `reports/week1.md`, "Update 2026-11-02 (later)". GPU job hygiene is `AGENTS.md` section 7.

On a machine you control, `pipeline/zero_shot.sh` runs the whole thing in one command (`pipeline/README.md`). On a SLURM cluster (no sudo, so no clock lock) use `remoteGPU/zero_shot.sh` (`sbatch remoteGPU/zero_shot.sh`), which runs steps 2-4 and 6 below with these checks built in.

`$D` below is the target machine's data directory (`data/<hw_id>` by default). If you use another one (e.g. a `pipeline/run.sh` run's `data/`), pass
`--out $D` to the `tune` commands and `--out $D` / `--data $D` to the `analysis.zero_shot` ones. `X` is the SM clock lock in MHz.

## 0. Prerequisites

- **Commit `9a2e9ab` or later.** `dab93ff` adds rank v2, the joint selector and `--lt-n`; `4319453` is needed for v2 `native` on a GPU smaller than the 3080
  (pool configs it cannot hold); `9a2e9ab` keeps picks the screen cut as `slow` in the statistics. Without that fix bad picks silently drop out of the tables.
- Package versions as in `requirements.txt` (torch 2.13.0+cu130, triton 3.7.1, lightgbm).
- GPU idle: `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` shows no process. Never overlap two GPU jobs.
- A clock lock the card holds under load without power-capping: `sudo nvidia-smi -lgc X,X` (the 2060 used 1755). Use the same X for every command; only compare
  sessions with the same lock and driver. Release afterwards with `sudo nvidia-smi -rgc`.
- Optional sanity check: `python -m pytest -q tests/test_zero_shot.py tests/test_cross_gpu.py` (CPU only).

## 1. Copy the 3080 models by hand

`data/` is gitignored. From `data/rtx3080_sm86_68sms_10gb/` copy

```
lgbm_rank.txt  lgbm_rank.json  lgbm_rank_v2.txt  lgbm_rank_v2.json        # about 71 MB in total
```

into a **separate** folder, e.g. `$D/src3080/`. If the machine already has its own `lgbm_rank.*` in `$D` (a retrained model), copying over it would overwrite it.
Check the copy with `md5sum` against the source. v2's analytic-feature constants are fixed model units: do not change them for the new GPU.

## 2. The oracle: this GPU's own held-out screen

```bash
python -m tune env
python -m tune --clock-lock X sweep --split heldout --per-suite 1000       # all 86 held-out shapes (3080: ~1.3 h)
```

- Smaller or slower card: `--per-suite 12` gives the 27-shape subset of the same 86 (the 2060 took 17.7 h for it, so the full 86 would take much longer).
  A subset keeps the comparison with the 3080 possible, because its shapes are 3080 held-out shapes.
- If the machine already has a `sweep_screen_heldout/` (e.g. from `pipeline/run.sh`), put it in `$D` and skip this step.
- `run` needs `$D/sweep_screen_heldout/subset.json`; never fit anything on these shapes.

## 3. Rank, compile and short-time on this GPU

```bash
python -m analysis.zero_shot --clock-lock X run --out $D \
  --ranker v1=$D/src3080/lgbm_rank v2=$D/src3080/lgbm_rank_v2 \
  --k 5 --lt-n 16 --fresh-cache
```

- About 20 min per model on the 2060 (27 shapes, both variants). Resumable per shape and k: after a CUDA fault just rerun the same command.
- `--clock-lock` goes before the subcommand. Do not start it while the sweep is still running.
- Variants: `as_is` (the ranker exactly as saved, with the 3080's SM count and limits in its features) and `native` (same booster, this GPU's numbers; what a
  deployment would do). `--k 5` stores the choice for every k' <= 5; `--lt-n 16` adds cuBLASLt's heuristic top-16 to the pool (joint selection).
- Output: `$D/zero_shot_{v1,v2}_{as_is,native}.jsonl` (+ `.raw.jsonl`).
- A bare `--ranker $D/src3080/lgbm_rank --k 3` is the original Triton-only experiment and keeps the old file names.

## 4. Score (CPU only)

```bash
python -m analysis.zero_shot score --data $D --report $R/k3 --k 3
python -m analysis.zero_shot score --data $D --report $R/k5 --k 5      # the frequent-call tier
```

Pass `--report` explicitly (`R` is any folder, e.g. `reports/zero_shot_<hw_id>`), otherwise the report lands in the repo's `reports/`. Output: `zero_shot_<dir name>.md`
(+ `_summary.json`): one fixed name per data directory, so **k = 3 and k = 5 must go to different folders** or the second overwrites the first.

## 5. Reading the report

1. `regret > 50%` and `unscored` first. Slow picks are scored (their one-sample screen time proves regret > 50%); `unscored` means the pick is not in this GPU's screen at
   all (pruned or failed there), and those are excluded from the statistics.
2. `native` against `as_is`: the gap is what the 3080's numbers in the features cost.
3. The `fixed` row (the 3080's B6 config reused as-is): if the ranker barely beats it, the transfer is weak.
4. Per-regime columns (>= 50 us, < 50 us): the < 50 us shapes are noisier and mostly launch-bound.
5. The "RTX 3080 reference" rows at the bottom are over the 3080's full 86 shapes. For a run on a subset compare on the shared shapes instead:

```bash
python -m analysis.cross_gpu \
  --side 3080-v1 data/rtx3080_sm86_68sms_10gb online_lgbm_rank_diverse.jsonl online_joint_diverse.jsonl results_lock1755_v2.jsonl results_lock1755_v2_graph.jsonl \
  --side 3080-v2 data/rtx3080_sm86_68sms_10gb online_joint_diverse_v2.jsonl online_joint_diverse_v2.jsonl results_lock1755_v2.jsonl results_lock1755_v2_graph.jsonl \
  --side zs-v1-native $D zero_shot_v1_native.jsonl zero_shot_v1_native.jsonl <stage2 cold file> <stage2 graph file> \
  --side zs-v2-native $D zero_shot_v2_native.jsonl zero_shot_v2_native.jsonl <stage2 cold file> <stage2 graph file>
```

(add `as_is` sides and a retrained-on-this-GPU side if you have one; the stage-2 files are the target's `results_validate*.jsonl` or the ones from step 6).

## 6. Optional: score the joint choice against B2 / B3 / B3+ (about 0.5 h)

```bash
python -m analysis.zero_shot --clock-lock X picks --data $D
```

It writes the stage-2 inputs and prints the two `tune benchmark` commands (cold, and graph for the shapes under 50 us) and the `analysis.ranker_compare`
command (it passes `--report`: without it `ranker_compare` overwrites the committed `reports/ranker_compare.md`, the 3080's). Add `cublaslt_deep:20` to `--backends` for B3+ (20 s per shape). Without a stage-2 session only cuBLASLt picks, and Triton picks that happen to be
timed there, have a full-protocol time.

## 7. Write it up

Add a dated `## Update YYYY-MM-DD: ...` at the top of `reports/week1.md`: session ids, row counts, invalid and skipped counts, the commits the rows record
(a resumed run can mix commits; say whether the difference matters), the tables, and what changed. Copy small report-relevant artifacts to `reports/snapshot/`.

## Caveats

- A 6 GB card records a too-large shape as `skipped` (counted, not scored).
- On Turing (sm_75) Triton's `tl.dot` compiled to FMAs, so a poor transfer there cannot be separated from the missing tensor-core path (check
  `online_triton_cache/*/*/_matmul_kernel.ptx` for `mma.sync`). A second tensor-core GPU is the clean test.
- The `native` pruner assumes Ampere-style stage buffers, so on Turing it may drop tiles that would run (conservative: a lower bound).
- Picks the 3080 model proposes can be absent from the new GPU's screen (pruned there but valid when run): they show up as `unscored`.
- n is small (27 or 86 shapes): read counts of picks > 50% off and intervals, not 1-point differences.
