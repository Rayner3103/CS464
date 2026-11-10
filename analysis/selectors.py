"""Trivial selector baselines on held-out shapes (plan 8.1 item 7).

Each selector maps (M, N, K) to one config of the Triton family. All are fit on the tune-split screen only
(data/<hw>/sweep_screen) and never see a held-out shape:

  * b6     - the best single config over the tune screen (lowest geomean regret), for reference.
  * nn1    - nearest tune shape in (log2 M, log2 N, log2 K), Euclidean; take that shape's fastest config.
  * knn5   - the 5 nearest tune shapes; take the config with the lowest mean log regret over them.
  * rule   - a hand-written rule (`rule_config`): tile from M and N, split-K to fill the SMs, warps from the tile.
  * nn1f / knnf - nn1 / k-NN on (log2 M, log2 N, log2 K) plus derived features (`shape_features`): how a few reference
             tiles fill the SMs (CTA count, wave tail), K steps left per split, and 8/16-alignment of each dim. Group
             weights and k are chosen on the tune shapes only, by grouped leave-one-out (`fit_features`).
  * knnc   - per-candidate k-NN (`knnc_select`): the same fill and split features computed for each candidate's own
             tile and split (`candidate_features`: waves at that config's residency, last-wave efficiency, K steps per
             split, M x N and K padding, EVEN_K) plus log2 M, N, K. Each runnable config is scored by its own regret on
             the k tune shapes where it saw the most similar conditions, so it can pick a config that was no
             neighbour's winner. dims weight and k are fit by grouped leave-one-out (`fit_candidate`, cached in
             data/<hw>/selector_knnc_fit.json because the fit takes minutes).

Leave-one-out on the tune shapes is grouped: a tune shape is never its own neighbour, nor is any tune shape sharing two
of (M, N, K) with it (S2's M sweeps over one layer would otherwise be near-copies), so in-sample numbers mimic novel shapes.

A selector only proposes configs that pass the static and shape pruning rules for the target shape (the same rules
the screen used), so a pick is always runnable; a neighbour's best config that is pruned on the target is skipped.

Scoring:
  * screen   - regret vs the held-out screen oracle (best of the full pruned space), as in analysis/portfolio.py.
  * stage 2  - one cold session that re-times, per held-out shape, the screen's top-8 tiles, every selector pick, B1,
               B3 and B3+ (`results_lock1755_selectors3.jsonl`). Regret vs the fastest Triton row; speedup vs B3+.
  * graph    - the same backends, hot, CUDA-graph amortized, on the held-out shapes whose screen oracle is under
               50 us (the methodology's protocol below the event tick; `results_lock1755_selectors3_graph.jsonl`).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from analysis.portfolio import REGRET_CAP, SMALL_US, _split_info, _stats, align, load_screen, regret_matrix

SELECTORS = ("b6", "nn1", "knn5", "rule", "nn1f", "knnf", "knnc")
FILL_TILES = ((16, 64), (16, 128), (32, 64), (64, 64), (64, 128), (128, 128))
FEATURE_GRID = {"fill": (0, 0.5, 1, 1.5, 2, 3), "align": (0, 0.25, 0.5, 1), "k": (1, 3, 5, 7, 10)}
LT_FAMILIES = ("cublaslt", "cublaslt_deep")


def hw_limits(data: Path):
    """HwLimits from the env capture, so picks can be computed without a GPU."""
    from kernels.config_space import HwLimits
    g = json.loads((Path(data) / "env.json").read_text())["gpu"]
    return HwLimits(g["smem_per_block_optin_bytes"], g["regs_per_sm"], g["sm_count"])


def runnable(cfg, M: int, N: int, K: int, hw) -> bool:
    from kernels.config_space import shape_violations, static_violations
    return not static_violations(cfg, hw) and not shape_violations(cfg, M, N, K)


def features(shapes: list[dict]) -> np.ndarray:
    return np.log2(np.array([[s["M"], s["N"], s["K"]] for s in shapes], float))


def shape_features(shapes: list[dict], sm_count: int) -> dict[str, np.ndarray]:
    """Feature groups per shape. dims: log2 of M, N, K. fill: for each FILL_TILES tile, log2(CTAs / SMs) and the wave
    efficiency CTAs / (waves * SMs) (low = a nearly empty last wave), plus log2 of the K steps (of 64) left per split
    if a 64x64 grid were split to fill the SMs. align: M, N, K divisible by 16 and by 8."""
    a = np.array([[s["M"], s["N"], s["K"]] for s in shapes], float)
    M, N, K = a.T
    fill = []
    for bm, bn in FILL_TILES:
        ctas = np.ceil(M / bm) * np.ceil(N / bn)
        fill += [np.log2(ctas / sm_count), ctas / (np.ceil(ctas / sm_count) * sm_count)]
    fill.append(np.log2(K / 64) - np.log2(np.maximum(1, sm_count / (np.ceil(M / 64) * np.ceil(N / 64)))))
    align_ = [x % 16 == 0 for x in (M, N, K)] + [x % 8 == 0 for x in (M, N, K)]
    return {"dims": np.log2(a), "fill": np.stack(fill, axis=1), "align": np.stack(align_, axis=1).astype(float)}


def embed(groups: dict[str, np.ndarray], ref: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Z-score each feature on the reference (tune) set; scale each group by weight / sqrt(group size), so a group's
    weight, not its feature count, sets its share of the distance. dims always has weight 1."""
    out = []
    for g, w in {"dims": 1.0, **weights}.items():
        if w > 0:
            mu, sd = ref[g].mean(axis=0), ref[g].std(axis=0)
            out.append(w * (groups[g] - mu) / np.where(sd > 0, sd, 1) / np.sqrt(groups[g].shape[1]))
    return np.concatenate(out, axis=1)


def share_two_dims(shapes: list[dict], others: list[dict]) -> np.ndarray:
    """[i, j] True when shapes[i] and others[j] agree on at least two of M, N, K (includes identical shapes)."""
    a = np.array([[s["M"], s["N"], s["K"]] for s in shapes])
    b = np.array([[s["M"], s["N"], s["K"]] for s in others])
    return (a[:, None, :] == b[None, :, :]).sum(axis=2) >= 2


def knn_select(X_train: np.ndarray, R_train: np.ndarray, X: np.ndarray, ok: np.ndarray, k: int,
               exclude: np.ndarray | None = None) -> np.ndarray:
    """Per target row of X, the column with the lowest mean log regret over its k nearest training shapes, among the
    columns marked runnable in ok[target]. k = 1 is plain nearest neighbour (the neighbour's fastest runnable config).
    exclude[i, j]: training shape j may not be a neighbour of target i (leave-one-out)."""
    logR = np.log(np.minimum(R_train, REGRET_CAP))
    D = ((X[:, None, :] - X_train[None, :, :]) ** 2).sum(axis=2)
    if exclude is not None:
        D = np.where(exclude, np.inf, D)
    nb = np.argsort(D, axis=1, kind="stable")[:, :k]
    return np.where(ok, logR[nb].mean(axis=1), np.inf).argmin(axis=1)


def fit_features(F: dict[str, np.ndarray], R: np.ndarray, ok: np.ndarray, exclude: np.ndarray,
                 grid: dict = FEATURE_GRID) -> list[dict]:
    """Grouped leave-one-out over the weight / k grid on the tune shapes. Returns every grid point with its in-sample
    regret stats, best (lowest capped mean regret) first."""
    import itertools
    out = []
    for wf, wa, k in itertools.product(grid["fill"], grid["align"], grid["k"]):
        w = {"fill": wf, "align": wa}
        X = embed(F, F, w)
        r = R[np.arange(len(R)), knn_select(X, R, X, ok, k, exclude)]
        out.append({"weights": w, "k": k, **_stats(r)})
    return sorted(out, key=lambda g: g["mean"])


CAND_FEATURES = ("log2_waves", "wave_eff", "log2_ksteps", "log2_pad_mn", "log2_pad_k", "even_k")
CAND_GRID = {"dims": (0, 0.25, 0.5, 1), "k": (3, 5, 10, 20)}


def candidate_features(shapes: list[dict], cfgs: list, gpu: dict) -> np.ndarray:
    """[shape, config, feature] for each candidate's own tile and split (CAND_FEATURES, then log2 M, N, K):
    waves = CTAs x SPLIT_K / (SMs x resident CTAs per SM), with residency bounded by shared memory (same smem estimate
    as the pruning rules), threads per SM and 16 CTAs per SM; wave_eff = the grid's share of its last full wave
    (low = a nearly empty last wave); K steps (of BLOCK_K) per split; M x N padded to whole tiles over M x N; K padded
    to whole BLOCK_K x SPLIT_K steps over K; whether the EVEN_K specialization applies."""
    a = np.array([[s["M"], s["N"], s["K"]] for s in shapes], float)
    M, N, K = (x[:, None] for x in a.T)
    c = np.array([[x.BLOCK_M, x.BLOCK_N, x.BLOCK_K, x.num_warps, x.num_stages, x.SPLIT_K] for x in cfgs], float)
    bm, bn, bk, warps, stages, sk = (x[None, :] for x in c.T)
    smem = stages * (bm * bk + bk * bn) * 2
    resident = np.maximum(1, np.minimum.reduce([gpu["smem_per_sm_bytes"] // smem,
                                                gpu["max_threads_per_sm"] // (32 * warps), np.full_like(smem, 16)]))
    tm, tn = np.ceil(M / bm), np.ceil(N / bn)
    ctas = tm * tn * sk
    cap = gpu["sm_count"] * resident
    ksteps = np.ceil(K / (sk * bk))
    feats = [np.log2(ctas / cap), ctas / (np.ceil(ctas / cap) * cap), np.log2(ksteps),
             np.log2(tm * bm * tn * bn / (M * N)), np.log2(ksteps * sk * bk / K), (K % (bk * sk) == 0).astype(float)]
    S, C = len(shapes), len(cfgs)
    feats += [np.broadcast_to(np.log2(x), (S, C)) for x in (M, N, K)]
    return np.stack(feats, axis=2).astype(np.float32)


def cand_scale(Fc: np.ndarray, timed: np.ndarray, dims_weight: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature (mean, multiplier) from the timed tune cells: z-score, then dims scaled by dims_weight /
    sqrt(3) and the candidate group by 1 / sqrt(its size), so dims_weight is the dims group's share of the distance."""
    x = Fc[timed]
    mu, sd = x.mean(axis=0), x.std(axis=0)
    nc = len(CAND_FEATURES)
    w = np.r_[np.full(nc, 1 / np.sqrt(nc)), np.full(3, dims_weight / np.sqrt(3))]
    return mu, w / np.where(sd > 0, sd, 1)


def knnc_select(Fc_train: np.ndarray, R_train: np.ndarray, Fc: np.ndarray, ok: np.ndarray, ks: tuple[int, ...],
                scale: tuple[np.ndarray, np.ndarray], exclude: np.ndarray | None = None) -> np.ndarray:
    """Per-candidate k-NN. For each target shape and each runnable config c, predict log regret as the mean over the k
    tune shapes where c was timed and c's own features (candidate_features) are closest to c's features on the target;
    pick the config with the lowest prediction. Unlike knn_select, a config wins on its own record, not by being some
    neighbour's winner. exclude[i, j]: tune shape j may not be used for target i. Returns [len(ks), targets]: the
    distances are computed once per target and shared by every k."""
    mu, w = scale
    timed = np.isfinite(R_train)
    logR = np.log(np.minimum(R_train, REGRET_CAP)).astype(np.float32)
    Zt = (Fc_train - mu) * w
    out = np.empty((len(ks), len(Fc)), int)
    for i in range(len(Fc)):
        cols = np.flatnonzero(ok[i])
        D = (((Zt[:, cols] - (Fc[i, cols] - mu) * w) ** 2).sum(axis=2))  # [tune shape, candidate]
        D[~timed[:, cols]] = np.inf
        if exclude is not None:
            D[exclude[i]] = np.inf
        order = np.argsort(D, axis=0, kind="stable")[:max(ks)]
        lr = np.where(np.isfinite(np.take_along_axis(D, order, axis=0)),
                      np.take_along_axis(logR[:, cols], order, axis=0), np.log(REGRET_CAP))
        csum = np.cumsum(lr, axis=0)
        for j, k in enumerate(ks):
            out[j, i] = cols[int(csum[k - 1].argmin())]  # mean over k neighbours: same argmin as the sum
    return out


def fit_candidate(Fc: np.ndarray, R: np.ndarray, ok: np.ndarray, exclude: np.ndarray,
                  grid: dict = CAND_GRID) -> list[dict]:
    """Grouped leave-one-out over dims weight x k on the tune shapes, best (lowest capped mean regret) first. Each
    entry keeps its leave-one-out picks (config column per tune shape) for the in-sample report."""
    timed = np.isfinite(R)
    out = []
    for wd in grid["dims"]:
        P = knnc_select(Fc, R, Fc, ok, grid["k"], cand_scale(Fc, timed, wd), exclude)
        for j, k in enumerate(grid["k"]):
            out.append({"dims_weight": wd, "k": k, **_stats(R[np.arange(len(R)), P[j]]), "picks": P[j].tolist()})
    return sorted(out, key=lambda g: g["mean"])


def rule_config(M: int, N: int, K: int, hw):
    """Hand-written rule. Written by looking at the tune-split oracle configs (never the held-out ones), then frozen.

    Tile: BLOCK_M follows M (16 / 32 / 64), BLOCK_N follows N (32 / 64 / 128; 64 for skinny-M up to N = 1024).
    BLOCK_K: short K -> 16 / 32; small tiles on long K -> 128 (few CTAs, so each must stream K fast); a grid that
    already fills the SMs -> 32; otherwise 64. SPLIT_K: double while the grid has fewer CTAs than SMs and each split
    keeps >= 2 K-steps. Warps: 2 for 16x32 tiles, 8 for >= 128x128, else 4. GROUP_M 8 whenever there is >1 tile row.
    If the result breaks a pruning rule, lower num_stages, then BLOCK_K, then SPLIT_K, then num_warps until it passes.
    """
    from kernels.triton_matmul import MatmulConfig
    cdiv = lambda a, b: -(-a // b)
    bm = 16 if M <= 16 else 32 if M <= 64 else 64
    bn = 32 if N <= 32 else 64 if (N <= 256 or (bm <= 32 and N <= 1024)) else 128
    tiles = cdiv(M, bm) * cdiv(N, bn)
    small = bm * bn <= 32 * 64 and bn <= 64
    if K <= 64:
        bk = 16 if K <= 16 else 32
    elif small and K >= 512:
        bk = 128
    elif tiles >= hw.sm_count:
        bk = 32
    else:
        bk = 64
    sk = 1
    while sk < 16 and tiles * sk < hw.sm_count and K >= 2 * sk * bk * 2:
        sk *= 2
    warps = 2 if bm * bn <= 16 * 32 else 8 if bm * bn >= 128 * 128 else 4
    stages = 4 if bk <= 32 else 3
    c = MatmulConfig(bm, bn, bk, num_warps=warps, num_stages=stages, GROUP_M=8 if cdiv(M, bm) > 1 else 1, SPLIT_K=sk)
    while not runnable(c, M, N, K, hw):
        if c.num_stages > 2:
            c = replace(c, num_stages=c.num_stages - 1)
        elif c.BLOCK_K > 16:
            c = replace(c, BLOCK_K=c.BLOCK_K // 2, num_stages=stages)
        elif c.SPLIT_K > 1:
            c = replace(c, SPLIT_K=c.SPLIT_K // 2)
        elif c.num_warps > 2:
            c = replace(c, num_warps=c.num_warps // 2)
        else:
            raise ValueError(f"rule found no runnable config for {M}x{N}x{K}")
    return c


def picks(data: Path, knn_ks=(1, 3, 5, 10)) -> dict:
    """Fit every selector on the tune screen; return their picks on the tune shapes (leave-one-out for the
    neighbour selectors) and on the held-out shapes, as column indices into the tune screen's config keys."""
    from tune.sweep import parse_key
    hw = hw_limits(data)
    shapes, keys, T = load_screen(data / "sweep_screen")
    R = regret_matrix(T)
    cfgs = [parse_key(k) for k in keys]
    hdir = data / "sweep_screen_heldout"
    hshapes, hkeys, hT = load_screen(hdir)
    hshapes = _split_info(hdir, hshapes)
    ok_tune = np.array([[runnable(c, s["M"], s["N"], s["K"], hw) for c in cfgs] for s in shapes])
    ok_held = np.array([[runnable(c, s["M"], s["N"], s["K"], hw) for c in cfgs] for s in hshapes])
    b6 = int(np.log(np.minimum(R, REGRET_CAP)).mean(axis=0).argmin())
    X, Xh = features(shapes), features(hshapes)
    loo = share_two_dims(shapes, shapes)
    F, Fh = shape_features(shapes, hw.sm_count), shape_features(hshapes, hw.sm_count)
    grid = fit_features(F, R, ok_tune, loo)
    fits = {"knnf": grid[0], "nn1f": next(g for g in grid if g["k"] == 1)}
    gpu = json.loads((Path(data) / "env.json").read_text())["gpu"]
    Fc, Fch = candidate_features(shapes, cfgs, gpu), candidate_features(hshapes, cfgs, gpu)
    tag = hashlib.sha1(json.dumps([keys, [(s["M"], s["N"], s["K"]) for s in shapes], CAND_FEATURES, CAND_GRID,
                                   gpu["sm_count"]]).encode()).hexdigest()
    cache = Path(data) / "selector_knnc_fit.json"
    cdoc = json.loads(cache.read_text()) if cache.exists() else {}
    if cdoc.get("tag") != tag or "p99" not in cdoc.get("grid", [{}])[0]:  # a fit cached before p99 was reported is stale
        cdoc = {"tag": tag, "grid": fit_candidate(Fc, R, ok_tune, loo)}
        cache.write_text(json.dumps(cdoc))
    cgrid = cdoc["grid"]
    fits["knnc"] = {k: v for k, v in cgrid[0].items() if k != "picks"}
    col = {k: j for j, k in enumerate(keys)}

    def rule_cols(ss):
        out = []
        for s in ss:
            key = rule_config(s["M"], s["N"], s["K"], hw).key()
            if key not in col:  # runnable but never screened on any tune shape: cannot happen with a full screen
                raise KeyError(f"rule picked {key} for {s}, which is not a tune-screen column")
            out.append(col[key])
        return np.array(out)

    tune = {"b6": np.full(len(shapes), b6), "rule": rule_cols(shapes)}
    held = {"b6": np.full(len(hshapes), b6), "rule": rule_cols(hshapes)}
    for k in knn_ks:
        name = "nn1" if k == 1 else f"knn{k}"
        tune[name] = knn_select(X, R, X, ok_tune, k, loo)
        held[name] = knn_select(X, R, Xh, ok_held, k)
    tune["knnc"] = np.array(cgrid[0]["picks"])
    held["knnc"] = knnc_select(Fc, R, Fch, ok_held, (cgrid[0]["k"],), cand_scale(Fc, np.isfinite(R), cgrid[0]["dims_weight"]))[0]
    for name, g in ((n, g) for n, g in fits.items() if n != "knnc"):
        Xf, Xfh = embed(F, F, g["weights"]), embed(Fh, F, g["weights"])
        tune[name] = knn_select(Xf, R, Xf, ok_tune, g["k"], loo)
        held[name] = knn_select(Xf, R, Xfh, ok_held, g["k"])
    return {"keys": keys, "shapes": shapes, "T": T, "hshapes": hshapes, "hkeys": hkeys, "hT": hT,
            "tune": tune, "held": held, "b6": keys[b6], "fits": fits, "grid": grid,
            "cgrid": [{k: v for k, v in g.items() if k != "picks"} for g in cgrid]}


def _rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def stage2(rows: list[dict], hshapes: list[dict], held_keys: dict[str, list[str]]) -> dict:
    """Per held-out shape from one benchmark session: Triton oracle (fastest valid Triton row), B3+ (fastest
    cuBLASLt row, heuristic or deep), B1, and each selector's pick time. Missing measurements are nan."""
    tri, lt, b1 = {}, {}, {}
    for r in rows:
        if not (r.get("valid") and r.get("time_median_us")):
            continue
        s, t = (r["M"], r["N"], r["K"]), r["time_median_us"]
        fam = r.get("kernel_family", "")
        if fam.startswith("triton_nn"):
            d = tri.setdefault(s, {})
            d[r["config_key"]] = min(d.get(r["config_key"], np.inf), t)
        elif fam in LT_FAMILIES:
            lt[s] = min(lt.get(s, np.inf), t)
        elif fam == "torch_matmul":
            b1[s] = min(b1.get(s, np.inf), t)
    sk = [(s["M"], s["N"], s["K"]) for s in hshapes]
    nan = float("nan")
    out = {"oracle": np.array([min(tri[s].values()) if s in tri else nan for s in sk]),
           "b3plus": np.array([lt.get(s, nan) for s in sk]), "b1": np.array([b1.get(s, nan) for s in sk])}
    for name, ks in held_keys.items():
        out[name] = np.array([tri.get(s, {}).get(k, nan) for s, k in zip(sk, ks)])
    return out


def _vs_bar(t: np.ndarray, bar: np.ndarray) -> dict:
    """Speedup bar / t over shapes where both exist (> 1: the selector's pick is faster than the bar)."""
    m = np.isfinite(t) & np.isfinite(bar)
    x = bar[m] / t[m]
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "median": float(np.median(x)), "p10": float(np.percentile(x, 10)),
            "p90": float(np.percentile(x, 90)), "min": float(x.min()), "geomean": float(np.exp(np.log(x).mean())),
            "ahead_gt5": int((x > 1.05).sum()), "within2": int((abs(x - 1) <= 0.02).sum()), "behind_gt5": int((x < 1 / 1.05).sum())}


def build(data: Path, report: Path, results: str = "results_lock1755_selectors3.jsonl",
          graph_results: str = "results_lock1755_selectors3_graph.jsonl") -> tuple[Path, dict]:
    P = picks(data)
    keys, shapes, T, hshapes = P["keys"], P["shapes"], P["T"], P["hshapes"]
    R = regret_matrix(T)
    oracle_h = P["hT"].min(axis=1)
    Rh = align(keys, P["hkeys"], P["hT"]) / oracle_h[:, None]
    names = [n for n in P["held"] if n in SELECTORS] + [n for n in P["held"] if n not in SELECTORS]

    # Picks file: what stage 2 times (backend triton:picks, env TUNE_PICKS).
    held_keys = {n: [keys[j] for j in P["held"][n]] for n in names}
    pk = {f"{s['M']}x{s['N']}x{s['K']}": sorted({held_keys[n][i] for n in SELECTORS}) for i, s in enumerate(hshapes)}
    (data / "selector_picks.json").write_text(json.dumps(
        {"selectors": list(SELECTORS), "feature_fits": P["fits"], "picks": pk,
         "by_selector": {n: {f"{s['M']}x{s['N']}x{s['K']}": held_keys[n][i] for i, s in enumerate(hshapes)}
                         for n in SELECTORS}}, indent=1))

    small = oracle_h < SMALL_US
    # Shapes for the graph-amortized stage 2 (< 50 us by the screen oracle).
    hdir = data / "sweep_screen_heldout"
    sub = json.loads((hdir / "subset.json").read_text())
    lt = {(s["M"], s["N"], s["K"]) for s, m in zip(hshapes, small) if m}
    (hdir / "subset_lt50us.json").write_text(json.dumps(
        {**sub, "note": f"held-out shapes whose screen oracle is < {SMALL_US:.0f} us",
         "shapes": [s for s in sub["shapes"] if (s["M"], s["N"], s["K"]) in lt]}, indent=1))
    groups = {"all": np.ones(len(hshapes), bool), f">= {SMALL_US:.0f} us": ~small, f"< {SMALL_US:.0f} us": small}
    for sp in sorted({s["split"] for s in hshapes if s["split"]}):
        groups[sp] = np.array([s["split"] == sp for s in hshapes])
    rows_i = np.arange(len(hshapes))
    screen = {n: {g: _stats(Rh[rows_i, P["held"][n]][m]) for g, m in groups.items()} for n in names}
    in_sample = {n: _stats(R[np.arange(len(shapes)), P["tune"][n]]) for n in names}
    agree = {n: float(np.mean([keys[j] == P["hkeys"][int(P["hT"][i].argmin())] for i, j in enumerate(P["held"][n])]))
             for n in names}

    summary = {"n_tune_shapes": len(shapes), "n_heldout_shapes": len(hshapes), "b6": P["b6"],
               "heldout_split_counts": {g: int(m.sum()) for g, m in groups.items()},
               "in_sample_grouped_loo": in_sample, "screen": screen, "exact_oracle_pick_rate": agree,
               "feature_fits": P["fits"], "feature_grid_top": P["grid"][:10], "candidate_grid": P["cgrid"]}

    pct = lambda x: f"{100 * x:.1f}%"
    md = ["# Trivial selector baselines on held-out shapes (auto-generated)\n",
          f"Selectors are fit on the tune-split screen ({len(shapes)} shapes x {len(keys)} configs) and scored on "
          f"{len(hshapes)} held-out shapes (" + ", ".join(f"{g} {int(m.sum())}" for g, m in groups.items() if g != "all")
          + "). Features: (log2 M, log2 N, log2 K), Euclidean, for nn1 / knn*; the same plus SM-fill and alignment groups "
          "(`shape_features`) for nn1f / knnf. Every pick passes the "
          "pruning rules for its target shape. `b6` = best single config on the tune screen "
          f"(`{P['b6']}`); `rule` = `analysis.selectors.rule_config`. Regret = t / oracle - 1.\n",
          "## Screen: regret vs the held-out per-shape best (full pruned space)\n",
          "| selector | in-sample (tune, grouped LOO) median / p95 / p99 | " + " | ".join(f"held-out {g} median / p95 / p99" for g in groups)
          + " | held-out mean | picks the oracle |", "|---|---|" + "---|" * len(groups) + "---|---|"]
    for n in names:
        md.append(f"| {n} | {pct(in_sample[n]['p50'])} / {pct(in_sample[n]['p95'])} / {pct(in_sample[n]['p99'])} | "
                  + " | ".join(f"{pct(screen[n][g]['p50'])} / {pct(screen[n][g]['p95'])} / {pct(screen[n][g]['p99'])}" for g in groups)
                  + f" | {pct(screen[n]['all']['mean'])} | {pct(agree[n])} |")
    md.append("\nIn-sample uses grouped leave-one-out for the neighbour selectors (no tune shape sharing two of M, N, K "
              "with the target is a neighbour); the rule was written looking at these tune "
              "shapes, so its in-sample number is optimistic. For reference, the tune-fitted 48-config greedy portfolio's "
              "held-out screen p95 is in `sweep_portfolio_heldout.md` (an oracle over 48 configs, not a selector).\n")
    fdesc = lambda g: f"fill weight {g['weights']['fill']}, align weight {g['weights']['align']}, k = {g['k']}"
    md.append("## Feature fit (tune shapes only)\n")
    md.append(f"Grid: fill weight {FEATURE_GRID['fill']}, align weight {FEATURE_GRID['align']}, k {FEATURE_GRID['k']} "
              "(dims weight 1). Chosen by the lowest capped mean regret under grouped leave-one-out on the tune shapes, "
              f"before scoring any held-out shape. **knnf: {fdesc(P['fits']['knnf'])}. nn1f: {fdesc(P['fits']['nn1f'])}.**\n")
    md.append("| fill | align | k | in-sample mean | median | p95 | p99 |")
    md.append("|---|---|---|---|---|---|---|")
    for g in P["grid"][:8]:
        md.append(f"| {g['weights']['fill']} | {g['weights']['align']} | {g['k']} | {pct(g['mean'])} | {pct(g['p50'])} | "
                  f"{pct(g['p95'])} | {pct(g.get('p99', float('nan')))} |")
    base = [g for g in P["grid"] if g["weights"] == {"fill": 0, "align": 0} and g["k"] in (1, 5)]
    for g in sorted(base, key=lambda g: g["k"]):
        md.append(f"| 0 | 0 | {g['k']} | {pct(g['mean'])} | {pct(g['p50'])} | {pct(g['p95'])} | {pct(g.get('p99', float('nan')))} |")
    md.append("\nThe last rows are the dims-only baselines (nn1, knn5) for comparison.\n")
    cf = P["fits"]["knnc"]
    md.append(f"Per-candidate k-NN (knnc), grid dims weight {CAND_GRID['dims']} x k {CAND_GRID['k']} (candidate features "
              f"{', '.join(CAND_FEATURES)} share weight 1), same selection rule. **knnc: dims weight {cf['dims_weight']}, "
              f"k = {cf['k']}.**\n")
    md.append("| dims weight | k | in-sample mean | median | p95 | p99 |")
    md.append("|---|---|---|---|---|---|")
    for g in P["cgrid"]:
        md.append(f"| {g['dims_weight']} | {g['k']} | {pct(g['mean'])} | {pct(g['p50'])} | {pct(g['p95'])} | {pct(g.get('p99', float('nan')))} |")
    md.append("")
    worst = {}
    for n in ("nn1", "knn5", "rule", "knnf", "knnc"):
        r = Rh[rows_i, P["held"][n]]
        worst[n] = [{"shape": f"{hshapes[i]['M']}x{hshapes[i]['N']}x{hshapes[i]['K']}", "split": hshapes[i]["split"],
                     "oracle_us": float(oracle_h[i]), "pick": keys[P["held"][n][i]], "regret": float(r[i] - 1)}
                    for i in np.argsort(-r)[:8]]
    summary["screen_worst"] = worst
    md.append("Worst held-out shapes (screen):\n")
    md.append("| selector | shape | split | oracle us | pick | regret |")
    md.append("|---|---|---|---|---|---|")
    for n, ws in worst.items():
        for w in ws[:5]:
            md.append(f"| {n} | {w['shape']} | {w['split']} | {w['oracle_us']:.1f} | `{w['pick']}` | {pct(w['regret'])} |")

    for label, fname, desc in (
            ("cold", results, "cold L2, per-launch, adaptive repeats, 3 input placements; all held-out shapes"),
            ("graph", graph_results, f"hot L2, CUDA-graph amortized (50 launches); held-out shapes with screen oracle "
                                     f"< {SMALL_US:.0f} us")):
        rows = _rows(data / fname)
        if not rows:
            md.append(f"\n## Stage 2 ({label}): `{fname}` not found\n")
            continue
        st = stage2(rows, hshapes, {n: held_keys[n] for n in SELECTORS})
        have = np.isfinite(st["oracle"]) & np.isfinite(st["b3plus"])
        small2 = st["oracle"] < SMALL_US
        g2 = {"all": have, f">= {SMALL_US:.0f} us": have & ~small2, f"< {SMALL_US:.0f} us": have & small2}
        if label == "graph":
            g2 = {"all": have, "B3+ < 5 us": have & (st["b3plus"] < 5), "B3+ 5-15 us": have & (st["b3plus"] >= 5) & (st["b3plus"] < 15),
                  "B3+ >= 15 us": have & (st["b3plus"] >= 15)}
        res = {"results": fname, "n_shapes": int(have.sum()), "n_rows": len(rows),
               "n_invalid_rows": sum(not r.get("valid") for r in rows),
               "missing_picks": {n: int((have & ~np.isfinite(st[n])).sum()) for n in SELECTORS},
               "regret": {n: {g: _stats(st[n][m] / st["oracle"][m]) for g, m in g2.items()} for n in SELECTORS},
               "vs_b3plus": {n: {g: _vs_bar(st[n][m], st["b3plus"][m]) for g, m in g2.items()} for n in ("oracle",) + SELECTORS},
               "b3plus_vs_b1": {g: _vs_bar(st["b3plus"][m], st["b1"][m]) for g, m in g2.items()}}
        summary[f"stage2_{label}"] = res
        md.append(f"\n## Stage 2 ({label}): re-measured, `{fname}`\n")
        md.append(f"{desc}. Per shape: the screen's top-8 tiles, every selector pick, B1, B3 (cuBLASLt heuristic top-16) "
                  f"and B3+ (B3 plus the deep cuBLASLt search), all in one session. {res['n_shapes']} shapes, "
                  f"{res['n_rows']} rows, {res['n_invalid_rows']} invalid. Oracle = fastest Triton row.\n")
        md.append("Regret vs the per-shape Triton oracle, median / p95 / p99:\n")
        md.append("| selector | " + " | ".join(f"{g} (n={int(m.sum())})" for g, m in g2.items()) + " |")
        md.append("|---|" + "---|" * len(g2))
        for n in SELECTORS:
            md.append(f"| {n} | " + " | ".join(
                f"{pct(res['regret'][n][g]['p50'])} / {pct(res['regret'][n][g]['p95'])} / {pct(res['regret'][n][g]['p99'])}" if res['regret'][n][g]['n'] else "-"
                for g in g2) + " |")
        md.append("\nSpeedup vs B3+ (t_B3+ / t_pick; > 1 = the pick is faster), median [p10, p90], and shapes "
                  "ahead by > 5% / within 2% / behind by > 5%:\n")
        md.append("| selector | " + " | ".join(g2) + " |")
        md.append("|---|" + "---|" * len(g2))
        for n in ("oracle",) + SELECTORS:
            cells = []
            for g in g2:
                v = res["vs_b3plus"][n][g]
                cells.append(f"{v['median']:.3f} [{v['p10']:.2f}, {v['p90']:.2f}] · {v['ahead_gt5']}/{v['within2']}/"
                             f"{v['behind_gt5']}" if v["n"] else "-")
            md.append(f"| {'Triton oracle' if n == 'oracle' else n} | " + " | ".join(cells) + " |")
        b = res["b3plus_vs_b1"]["all"]
        if b["n"]:
            md.append(f"\nB3+ vs B1 on these shapes: median {b['median']:.3f} [{b['p10']:.2f}, {b['p90']:.2f}].")
        if any(res["missing_picks"].values()):
            md.append(f"\nMissing pick measurements: {res['missing_picks']}.")
    report.mkdir(parents=True, exist_ok=True)
    (report / "selector_baselines_summary.json").write_text(json.dumps(summary, indent=1))
    out = report / "selector_baselines.md"
    out.write_text("\n".join(md) + "\n")
    return out, summary
