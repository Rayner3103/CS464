"""Learned selectors v0 (AGENTS.md next step 1): decision tree, linear SVM, LightGBM, random forest and logistic
regression, each in the two forms a supervised model can take here, fit on the tune-split screen only.

  * cls  - shape -> config classification over a label set. The label set is the first P members of the greedy
           portfolio refit on the training shapes (analysis.portfolio.greedy_portfolio); a shape's label is the first
           member, in greedy order, within LABEL_EPS of the shape's best member, so near-ties map to the broadest
           config instead of a noisy argmin. Inputs: the 22 shape features (`shape_features`: dims, fill, align).
           Pick = the highest-scoring member that is runnable on the target. Bounded by the portfolio oracle (which
           plateaus at ~8-9% held-out p95, AGENTS.md G5).
  * rank - per-candidate scoring. Candidates = the union of the top POOL_TOP configs of each training shape; one row
           per (shape, timed candidate) with the 22 shape features, the 7 config parameters (log2 where geometric)
           and the 6 per-candidate fill/split features (`candidate_features`). Trees and LightGBM regress capped log
           regret (LightGBM may instead use a lambdarank objective on regret bins, grouped by shape); the linear SVM
           (LinearSVR) regresses log regret and logistic regression classifies "within RANK_POS of the oracle", both
           on the standardized features plus every shape x candidate product, so a linear model can express "this
           tile on this shape". Pick = best predicted runnable pool member. Bounded by the pool oracle.

Hyperparameters are chosen per (model, form) on tune shapes only, by grouped N_FOLDS-fold cross-validation: a fold's
shapes are scored by a model trained on the other tune shapes minus any that share two of (M, N, K) with a fold shape
(`share_two_dims`, as the k-NN selectors' grouped leave-one-out). The share-two-dims graph has one component of 258
shapes, so grouping by component is not an option. With 20 folds each fit sees ~320 of the 477 tune shapes (grouped
LOO sees ~466), so CV numbers are a little pessimistic and favour data-light models; the held-out screen and stage 2
are the comparison. Selection rule is the k-NN selectors' (lowest capped mean regret). knnf / knn5 are scored under
the same folds as a reference (with their weights and k as fit on all tune shapes, which slightly favours them).

Held-out scoring reuses analysis/selectors.py: regret vs the held-out screen oracle (full pruned space), then stage 2
re-times every pick with the full protocol (`triton:picks` with TUNE_PICKS=data/<hw>/learned_picks.json).
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np

from analysis.portfolio import REGRET_CAP, SMALL_US, _split_info, _stats, align, greedy_portfolio, load_screen, regret_matrix
from analysis.selectors import (CAND_FEATURES, candidate_features, embed, hw_limits, knn_select, runnable,
                                shape_features, share_two_dims, stage2, _rows, _vs_bar)

MODELS = ("dt", "svm", "lgbm", "rf", "logreg")
FORMS = ("cls", "rank")
LABEL_EPS = 0.02   # cls: members within 2% of the shape's best member are ties (the >= 50 us tie margin)
POOL_TOP = 5       # rank: candidate pool = union of each training shape's top-5 configs (~900 of 10,840)
RANK_POS = 1.05    # rank, logistic: positive = within 5% of the shape's oracle
REL_BINS = (1.02, 1.05, 1.10, 1.30)  # rank, lambdarank: relevance 4..0 by regret bin
N_FOLDS = 20
SEED = 464
PARAM_NAMES = ("log2_bm", "log2_bn", "log2_bk", "log2_warps", "stages", "log2_group_m", "log2_split_k")

# Grids were widened once (2026-10-19) where the first pass's CV winner sat on an edge; tune shapes only.
GRIDS = {
    ("dt", "cls"): {"P": (8, 16, 32, 48), "max_depth": (4, 6, 8, None), "min_samples_leaf": (1, 3, 5, 10)},
    ("svm", "cls"): {"P": (8, 16, 32, 48), "C": (0.1, 1, 10, 100)},
    ("lgbm", "cls"): {"P": (8, 16, 32), "num_leaves": (4, 8, 16), "n_estimators": (100, 300), "min_child_samples": (3, 10)},
    ("rf", "cls"): {"P": (8, 16, 32, 48), "max_features": ("sqrt", 0.5, 1.0), "min_samples_leaf": (1, 3, 5)},
    ("logreg", "cls"): {"P": (8, 16, 32, 48), "C": (1, 10, 100, 1000)},
    ("dt", "rank"): {"max_depth": (12, 16, 24, None), "min_samples_leaf": (5, 20, 50)},
    ("svm", "rank"): {"C": (0.001, 0.01, 0.1)},  # flat over 100x in C (22.7% -> 22.4% CV mean); not widened
    ("lgbm", "rank"): {"objective": ("regression", "lambdarank"), "num_leaves": (63, 127, 255), "n_estimators": (800, 1500)},
    ("rf", "rank"): {"max_features": (0.33, 0.66, 1.0), "min_samples_leaf": (2, 5, 20)},
    ("logreg", "rank"): {"C": (0.01, 0.1, 1)},
}


# ---------------------------------------------------------------- features

def shape_matrix(shapes: list[dict], sm_count: int) -> np.ndarray:
    """[shape, 22]: dims (3), fill (13), align (6) from `shape_features`, unscaled."""
    F = shape_features(shapes, sm_count)
    return np.concatenate([F["dims"], F["fill"], F["align"]], axis=1)


def config_params(cfgs: list) -> np.ndarray:
    """[config, 7] tunables (PARAM_NAMES): log2 of the geometric ones, num_stages as is."""
    a = np.array([[c.BLOCK_M, c.BLOCK_N, c.BLOCK_K, c.num_warps, c.num_stages, c.GROUP_M, c.SPLIT_K] for c in cfgs], float)
    a[:, [0, 1, 2, 3, 5, 6]] = np.log2(a[:, [0, 1, 2, 3, 5, 6]])
    return a


def pair_rows(Xs: np.ndarray, P: np.ndarray, Fc: np.ndarray, si: np.ndarray, cj: np.ndarray) -> np.ndarray:
    """Per-candidate rows for (shape si[n], config cj[n]): shape features, config params, candidate features."""
    nc = len(CAND_FEATURES)
    return np.concatenate([Xs[si], P[cj], Fc[si, cj, :nc]], axis=1).astype(np.float32)


class Crossed:
    """Standardize, then append every shape-feature x candidate-feature product (z-scored inputs), so a linear model
    can weigh a tile's fit to a shape rather than shape and tile separately. n_shape = leading shape columns."""

    def __init__(self, n_shape: int):
        self.n_shape = n_shape

    def fit(self, X):
        self.mu, sd = X.mean(axis=0), X.std(axis=0)
        self.sd = np.where(sd > 0, sd, 1)
        Z = self._cross((X - self.mu) / self.sd)
        self.mu2, sd2 = Z.mean(axis=0), Z.std(axis=0)
        self.sd2 = np.where(sd2 > 0, sd2, 1)
        return self

    def _cross(self, Z):
        s, c = Z[:, :self.n_shape], Z[:, self.n_shape:]
        return np.concatenate([Z, (s[:, :, None] * c[:, None, :]).reshape(len(Z), -1)], axis=1).astype(np.float32)

    def transform(self, X):
        return ((self._cross((X - self.mu) / self.sd) - self.mu2) / self.sd2).astype(np.float32)


# ---------------------------------------------------------------- models

def make_model(model: str, form: str, hp: dict):
    """An unfitted estimator for (model, form) with hyperparameters hp (P, the cls label-set size, is not one)."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    hp = {k: v for k, v in hp.items() if k != "P"}
    if model == "dt":
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        return (DecisionTreeClassifier if form == "cls" else DecisionTreeRegressor)(random_state=SEED, **hp)
    if model == "rf":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        if form == "cls":
            return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED, **hp)
        return RandomForestRegressor(n_estimators=100, max_samples=0.3, n_jobs=-1, random_state=SEED, **hp)
    if model == "lgbm":
        import lightgbm as lgb
        if form == "cls":
            return lgb.LGBMClassifier(learning_rate=0.05, n_jobs=-1, random_state=SEED, verbose=-1, **hp)
        return lgb.LGBMRegressor(learning_rate=0.05, min_child_samples=20, subsample=0.8, subsample_freq=1,
                                 colsample_bytree=0.8, n_jobs=-1, random_state=SEED, verbose=-1, **hp) \
            if hp.get("objective") == "regression" else \
            lgb.LGBMRanker(learning_rate=0.05, min_child_samples=20, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, n_jobs=-1, random_state=SEED, verbose=-1, **hp)
    if model == "svm":
        from sklearn.svm import LinearSVC, LinearSVR
        if form == "cls":
            return make_pipeline(StandardScaler(), LinearSVC(C=hp["C"], max_iter=20000, random_state=SEED))
        return LinearSVR(C=hp["C"], epsilon=0.0, loss="squared_epsilon_insensitive", dual=False, max_iter=5000,
                         random_state=SEED)
    if model == "logreg":
        from sklearn.linear_model import LogisticRegression
        if form == "cls":
            return make_pipeline(StandardScaler(), LogisticRegression(C=hp["C"], max_iter=5000))
        return LogisticRegression(C=hp["C"], max_iter=2000)
    raise ValueError(model)


def canonical_labels(R: np.ndarray, members: list[int], eps: float = LABEL_EPS) -> np.ndarray:
    """Per shape, the index (into members, greedy order) of the first member within eps of the best member."""
    Rm = R[:, members]
    return (Rm <= Rm.min(axis=1, keepdims=True) * (1 + eps)).argmax(axis=1)


def _scores(est, X: np.ndarray) -> np.ndarray:
    return est.predict_proba(X) if hasattr(est, "predict_proba") else est.decision_function(X)


def fit_predict_cls(model: str, hp: dict, Xs: np.ndarray, R: np.ndarray, train: np.ndarray, Xt: np.ndarray,
                    ok_t: np.ndarray, fallback: int, full: bool = False):
    """Fit on shapes `train`, return a config column per target row of Xt (ok_t: [target, config] runnable).
    full: also return the label set (config columns) and the fitted estimator."""
    members = greedy_portfolio(R[train], hp["P"])
    y = canonical_labels(R[train], members)
    S = np.full((len(Xt), len(members)), -np.inf)
    est = None
    if len(np.unique(y)) == 1:
        S[:, y[0]] = 0
    else:
        est = make_model(model, "cls", hp).fit(Xs[train], y)
        sc = _scores(est, Xt)
        sc = sc[:, None] if sc.ndim == 1 else sc
        if sc.shape[1] == 1 and len(est.classes_) == 2:  # binary decision_function: score of classes_[1]
            sc = np.stack([-sc[:, 0], sc[:, 0]], axis=1)
        S[:, est.classes_] = sc
    S = np.where(ok_t[:, members], S, -np.inf)
    pick = np.where(np.isfinite(S.max(axis=1)), np.array(members)[S.argmax(axis=1)], fallback)
    return (pick, np.array(members), est) if full else pick


def rank_targets(R: np.ndarray, model: str, hp: dict) -> np.ndarray:
    if model == "logreg":
        return (R <= RANK_POS).astype(int)
    if hp.get("objective") == "lambdarank":
        return (len(REL_BINS) - np.searchsorted(np.array(REL_BINS), R, side="left")).astype(int)
    return np.log(np.minimum(R, REGRET_CAP))


def fit_predict_rank(model: str, hp: dict, Xs: np.ndarray, Pc: np.ndarray, Fc: np.ndarray, R: np.ndarray,
                     train: np.ndarray, Xt: np.ndarray, Fct: np.ndarray, ok_t: np.ndarray, fallback: int,
                     full: bool = False, topk: int = 1, scores: bool = False):
    """Fit per-candidate rows of shapes `train` over their top-POOL_TOP pool; pick the best-predicted runnable pool
    member per target (Fct: target candidate features [target, config, feature]). full: also return the pool
    (config columns) and the fitted estimator. topk > 1: return [target, topk] columns, best first (-1 past the
    last runnable pool member). scores: return (S [target, pool] predicted score, lower = better, inf where not
    runnable; pool; estimator) instead of picks."""
    pool = np.unique(np.argsort(R[train], axis=1, kind="stable")[:, :POOL_TOP])
    Rt = R[np.ix_(train, pool)]
    si, cj = np.nonzero(np.isfinite(Rt))  # timed cells only (pruned / invalid have no time)
    X = pair_rows(Xs, Pc, Fc, train[si], pool[cj])
    y = rank_targets(Rt[si, cj], model, hp)
    tf = Crossed(Xs.shape[1]).fit(X) if model in ("svm", "logreg") else None
    est = make_model(model, "rank", hp)
    if hp.get("objective") == "lambdarank":
        n = np.bincount(si, minlength=len(train))
        est.fit(X, y, group=n[n > 0])  # rows are grouped by shape (np.nonzero is row-major)
    else:
        est.fit(tf.transform(X) if tf else X, y)
    ti, tj = np.meshgrid(np.arange(len(Xt)), np.arange(len(pool)), indexing="ij")
    Xq = pair_rows(Xt, Pc, Fct, ti.ravel(), pool[tj.ravel()])
    Xq = tf.transform(Xq) if tf else Xq
    if model == "logreg":
        s = -est.predict_proba(Xq)[:, 1]
    elif hp.get("objective") == "lambdarank":
        s = -est.predict(Xq)
    else:
        s = est.predict(Xq)
    S = np.where(ok_t[:, pool], s.reshape(len(Xt), len(pool)), np.inf)
    if scores:
        return S, pool, est
    if topk > 1:
        order = np.argsort(S, axis=1, kind="stable")[:, :topk]
        pick = np.where(np.isfinite(np.take_along_axis(S, order, axis=1)), pool[order], -1)
    else:
        pick = np.where(np.isfinite(S.min(axis=1)), pool[S.argmin(axis=1)], fallback)
    return (pick, pool, est) if full else pick


# ---------------------------------------------------------------- cross-validation on tune shapes

def grouped_folds(shapes: list[dict], n_folds: int = N_FOLDS, seed: int = SEED) -> list[tuple[np.ndarray, np.ndarray]]:
    """(train, test) index pairs: random folds; train drops any shape sharing two of (M, N, K) with a test shape."""
    ex = share_two_dims(shapes, shapes)
    fold = np.empty(len(shapes), int)
    fold[np.random.default_rng(seed).permutation(len(shapes))] = np.arange(len(shapes)) % n_folds
    out = []
    for k in range(n_folds):
        test = np.flatnonzero(fold == k)
        out.append((np.flatnonzero((fold != k) & ~ex[test].any(axis=0)), test))
    return out


def grid_points(grid: dict) -> list[dict]:
    return [dict(zip(grid, v)) for v in itertools.product(*grid.values())]


def cv_picks(model: str, form: str, hp: dict, ctx: dict, folds) -> np.ndarray:
    picks = np.empty(len(ctx["R"]), int)
    for train, test in folds:
        if form == "cls":
            picks[test] = fit_predict_cls(model, hp, ctx["Xs"], ctx["R"], train, ctx["Xs"][test], ctx["ok"][test],
                                          ctx["b6"])
        else:
            picks[test] = fit_predict_rank(model, hp, ctx["Xs"], ctx["Pc"], ctx["Fc"], ctx["R"], train,
                                           ctx["Xs"][test], ctx["Fc"][test], ctx["ok"][test], ctx["b6"])
    return picks


def knn_cv_picks(F: dict, R: np.ndarray, ok: np.ndarray, weights: dict, k: int, folds) -> np.ndarray:
    """knnf / knn5 under the same folds (weights and k fixed as fit on all tune shapes)."""
    X = embed(F, F, weights)
    picks = np.empty(len(R), int)
    for train, test in folds:
        picks[test] = knn_select(X[train], R[train], X[test], ok[test], k)
    return picks


# ---------------------------------------------------------------- build

def context(data: Path) -> dict:
    """Tune and held-out screens, features and runnable masks (the masks take ~1 min of pruning-rule calls)."""
    from tune.sweep import parse_key
    hw = hw_limits(data)
    gpu = json.loads((Path(data) / "env.json").read_text())["gpu"]
    shapes, keys, T = load_screen(data / "sweep_screen")
    hdir = data / "sweep_screen_heldout"
    hshapes, hkeys, hT = load_screen(hdir)
    hshapes = _split_info(hdir, hshapes)
    cfgs = [parse_key(k) for k in keys]
    R = regret_matrix(T)
    return {"hw": hw, "shapes": shapes, "keys": keys, "T": T, "R": R, "hshapes": hshapes, "hkeys": hkeys, "hT": hT,
            "Rh": align(keys, hkeys, hT) / hT.min(axis=1)[:, None],
            "Xs": shape_matrix(shapes, hw.sm_count), "Xh": shape_matrix(hshapes, hw.sm_count),
            "F": shape_features(shapes, hw.sm_count), "Fh": shape_features(hshapes, hw.sm_count),
            "Pc": config_params(cfgs), "Fc": candidate_features(shapes, cfgs, gpu),
            "Fch": candidate_features(hshapes, cfgs, gpu),
            "ok": np.array([[runnable(c, s["M"], s["N"], s["K"], hw) for c in cfgs] for s in shapes]),
            "okh": np.array([[runnable(c, s["M"], s["N"], s["K"], hw) for c in cfgs] for s in hshapes]),
            "b6": int(np.log(np.minimum(R, REGRET_CAP)).mean(axis=0).argmin())}


def search(ctx: dict, log=print, only: tuple[str, ...] = ()) -> dict:
    """Grid search per (model, form) under grouped CV; returns {name: {"grid": [...best first], "picks": cv picks}}."""
    folds = grouped_folds(ctx["shapes"])
    out = {}
    R, rows = ctx["R"], np.arange(len(ctx["R"]))
    for (model, form), grid in GRIDS.items():
        name = f"{model}_{form}"
        if only and name not in only:
            continue
        res = []
        for hp in grid_points(grid):
            t0 = time.time()
            p = cv_picks(model, form, hp, ctx, folds)
            res.append({"hp": hp, **_stats(R[rows, p]), "fit_s": time.time() - t0, "picks": p.tolist()})
            log(f"{name} {hp}: mean {100 * res[-1]['mean']:.1f}% p50 {100 * res[-1]['p50']:.1f}% "
                f"p95 {100 * res[-1]['p95']:.1f}% ({res[-1]['fit_s']:.0f}s)")
        res.sort(key=lambda g: g["mean"])
        out[name] = {"grid": res}
    return out


def fit_final(ctx: dict, name: str, hp: dict):
    """Fit on all tune shapes; (held-out picks, label set or pool as config columns, estimator)."""
    model, form = name.rsplit("_", 1)
    train = np.arange(len(ctx["R"]))
    if form == "cls":
        return fit_predict_cls(model, hp, ctx["Xs"], ctx["R"], train, ctx["Xh"], ctx["okh"], ctx["b6"], full=True)
    return fit_predict_rank(model, hp, ctx["Xs"], ctx["Pc"], ctx["Fc"], ctx["R"], train, ctx["Xh"], ctx["Fch"],
                            ctx["okh"], ctx["b6"], full=True)


def feature_names() -> tuple[list[str], list[str]]:
    """(shape feature names, per-candidate row names) in the column order of shape_matrix / pair_rows."""
    from analysis.selectors import FILL_TILES
    sh = ["log2_M", "log2_N", "log2_K"]
    for bm, bn in FILL_TILES:
        sh += [f"log2_ctas_per_sm_{bm}x{bn}", f"wave_eff_{bm}x{bn}"]
    sh += ["log2_ksteps_per_split_64x64"] + [f"{d}_mod{a}" for a in (16, 8) for d in "MNK"]
    return sh, sh + list(PARAM_NAMES) + list(CAND_FEATURES)


def importances(est, names: list[str], top: int = 10) -> list[tuple[str, float]]:
    """Top features by the estimator's own importance (split gain for LightGBM, impurity for sklearn trees), as
    shares of the total; [] for linear models (their crossed coefficients are not comparable per input)."""
    if est is None:
        return []
    if hasattr(est, "booster_"):
        imp = est.booster_.feature_importance(importance_type="gain")
    elif hasattr(est, "feature_importances_"):
        imp = est.feature_importances_
    else:
        return []
    imp = np.asarray(imp, float) / max(np.sum(imp), 1e-12)
    return [(names[i], float(imp[i])) for i in np.argsort(-imp)[:top]]


def search_tag(ctx: dict) -> str:
    """Cache key for everything a grid entry depends on except its own grid."""
    import hashlib
    return hashlib.sha1(json.dumps([ctx["keys"], [(s["M"], s["N"], s["K"]) for s in ctx["shapes"]], LABEL_EPS,
                                    POOL_TOP, RANK_POS, REL_BINS, N_FOLDS, SEED]).encode()).hexdigest()


def search_cached(ctx: dict, data: Path, log=print) -> dict:
    """search(), cached per (model, form) in data/<hw>/learned_search.json; an entry is refit only when its grid or
    the shared tag (screen, constants) changes. The full grid takes ~1 h on 12 cores, most of it LinearSVR."""
    tag = search_tag(ctx)
    cache = Path(data) / "learned_search.json"
    doc = json.loads(cache.read_text()) if cache.exists() else {}
    if doc.get("tag") != tag:
        doc = {"tag": tag, "search": {}}
    for (model, form), grid in GRIDS.items():
        name = f"{model}_{form}"
        spec = json.loads(json.dumps(grid))
        if doc["search"].get(name, {}).get("grid_def") != spec:
            doc["search"][name] = {"grid_def": spec, **search(ctx, log, only=(name,))[name]}
            cache.write_text(json.dumps(doc))
    return doc["search"]


class Ranker:
    """The rank-form LightGBM selector as a standalone object: its booster plus the candidate pool (config keys) and
    the GPU numbers the features need, so it can rank configs for any (M, N, K) without the screen. Saved as
    <stem>.txt (LightGBM model text) + <stem>.json (pool, hyperparameters, gpu, training-set tag)."""

    def __init__(self, booster, pool_keys: list[str], hp: dict, gpu: dict, meta: dict | None = None):
        from tune.sweep import parse_key
        self.booster, self.pool_keys, self.hp, self.gpu, self.meta = booster, list(pool_keys), hp, gpu, meta or {}
        self.cfgs = [parse_key(k) for k in self.pool_keys]
        self.Pc = config_params(self.cfgs)
        from kernels.config_space import HwLimits
        self.hw = HwLimits(gpu["smem_per_block_optin_bytes"], gpu["regs_per_sm"], gpu["sm_count"])

    def scores(self, M: int, N: int, K: int) -> np.ndarray:
        """Predicted log regret per pool member (lower is better; inf where the pruning rules reject it)."""
        s = [{"M": M, "N": N, "K": K}]
        X = pair_rows(shape_matrix(s, self.gpu["sm_count"]), self.Pc, candidate_features(s, self.cfgs, self.gpu),
                      np.zeros(len(self.cfgs), int), np.arange(len(self.cfgs)))
        pred = self.booster.predict(X)
        pred = -pred if self.hp.get("objective") == "lambdarank" else pred
        ok = np.array([runnable(c, M, N, K, self.hw) for c in self.cfgs])
        return np.where(ok, pred, np.inf)

    def rank(self, M: int, N: int, K: int, k: int = 5, order: str = "plain") -> list[str]:
        """Top k pool configs: plain score order, or "diverse" (the CV-chosen rule in meta["diverse_rule"],
        `diverse_order`)."""
        sc = self.scores(M, N, K)
        rule = PLAIN_RULE if order == "plain" else self.meta["diverse_rule"]
        groups = config_groups(self.cfgs, rule["group"])
        return [self.pool_keys[j] for j in diverse_order(sc, groups, k, rule)]

    def save(self, stem: Path) -> Path:
        stem = Path(stem)
        self.booster.save_model(str(stem.with_suffix(".txt")))
        stem.with_suffix(".json").write_text(json.dumps(
            {"pool": self.pool_keys, "hp": self.hp, "gpu": self.gpu, "meta": self.meta}, indent=1))
        return stem.with_suffix(".json")

    @classmethod
    def load(cls, stem: Path) -> "Ranker":
        import lightgbm as lgb
        stem = Path(stem)
        d = json.loads(stem.with_suffix(".json").read_text())
        return cls(lgb.Booster(model_file=str(stem.with_suffix(".txt"))), d["pool"], d["hp"], d["gpu"], d["meta"])


def train_ranker(data: Path, ctx: dict | None = None, log=print) -> tuple["Ranker", dict]:
    """Refit lgbm_rank on every tune shape with its CV-chosen hyperparameters; save it as data/<hw>/lgbm_rank.*"""
    ctx = ctx or context(data)
    S = search_cached(ctx, data, log)
    hp = S["lgbm_rank"]["grid"][0]["hp"]
    cols, pool, est = fit_predict_rank("lgbm", hp, ctx["Xs"], ctx["Pc"], ctx["Fc"], ctx["R"], np.arange(len(ctx["R"])),
                                       ctx["Xh"], ctx["Fch"], ctx["okh"], ctx["b6"], full=True, topk=TOPK)
    gpu = json.loads((Path(data) / "env.json").read_text())["gpu"]
    # Diversification rule for top-k, chosen on out-of-fold tune scores only (best-of-DIVERSE_K mean regret).
    rules = fit_diverse_rule(ctx, cv_scores(ctx, data, hp, log)["S"], DIVERSE_K)
    rk = Ranker(est.booster_, [ctx["keys"][j] for j in pool], hp, gpu,
                {"tag": search_tag(ctx), "n_tune_shapes": len(ctx["shapes"]), "pool_top": POOL_TOP,
                 "diverse_rule": rules[0]["rule"], "diverse_k": DIVERSE_K})
    rk.save(Path(data) / "lgbm_rank")
    return rk, {"ctx": ctx, "cols": cols, "pool": pool, "rules": rules}


TOPK = 5
REFERENCE = ("b6", "knn5", "knnf")

# ---------------------------------------------------------------- diversified top-k
# Plain top-k of lgbm_rank mostly re-times near-duplicates: its top 5 average 1.9 distinct (BLOCK_M, BLOCK_N, SPLIT_K)
# choices, and the worst held-out shapes are ones where the first-ranked tile x split is wrong, so no k fixes them.
# A diversified order keeps the first `plain` picks, then takes the best-scored config of each not-yet-seen group
# (GROUPINGS), as long as its predicted log regret is within `delta` of the top pick's; then falls back to score
# order. The rule is chosen on tune shapes only, by best-of-k screen regret under the same grouped CV folds.

GROUPINGS = {"tile_split": (0, 1, 6), "tile": (0, 1), "split": (6,), "tile_split_bk": (0, 1, 2, 6)}  # PARAMS index
DIVERSE_GRID = {"group": tuple(GROUPINGS), "delta": (0.02, 0.03, 0.05, 0.1, 0.2, 0.5, float("inf")), "plain": (1, 2)}
PLAIN_RULE = {"group": None, "delta": 0.0, "plain": TOPK}
DIVERSE_K = 3  # the rule is fit for best-of-3 (the online search's default budget)


def config_groups(cfgs: list, group: str | None) -> np.ndarray:
    """An integer group id per config for `group` (GROUPINGS name); every config its own group for None."""
    if group is None:
        return np.arange(len(cfgs))
    t = [tuple((c.BLOCK_M, c.BLOCK_N, c.BLOCK_K, c.num_warps, c.num_stages, c.GROUP_M, c.SPLIT_K)[i]
               for i in GROUPINGS[group]) for c in cfgs]
    ids = {g: i for i, g in enumerate(dict.fromkeys(t))}
    return np.array([ids[g] for g in t])


def score_order(score: np.ndarray) -> list[int]:
    """Runnable indices (finite score) best first."""
    fin = np.flatnonzero(np.isfinite(score))
    return fin[np.argsort(score[fin], kind="stable")].tolist()


def diverse_order(score: np.ndarray, groups: np.ndarray, k: int, rule: dict, order: list[int] | None = None) -> list[int]:
    """Indices of the k picks (into score / groups) under a diversification rule; score lower = better, inf =
    not runnable. rule: {"group", "delta", "plain"} (PLAIN_RULE = plain top-k). order: score_order(score), if known."""
    order = score_order(score) if order is None else order
    if not order:
        return []
    picked = order[:min(rule["plain"], k)]
    if rule["group"] is not None:
        seen = {groups[j] for j in picked}
        for j in order[len(picked):]:
            if len(picked) >= k or score[j] - score[order[0]] > rule["delta"]:
                break
            if groups[j] not in seen:
                picked.append(j)
                seen.add(groups[j])
    for j in order:  # fill from score order
        if len(picked) >= k:
            break
        if j not in picked:
            picked.append(j)
    return picked


def cv_scores(ctx: dict, data: Path, hp: dict, log=print) -> dict:
    """Out-of-fold lgbm_rank scores on every tune shape (grouped N_FOLDS folds, the search's folds) with
    hyperparameters hp, cached in data/<hw>/lgbm_rank_cv_scores.npz. Returns {"S": [shape, config] score, inf outside
    the fold's pool or not runnable}."""
    import hashlib
    tag = hashlib.sha1(json.dumps([search_tag(ctx), hp]).encode()).hexdigest()
    cache = Path(data) / "lgbm_rank_cv_scores.npz"
    if cache.exists():
        z = np.load(cache)
        if str(z["tag"]) == tag:
            return {"S": z["S"]}
    S = np.full(ctx["R"].shape, np.inf, np.float32)
    for f, (train, test) in enumerate(grouped_folds(ctx["shapes"])):
        Sf, pool, _ = fit_predict_rank("lgbm", hp, ctx["Xs"], ctx["Pc"], ctx["Fc"], ctx["R"], train, ctx["Xs"][test],
                                       ctx["Fc"][test], ctx["ok"][test], ctx["b6"], scores=True)
        S[np.ix_(test, pool)] = Sf
        log(f"cv scores fold {f + 1}/{N_FOLDS}")
    np.savez_compressed(cache, S=S, tag=tag)
    return {"S": S}


def best_of_k(R: np.ndarray, S: np.ndarray, groups_by_rule: dict, rule: dict, k: int, orders=None) -> np.ndarray:
    """Per shape, the screen regret of the fastest of the k configs the rule picks from scores S."""
    g = groups_by_rule[rule["group"]]
    orders = orders or [score_order(x) for x in S]
    return np.array([R[i, diverse_order(S[i], g, k, rule, orders[i])].min() if orders[i] else REGRET_CAP
                     for i in range(len(R))])


def fit_diverse_rule(ctx: dict, S: np.ndarray, k: int = 3, grid: dict = DIVERSE_GRID) -> list[dict]:
    """Every diversification rule (and plain top-k) scored by best-of-k regret on the tune shapes' out-of-fold scores,
    best (lowest capped mean) first. Ties on the mean go to the simpler rule (plain, then fewer diverse picks)."""
    from tune.sweep import parse_key
    cfgs = [parse_key(x) for x in ctx["keys"]]
    G = {gname: config_groups(cfgs, gname) for gname in (None, *GROUPINGS)}
    rules = [PLAIN_RULE] + [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
    orders = [score_order(x) for x in S]
    out = []
    for r in rules:
        out.append({"rule": r, **{f"k{j}": _stats(best_of_k(ctx["R"], S, G, r, j, orders)) for j in sorted({1, k, TOPK})}})
    return sorted(out, key=lambda e: (round(e[f"k{k}"]["mean"], 4), e["rule"]["group"] is not None, -e["rule"]["plain"]))


def build(data: Path, report: Path, results: str = "results_lock1755_learned.jsonl",
          graph_results: str = "results_lock1755_learned_graph.jsonl", log=print) -> tuple[Path, dict]:
    ctx = context(data)
    S = search_cached(ctx, data, log)
    keys, shapes, hshapes, R, Rh = ctx["keys"], ctx["shapes"], ctx["hshapes"], ctx["R"], ctx["Rh"]
    col = {k: j for j, k in enumerate(keys)}
    names = [f"{m}_{f}" for f in FORMS for m in MODELS]
    sh_names, row_names = feature_names()
    tune_rows, held_rows = np.arange(len(shapes)), np.arange(len(hshapes))

    # Reference selectors: held-out picks as written by `tune selectors`; CV picks under the same folds.
    sel = json.loads((Path(data) / "selector_picks.json").read_text())
    folds = grouped_folds(shapes)
    knnf_fit = sel["feature_fits"]["knnf"]
    held = {n: np.array([col[sel["by_selector"][n][f"{s['M']}x{s['N']}x{s['K']}"]] for s in hshapes]) for n in REFERENCE}
    cvp = {"b6": np.full(len(shapes), ctx["b6"]),
           "knn5": knn_cv_picks(ctx["F"], R, ctx["ok"], {"fill": 0, "align": 0}, 5, folds),
           "knnf": knn_cv_picks(ctx["F"], R, ctx["ok"], knnf_fit["weights"], knnf_fit["k"], folds)}
    fits, ceil = {}, {}
    for n in names:
        best = S[n]["grid"][0]
        pick, cand, est = fit_final(ctx, n, best["hp"])
        held[n], cvp[n] = pick, np.array(best["picks"])
        imp = importances(est, sh_names if n.endswith("_cls") else row_names)
        fits[n] = {"hp": best["hp"], "n_candidates": int(len(cand)), "importance": imp,
                   "grid": [{k: v for k, v in g.items() if k != "picks"} for g in S[n]["grid"]]}
        # Ceiling: the best label-set / pool member per held-out shape (an oracle over the candidates, not a pick).
        ceil[n] = Rh[:, cand].min(axis=1)
    order = list(REFERENCE) + names

    small = ctx["hT"].min(axis=1) < SMALL_US
    groups = {"all": np.ones(len(hshapes), bool), f">= {SMALL_US:.0f} us": ~small, f"< {SMALL_US:.0f} us": small}
    for sp in sorted({s["split"] for s in hshapes if s["split"]}):
        groups[sp] = np.array([s["split"] == sp for s in hshapes])
    screen = {n: {g: _stats(Rh[held_rows, held[n]][m]) for g, m in groups.items()} for n in order}
    cv = {n: _stats(R[tune_rows, cvp[n]]) for n in order}
    hbest = ctx["hT"].argmin(axis=1)
    hcol = np.array([col.get(ctx["hkeys"][j], -1) for j in hbest])
    agree = {n: float(np.mean(held[n] == hcol)) for n in order}
    ceilings = {n: {g: _stats(ceil[n][m]) for g, m in groups.items()} for n in names}

    held_keys = {n: [keys[j] for j in held[n]] for n in order}
    tag = lambda s: f"{s['M']}x{s['N']}x{s['K']}"
    (Path(data) / "learned_picks.json").write_text(json.dumps(
        {"selectors": order, "fits": {n: {k: v for k, v in f.items() if k != "grid"} for n, f in fits.items()},
         "picks": {tag(s): sorted({held_keys[n][i] for n in order}) for i, s in enumerate(hshapes)},
         "by_selector": {n: {tag(s): held_keys[n][i] for i, s in enumerate(hshapes)} for n in order}}, indent=1))

    summary = {"n_tune_shapes": len(shapes), "n_heldout_shapes": len(hshapes), "n_folds": N_FOLDS,
               "cv_train_shapes": [int(len(tr)) for tr, _ in folds], "cv": cv, "screen": screen,
               "exact_oracle_pick_rate": agree, "candidate_ceiling": ceilings, "fits": fits,
               "distinct_picks": {n: int(len(set(held[n].tolist()))) for n in order}}
    pct = lambda x: f"{100 * x:.1f}%"
    ms = lambda d: f"{pct(d['p50'])} / {pct(d['p95'])}" if d.get("n") else "-"
    md = ["# Learned selectors v0 on held-out shapes (auto-generated)\n",
          f"Five model families x two forms, fit on the tune-split screen ({len(shapes)} shapes x {len(keys)} configs) and "
          f"scored on {len(hshapes)} held-out shapes. **cls** = shape -> config classification over the first P greedy "
          f"portfolio members (labels: first member within {pct(LABEL_EPS)} of the shape's best); **rank** = per-candidate "
          f"scoring over each training shape's top-{POOL_TOP} pool (trees / LightGBM regress log regret or lambdarank; "
          f"LinearSVR regresses log regret and logistic regression classifies regret <= {RANK_POS}, both on standardized "
          "features plus shape x candidate products). Hyperparameters by grouped "
          f"{N_FOLDS}-fold CV on tune shapes (each fit sees {min(summary['cv_train_shapes'])}-"
          f"{max(summary['cv_train_shapes'])} shapes; none sharing two of M, N, K with a fold shape), lowest capped mean "
          "regret. b6 / knn5 / knnf are the existing baselines (`selector_baselines.md`), CV-scored under the same folds "
          "with their weights fixed. Regret = t / oracle - 1.\n",
          "## Screen: regret vs the held-out per-shape best (full pruned space)\n",
          "| selector | tune CV mean | tune CV median / p95 | " + " | ".join(f"held-out {g} median / p95" for g in groups)
          + " | held-out mean | max | picks the oracle | distinct picks |",
          "|---|---|---|" + "---|" * len(groups) + "---|---|---|---|"]
    for n in order:
        md.append(f"| {n} | {pct(cv[n]['mean'])} | {ms(cv[n])} | " + " | ".join(ms(screen[n][g]) for g in groups)
                  + f" | {pct(screen[n]['all']['mean'])} | {pct(min(screen[n]['all']['max'], REGRET_CAP - 1))} | "
                  f"{pct(agree[n])} | {summary['distinct_picks'][n]} |")
    md.append("\n## Chosen hyperparameters and candidate ceilings\n")
    md.append("Ceiling = regret of the best member of the selector's own candidate set (cls label set, rank pool) on each "
              "held-out shape: the floor any pick from that set can reach.\n")
    md.append("| selector | hyperparameters | candidates | grid points | ceiling held-out median / p95 / mean | "
              "CV mean, best / worst grid point |")
    md.append("|---|---|---|---|---|---|")
    for n in names:
        f = fits[n]
        md.append(f"| {n} | `{json.dumps(f['hp'])}` | {f['n_candidates']} | {len(f['grid'])} | "
                  f"{ms(ceilings[n]['all'])} / {pct(ceilings[n]['all']['mean'])} | "
                  f"{pct(f['grid'][0]['mean'])} / {pct(f['grid'][-1]['mean'])} |")
    md.append("\n## Feature importance (final fits; share of total gain / impurity decrease)\n")
    for n in names:
        if fits[n]["importance"]:
            md.append(f"- **{n}**: " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in fits[n]["importance"][:8]))
    worst = {}
    for n in order:
        r = Rh[held_rows, held[n]]
        worst[n] = [{"shape": tag(hshapes[i]), "split": hshapes[i]["split"], "oracle_us": float(ctx["hT"][i].min()),
                     "pick": keys[held[n][i]], "regret": float(min(r[i], REGRET_CAP) - 1)} for i in np.argsort(-r)[:5]]
    summary["screen_worst"] = worst
    md.append("\n## Worst held-out shapes (screen)\n")
    md.append("| selector | shape | split | oracle us | pick | regret |")
    md.append("|---|---|---|---|---|---|")
    for n in order:
        for w in worst[n][:3]:
            md.append(f"| {n} | {w['shape']} | {w['split']} | {w['oracle_us']:.1f} | `{w['pick']}` | {pct(w['regret'])} |")

    for label, fname, desc in (
            ("cold", results, "cold L2, per-launch, adaptive repeats, 3 input placements; all held-out shapes"),
            ("graph", graph_results, f"hot L2, CUDA-graph amortized (50 launches); held-out shapes with screen oracle "
                                     f"< {SMALL_US:.0f} us")):
        rows = _rows(Path(data) / fname)
        if not rows:
            md.append(f"\n## Stage 2 ({label}): `{fname}` not found\n")
            continue
        st = stage2(rows, hshapes, held_keys)
        have = np.isfinite(st["oracle"]) & np.isfinite(st["b3plus"])
        small2 = st["oracle"] < SMALL_US
        g2 = {"all": have, f">= {SMALL_US:.0f} us": have & ~small2, f"< {SMALL_US:.0f} us": have & small2}
        if label == "graph":
            g2 = {"all": have, "B3+ < 5 us": have & (st["b3plus"] < 5),
                  "B3+ 5-15 us": have & (st["b3plus"] >= 5) & (st["b3plus"] < 15), "B3+ >= 15 us": have & (st["b3plus"] >= 15)}
        res = {"results": fname, "n_shapes": int(have.sum()), "n_rows": len(rows),
               "n_invalid_rows": sum(not r.get("valid") for r in rows),
               "power_capped_rows": sum(bool(int(r.get("throttle_reasons") or 0) & 0x4) for r in rows),
               "missing_picks": {n: int((have & ~np.isfinite(st[n])).sum()) for n in order},
               "regret": {n: {g: _stats(st[n][m] / st["oracle"][m]) for g, m in g2.items()} for n in order},
               "vs_b3plus": {n: {g: _vs_bar(st[n][m], st["b3plus"][m]) for g, m in g2.items()} for n in ("oracle",) + tuple(order)}}
        summary[f"stage2_{label}"] = res
        md.append(f"\n## Stage 2 ({label}): re-measured, `{fname}`\n")
        md.append(f"{desc}. Per shape: the screen's top-8 tiles, every pick above, B1, B3 and B3+ in one session. "
                  f"{res['n_shapes']} shapes, {res['n_rows']} rows, {res['n_invalid_rows']} invalid, "
                  f"{res['power_capped_rows']} power-capped. Oracle = fastest Triton row.\n")
        md.append("Regret vs the per-shape Triton oracle, median / p95 (mean of all):\n")
        md.append("| selector | " + " | ".join(f"{g} (n={int(m.sum())})" for g, m in g2.items()) + " | mean (all) |")
        md.append("|---|" + "---|" * (len(g2) + 1))
        for n in order:
            md.append(f"| {n} | " + " | ".join(ms(res["regret"][n][g]) for g in g2)
                      + f" | {pct(res['regret'][n]['all']['mean']) if res['regret'][n]['all']['n'] else '-'} |")
        md.append("\nSpeedup vs B3+ (t_B3+ / t_pick; > 1 = faster), median [p10, p90] and shapes ahead by > 5% / within "
                  "2% / behind by > 5%:\n")
        md.append("| selector | " + " | ".join(g2) + " |")
        md.append("|---|" + "---|" * len(g2))
        for n in ("oracle",) + tuple(order):
            cells = []
            for g in g2:
                v = res["vs_b3plus"][n][g]
                cells.append(f"{v['median']:.3f} [{v['p10']:.2f}, {v['p90']:.2f}] · {v['ahead_gt5']}/{v['within2']}/"
                             f"{v['behind_gt5']}" if v["n"] else "-")
            md.append(f"| {'Triton oracle' if n == 'oracle' else n} | " + " | ".join(cells) + " |")
        if any(res["missing_picks"].values()):
            md.append(f"\nMissing pick measurements: {res['missing_picks']}.")
    Path(report).mkdir(parents=True, exist_ok=True)
    (Path(report) / "learned_selectors_summary.json").write_text(json.dumps(summary, indent=1))
    out = Path(report) / "learned_selectors.md"
    out.write_text("\n".join(md) + "\n")
    return out, summary


# ---------------------------------------------------------------- top-k + short online search (lgbm_rank)

TOPK_REFS = ("knn5", "knnf")


def _nanmin_cols(st: dict, names: list[str]) -> np.ndarray:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmin(np.stack([st[n] for n in names], axis=1), axis=1)


VARIANTS = {"plain": "lgbm", "diverse": "lgbmd"}  # top-k order -> pick-name prefix


def build_topk(data: Path, report: Path, results: str = "results_lock1755_topk.jsonl",
               graph_results: str = "results_lock1755_topk_graph.jsonl", online: str = "online_lgbm_rank.jsonl",
               online_diverse: str = "online_lgbm_rank_diverse.jsonl", log=print) -> tuple[Path, dict]:
    """Train + save the ranker (with its CV-chosen diversification rule), write its plain and diverse top-TOPK picks
    per held-out shape (data/<hw>/learned_topk_picks.json, for `triton:picks`), and score best-of-k for each order:
    on the screen, with perfect (full-protocol stage-2) measurement, and with the short online search (`tune online
    --order ...`) whose choice is priced at its stage-2 time."""
    rk, fit = train_ranker(data, log=log)
    ctx, cols = fit["ctx"], fit["cols"]
    keys, hshapes, Rh = ctx["keys"], ctx["hshapes"], ctx["Rh"]
    col = {k: j for j, k in enumerate(keys)}
    tag = lambda s: f"{s['M']}x{s['N']}x{s['K']}"
    ranked = {v: [rk.rank(s["M"], s["N"], s["K"], TOPK, order=v) for s in hshapes] for v in VARIANTS}
    mismatch = sum(a != [keys[j] for j in c if j >= 0] for a, c in zip(ranked["plain"], cols))
    if mismatch:
        raise RuntimeError(f"saved Ranker disagrees with the fitted model on {mismatch} held-out shapes")
    sel = json.loads((Path(data) / "selector_picks.json").read_text())["by_selector"]
    names = {v: [f"{p}@{r}" for r in range(1, TOPK + 1)] for v, p in VARIANTS.items()}
    by = {n: {tag(s): (ranked[v][i][r] if r < len(ranked[v][i]) else None) for i, s in enumerate(hshapes)}
          for v in VARIANTS for r, n in enumerate(names[v])}
    by.update({n: sel[n] for n in TOPK_REFS})
    (Path(data) / "learned_topk_picks.json").write_text(json.dumps(
        {"selectors": [n for v in VARIANTS for n in names[v]] + list(TOPK_REFS), "ranker": str(Path(data) / "lgbm_rank"),
         "hp": rk.hp, "diverse_rule": rk.meta["diverse_rule"],
         "picks": {tag(s): sorted({by[n][tag(s)] for n in by if by[n][tag(s)]}) for s in hshapes},
         "by_selector": {n: {t: k for t, k in d.items() if k} for n, d in by.items()}}, indent=1))

    small = ctx["hT"].min(axis=1) < SMALL_US
    pct = lambda x: f"{100 * x:.1f}%"
    ms = lambda d: f"{pct(d['p50'])} / {pct(d['p95'])} ({pct(d['mean'])})" if d.get("n") else "-"
    from tune.sweep import parse_key
    tiles = {v: [len({(c.BLOCK_M, c.BLOCK_N, c.SPLIT_K) for c in map(parse_key, ks[:DIVERSE_K])}) for ks in ranked[v]]
             for v in VARIANTS}
    summary = {"hp": rk.hp, "n_heldout_shapes": len(hshapes), "pool": len(rk.pool_keys),
               "diverse_rule": rk.meta["diverse_rule"], "diverse_rule_grid_top": fit["rules"][:12],
               "diverse_rule_plain": next(e for e in fit["rules"] if e["rule"]["group"] is None),
               "distinct_tile_split_in_top3": {v: float(np.mean(t)) for v, t in tiles.items()}, "screen_best_of_k": {}}
    md = ["# Learned ranker: top-k + short online search on held-out shapes (auto-generated)\n",
          f"lgbm_rank (`{json.dumps(rk.hp)}`) refit on all {len(ctx['shapes'])} tune shapes and saved as "
          f"`data/<hw>/lgbm_rank.txt/.json` (pool of {len(rk.pool_keys)} configs). Per held-out shape it ranks its runnable "
          f"pool; best-of-k = the fastest of the first k. Two orders: **plain** (score order) and **diverse** "
          f"(`diverse_order`, rule `{json.dumps(rk.meta['diverse_rule'])}`: keep the top `plain` picks, then the best "
          "config of each new group within `delta` predicted log regret of the top pick, then score order). "
          "Regret = t / oracle - 1, median / p95 (mean).\n",
          f"## Diversification rule (tune shapes only: best-of-{DIVERSE_K} screen regret on out-of-fold scores, "
          f"{N_FOLDS} grouped folds)\n",
          f"| rule | best-of-1 | best-of-{DIVERSE_K} | best-of-{TOPK} |", "|---|---|---|---|"]
    for e in fit["rules"][:8] + [summary["diverse_rule_plain"]]:
        r = e["rule"]
        md.append(f"| {'plain' if r['group'] is None else json.dumps(r)} | {ms(e['k1'])} | {ms(e[f'k{DIVERSE_K}'])} | "
                  f"{ms(e[f'k{TOPK}'])} |")
    md.append(f"\nDistinct (BLOCK_M, BLOCK_N, SPLIT_K) among the held-out top {DIVERSE_K}: plain "
              f"{summary['distinct_tile_split_in_top3']['plain']:.2f}, diverse {summary['distinct_tile_split_in_top3']['diverse']:.2f}.\n")
    md += ["## Screen (regret vs the held-out screen oracle, full pruned space)\n",
           "| order | k | all | >= 50 us | < 50 us |", "|---|---|---|---|---|"]
    for v in VARIANTS:
        Rk = np.array([[Rh[i, col[x]] for x in ks] + [np.inf] * (TOPK - len(ks)) for i, ks in enumerate(ranked[v])])
        summary["screen_best_of_k"][v] = {k: _stats(Rk[:, :k].min(axis=1)) for k in range(1, TOPK + 1)}
        for k in range(1, TOPK + 1):
            r = Rk[:, :k].min(axis=1)
            md.append(f"| {v} | {k} | {ms(_stats(r))} | {ms(_stats(r[~small]))} | {ms(_stats(r[small]))} |")
    held_keys = {n: [by[n][tag(s)] for s in hshapes] for n in by}
    on_rows = {}
    for v, fname in (("plain", online), ("diverse", online_diverse)):
        on_rows[v] = {(r["M"], r["N"], r["K"]): r for r in _rows(Path(data) / fname)
                      if r.get("k") == TOPK and r.get("order", "plain") == v}
        for k in range(1, TOPK + 1):  # the online choice for each k, priced below at its stage-2 time
            held_keys[f"online-{v}@{k}"] = [on_rows[v].get((s["M"], s["N"], s["K"]), {}).get("choice", {}).get(str(k), {}).get("key")
                                            for s in hshapes]
        if on_rows[v]:
            md.append(f"\nOnline search ({v}): {len(on_rows[v])} shapes from `{fname}`.")
    for label, fname in (("cold", results), ("graph", graph_results)):
        rows = _rows(Path(data) / fname)
        if not rows:
            md.append(f"\n## Stage 2 ({label}): `{fname}` not found\n")
            continue
        st = stage2(rows, hshapes, held_keys)
        have = np.isfinite(st["oracle"])
        small2 = st["oracle"] < SMALL_US
        g2 = ({"all": have, f">= {SMALL_US:.0f} us": have & ~small2, f"< {SMALL_US:.0f} us": have & small2}
              if label == "cold" else {"all (< 50 us shapes)": have})
        res = {"results": fname, "n_shapes": int(have.sum()), "n_rows": len(rows),
               "n_invalid_rows": sum(not r.get("valid") for r in rows),
               "power_capped_rows": sum(bool(int(r.get("throttle_reasons") or 0) & 0x4) for r in rows),
               "session_ids": sorted({r["session_id"] for r in rows}),
               "harness": sorted({(r.get("harness_commit") or "")[:7] + ("-dirty" if r.get("harness_dirty") else "") for r in rows}),
               "perfect": {}, "online": {}, "refs": {}, "vs_b3plus": {}, "worst": {}}
        md.append(f"\n## Stage 2 ({label}): `{fname}`\n")
        md.append(f"{res['n_shapes']} shapes, {res['n_rows']} rows, {res['n_invalid_rows']} invalid, "
                  f"{res['power_capped_rows']} power-capped; sessions {', '.join(res['session_ids'])} "
                  f"(harness {', '.join(res['harness'])}). Oracle = fastest Triton row (screen top-8 + every pick).\n")
        md.append("| selector | " + " | ".join(f"{g} (n={int(m.sum())})" for g, m in g2.items()) + " | vs B3+ median [p10, p90] |")
        md.append("|---|" + "---|" * (len(g2) + 1))
        lines = []
        for n in TOPK_REFS:
            res["refs"][n] = {g: _stats(st[n][m] / st["oracle"][m]) for g, m in g2.items()}
            lines.append((n, st[n], res["refs"][n]))
        for v in VARIANTS:
            for k in range(1, TOPK + 1):
                t = _nanmin_cols(st, names[v][:k])
                res["perfect"][f"{v}@{k}"] = {g: _stats(t[m] / st["oracle"][m]) for g, m in g2.items()}
                lines.append((f"{v} best-of-{k}, full-protocol timing", t, res["perfect"][f"{v}@{k}"]))
        for v in VARIANTS:
            if not on_rows[v]:
                continue
            for k in range(1, TOPK + 1):
                t = st[f"online-{v}@{k}"]
                ok = np.isfinite(t)
                res["online"][f"{v}@{k}"] = {g: _stats(t[m & ok] / st["oracle"][m & ok]) for g, m in g2.items()}
                res["online"][f"{v}@{k}"]["missing"] = int((have & ~ok).sum())
                lines.append((f"{v} online top-{k} (short timing picks)", t, res["online"][f"{v}@{k}"]))
        for name, t, d in lines:
            v = _vs_bar(t[have], st["b3plus"][have])
            res["vs_b3plus"][name] = v
            md.append(f"| {name} | " + " | ".join(ms(d[g]) for g in g2)
                      + (f" | {v['median']:.3f} [{v['p10']:.2f}, {v['p90']:.2f}] |" if v["n"] else " | - |"))
        # The worst shapes of plain online top-3 (or plain best-of-3 without online rows), and what diverse does there.
        ref = st[f"online-plain@{DIVERSE_K}"] if on_rows["plain"] else _nanmin_cols(st, names["plain"][:DIVERSE_K])
        alt = st[f"online-diverse@{DIVERSE_K}"] if on_rows["diverse"] else _nanmin_cols(st, names["diverse"][:DIVERSE_K])
        rr, ra = ref / st["oracle"], alt / st["oracle"]
        worst = [i for i in np.argsort(-np.where(np.isfinite(rr), rr, -1))[:8]]
        res["worst"] = [{"shape": tag(hshapes[i]), "oracle_us": float(st["oracle"][i]), "plain": float(rr[i] - 1),
                         "diverse": float(ra[i] - 1), "plain_top3": ranked["plain"][i][:DIVERSE_K],
                         "diverse_top3": ranked["diverse"][i][:DIVERSE_K]} for i in worst]
        md.append(f"\nWorst shapes of plain top-{DIVERSE_K} ({'online' if on_rows['plain'] else 'full-protocol'}) and the "
                  f"diverse top-{DIVERSE_K} there:\n")
        md.append(f"| shape | oracle us | plain | diverse | diverse top-{DIVERSE_K} |")
        md.append("|---|---|---|---|---|")
        for w in res["worst"]:
            md.append(f"| {w['shape']} | {w['oracle_us']:.1f} | {pct(w['plain'])} | {pct(w['diverse'])} | "
                      + ", ".join(f"`{x}`" for x in w["diverse_top3"]) + " |")
        summary[f"stage2_{label}"] = res
    for v in VARIANTS:
        if not on_rows[v]:
            continue
        md.append(f"\n## Online search cost per shape ({v}; wall clock, empty Triton cache)\n")
        md.append("Compile = first call + correctness check per candidate (JIT); timing = short protocol "
                  f"({_short_desc()}).\n")
        md.append("| k | compile s median / p90 | timing s median / p90 | total s median / p90 / sum over shapes | picked rank (0-based) counts |")
        md.append("|---|---|---|---|---|")
        cost = {}
        for k in range(1, TOPK + 1):
            ch = [r["choice"][str(k)] for r in on_rows[v].values()]
            c, tm = np.array([x["compile_s"] for x in ch]), np.array([x["time_s"] for x in ch])
            q = lambda x: f"{np.median(x):.2f} / {np.percentile(x, 90):.2f}"
            rk_counts = np.bincount([x["rank"] for x in ch if x.get("rank") is not None], minlength=k).tolist()
            cost[k] = {"compile_s_median": float(np.median(c)), "time_s_median": float(np.median(tm)),
                       "total_s_sum": float((c + tm).sum()), "picked_rank_counts": rk_counts}
            md.append(f"| {k} | {q(c)} | {q(tm)} | {q(c + tm)} / {(c + tm).sum():.0f} | {rk_counts} |")
        summary[f"online_cost_{v}"] = cost
        caches = sorted({str(r.get("triton_cache_dir")) for r in on_rows[v].values()})
        md.append(f"\nTriton cache dir(s): {', '.join(f'`{c}`' for c in caches)}. Ranking itself: median "
                  f"{1e3 * np.median([r['rank_s'] for r in on_rows[v].values()]):.0f} ms per shape.")
    Path(report).mkdir(parents=True, exist_ok=True)
    (Path(report) / "learned_topk_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    out = Path(report) / "learned_topk.md"
    out.write_text("\n".join(md) + "\n")
    return out, summary


def _short_desc() -> str:
    from tune import online as o
    return (f"{o.SHORT_REPS} cold samples after {o.SHORT_WARMUP_MS:g} ms warmup, one placement; graph-amortized "
            f"{o.GRAPH_BATCH} x {o.GRAPH_REPS} hot when the fastest is < {o.SMALL_US:g} us")
