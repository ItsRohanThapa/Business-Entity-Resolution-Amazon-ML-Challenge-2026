"""Gradient-boosted pair classifier with grouped out-of-fold predictions.

Backends (both licence-compatible with the challenge rules):
  xgb  - XGBoost (Apache-2.0), runs on the GPU (device="cuda") when one is available
  lgbm - LightGBM (MIT), CPU
"""
import gc
import os
import shutil
import warnings

import numpy as np
import pandas as pd

from .progress import bar

# Monotone constraints: "more similar can never mean less likely to match". They stop the
# trees from memorising country-specific quirks - the main guard for the unseen test country.
_POS_EXACT = {
    "n_ratio", "n_partial", "n_tsort", "n_tset", "n_jw", "n_lev", "nf_tset", "sk_ratio",
    "sk_tset", "a_ratio", "a_partial", "a_tsort", "a_tset", "a_jw", "as_tset", "n_dice",
    "n_soft", "a_dice", "a_soft", "alias_best", "acronym", "postal_eq", "postal_pre3",
    "num_jacc", "first_num_eq", "n_first_eq", "n_nospace", "a_nospace", "bs", "qn", "qa",
    "p1", "ce", "bsn", "bsa", "es", "er",
}
_NEG_EXACT = {
    "n_unm_idf1", "n_unm_idf2", "a_unm_idf1", "a_unm_idf2", "n_unm_cnt1", "n_unm_cnt2",
    "a_unm_cnt1", "a_unm_cnt2", "num_conflict", "name_num_conflict",
}
_COS_SPACES = {"name_char", "name_word", "name_skel", "addr_char", "addr_skel", "full_char"}


def monotone_vector(columns):
    out = []
    for c in columns:
        if c in _POS_EXACT or c.endswith(("_gapmax", "_gap2", "_gapmax_rev", "_gap2_rev")):
            out.append(1)
        elif c in _NEG_EXACT or c.endswith(("_rk", "_rk_src", "_rk_rev")):
            out.append(-1)
        elif c.startswith("cos_") and c[4:] in _COS_SPACES:
            out.append(1)
        else:
            out.append(0)
    return out


def cuda_available():
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return shutil.which("nvidia-smi") is not None


def xgb_gpu_works():
    """Tiny real training run on the GPU (catches driver / CUDA-build mismatches)."""
    try:
        import xgboost as xgb
        X = np.random.RandomState(0).rand(512, 4).astype(np.float32)
        y = (X[:, 0] > 0.5).astype(np.float32)
        bst = xgb.train({"device": "cuda", "tree_method": "hist", "verbosity": 0},
                        xgb.DMatrix(X, label=y), num_boost_round=2)
        return '"cuda' in bst.save_config().replace(" ", "")
    except Exception:  # noqa: BLE001
        return False


def resolve_backend(cfg, log=print):
    """Fills cfg.backend / cfg.device when they are 'auto'; falls back to CPU if the GPU
    cannot actually be used."""
    if cfg.device == "auto":
        cfg.device = "cuda" if cuda_available() else "cpu"
    if cfg.backend == "auto":
        try:
            import xgboost  # noqa: F401
            cfg.backend = "xgb" if cfg.device == "cuda" else "lgbm"
        except ImportError:
            cfg.backend = "lgbm"
    if cfg.backend == "xgb" and cfg.device == "cuda" and not xgb_gpu_works():
        log("WARNING: XGBoost could not train on the GPU (driver / CUDA build mismatch?) "
            "-> using LightGBM on CPU")
        cfg.backend = "lgbm"
    log(f"device: {cfg.device} | booster backend: {cfg.backend}"
        f"{' (GPU)' if cfg.backend == 'xgb' and cfg.device == 'cuda' else ' (CPU)'}")


def make_folds(groups, n_folds, seed=42):
    """Folds grouped by Source 1 entity, so all candidates of an entity share a fold."""
    uniq = pd.unique(groups)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(uniq))
    fold_of = pd.Series(np.arange(len(uniq)) % n_folds, index=uniq[perm])
    fid = fold_of.reindex(groups).values
    return [(np.where(fid != f)[0], np.where(fid == f)[0]) for f in range(n_folds)], fid


class FeatureMatrix:
    """float32 feature array + column names: lets the caller drop the pandas copy of a large
    feature table before training (XGBoost needs only the array)."""

    def __init__(self, X):
        self.columns = list(X.columns)
        self.A = X.to_numpy(np.float32)

    def __len__(self):
        return len(self.A)


class Predictor:
    """Averages the full-data boosters of all seeds; works for both backends."""

    def __init__(self, backend, boosters, columns, device="cpu"):
        self.backend, self.boosters, self.columns, self.device = backend, boosters, columns, device

    def predict(self, X, chunk=5_000_000):
        X = X[self.columns]
        p = np.zeros(len(X))
        for s in bar(range(0, len(X), chunk), f"predict ({len(X):,} rows)", unit="chunk",
                     leave=False):
            part = X.iloc[s:s + chunk]
            for b in self.boosters:
                if self.backend == "xgb":
                    import xgboost as xgb
                    p[s:s + chunk] += b.predict(xgb.DMatrix(part.to_numpy(np.float32),
                                                            missing=np.nan,
                                                            feature_names=self.columns))
                else:
                    p[s:s + chunk] += b.predict(part)
        return (p / max(1, len(self.boosters))).astype(np.float32)

    def save(self, path):
        import json
        os.makedirs(path, exist_ok=True)
        ext = "ubj" if self.backend == "xgb" else "txt"
        for i, b in enumerate(self.boosters):
            b.save_model(os.path.join(path, f"booster_{i}.{ext}"))
        with open(os.path.join(path, "predictor.json"), "w") as f:
            json.dump({"backend": self.backend, "columns": self.columns,
                       "n": len(self.boosters)}, f)

    @classmethod
    def load(cls, path, device="cpu"):
        import json
        with open(os.path.join(path, "predictor.json")) as f:
            meta = json.load(f)
        boosters = []
        for i in range(meta["n"]):
            if meta["backend"] == "xgb":
                import xgboost as xgb
                b = xgb.Booster()
                b.load_model(os.path.join(path, f"booster_{i}.ubj"))
            else:
                import lightgbm as lgb
                b = lgb.Booster(model_file=os.path.join(path, f"booster_{i}.txt"))
            boosters.append(b)
        return cls(meta["backend"], boosters, meta["columns"], device)


# ----------------------------------------------------------------------------- LightGBM
def _lgb_params(columns, cfg, seed):
    params = dict(cfg.lgb_params, seed=seed)
    if cfg.monotone:
        params["monotone_constraints"] = monotone_vector(list(columns))
        params["monotone_constraints_method"] = "intermediate"
    return params


def _train_lgbm(X, y, folds, cfg, log, tag, refit):
    import lightgbm as lgb
    oof = np.zeros(len(X))
    boosters, imp = [], pd.Series(0.0, index=X.columns)
    for seed in cfg.seeds:
        params = _lgb_params(X.columns, cfg, seed)
        its = []
        for f, (tr, va) in enumerate(folds, 1):
            dtr = lgb.Dataset(X.iloc[tr], y[tr], free_raw_data=True)
            dva = lgb.Dataset(X.iloc[va], y[va], reference=dtr)
            b = bar(desc=f"{tag} seed {seed} fold {f}/{len(folds)}", total=cfg.num_boost_round,
                    unit="round", leave=False)
            m = lgb.train(params, dtr, num_boost_round=cfg.num_boost_round, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False),
                                     lambda env: b.update(1)])
            b.close()
            oof[va] += m.predict(X.iloc[va], num_iteration=m.best_iteration) / len(cfg.seeds)
            its.append(m.best_iteration)
        n_iter = max(50, int(np.mean(its) * 1.1))
        if refit:
            b = bar(desc=f"{tag} seed {seed} refit", total=n_iter, unit="round", leave=False)
            full = lgb.train(params, lgb.Dataset(X, y), num_boost_round=n_iter,
                             callbacks=[lambda env: b.update(1)])
            b.close()
            boosters.append(full)
            imp += pd.Series(full.feature_importance("gain"), index=X.columns)
        log(f"    {tag} seed {seed}: best iterations {its} -> refit {n_iter}")
    return oof, boosters, imp


# ----------------------------------------------------------------------------- XGBoost
def _xgb_bar(total, desc):
    """XGBoost callback: one progress-bar step per boosting round, validation loss shown."""
    import xgboost as xgb

    class _Bar(xgb.callback.TrainingCallback):
        def __init__(self):
            super().__init__()
            self.b = bar(desc=desc, total=total, unit="round", leave=False)

        def after_iteration(self, model, epoch, evals_log):
            self.b.update(1)
            for data in evals_log.values():
                for metric, vals in data.items():
                    self.b.set_postfix_str(f"{metric} {vals[-1]:.5f}", refresh=False)
            return False

        def after_training(self, model):
            self.b.close()
            return model

    return _Bar()


def _xgb_params(columns, cfg, seed):
    params = dict(cfg.xgb_params, seed=seed, device=cfg.device, tree_method="hist")
    if cfg.monotone:
        params["monotone_constraints"] = "(" + ",".join(map(str, monotone_vector(columns))) + ")"
    return params


def _xgb_fit(params, dtrain, log, **kw):
    """xgb.train on the GPU; if the GPU runs out of memory, the same call on the CPU (slower,
    same model) instead of losing the whole run."""
    import xgboost as xgb
    try:
        return xgb.train(params, dtrain, **kw)
    except xgb.core.XGBoostError as e:
        if params.get("device", "cpu") == "cpu" or "out of memory" not in str(e):
            raise
        if dtrain.num_row() * dtrain.num_col() > 3e9:      # CPU would take days at this size
            raise RuntimeError(
                f"XGBoost out of GPU memory on {dtrain.num_row():,} rows x {dtrain.num_col()} "
                f"features - lower --train-sample or --max-train-rows") from e
        log(f"WARNING: XGBoost out of GPU memory -> retrying this model on the CPU "
            f"({str(e).splitlines()[0][:160]})")
        params.update(device="cpu", nthread=os.cpu_count())   # rest of this seed stays on the CPU
        gc.collect()
        for cb in kw.get("callbacks") or []:
            if hasattr(cb, "b"):
                cb.b.reset()                       # progress bar restarts at round 0
        return xgb.train(params, dtrain, **kw)


def train_row_sample(y, hard, max_rows, seed=1, log=print, tag="stage"):
    """(keep mask, weights) for training at most max_rows rows (GPU memory budget): every
    positive and - if they fit - every hard negative (look-alikes: the decoys) are kept, easy
    negatives are sampled; sampled rows get weight 1/rate so probabilities stay calibrated.
    max_rows <= 0 or >= len(y): all rows, weight 1."""
    n = len(y)
    w = np.ones(n, np.float32)
    if max_rows <= 0 or n <= max_rows:
        return np.ones(n, bool), w
    rng = np.random.RandomState(seed)
    pos = y > 0
    hn = ~pos & hard
    en = ~pos & ~hard
    room = max_rows - int(pos.sum())
    r_hard = min(1.0, max(0.05, room * 0.8 / max(1, hn.sum())))
    room -= int(r_hard * hn.sum())
    r_easy = min(1.0, max(0.01, room / max(1, en.sum())))
    u = rng.rand(n)
    keep = pos | (hn & (u < r_hard)) | (en & (u < r_easy))
    w[hn] = 1.0 / r_hard
    w[en] = 1.0 / r_easy
    log(f"    {tag}: GPU row budget {max_rows:,}: training on {int(keep.sum()):,} of {n:,} rows "
        f"(all {int(pos.sum()):,} positives, hard negatives x{r_hard:.2f}, easy negatives "
        f"x{r_easy:.3f}, weighted); out-of-fold predictions still cover all rows")
    return keep, w


def _predict_rows(m, A, rows, best, chunk=2_000_000):
    import xgboost as xgb
    out = np.empty(len(rows), np.float32)
    for s in range(0, len(rows), chunk):
        part = A[rows[s:s + chunk]]
        out[s:s + chunk] = m.predict(xgb.DMatrix(part, missing=np.nan,
                                                 feature_names=m.feature_names),
                                     iteration_range=(0, best))
    return out


def _load_booster(path):
    import xgboost as xgb
    b = xgb.Booster()
    b.load_model(path)
    return b


def _train_xgb(X, y, folds, cfg, log, tag, refit, ckpt=None, keep=None, weight=None):
    """ckpt: folder where every finished fold / refit is saved at once, so a crash or kill
    loses at most the model being trained (None = no fold checkpoints)."""
    import xgboost as xgb
    cols = list(X.columns)
    A = X.A if isinstance(X, FeatureMatrix) else X.to_numpy(np.float32)
    oof = np.zeros(len(X))
    boosters, imp = [], pd.Series(0.0, index=cols)
    if ckpt:
        os.makedirs(ckpt, exist_ok=True)
    for seed in cfg.seeds:
        params = _xgb_params(cols, cfg, seed)
        its = []
        for f, (tr, va) in enumerate(folds, 1):
            fp = os.path.join(ckpt, f"seed{seed}_fold{f}") if ckpt else None
            if fp and os.path.exists(fp + ".done"):
                pv = np.load(fp + "_oof.npy")
                best = int(open(fp + ".done").read())
                oof[va] += pv / len(cfg.seeds)
                its.append(best)
                log(f"    {tag} seed {seed} fold {f}/{len(folds)}: from checkpoint (best round "
                    f"{best})")
                continue
            trk, vak = (tr, va) if keep is None else (tr[keep[tr]], va[keep[va]])
            wt = None if weight is None else weight[trk]
            wv = None if weight is None else weight[vak]
            dtr = xgb.QuantileDMatrix(A[trk], y[trk], weight=wt, missing=np.nan,
                                      feature_names=cols, max_bin=cfg.xgb_max_bin)
            dva = xgb.QuantileDMatrix(A[vak], y[vak], weight=wv, missing=np.nan,
                                      feature_names=cols, ref=dtr)
            m = _xgb_fit(params, dtr, log, num_boost_round=cfg.num_boost_round,
                          evals=[(dva, "valid")], early_stopping_rounds=cfg.early_stopping_rounds,
                          verbose_eval=False,
                          callbacks=[_xgb_bar(cfg.num_boost_round,
                                              f"{tag} seed {seed} fold {f}/{len(folds)}")])
            best = m.best_iteration + 1
            log(f"    {tag} seed {seed} fold {f}/{len(folds)}: best round {best} of "
                f"{cfg.num_boost_round}, valid logloss {m.best_score:.5f}")
            if keep is None:
                pv = m.predict(dva, iteration_range=(0, best))
            else:                                 # out-of-fold for ALL rows of the fold
                del dtr, dva
                dtr = dva = None
                pv = _predict_rows(m, A, va, best)
            oof[va] += pv / len(cfg.seeds)
            its.append(best)
            if fp:
                np.save(fp + "_oof.npy", pv)
                with open(fp + ".done", "w") as fh:
                    fh.write(str(best))
            del dtr, dva, m
            gc.collect()
        n_iter = max(50, int(np.mean(its) * 1.1))
        if refit:
            rp = os.path.join(ckpt, f"seed{seed}_refit{n_iter}.ubj") if ckpt else None
            if rp and os.path.exists(rp):
                full = _load_booster(rp)
                log(f"    {tag} seed {seed} refit: from checkpoint")
            else:
                ka = np.arange(len(A)) if keep is None else np.where(keep)[0]
                dall = xgb.QuantileDMatrix(A[ka], y[ka], missing=np.nan, feature_names=cols,
                                           weight=None if weight is None else weight[ka],
                                           max_bin=cfg.xgb_max_bin)
                full = _xgb_fit(params, dall, log, num_boost_round=n_iter,
                                callbacks=[_xgb_bar(n_iter, f"{tag} seed {seed} refit")])
                del dall
                if rp:
                    full.save_model(rp + ".tmp")
                    os.replace(rp + ".tmp", rp)
            boosters.append(full)
            gain = full.get_score(importance_type="total_gain")
            imp += pd.Series(gain).reindex(cols).fillna(0.0)
        log(f"    {tag} seed {seed}: best iterations {its} -> refit {n_iter}")
    return oof, boosters, imp


def train_stage(X, y, folds, cfg, log=print, tag="stage", refit=True, ckpt=None, keep=None,
                weight=None):
    """Returns (out-of-fold probabilities, Predictor for new data, gain importance)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        if cfg.backend == "xgb":
            oof, boosters, imp = _train_xgb(X, y, folds, cfg, log, tag, refit, ckpt, keep,
                                            weight)
        else:
            oof, boosters, imp = _train_lgbm(X, y, folds, cfg, log, tag, refit)
    return oof, Predictor(cfg.backend, boosters, list(X.columns), cfg.device), \
        imp.sort_values(ascending=False)
