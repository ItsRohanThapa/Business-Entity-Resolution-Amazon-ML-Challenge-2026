"""From pair probabilities to final match lists (array based, scales to 100M pairs).

1. One-to-one assignment: Source 1 is deduplicated, so a Source 2/3 record can belong to
   at most one Source 1 entity -> keep only its most probable S1 entity.
2. Selection per Source 1 entity, two alternatives tuned on out-of-fold data:
   * threshold : keep candidates with p >= t (separate t for Source 2 / Source 3)
   * expected-F: choose the top-k set that maximises the EXPECTED F0.5 of that entity
                 (k = 0 means "no match"), after a tuned logit bias on p.
   The better one on OOF macro-F0.5 is used for the test set.
"""
import numpy as np

from .metrics import fbeta_from_counts
from .progress import bar


def one_to_one_mask(cand, p):
    """True where p is the maximum probability among all S1 entities claiming `cand`."""
    order = np.lexsort((-p, cand))
    cs = cand[order]
    first = np.r_[True, cs[1:] != cs[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(cs)), 0))
    best = np.empty(len(p), dtype=p.dtype)
    best[order] = p[order][start]
    return p >= best


def _pb_dist(ps):
    dist = np.array([1.0])
    for q in ps:
        dist = np.convolve(dist, [1.0 - q, q])
    return dist


def expected_f_best_k(probs, beta=0.5):
    """probs sorted descending. Returns k (0 = predict empty) maximising E[F_beta]."""
    n = len(probs)
    b2 = beta * beta
    best_k, best_v = 0, float(np.prod(1.0 - probs))
    for k in range(1, n + 1):
        ds, dr = _pb_dist(probs[:k]), _pb_dist(probs[k:])
        a = np.arange(k + 1)[:, None]
        b = np.arange(n - k + 1)[None, :]
        f = np.where(a > 0, (1 + b2) * a / (b2 * (a + b) + k), 0.0)
        v = float((ds[:, None] * dr[None, :] * f).sum())
        if v > best_v:
            best_k, best_v = k, v
    return best_k


def _logit_shift(p, bias):
    p = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)
    return 1.0 / (1.0 + np.exp(-(np.log(p / (1 - p)) + bias)))


def select_threshold(src2, p, t2, t3):
    return p >= np.where(src2 == 2, t2, t3)


def select_expected_f(s1, p, bias, min_p, beta):
    q = _logit_shift(p, bias)
    sel = np.zeros(len(p), dtype=bool)
    idx = np.where(q >= min_p)[0]
    if len(idx) == 0:
        return sel
    order = idx[np.lexsort((-q[idx], s1[idx]))]
    ss = s1[order]
    bounds = np.flatnonzero(ss[1:] != ss[:-1]) + 1
    for grp in bar(np.split(order, bounds), "expected-F per S1", total=len(bounds) + 1,
                   unit="S1", leave=False):
        k = expected_f_best_k(q[grp], beta)
        if k:
            sel[grp[:k]] = True
    return sel


class Evaluator:
    """Fast macro-F evaluation of a boolean row selection."""

    def __init__(self, s1_code, label, ntrue, beta):
        self.code, self.label, self.ntrue, self.beta = s1_code, label.astype(bool), ntrue, beta

    def per_entity(self, sel):
        n = len(self.ntrue)
        npred = np.bincount(self.code[sel], minlength=n)
        tp = np.bincount(self.code[sel & self.label], minlength=n)
        return fbeta_from_counts(tp, npred, self.ntrue, self.beta)

    def score(self, sel):
        return float(self.per_entity(sel).mean())


def tune_decision(s1_code, cand, src2, p, label, ntrue, cfg, log=print, max_entities=300_000):
    """One-to-one uses ALL rows (full competition); the grid search is scored on a random
    subset of at most max_entities S1 entities to keep it fast on millions of entities."""
    keep = one_to_one_mask(cand, p) if cfg.one_to_one else np.ones(len(p), bool)
    n_ent = len(ntrue)
    if n_ent > max_entities:
        chosen = np.zeros(n_ent, dtype=bool)
        chosen[np.random.RandomState(1).choice(n_ent, max_entities, replace=False)] = True
        new_code = np.cumsum(chosen) - 1
        keep &= chosen[s1_code]
        ev = Evaluator(new_code[s1_code[keep]], label[keep], ntrue[chosen], cfg.beta)
        s1k = new_code[s1_code[keep]]
    else:
        ev = Evaluator(s1_code[keep], label[keep], ntrue, cfg.beta)
        s1k = s1_code[keep]
    srck, pk = src2[keep], p[keep]
    # coarse-to-fine search (~4x fewer evaluations than the full grid)
    grid = np.array(cfg.threshold_grid)
    best_t = (-1.0, None)
    for t2 in bar(grid[::2], "threshold grid (coarse)", unit="t2", leave=False):
        for t3 in grid[::2]:
            sc = ev.score(select_threshold(srck, pk, t2, t3))
            if sc > best_t[0]:
                best_t = (sc, (float(t2), float(t3)))
    c2, c3 = best_t[1]
    fine = lambda c: [round(c + d, 4) for d in np.arange(-0.05, 0.0501, 0.0125) if 0.02 <= c + d <= 0.99]  # noqa: E731
    for t2 in bar(fine(c2), "threshold grid (fine)", unit="t2", leave=False):
        for t3 in fine(c3):
            sc = ev.score(select_threshold(srck, pk, t2, t3))
            if sc > best_t[0]:
                best_t = (sc, (t2, t3))
    log(f"    threshold rule : OOF macro-F0.5 {best_t[0]:.5f} at t2={best_t[1][0]}, "
        f"t3={best_t[1][1]}")
    best_e = (-1.0, None)
    for b in cfg.expf_bias_grid:
        sc = ev.score(select_expected_f(s1k, pk, b, cfg.expf_min_p, cfg.beta))
        if sc > best_e[0]:
            best_e = (sc, b)
    log(f"    expected-F rule: OOF macro-F0.5 {best_e[0]:.5f} at bias={best_e[1]}")
    rep = {"threshold": {"score": best_t[0], "t2": best_t[1][0], "t3": best_t[1][1]},
           "expected_f": {"score": best_e[0], "bias": best_e[1]}}
    if best_e[0] > best_t[0]:
        return {"method": "expected_f", "bias": best_e[1], "score": best_e[0]}, rep
    return {"method": "threshold", "t2": best_t[1][0], "t3": best_t[1][1],
            "score": best_t[0]}, rep


def apply_decision(s1_code, cand, src2, p, params, cfg):
    """Boolean mask over all rows: the selected (S1, candidate) matches."""
    keep = one_to_one_mask(cand, p) if cfg.one_to_one else np.ones(len(p), bool)
    idx = np.where(keep)[0]
    if params["method"] == "threshold":
        sel = select_threshold(src2[idx], p[idx], params["t2"], params["t3"])
    else:
        sel = select_expected_f(s1_code[idx], p[idx], params["bias"], cfg.expf_min_p, cfg.beta)
    out = np.zeros(len(p), dtype=bool)
    out[idx[sel]] = True
    return out
