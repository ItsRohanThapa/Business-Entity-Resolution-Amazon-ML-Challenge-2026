"""Macro F-beta exactly as the challenge defines it (per Source 1 entity, then averaged)."""
import numpy as np


def fbeta_row(pred, true, beta=0.5):
    pred, true = set(pred), set(true)
    if not true and not pred:
        return 1.0          # correctly predicted singleton
    if not true or not pred:
        return 0.0          # false merge on a singleton, or everything missed
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def macro_fbeta(pred_map, true_map, s1_ids, beta=0.5):
    if len(s1_ids) == 0:
        return float("nan")
    return float(np.mean([fbeta_row(pred_map.get(s, ()), true_map.get(s, ()), beta)
                          for s in s1_ids]))


def fbeta_from_counts(tp, npred, ntrue, beta=0.5):
    """Vectorised per-row F-beta from counts (numpy arrays)."""
    tp = tp.astype(float)
    npred = npred.astype(float)
    ntrue = ntrue.astype(float)
    b2 = beta * beta
    denom = b2 * ntrue + npred
    f = np.where(denom > 0, (1 + b2) * tp / np.maximum(denom, 1e-12), 1.0)
    return np.where((npred == 0) & (ntrue == 0), 1.0, f)
