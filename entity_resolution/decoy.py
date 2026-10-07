"""Label-free decoy signals.

Look-alike businesses in the data often differ from a true match only by a house number
shifted by a small amount (1-50, often several digits) and/or one added generic word
(e.g. "Group", "Holdings"), while true duplicates keep the number and differ only by noise
(DBA markers, initials, typos).

Everything here is computed from the split's own text, never from labels or fixed word lists,
so the same code runs on train and test, and the decoy words of an unseen country are learned
from that country's own test records.

  record level : first non-postal house number (value + digit count), twin counts
                 (how many records share core name + house number, or core name only).
  pair level   : number difference / small-shift flag / changed digits, and inside each
                 S1 group how many look-alike candidates share the S1's number vs this
                 candidate's number ("support": decoys are outvoted by the real copies).
  decoy words  : from near-twin candidate pairs (similar name + similar address, both with
                 a house number): per word added on the candidate side, the share of those
                 pairs whose number is shifted by 1-50 -> decoy-word score (per country,
                 smoothed); used by pair_features as dw2_max / dw1_max.
"""
import numpy as np
import pandas as pd

from .progress import bar

DECOY_VERSION = 1                       # bump when these features change (checkpoint key)
SMALL_SHIFT = (1, 50)
TWIN_QN, TWIN_QA = 0.85, 0.80           # near-twin pair: quick name / address similarity
SUPPORT_QN = 0.80                       # look-alike candidate for the number-support counts
MIN_WORD_PAIRS = 30                     # a word needs this many near-twin pairs for a score
SMOOTH = 20.0                           # pseudo-count towards the country's base rate
MAX_TWIN_PAIRS = 8_000_000              # sample size for learning the word table


def _first_number(addr, postal):
    """First non-postal number of every record: (value int64, -1 = none), digit count."""
    n = len(addr)
    val = np.full(n, -1, np.int64)
    ln = np.zeros(n, np.int8)
    for i in bar(range(n), "house numbers", unit="rec", leave=False):
        p = postal[i]
        for t in addr[i].split():
            if t.isdigit() and t != p:
                t = t.lstrip("0") or "0"
                val[i] = int(t[:15])
                ln[i] = min(len(t), 15)
                break
    return val, ln


def name_words(split_ctx, rows):
    core, extra = split_ctx["core"], split_ctx["extra"]
    return [set(f"{core[r]} {extra[r]}".split()) for r in rows]


def record_context(split_ctx, log=print):
    """Per-record arrays (added to split_ctx): house number, digit count, twin counts."""
    num, nlen = _first_number(split_ctx["addr"], split_ctx["postal"])
    ck = pd.factorize(split_ctx["ck"])[0].astype(np.int64)
    core = pd.factorize(split_ctx["core"])[0].astype(np.int64)
    src = split_ctx["src"]
    num_code = pd.factorize(num)[0].astype(np.int64)
    key_name = ck * (core.max() + 1) + core
    key_nn = pd.factorize(key_name * (num_code.max() + 1) + num_code)[0]
    _, inv, cnt = np.unique(key_nn, return_inverse=True, return_counts=True)
    tw_name_num = cnt[inv.ravel()].astype(np.float32)                  # same name + number
    tw_name_num[num < 0] = np.nan
    s1 = src == 1
    kn = pd.factorize(key_name)[0]
    s1_cnt = np.bincount(kn[s1], minlength=kn.max() + 1)
    ot_cnt = np.bincount(kn[~s1], minlength=kn.max() + 1)
    split_ctx.update({"hnum": num, "hnum_len": nlen,
                      "tw_name_num": tw_name_num,
                      "tw_s1_name": s1_cnt[kn].astype(np.float32),      # S1 with same name
                      "tw_ot_name": ot_cnt[kn].astype(np.float32)})     # S2/S3 same name
    log(f"    decoy: house numbers for {int((num >= 0).sum()):,} of {len(num):,} records")


def _digit_changes(a, b, la, lb):
    """Changed digits between two numbers of the same length (NaN otherwise)."""
    out = np.full(len(a), np.nan, np.float32)
    same = (la == lb) & (a >= 0) & (b >= 0)
    x, y = a[same], b[same]
    d = np.zeros(len(x), np.float32)
    for _ in range(15):
        d += (x % 10) != (y % 10)
        x, y = x // 10, y // 10
    out[same] = d
    return out


def _support(i1, keycol, keyval, ok):
    """For each pair: number of look-alike candidates (ok) of the same S1 whose house number
    equals keyval (the S1's own number or this candidate's)."""
    codes, uniq = pd.factorize(np.concatenate([keycol, keyval]))
    kc, kv = codes[:len(keycol)].astype(np.int64), codes[len(keycol):].astype(np.int64)
    m = len(uniq) + 1
    k_ok = i1[ok].astype(np.int64) * m + kc[ok]
    u, cnt = np.unique(k_ok, return_counts=True)
    q = i1.astype(np.int64) * m + kv
    pos = np.searchsorted(u, q)
    pos = np.minimum(pos, len(u) - 1)
    return np.where(u[pos] == q, cnt[pos], 0).astype(np.float32)


def pair_context(split_ctx, pairs, gctx, log=print):
    """Pair-level decoy features for ALL pairs (dict of float16 arrays, merged into gctx)."""
    i1, i2 = pairs["i1"], pairs["i2"]
    num, nlen = split_ctx["hnum"], split_ctx["hnum_len"]
    a, b = num[i1], num[i2]
    both = (a >= 0) & (b >= 0)
    diff = np.where(both, np.abs(a - b), -1)
    out = {
        "hn_logdiff": np.where(both, np.log1p(np.maximum(diff, 0).astype(np.float64)), np.nan),
        "hn_small": np.where(both, (diff >= SMALL_SHIFT[0]) & (diff <= SMALL_SHIFT[1]), np.nan),
        "hn_digits": _digit_changes(a, b, nlen[i1], nlen[i2]),
        "hn_lendiff": np.where(both, np.abs(nlen[i1].astype(np.int16) - nlen[i2]), np.nan),
    }
    qn = gctx["qn"].astype(np.float32)
    look = (qn >= SUPPORT_QN) & (b >= 0)
    s1_sup = _support(i1, b, a, look)             # look-alikes carrying the S1's number
    cand_sup = _support(i1, b, b, look)           # look-alikes carrying this candidate's number
    out["hn_sup_s1"] = np.where(a >= 0, s1_sup, np.nan)
    out["hn_sup_cand"] = np.where(b >= 0, cand_sup, np.nan)
    out["hn_sup_gap"] = np.where(both, s1_sup - cand_sup, np.nan)
    for k in ("tw_name_num", "tw_s1_name", "tw_ot_name"):
        out[f"{k}1"] = split_ctx[k][i1]
        out[f"{k}2"] = split_ctx[k][i2]
    out = {k: np.asarray(v, np.float32).astype(np.float16) for k, v in out.items()}
    log(f"    decoy: {len(out)} pair features (number shift / support / twins)")
    return out


def learn_decoy_words(split_ctx, pairs, gctx, log=print, seed=0):
    """{country_key: {word: score}} learned from near-twin pairs of this split (no labels).
    score = smoothed share of near-twin pairs with a 1-50 house-number shift among the pairs
    where the word appears only on the candidate side."""
    i1, i2 = pairs["i1"], pairs["i2"]
    num = split_ctx["hnum"]
    qn, qa = gctx["qn"].astype(np.float32), gctx["qa"].astype(np.float32)
    a, b = num[i1], num[i2]
    twin = np.where((qn >= TWIN_QN) & (qa >= TWIN_QA) & (a >= 0) & (b >= 0))[0]
    if len(twin) > MAX_TWIN_PAIRS:
        twin = np.sort(np.random.RandomState(seed).choice(twin, MAX_TWIN_PAIRS, replace=False))
    diff = np.abs(a[twin] - b[twin])
    small = (diff >= SMALL_SHIFT[0]) & (diff <= SMALL_SHIFT[1])
    ck = split_ctx["ck"][i1[twin]]
    core, extra = split_ctx["core"], split_ctx["extra"]
    rows = []
    for k in bar(range(len(twin)), "decoy words (near-twin pairs)", unit="pair", leave=False):
        r1, r2 = i1[twin[k]], i2[twin[k]]
        added = set(f"{core[r2]} {extra[r2]}".split()) - set(f"{core[r1]} {extra[r1]}".split())
        if 0 < len(added) <= 2:
            for w in added:
                if not w.isdigit() and len(w) > 1:
                    rows.append((ck[k], w, small[k]))
    table = {}
    if not rows:
        log("    decoy words: no near-twin pairs with an added word")
        return table
    df = pd.DataFrame(rows, columns=["ck", "word", "small"])
    for c, g in df.groupby("ck"):
        base = g["small"].mean()
        st = g.groupby("word")["small"].agg(["size", "sum"])
        st = st[st["size"] >= MIN_WORD_PAIRS]
        score = (st["sum"] + SMOOTH * base) / (st["size"] + SMOOTH)
        table[str(c)] = score.astype(float).to_dict()
        top = score.sort_values(ascending=False).head(12)
        log(f"    decoy words {c}: {len(score):,} words (base shift share {base:.3f}); top: "
            + ", ".join(f"{w} {s:.2f}" for w, s in top.items()))
    log(f"    decoy words learned from {len(twin):,} near-twin pairs")
    return table


def decoy_word_features(W1, W2, ck1, table):
    """Per pair: max decoy score of words only on the candidate side (dw2_max) / only on the
    S1 side (dw1_max), and how many such words have a score (dw2_known)."""
    m = len(W1)
    d2 = np.full(m, np.nan, np.float32)
    d1 = np.full(m, np.nan, np.float32)
    kn = np.zeros(m, np.float32)
    for k in range(m):
        t = table.get(str(ck1[k]))
        if not t:
            continue
        a, b = W1[k], W2[k]
        s2 = [t[w] for w in b - a if w in t]
        s1 = [t[w] for w in a - b if w in t]
        if s2:
            d2[k] = max(s2)
            kn[k] = len(s2)
        if s1:
            d1[k] = max(s1)
    return d2, d1, kn
