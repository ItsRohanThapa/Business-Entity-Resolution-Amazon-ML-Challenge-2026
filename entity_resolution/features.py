"""Pair features, computed in parallel chunks (a chunk always holds complete S1 groups).

Global context (all pairs, vectorised, cheap):
  block score + quick name/address token-set similarity, and for each of them the rank /
  margin of the candidate inside its S1 group AND of the S1 inside the candidate's group
  (competition between look-alike S1 entities), group sizes, which blocking pass found it.
Per-pair features (chunked):
  rapidfuzz name / skeleton / address similarities, IDF-weighted exact + soft token overlap
  and the weight of the most "surprising" unmatched token on each side (catches look-alikes
  such as "Acme Robotics" vs "Acme Bakery"), postal / house-number agreement and conflicts,
  DBA aliases, acronyms, legal forms, landmarks, lengths.
"""
import multiprocessing as mp

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

from .decoy import decoy_word_features
from .normalize import addr_view, name_view, token_idf
from .progress import bar

NAN = np.float32(np.nan)
SIDE_WORDS = ["svc", "ctr", "partner", "grp", "hldg", "ent", "com", "inc", "llc", "ltd", "pvt",
              "corp", "co", "lp", "llp", "pc"]
_CTX = {}   # shared with forked worker processes


# ----------------------------------------------------------------------------- helpers
def _rf(scorer, a, b):
    return process.cpdist(a, b, scorer=scorer, dtype=np.float32,
                          workers=_CTX.get("rf_workers", 1))


def group_stats(g, v):
    """rank (1 = best), v - max, v - second best inside groups g (numpy, any size)."""
    n = len(g)
    order = np.lexsort((-v, g))
    gs, vs = g[order], v[order]
    first = np.r_[True, gs[1:] != gs[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(n), 0))
    nxt = np.minimum(start + 1, n - 1)
    has2 = (start + 1 < n) & (gs[nxt] == gs)
    second = np.where(has2, vs[nxt], 0.0)
    rank = np.empty(n, np.float32)
    gapmax = np.empty(n, np.float32)
    gap2 = np.empty(n, np.float32)
    rank[order] = np.arange(n) - start + 1
    gapmax[order] = vs - vs[start]
    gap2[order] = vs - second
    return rank, gapmax, gap2


def _tok_sim(a, b):
    s = JaroWinkler.normalized_similarity(a, b)
    if min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a)):
        s = max(s, 0.92)
    return s


def token_overlap(TA, TB, idf, default, thr=0.88):
    """IDF-weighted exact Dice, soft Dice and unmatched-token statistics."""
    n = len(TA)
    ex, so = np.zeros(n, np.float32), np.zeros(n, np.float32)
    um1, um2 = np.zeros(n, np.float32), np.zeros(n, np.float32)
    uc1, uc2 = np.zeros(n, np.float32), np.zeros(n, np.float32)
    first = np.zeros(n, np.float32)
    for i in range(n):
        A, B = TA[i], TB[i]
        if not A or not B:
            um1[i] = max((idf.get(t, default) for t in A), default=0.0)
            um2[i] = max((idf.get(t, default) for t in B), default=0.0)
            uc1[i], uc2[i] = len(A), len(B)
            continue
        first[i] = 1.0 if A[0] == B[0] else float(_tok_sim(A[0], B[0]) >= 0.9)
        sa, sb = set(A), set(B)
        wa = {t: idf.get(t, default) for t in sa}
        wb = {t: idf.get(t, default) for t in sb}
        inter = sa & sb
        w_inter = sum(wa[t] for t in inter)
        tot = sum(wa.values()) + sum(wb.values())
        ex[i] = 2 * w_inter / tot
        ra = [t for t in sa if t not in inter]
        rb = [t for t in sb if t not in inter]
        soft, ma, mb = 0.0, set(), set()
        if ra and rb:
            cands = []
            for a in ra:
                for b in rb:
                    s = _tok_sim(a, b)
                    if s >= thr:
                        cands.append((s, a, b))
            cands.sort(reverse=True)
            for s, a, b in cands:
                if a in ma or b in mb:
                    continue
                ma.add(a)
                mb.add(b)
                soft += s * (wa[a] + wb[b])
        so[i] = (2 * w_inter + soft) / tot
        ua = [wa[t] for t in ra if t not in ma]
        ub = [wb[t] for t in rb if t not in mb]
        um1[i], um2[i] = max(ua, default=0.0), max(ub, default=0.0)
        uc1[i], uc2[i] = len(ua), len(ub)
    return ex, so, um1, um2, uc1, uc2, first


# ----------------------------------------------------------------------------- setup
def prepare_context(prof, n_jobs=1, log=print):
    """Per-split arrays and token IDF tables used by the workers."""
    core = prof["name_core"].values
    addr = prof["addr_norm"].values
    idf_n, dn = token_idf((c.split() for c in core), len(core), "name token IDF")
    idf_a, da = token_idf(((t for t in a.split() if not t.isdigit()) for a in addr), len(addr),
                          "address token IDF")
    log(f"    token IDF tables: {len(idf_n):,} name tokens, {len(idf_a):,} address tokens")
    return {
        "core": core, "legal": prof["name_legal"].values, "alias": prof["name_alias"].values,
        "extra": prof["name_extra"].values if "name_extra" in prof else np.full(len(core), ""),
        "raw_name": prof["raw_name"].values if "raw_name" in prof else None,
        "raw_addr": prof["raw_addr"].values if "raw_addr" in prof else None,
        "addr": addr, "postal": prof["addr_postal"].values,
        "lm": prof["addr_landmark"].values.astype(np.float32),
        "src": prof["src"].values, "ck": prof["country_key"].values,
        "idf_n": idf_n, "dn": dn, "idf_a": idf_a, "da": da,
    }


def global_context(split_ctx, pairs, chunk=5_000_000, log=print):
    """Cheap features for ALL candidate pairs (needed for competition statistics)."""
    i1, i2 = pairs["i1"], pairs["i2"]
    core, addr = split_ctx["core"], split_ctx["addr"]
    n = len(i1)
    qn = np.empty(n, np.float32)
    qa = np.empty(n, np.float32)
    _CTX["rf_workers"] = -1
    for s in bar(range(0, n, chunk), "quick name/address similarity", unit="chunk"):
        e = min(n, s + chunk)
        qn[s:e] = _rf(fuzz.token_set_ratio, core[i1[s:e]].tolist(), core[i2[s:e]].tolist())
        qa[s:e] = _rf(fuzz.token_set_ratio, addr[i1[s:e]].tolist(), addr[i2[s:e]].tolist())
    qn /= 100.0
    qa /= 100.0
    ctx = {"bs": pairs["bs"].astype(np.float32), "qn": qn, "qa": qa,
           "bp_fwd": pairs["fwd"].astype(np.float32), "bp_rev": pairs["rev"].astype(np.float32)}
    for e in ("es", "er"):                   # embedding cosines (emb_blocking.py)
        if e in pairs:
            ctx[e] = pairs[e].astype(np.float32)
    if "mask" in pairs:
        ctx["bp_emb"] = ((pairs["mask"] & 16) > 0).astype(np.float32)
        ctx["bp_emb_raw"] = ((pairs["mask"] & 32) > 0).astype(np.float32)
        ctx["bsn"] = pairs["bsn"].astype(np.float32)
        ctx["bsa"] = pairs["bsa"].astype(np.float32)
        ctx["bp_all"] = ((pairs["mask"] & 1) > 0).astype(np.float32)
        ctx["bp_name"] = ((pairs["mask"] & 2) > 0).astype(np.float32)
        ctx["bp_addr"] = ((pairs["mask"] & 4) > 0).astype(np.float32)
    ranked = [("bs", ctx["bs"]), ("qn", qn), ("qa", qa)]
    if "bsa" in ctx:
        ranked += [("bsn", ctx["bsn"]), ("bsa", ctx["bsa"])]
    ranked += [(e, ctx[e]) for e in ("es", "er") if e in ctx]
    for name, v in bar(ranked, "competition statistics", unit="score"):
        for suffix, g in (("", i1), ("_rev", i2)):
            rk, gm, g2 = group_stats(g, v)
            ctx[f"{name}_rk{suffix}"] = rk
            ctx[f"{name}_gapmax{suffix}"] = gm
            ctx[f"{name}_gap2{suffix}"] = g2
    _, inv1, cnt1 = np.unique(i1, return_inverse=True, return_counts=True)
    _, inv2, cnt2 = np.unique(i2, return_inverse=True, return_counts=True)
    ctx["n_cand_s1"] = cnt1[inv1.ravel()].astype(np.float32)
    ctx["n_s1_for_cand"] = cnt2[inv2.ravel()].astype(np.float32)
    # float16 storage keeps tens of millions of pairs affordable
    ctx = {k: v.astype(np.float16) for k, v in ctx.items()}
    log(f"    global context: {len(ctx)} features x {n:,} pairs")
    return ctx


# ----------------------------------------------------------------------------- per pair
def pair_features(i1, i2):
    C = _CTX
    core, legal, alias, addr = C["core"], C["legal"], C["alias"], C["addr"]
    postal, src, ck = C["postal"], C["src"], C["ck"]
    uniq = np.unique(np.concatenate([i1, i2]))
    nv = {u: name_view(core[u]) for u in uniq}
    av = {u: addr_view(addr[u], postal[u]) for u in uniq}
    NV1, NV2 = [nv[u] for u in i1], [nv[u] for u in i2]
    AV1, AV2 = [av[u] for u in i1], [av[u] for u in i2]
    F = {}
    n1, n2 = core[i1].tolist(), core[i2].tolist()
    F["n_ratio"] = _rf(fuzz.ratio, n1, n2)
    F["n_partial"] = _rf(fuzz.partial_ratio, n1, n2)
    F["n_tsort"] = _rf(fuzz.token_sort_ratio, n1, n2)
    F["n_tset"] = _rf(fuzz.token_set_ratio, n1, n2)
    F["n_jw"] = _rf(JaroWinkler.normalized_similarity, n1, n2)
    F["n_lev"] = _rf(Levenshtein.normalized_similarity, n1, n2)
    F["n_nospace"] = _rf(fuzz.ratio, [x.replace(" ", "") for x in n1],
                         [x.replace(" ", "") for x in n2])
    L1, L2 = legal[i1], legal[i2]
    F["nf_tset"] = _rf(fuzz.token_set_ratio, [f"{a} {b}".strip() for a, b in zip(n1, L1)],
                       [f"{a} {b}".strip() for a, b in zip(n2, L2)])
    k1, k2 = [v[1] for v in NV1], [v[1] for v in NV2]
    F["sk_ratio"] = _rf(fuzz.ratio, k1, k2)
    F["sk_tset"] = _rf(fuzz.token_set_ratio, k1, k2)
    a1, a2 = addr[i1].tolist(), addr[i2].tolist()
    F["a_ratio"] = _rf(fuzz.ratio, a1, a2)
    F["a_partial"] = _rf(fuzz.partial_ratio, a1, a2)
    F["a_tsort"] = _rf(fuzz.token_sort_ratio, a1, a2)
    F["a_jw"] = _rf(JaroWinkler.normalized_similarity, a1, a2)
    F["a_nospace"] = _rf(fuzz.partial_ratio, [x.replace(" ", "") for x in a1],
                         [x.replace(" ", "") for x in a2])
    F["as_tset"] = _rf(fuzz.token_set_ratio, [v[3] for v in AV1], [v[3] for v in AV2])
    F["x_name1_addr2"] = _rf(fuzz.partial_ratio, n1, a2)
    F["x_name2_addr1"] = _rf(fuzz.partial_ratio, n2, a1)

    ex, so, um1, um2, uc1, uc2, first = token_overlap(
        [v[0] for v in NV1], [v[0] for v in NV2], C["idf_n"], C["dn"])
    F.update({"n_dice": ex, "n_soft": so, "n_unm_idf1": um1, "n_unm_idf2": um2,
              "n_unm_cnt1": uc1, "n_unm_cnt2": uc2, "n_first_eq": first})
    ex, so, um1, um2, uc1, uc2, _ = token_overlap(
        [v[0] for v in AV1], [v[0] for v in AV2], C["idf_a"], C["da"])
    F.update({"a_dice": ex, "a_soft": so, "a_unm_idf1": um1, "a_unm_idf2": um2,
              "a_unm_cnt1": uc1, "a_unm_cnt2": uc2})

    m = len(i1)
    postal_eq, postal_pre3 = np.full(m, NAN), np.full(m, NAN)
    num_jac, num_conf = np.full(m, NAN), np.full(m, NAN)
    first_eq, nname_conf = np.full(m, NAN), np.full(m, NAN)
    alias_best = F["n_tset"].copy()
    acr = np.zeros(m, np.float32)
    P1, P2 = postal[i1], postal[i2]
    AL1, AL2 = alias[i1], alias[i2]
    for k in range(m):
        pa, pb = P1[k], P2[k]
        if pa and pb:
            postal_eq[k] = float(pa == pb)
            postal_pre3[k] = float(pa[:3] == pb[:3])
        na, nb = AV1[k][1], AV2[k][1]
        if na and nb:
            inter = len(na & nb)
            num_jac[k] = inter / len(na | nb)
            num_conf[k] = float(inter == 0)
        fa, fb = AV1[k][2], AV2[k][2]
        if fa and fb:
            first_eq[k] = float(fa == fb)
        xa, xb = NV1[k][2], NV2[k][2]
        if xa or xb:
            nname_conf[k] = float(not (xa & xb))
        if AL1[k] or AL2[k]:
            la = AL1[k].split(" | ") if AL1[k] else [n1[k]]
            lb = AL2[k].split(" | ") if AL2[k] else [n2[k]]
            alias_best[k] = max(fuzz.token_set_ratio(x, y) for x in la for y in lb)
        ca, cb = NV1[k][3], NV2[k][3]
        if (ca and ca in NV2[k][0]) or (cb and cb in NV1[k][0]) or (ca and ca == cb and len(ca) >= 3):
            acr[k] = 1.0
    F.update({"postal_eq": postal_eq, "postal_pre3": postal_pre3, "num_jacc": num_jac,
              "num_conflict": num_conf, "first_num_eq": first_eq,
              "name_num_conflict": nname_conf, "alias_best": alias_best, "acronym": acr})
    # which generic / legal words appear on only one side ("X Group" vs "X" = often a sibling
    # entity, "X LLC" vs "X Inc" = often noise) -> the trees learn which ones matter
    E1, E2 = C["extra"][i1], C["extra"][i2]
    W1 = [set(f"{a} {b}".split()) for a, b in zip(L1, E1)]
    W2 = [set(f"{a} {b}".split()) for a, b in zip(L2, E2)]
    for w in SIDE_WORDS:
        F[f"w_{w}"] = np.fromiter(((w in b) - (w in a) for a, b in zip(W1, W2)),
                                  dtype=np.float32, count=m)
    F["w_diff_cnt"] = np.fromiter((len(a ^ b) for a, b in zip(W1, W2)), dtype=np.float32,
                                  count=m)
    if C.get("decoy_words") is not None:     # label-free decoy-word score (decoy.py)
        D1 = [set(v[0]) | set(e.split()) for v, e in zip(NV1, E1)]
        D2 = [set(v[0]) | set(e.split()) for v, e in zip(NV2, E2)]
        F["dw2_max"], F["dw1_max"], F["dw2_known"] = decoy_word_features(
            D1, D2, ck[i1], C["decoy_words"])
    both = (L1 != "") & (L2 != "")
    F["legal_both"] = both.astype(np.float32)
    F["legal_eq"] = np.where(both, (L1 == L2).astype(np.float32), NAN)
    F["lm1"], F["lm2"] = C["lm"][i1], C["lm"][i2]
    F["n_len1"] = np.array([len(v[0]) for v in NV1], np.float32)
    F["n_len2"] = np.array([len(v[0]) for v in NV2], np.float32)
    F["a_len1"] = np.array([len(v[0]) for v in AV1], np.float32)
    F["a_len2"] = np.array([len(v[0]) for v in AV2], np.float32)
    F["has_postal1"] = (P1 != "").astype(np.float32)
    F["has_postal2"] = (P2 != "").astype(np.float32)
    F["same_country"] = (ck[i1] == ck[i2]).astype(np.float32)
    F["is_s3"] = (src[i2] == 3).astype(np.float32)
    # rank / margin inside the S1 group for the strongest per-pair similarities
    for c in ("n_soft", "a_soft", "n_tset", "sk_tset"):
        rk, gm, g2 = group_stats(i1, F[c].astype(np.float64))
        F[f"{c}_rk"], F[f"{c}_gapmax"], F[f"{c}_gap2"] = rk, gm, g2
    return F


def _chunk_job(args):
    a, b = args
    i1, i2 = _CTX["i1"][a:b], _CTX["i2"][a:b]
    F = pair_features(i1, i2)
    for k, v in _CTX["gctx"].items():
        F[k] = v[a:b].astype(np.float32)
    return a, pd.DataFrame(F)


def _chunk_bounds(i1, size):
    """[a, b) ranges of about `size` pairs that never split an S1 group."""
    n, bounds, a = len(i1), [], 0
    while a < n:
        b = min(n, a + size)
        if b < n:
            b = a + int(np.searchsorted(i1[a:], i1[b], side="left"))
            if b == a:
                b = a + int(np.searchsorted(i1[a:], i1[a], side="right"))
        bounds.append((a, b))
        a = b
    return bounds


def run_chunks(split_ctx, pairs, gctx, cfg, n_jobs, predict_fn=None, keep=True, log=print):
    """Features for all pairs in parallel chunks (workers = CPU processes).
    predict_fn(X) -> probabilities is applied in the MAIN process (so a GPU model can be used);
    keep=False discards the features after prediction (test set). Returns (X or None, p or None)."""
    _CTX.clear()
    _CTX.update(split_ctx)
    _CTX.update({"i1": pairs["i1"], "i2": pairs["i2"], "gctx": gctx, "rf_workers": 1})
    jobs = _chunk_bounds(pairs["i1"], cfg.feature_chunk_pairs)
    desc = f"pair features ({len(pairs['i1']):,} pairs" + (", + predict)" if predict_fn else ")")
    p = np.empty(len(pairs["i1"]), dtype=np.float32) if predict_fn is not None else None
    kept = []

    def handle(res):
        a, X = res
        if predict_fn is not None:
            p[a:a + len(X)] = predict_fn(X)
        if keep:
            kept.append((a, X))

    if n_jobs > 1 and len(jobs) > 1:
        # maxtasksperchild: workers are recycled so copy-on-write pages of the big string
        # arrays never accumulate (keeps RAM flat on 10M+ record splits)
        with mp.get_context("fork").Pool(n_jobs, maxtasksperchild=4) as pool:
            for res in bar(pool.imap_unordered(_chunk_job, jobs), desc, total=len(jobs),
                           unit="chunk"):
                handle(res)
    else:
        _CTX["rf_workers"] = -1
        for j in bar(jobs, desc, unit="chunk"):
            handle(_chunk_job(j))
    _CTX.clear()
    X = None
    if keep:
        kept.sort(key=lambda t: t[0])
        X = pd.concat([x for _, x in kept], ignore_index=True)
    return X, p


def stage2_features(i1, i2, src2, p1, gctx, ce=None):
    """Compact stacking features: stage-1 probability (and cross-encoder score) plus their
    rank / margin inside the S1 group and inside the candidate's group (competition)."""
    F = {"p1": p1.astype(np.float32)}
    vals = [("p1", p1.astype(np.float64))]
    if ce is not None:
        F["ce"] = ce.astype(np.float32)
        F["has_ce"] = (~np.isnan(ce)).astype(np.float32)
        vals.append(("ce", np.nan_to_num(ce.astype(np.float64), nan=-1.0)))
    for name, v in vals:
        for suffix, g in (("", i1), ("_rev", i2)):
            rk, gm, g2 = group_stats(g, v)
            F[f"{name}_rk{suffix}"], F[f"{name}_gapmax{suffix}"], F[f"{name}_gap2{suffix}"] = \
                rk, gm, g2
    for k in ("bs", "qn", "qa", "n_cand_s1", "n_s1_for_cand", "es", "er",
              "hn_small", "hn_digits", "hn_sup_s1", "hn_sup_cand", "hn_sup_gap",
              "tw_name_num2", "tw_s1_name1"):
        if k in gctx:
            F[k] = gctx[k].astype(np.float32)
    F["is_s3"] = (src2 == 3).astype(np.float32)
    return pd.DataFrame(F)
