"""Scalable candidate generation (blocking) for millions of records.

Every record gets a bag of weighted blocking KEYS
  name    n token   b bigram   k phonetic skeleton   f 4-char prefix
          g character 4-grams of the space-free name (domains / handles / glued or
            re-ordered words: "spanglerdelbridge.com" vs "Delbridge & Spangler")
          r acronym ("TB" vs "Tech Business")       - for every DBA / "formerly" alias too
  address a token   w token bigram   h house-number x token   p postal   q house-number x postal
  combo   c name-skeleton x postal   d name-skeleton x address-token skeleton
Keys are IDF-weighted inside the country group; a key shared by too many pairs
(#S1-with-key x #S2/S3-with-key > max_key_pairs) is dropped.

Three retrieval passes run for every S1 record and every source (S2, S3):
  ALL (every key)  |  NAME (name keys only)  |  ADDRESS (address keys only)
The ADDRESS pass finds records whose name was replaced / re-branded / written in another
script; the NAME pass finds records with a missing or very different address. A reverse pass
(every S2/S3 record proposes its top S1 records) adds more. The union is the candidate set the
model scores and is exactly what candidate_pairs.tsv contains. Each pair also gets its cosine
in all three key spaces (bs, bsn, bsa) as features.

Two GPU EMBEDDING passes are added (emb_blocking.py): multilingual-e5 embeddings of the
normalised text (emb) and of the ORIGINAL text (emb_raw); their cosines (es, er) are features.
"""
import multiprocessing as mp

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize as l2_normalize

from .emb_blocking import rowwise_cosine, topk_cosine
from .normalize import skeleton_token
from .progress import bar

TYPE_MULT = {"n": 1.0, "b": 1.2, "k": 0.7, "f": 0.4, "g": 0.35, "r": 0.8,
             "a": 0.5, "w": 0.8, "h": 1.5, "p": 0.8, "q": 1.5, "c": 1.2, "d": 0.9}
NAME_TYPES, ADDR_TYPES = "nbkfgr", "awhpq"
_TYPES = list(TYPE_MULT)
_TYPE_ID = {t: i for i, t in enumerate(_TYPES)}
_MULT = np.array([TYPE_MULT[t] for t in _TYPES], dtype=np.float32)
PASS_BITS = {"all": 1, "name": 2, "addr": 4, "reverse": 8, "emb": 16, "emb_raw": 32}
EMB_FEATURE = {"emb": "es", "emb_raw": "er"}


def _name_keys(name):
    toks = name.split()
    alpha = [t for t in toks if not t.isdigit()]
    keys = ["n" + t for t in toks]
    keys += ["b" + a + "_" + b for a, b in zip(toks, toks[1:])]
    sk = [skeleton_token(t) for t in alpha]
    keys += ["k" + x for x in sk if len(x) >= 2]
    keys += ["f" + t[:4] for t in alpha if len(t) >= 6]
    glued = "".join(alpha)
    if len(glued) >= 4:
        keys += ["g" + glued[i:i + 4] for i in range(len(glued) - 3)]
    if len(alpha) >= 2:
        keys.append("r" + "".join(t[0] for t in alpha))
    elif alpha and 2 <= len(alpha[0]) <= 4:
        keys.append("r" + alpha[0])
    return keys, sk


def record_keys(core, alias, addr, postal):
    names = [core] + [a for a in alias.split(" | ") if a and a != core] if alias else [core]
    keys, sk = [], []
    for j, nm in enumerate(names):
        k, s_ = _name_keys(nm)
        keys += k
        if j == 0:
            sk = s_
    atok = addr.split()
    aalpha = [t for t in atok if len(t) >= 3 and not t.isdigit()]
    anum = list(dict.fromkeys(t for t in atok if t.isdigit() and t != postal))[:3]
    keys += ["a" + t for t in aalpha]
    keys += ["w" + a + "_" + b for a, b in zip(aalpha, aalpha[1:])]
    ashort = [t for t in atok if len(t) >= 2 and not t.isdigit()]
    for num in anum:
        keys += ["h" + num + "|" + t for t in ashort]
    if postal:
        keys.append("p" + postal)
        keys += ["c" + x + "|" + postal for x in sk[:3]]
        keys += ["q" + num + "|" + postal for num in anum]
    ask = {skeleton_token(t) for t in aalpha}
    for x in sk[:2]:
        keys += ["d" + x + "|" + t for t in ask if len(t) >= 2]
    return keys


def _keys_chunk(args):
    offset, cores, aliases, addrs, postals = args
    rows, hashes, types = [], [], []
    for j, (c, al, a, p) in enumerate(zip(cores, aliases, addrs, postals)):
        ks = record_keys(c, al, a, p)
        rows.extend([offset + j] * len(ks))
        hashes.extend([hash(k) for k in ks])   # forked workers share the hash seed
        types.extend([_TYPE_ID[k[0]] for k in ks])
    return (np.asarray(rows, dtype=np.int32), np.asarray(hashes, dtype=np.int64),
            np.asarray(types, dtype=np.int8))


def _pool(n_jobs):
    return mp.get_context("fork").Pool(n_jobs)


def _record_key_arrays(prof, rows, n_jobs, chunk=100_000):
    core = prof["name_core"].values
    alias = prof["name_alias"].values
    addr = prof["addr_norm"].values
    post = prof["addr_postal"].values
    jobs = [(k, list(core[rows[k:k + chunk]]), list(alias[rows[k:k + chunk]]),
             list(addr[rows[k:k + chunk]]), list(post[rows[k:k + chunk]]))
            for k in range(0, len(rows), chunk)]
    if n_jobs > 1 and len(jobs) > 1:
        with _pool(n_jobs) as pool:
            parts = list(bar(pool.imap(_keys_chunk, jobs), "blocking keys", total=len(jobs),
                             unit="chunk"))
    else:
        parts = [_keys_chunk(j) for j in bar(jobs, "blocking keys", unit="chunk")]
    return (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
            np.concatenate([p[2] for p in parts]))


def csr_topk(C, k, min_score):
    """Top-k entries of every row of a CSR matrix -> (row, col, value)."""
    counts = np.diff(C.indptr)
    row = np.repeat(np.arange(C.shape[0], dtype=np.int32), counts)
    data, col = C.data, C.indices
    m = data >= min_score
    row, col, data = row[m], col[m], data[m]
    if len(row) == 0:
        return row, col, data
    order = np.lexsort((-data, row))
    rs = row[order]
    first = np.r_[True, rs[1:] != rs[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(rs)), 0))
    keep = order[(np.arange(len(rs)) - start) < k]
    return row[keep], col[keep], data[keep]


_G = {}  # matrices shared with forked workers

try:   # multi-threaded C++ "top-k while multiplying" (Apache-2.0); ~5-10x faster, far less RAM
    from sparse_dot_topn import sp_matmul_topn
except ImportError:  # pragma: no cover
    sp_matmul_topn = None


def _product_chunk(args):
    a, b, k, min_score = args
    L = _G["L"][a:b]
    if sp_matmul_topn is not None:
        try:
            C = sp_matmul_topn(L, _G["RT"], top_n=k, threshold=min_score, sort=False,
                               n_threads=1).tocsr()
            r = np.repeat(np.arange(C.shape[0], dtype=np.int64), np.diff(C.indptr))
            return r + a, C.indices.astype(np.int64), C.data.astype(np.float32)
        except Exception:  # noqa: BLE001 - fall back to plain scipy
            pass
    C = (L @ _G["RT"]).tocsr()
    r, c, v = csr_topk(C, k, min_score)
    return r.astype(np.int64) + a, c.astype(np.int64), v.astype(np.float32)


def _chunks_by_cost(cost, budget):
    csum = np.cumsum(cost)
    bounds, start = [], 0
    while start < len(cost):
        base = csum[start - 1] if start else 0.0
        end = int(np.searchsorted(csum, base + budget, side="right"))
        end = max(end, start + 1)
        bounds.append((start, min(end, len(cost))))
        start = end
    return bounds


def _topk_products(left, right, k, cfg, n_jobs):
    """top-k columns of (left @ right.T) per left row, chunked by estimated work."""
    if left.shape[0] == 0 or right.shape[0] == 0 or k <= 0:
        e = np.array([], dtype=np.int64)
        return e, e, np.array([], dtype=np.float32)
    df_right = np.asarray((right > 0).sum(axis=0)).ravel().astype(np.float64)
    cost = (left > 0).astype(np.float64) @ df_right + 1.0
    budget = cfg.block_chunk_products * (5 if sp_matmul_topn is not None else 1)
    chunks = _chunks_by_cost(cost, budget)
    RT = right.T.tocsr()
    if left.indices.dtype != RT.indices.dtype:
        left = left.copy()
        left.indices = left.indices.astype(np.int64)
        left.indptr = left.indptr.astype(np.int64)
        RT.indices = RT.indices.astype(np.int64)
        RT.indptr = RT.indptr.astype(np.int64)
    _G["L"], _G["RT"] = left, RT
    jobs = [(a, b, k, cfg.min_block_score) for a, b in chunks]
    desc = f"key search {left.shape[0]:,}x{right.shape[0]:,} top{k}"
    if n_jobs > 1 and len(jobs) > 1:
        with _pool(n_jobs) as pool:
            parts = list(bar(pool.imap(_product_chunk, jobs), desc, total=len(jobs),
                             unit="chunk", leave=False))
    else:
        parts = [_product_chunk(j) for j in bar(jobs, desc, unit="chunk", leave=False)]
    _G.clear()
    return (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
            np.concatenate([p[2] for p in parts]))


def _group_matrices(prof, s1_rows, pool_rows, cfg, n_jobs):
    """IDF-weighted, pruned, L2-normalised key matrices for one country group."""
    rows_all = np.concatenate([s1_rows, pool_rows])
    n1, n_docs = len(s1_rows), len(rows_all)
    r, h, t = _record_key_arrays(prof, rows_all, n_jobs)
    uniq, col = np.unique(h, return_inverse=True)
    del h
    col = col.ravel()
    nk = len(uniq)
    # document frequency per key on each side (a record counts once per key)
    rc = np.unique(r.astype(np.int64) * nk + col)
    rr, cc = rc // nk, rc % nk
    df1 = np.bincount(cc[rr < n1], minlength=nk).astype(np.float64)
    df2 = np.bincount(cc[rr >= n1], minlength=nk).astype(np.float64)
    del rc, rr, cc
    keep_key = (df1 > 0) & (df2 > 0) & (df1 * df2 <= cfg.max_key_pairs)
    key_type = np.zeros(nk, dtype=np.int8)
    key_type[col] = t
    w = (np.log1p(n_docs / (df1 + df2 + 1.0)) * _MULT[key_type]).astype(np.float32)
    new_col = np.cumsum(keep_key) - 1
    m = keep_key[col]
    r, c = r[m], new_col[col[m]]
    ncol = int(keep_key.sum())
    X = sp.csr_matrix((w[keep_key][c], (r, c)), shape=(n_docs, max(ncol, 1)), dtype=np.float32)
    X.sum_duplicates()
    col_type = key_type[keep_key] if ncol else np.zeros(1, np.int8)
    return X[:n1].tocsr(), X[n1:].tocsr(), col_type


def _pass_matrix(Xraw, col_type, types):
    if types is not None:
        cols = np.where(np.isin(col_type, [_TYPE_ID[t] for t in types]))[0]
        Xraw = Xraw[:, cols]
    return l2_normalize(Xraw.astype(np.float32), norm="l2", copy=True).tocsr()


def _rowwise(A, B, ia, ib, chunk=2_000_000):
    out = np.empty(len(ia), dtype=np.float32)
    for s0 in bar(range(0, len(ia), chunk), "pair key cosines", unit="chunk", leave=False):
        out[s0:s0 + chunk] = np.asarray(
            A[ia[s0:s0 + chunk]].multiply(B[ib[s0:s0 + chunk]]).sum(axis=1)).ravel()
    return out


def _country_groups(prof, s1_query, cfg):
    src = prof["src"].values
    if not cfg.block_within_country:
        return [(s1_query, np.where(src != 1)[0])]
    key = prof["country_key"].values
    q_keys = key[s1_query]
    wildcard = np.where((src != 1) & (key == ""))[0]
    groups = []
    for g in sorted(set(q_keys)):
        s1_rows = s1_query[q_keys == g]
        if g == "":
            pool = np.where(src != 1)[0]
        else:
            pool = np.union1d(np.where((src != 1) & (key == g))[0], wildcard)
        groups.append((s1_rows, pool))
    return groups


def generate_candidates(prof, s1_query, cfg, n_jobs=1, log=print, emb=None):
    """Candidate pairs for the S1 rows in s1_query (row positions of prof).
    Key statistics, pruning and the reverse pass always use ALL S1 records of the split, so a
    --dev run on a subset of S1 sees exactly the same candidates as the full run would.
    Returns aligned arrays sorted by S1: i1, i2 (prof rows), bs / bsn / bsa (cosine in the
    all / name / address key space), mask (bit per pass), fwd, rev.
    emb: optional {"emb": E, "emb_raw": E} record embeddings -> extra GPU passes + es / er."""
    src = prof["src"].values
    s1_all = np.where(src == 1)[0]
    is_query = np.zeros(len(prof), dtype=bool)
    is_query[np.asarray(s1_query)] = True
    passes = [("all", None, cfg.k_forward), ("name", NAME_TYPES, cfg.k_name),
              ("addr", ADDR_TYPES, cfg.k_addr)]
    emb = emb or {}
    res = {k: [] for k in ("i1", "i2", "bs", "bsn", "bsa", "mask") + tuple(EMB_FEATURE[e] for e in emb)}
    for s1_rows, pool_rows in _country_groups(prof, s1_all, cfg):
        q = np.where(is_query[s1_rows])[0]
        if len(q) == 0 or len(pool_rows) == 0:
            continue
        log(f"    country {prof['country'].values[s1_rows[0]]!s}: key matrices for "
            f"{len(s1_rows):,} S1 + {len(pool_rows):,} S2/S3 records")
        X1raw, X2raw, col_type = _group_matrices(prof, s1_rows, pool_rows, cfg, n_jobs)
        mats = {}
        for name, types, _ in passes:
            mats[name] = (_pass_matrix(X1raw, col_type, types), _pass_matrix(X2raw, col_type, types))
        del X1raw, X2raw
        li, lj, lm, counts = [], [], [], {}
        for name, types, k in passes:
            A, B = mats[name]
            Aq = A if len(q) == len(s1_rows) else A[q]
            for s_ in (2, 3):
                sel = np.where(src[pool_rows] == s_)[0]
                if len(sel) == 0 or k <= 0:
                    continue
                r, c, _ = _topk_products(Aq, B[sel], k, cfg, n_jobs)
                li.append(q[r])
                lj.append(sel[c])
                lm.append(np.full(len(r), PASS_BITS[name], np.int8))
                counts[name] = counts.get(name, 0) + len(r)
        A, B = mats["all"]
        r, c, _ = _topk_products(B, A, cfg.k_reverse, cfg, n_jobs)
        keep = is_query[s1_rows[c]]
        li.append(c[keep])
        lj.append(r[keep])
        lm.append(np.full(int(keep.sum()), PASS_BITS["reverse"], np.int8))
        counts["reverse"] = int(keep.sum())
        for kind, E in emb.items():                      # GPU embedding passes
            A = E[s1_rows[q]]
            for s_ in (2, 3):
                sel = np.where(src[pool_rows] == s_)[0]
                if len(sel) == 0 or cfg.k_emb <= 0:
                    continue
                idx, sc = topk_cosine(A, E[pool_rows[sel]], cfg.k_emb, cfg.device, log=log,
                                      label=f"{kind} S{s_}")
                ok = (sc >= cfg.min_emb_score).ravel()
                li.append(np.repeat(q, idx.shape[1])[ok])
                lj.append(sel[idx.ravel()][ok])
                lm.append(np.full(int(ok.sum()), PASS_BITS[kind], np.int8))
                counts[kind] = counts.get(kind, 0) + int(ok.sum())
        li, lj, lm = np.concatenate(li), np.concatenate(lj), np.concatenate(lm)
        key = li.astype(np.int64) * len(pool_rows) + lj
        order = np.argsort(key, kind="stable")
        key, li, lj, lm = key[order], li[order], lj[order], lm[order]
        first = np.r_[True, key[1:] != key[:-1]]
        starts = np.flatnonzero(first)
        mask = np.bitwise_or.reduceat(lm, starts)
        li, lj = li[starts], lj[starts]
        res["i1"].append(s1_rows[li])
        res["i2"].append(pool_rows[lj])
        res["mask"].append(mask.astype(np.int8))
        for name, col in (("all", "bs"), ("name", "bsn"), ("addr", "bsa")):
            A, B = mats[name]
            res[col].append(_rowwise(A, B, li, lj))
        for kind, E in emb.items():
            res[EMB_FEATURE[kind]].append(rowwise_cosine(E, s1_rows[li], pool_rows[lj]))
        log(f"    country {prof['country'].values[s1_rows[0]]!s:>10}: {len(q):,} of "
            f"{len(s1_rows):,} S1 x {len(pool_rows):,} S2/S3 | kept keys {mats['all'][0].shape[1]:,}"
            f" | pairs {len(li):,} " + str({k: f"{v:,}" for k, v in counts.items()}))
        del mats
    if not res["i1"]:
        e = np.array([], dtype=np.int64)
        f = np.array([], np.float32)
        out = {"i1": e, "i2": e, "bs": f, "bsn": f, "bsa": f, "mask": np.array([], np.int8),
               "fwd": np.array([], bool), "rev": np.array([], bool)}
        out.update({EMB_FEATURE[k]: f for k in emb})
        return out
    out = {k: np.concatenate(v) for k, v in res.items()}
    out["i1"], out["i2"] = out["i1"].astype(np.int64), out["i2"].astype(np.int64)
    o = np.lexsort((out["i2"], out["i1"]))
    out = {k: v[o] for k, v in out.items()}
    out["fwd"] = (out["mask"] & ~PASS_BITS["reverse"]) > 0      # every pass except reverse
    out["rev"] = (out["mask"] & 8) > 0
    return out


def blocking_report(pairs, truth, s1_query_ids, ids, n_pool, src, k_forward):
    """Pair recall, recall@k of the forward pass, oracle macro-F0.5 ceiling (train only).
    Returns (report dict, set of missed true pairs)."""
    from .features import group_stats
    from .metrics import macro_fbeta
    true_pairs = {(s, m) for s in s1_query_ids for m in truth.get(s, ())}
    n_true = max(1, len(true_pairs))
    s_ids, c_ids = ids[pairs["i1"]], ids[pairs["i2"]]
    got = set(zip(s_ids, c_ids))
    f = pairs["fwd"]
    rep = {
        "s1_entities": len(s1_query_ids),
        "true_pairs": len(true_pairs),
        "pair_recall": len(true_pairs & got) / n_true,
        "pair_recall_forward_only": len(true_pairs & set(zip(s_ids[f], c_ids[f]))) / n_true,
        "candidate_pairs": int(len(pairs["i1"])),
        "avg_candidates_per_s1": len(pairs["i1"]) / max(1, len(s1_query_ids)),
        "reduction_ratio": 1.0 - len(pairs["i1"]) / max(1, len(s1_query_ids) * max(n_pool, 1)),
    }
    if f.any():
        g = pairs["i1"][f] * 4 + src[pairs["i2"][f]]
        rk, _, _ = group_stats(g, pairs["bs"][f].astype(np.float64))
        fs, fc = s_ids[f], c_ids[f]
        for k in (1, 3, 5, 10, 20, 30, 50, 100, 200):
            if k <= k_forward:
                m = rk <= k
                rep[f"recall_forward_top{k}_per_source"] = \
                    len(true_pairs & set(zip(fs[m], fc[m]))) / n_true
    if "mask" in pairs:
        for name, bit in PASS_BITS.items():
            m = (pairs["mask"] & bit) > 0
            rep[f"recall_pass_{name}"] = len(true_pairs & set(zip(s_ids[m], c_ids[m]))) / n_true
    cand_by_s1 = {}
    for s, m in got:
        cand_by_s1.setdefault(s, set()).add(m)
    oracle = {s: truth.get(s, set()) & cand_by_s1.get(s, set()) for s in s1_query_ids}
    rep["oracle_macro_f05"] = macro_fbeta(oracle, truth, list(s1_query_ids))
    return rep, true_pairs - got
