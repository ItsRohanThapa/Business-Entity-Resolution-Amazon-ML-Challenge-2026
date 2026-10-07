"""Embedding blocking passes on the GPU (added to the key-based passes of blocking.py).

Two text views per record, encoded with intfloat/multilingual-e5-small (MIT, 118M params):
  emb      normalised name + address ("name_core | addr_norm")
  emb_raw  ORIGINAL name + address as written - the multilingual model reads Hindi /
           Telugu / Bengali names directly instead of their transliteration
For every S1 record the top-k most similar S2 and S3 records (cosine, same country group)
become candidates; the two cosines are also pair features.
"""
import os
import time

import numpy as np

from .progress import bar

KINDS = ("emb", "emb_raw")


def record_texts(prof, kind):
    if kind == "emb":
        return (prof["name_core"].astype(str) + " | " + prof["addr_norm"].astype(str)).tolist()
    return (prof["raw_name"].fillna("").astype(str) + " | "
            + prof["raw_addr"].fillna("").astype(str)).tolist()


def encode(texts, cfg, device, log=print):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(cfg.emb_model, device=device)
    if device == "cuda":
        model = model.half()
    E = model.encode(["query: " + t for t in texts], batch_size=cfg.emb_batch,
                     normalize_embeddings=True, convert_to_numpy=True,
                     show_progress_bar=True).astype(np.float16)
    del model
    if device == "cuda":
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()           # give the GPU back to XGBoost
    return E


def load_embeddings(prof, cfg, device, work_dir, split, norm_version, log=print):
    """{kind: float16 [n_records x dim]} for every record of the split (cached in work_dir)."""
    out = {}
    for kind in KINDS:
        path = os.path.join(work_dir, f"{split}_{kind}_v{norm_version}_{cfg.emb_model.split('/')[-1]}.npy")
        if os.path.exists(path):
            E = np.load(path)
            if len(E) == len(prof):
                log(f"    {kind}: embeddings from cache {path}")
                out[kind] = E
                continue
        t0 = time.time()
        log(f"    {kind}: encoding {len(prof):,} records on {device}")
        out[kind] = encode(record_texts(prof, kind), cfg, device, log)
        np.save(path, out[kind])
        log(f"    {kind}: done in {(time.time() - t0) / 60:.1f} min")
    return out


# ----------------------------------------------------------------------------- GPU top-k
def _topk_part(A, B, k, device, mem_budget, log, label):
    import torch
    n = A.shape[0]
    idx = np.empty((n, k), dtype=np.int64)
    sc = np.empty((n, k), dtype=np.float32)
    dt = torch.float16 if device == "cuda" else torch.float32
    with torch.no_grad():
        Bt = torch.from_numpy(np.ascontiguousarray(B)).to(device, dt)
        if device == "cuda":
            free, _ = torch.cuda.mem_get_info()
            mem_budget = max(2 ** 22, min(mem_budget, free // 4 // 8))
        chunk = max(32, min(16384, mem_budget // max(B.shape[0], 1)))
        prog = bar(desc=f"{label} search {n:,}x{B.shape[0]:,}", total=n, unit="S1", leave=False)
        for s in range(0, n, chunk):
            a = torch.from_numpy(np.ascontiguousarray(A[s:s + chunk])).to(device, dt)
            v, i = torch.topk((a @ Bt.T).float(), k, dim=1)
            idx[s:s + chunk] = i.cpu().numpy()
            sc[s:s + chunk] = v.cpu().numpy()
            prog.update(len(a))
        prog.close()
        del Bt
        if device == "cuda":
            torch.cuda.empty_cache()
    return idx, sc


def topk_cosine(A, B, k, device, mem_budget=2 ** 31, log=print, label="emb"):
    """Top-k rows of B by cosine for every row of A (both L2-normalised). B is split into
    parts when it would take more than ~40% of the free GPU memory (12 GB cards)."""
    k = min(k, B.shape[0])
    if k <= 0 or A.shape[0] == 0:
        return np.zeros((A.shape[0], 0), np.int64), np.zeros((A.shape[0], 0), np.float32)
    parts = 1
    if device == "cuda":
        import torch
        parts = int(np.ceil(B.shape[0] * B.shape[1] * 2 / (0.4 * torch.cuda.mem_get_info()[0])))
    if parts <= 1:
        return _topk_part(A, B, k, device, mem_budget, log, label)
    bounds = np.linspace(0, B.shape[0], parts + 1).astype(int)
    idx = sc = None
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        i, v = _topk_part(A, B[lo:hi], min(k, hi - lo), device, mem_budget, log, label)
        i += lo
        if idx is None:
            idx, sc = i, v
            continue
        ci, cs = np.hstack([idx, i]), np.hstack([sc, v])
        order = np.argsort(-cs, axis=1, kind="stable")[:, :k]
        idx, sc = np.take_along_axis(ci, order, 1), np.take_along_axis(cs, order, 1)
    return idx, sc


def rowwise_cosine(E, a, b, chunk=4_000_000):
    """cosine(E[a[i]], E[b[i]]) for aligned index arrays (rows are already L2-normalised)."""
    out = np.empty(len(a), dtype=np.float32)
    for s in bar(range(0, len(a), chunk), "embedding cosines", unit="chunk", leave=False):
        x = E[a[s:s + chunk]].astype(np.float32)
        y = E[b[s:s + chunk]].astype(np.float32)
        out[s:s + chunk] = np.einsum("ij,ij->i", x, y)
    return out
