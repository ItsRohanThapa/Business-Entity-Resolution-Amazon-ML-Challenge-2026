"""GPU cross-encoder (enable with --cross-encoder).

A multilingual transformer (default intfloat/multilingual-e5-base: MIT licence, 278M
parameters) with a classification head is fine-tuned on (S1 record, candidate record)
text pairs and its match probability becomes an extra feature for the stage-2 booster.
It reads both records jointly, so it learns patterns the hand-made features miss
(transliterations, abbreviations, landmark-only addresses) and brings multilingual
pretraining for the unseen French test data.

Leakage control: the model is CROSS-FITTED on two halves of the training S1 entities
(train on half A -> score half B and vice versa), so the stage-2 booster only ever sees
out-of-fold cross-encoder scores; test pairs get the average of both models.
Only promising pairs are scored (top-n per S1 by stage-1 probability + every pair above
a small probability), the same label-free rule for train and test.
"""
import gc
import math
import os
import time

import numpy as np

from .progress import bar


def record_texts(split_ctx, rows):
    """'name ; address' text for the given profile rows - the ORIGINAL text when available
    (the multilingual model reads native scripts, and sibling words such as "Group" or
    "Enterprises" matter), otherwise the normalised text."""
    rn, ra = split_ctx.get("raw_name"), split_ctx.get("raw_addr")
    if rn is not None and ra is not None:
        return {int(r): f"{rn[r]} ; {ra[r]}" for r in rows}
    core, legal, addr = split_ctx["core"], split_ctx["legal"], split_ctx["addr"]
    return {int(r): (f"{core[r]} {legal[r]}".strip() + " ; " + addr[r]) for r in rows}


def select_for_ce(i1, p1, topn, min_p):
    """Label-free selection of the pairs worth scoring with the cross-encoder."""
    from .features import group_stats
    rk, _, _ = group_stats(i1, p1.astype(np.float64))
    return (rk <= topn) | (p1 >= min_p)


class CrossEncoder:
    def __init__(self, cfg, device, log=print, path=None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch, self.cfg, self.device, self.log = torch, cfg, device, log
        src = path or cfg.ce_model                 # path: a model saved by .save()
        self.tok = AutoTokenizer.from_pretrained(src)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            src, num_labels=1, ignore_mismatched_sizes=True).to(device)
        self.amp = device.startswith("cuda")
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        if self.amp:
            torch.backends.cuda.matmul.allow_tf32 = True

    def _tokenize_chunk(self, ta, tb, idx):
        return self.tok([ta[i] for i in idx], [tb[i] for i in idx], truncation=True,
                        max_length=self.cfg.ce_max_len, padding=False)

    def _to_batch(self, enc, rows):
        torch = self.torch
        ids = [enc["input_ids"][j] for j in rows]
        L = max(len(x) for x in ids)
        arr = np.full((len(ids), L), self.pad_id, dtype=np.int64)
        att = np.zeros((len(ids), L), dtype=np.int64)
        for r, x in enumerate(ids):
            arr[r, :len(x)] = x
            att[r, :len(x)] = 1
        batch = {"input_ids": torch.from_numpy(arr), "attention_mask": torch.from_numpy(att)}
        if "token_type_ids" in enc:
            tt = np.zeros((len(ids), L), dtype=np.int64)
            for r, j in enumerate(rows):
                x = enc["token_type_ids"][j]
                tt[r, :len(x)] = x
            batch["token_type_ids"] = torch.from_numpy(tt)
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def _logits(self, enc):
        torch = self.torch
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.amp):
            return self.model(**enc).logits.squeeze(-1).float()

    def fit(self, ta, tb, y, seed=42, chunk=65_536):
        torch = self.torch
        from transformers import get_linear_schedule_with_warmup
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)
        cfg, n = self.cfg, len(y)
        bs = cfg.ce_batch
        steps = math.ceil(n / bs) * cfg.ce_epochs
        self.model.to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.ce_lr, weight_decay=0.01)
        sched = get_linear_schedule_with_warmup(opt, max(1, int(0.05 * steps)), steps)
        self.model.train()
        t0, step, run_loss = time.time(), 0, 0.0
        pb = bar(desc=f"cross-encoder training ({n:,} pairs)", total=steps, unit="step")
        for _ in range(cfg.ce_epochs):
            order = rng.permutation(n)
            for c0 in range(0, n, chunk):
                idx = order[c0:c0 + chunk]
                enc = self._tokenize_chunk(ta, tb, idx)
                for s in range(0, len(idx), bs):
                    rows = np.arange(s, min(s + bs, len(idx)))
                    batch = self._to_batch(enc, rows)
                    yb = torch.tensor(y[idx[rows]], dtype=torch.float32, device=self.device)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        self._logits(batch), yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    opt.step()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
                    step += 1
                    run_loss = 0.98 * run_loss + 0.02 * float(loss) if step > 1 else float(loss)
                    pb.update(1)
                    if step % 50 == 0:
                        pb.set_postfix_str(f"loss {run_loss:.4f}", refresh=False)
        pb.close()
        del opt, sched
        self.log(f"      cross-encoder trained: {steps:,} steps, loss {run_loss:.4f} "
                 f"({(time.time() - t0) / 60:.1f} min)")
        return self

    def predict(self, ta, tb, chunk=131_072):
        torch = self.torch
        self.model.to(self.device)
        self.model.eval()
        n = len(ta)
        out = np.empty(n, dtype=np.float32)
        t0 = time.time()
        pb = bar(desc=f"cross-encoder scoring ({n:,} pairs)", total=n, unit="pair")
        with torch.inference_mode():
            for c0 in range(0, n, chunk):
                idx = np.arange(c0, min(n, c0 + chunk))
                enc = self._tokenize_chunk(ta, tb, idx)
                lens = np.array([len(x) for x in enc["input_ids"]])
                order = np.argsort(lens, kind="stable")
                for s in range(0, len(idx), self.cfg.ce_infer_batch):
                    rows = order[s:s + self.cfg.ce_infer_batch]
                    out[idx[rows]] = torch.sigmoid(self._logits(self._to_batch(enc, rows))).cpu().numpy()
                    pb.update(len(rows))
        pb.close()
        if n > 500_000:
            self.log(f"      cross-encoder scored {n:,} pairs in {(time.time() - t0) / 60:.1f} min")
        return out

    def offload(self):
        """Moves the model to host RAM and returns PyTorch's cached GPU memory, so XGBoost
        (stage 2) gets the whole GPU; fit/predict move it back to the GPU when needed."""
        self.model.to("cpu")
        gc.collect()
        if self.amp:
            self.torch.cuda.empty_cache()

    def save(self, path):
        self.model.save_pretrained(path)
        self.tok.save_pretrained(path)

    def release(self):
        del self.model
        gc.collect()
        if self.amp:
            self.torch.cuda.empty_cache()


def cross_fit_train(split_ctx, i1, i2, y, sel, half, cfg, device, log=print):
    """Out-of-fold cross-encoder scores for the selected train pairs (NaN elsewhere) and
    the two fitted models (kept for scoring the test set)."""
    ce = np.full(len(i1), np.nan, dtype=np.float32)
    texts = record_texts(split_ctx, np.unique(np.concatenate([i1[sel | (y > 0)],
                                                              i2[sel | (y > 0)]])))
    rng = np.random.RandomState(0)
    models = []
    for h in (0, 1):
        fit_idx = np.where((half == h) & (sel | (y > 0)))[0]
        if len(fit_idx) > cfg.ce_max_train_pairs:
            fit_idx = np.sort(rng.choice(fit_idx, cfg.ce_max_train_pairs, replace=False))
        log(f"    cross-encoder half {h}: fitting on {len(fit_idx):,} pairs "
            f"({int(y[fit_idx].sum()):,} positives)")
        m = CrossEncoder(cfg, device, log).fit([texts[int(i1[k])] for k in fit_idx],
                                               [texts[int(i2[k])] for k in fit_idx],
                                               y[fit_idx].astype(np.float32), seed=42 + h)
        sc_idx = np.where((half != h) & sel)[0]
        ce[sc_idx] = m.predict([texts[int(i1[k])] for k in sc_idx],
                               [texts[int(i2[k])] for k in sc_idx])
        m.offload()
        models.append(m)
    return ce, models


def score_test(models, split_ctx, i1, i2, sel):
    ce = np.full(len(i1), np.nan, dtype=np.float32)
    idx = np.where(sel)[0]
    texts = record_texts(split_ctx, np.unique(np.concatenate([i1[idx], i2[idx]])))
    ta = [texts[int(i1[k])] for k in idx]
    tb = [texts[int(i2[k])] for k in idx]
    scores = []
    for m in models:
        scores.append(m.predict(ta, tb))
        m.offload()
    ce[idx] = np.mean(scores, axis=0)
    return ce


def top_up(models, split_ctx, i1, i2, ce, need, half=None):
    """Scores the pairs in `need` that have no cross-encoder score yet (cached models reused
    after stage 1 changed its selection). half: cross-fitting half of each pair (training
    entities -> scored by the OTHER half's model); None = average of both models."""
    todo = np.where(need & np.isnan(ce))[0]
    if len(todo) == 0:
        return ce, 0
    texts = record_texts(split_ctx, np.unique(np.concatenate([i1[todo], i2[todo]])))
    if half is None:
        groups = [(todo, models)]
    else:
        groups = [(todo[half[todo] == h], [models[1 - h]]) for h in (0, 1)]
    for rows, ms in groups:
        if len(rows) == 0:
            continue
        ta = [texts[int(i1[k])] for k in rows]
        tb = [texts[int(i2[k])] for k in rows]
        sc = []
        for m in ms:
            sc.append(m.predict(ta, tb))
            m.offload()
        ce[rows] = np.mean(sc, axis=0)
    return ce, len(todo)


def save_models(models, ce, path):
    os.makedirs(path, exist_ok=True)
    for h, m in enumerate(models):
        m.save(os.path.join(path, f"model_{h}"))
    np.save(os.path.join(path, "ce.npy"), ce)
    open(os.path.join(path, "DONE"), "w").close()


def load_models(cfg, device, path, log=print):
    """(train ce scores, models) saved by save_models, or None when there is no complete cache."""
    if not os.path.exists(os.path.join(path, "DONE")):
        return None
    models = []
    for h in (0, 1):
        m = CrossEncoder(cfg, device, log, path=os.path.join(path, f"model_{h}"))
        m.offload()
        models.append(m)
    return np.load(os.path.join(path, "ce.npy")), models
