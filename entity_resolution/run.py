"""End-to-end pipeline:  data -> blocking (keys + GPU embeddings) -> features -> boosters
(+ cross-encoder) -> decision -> output.

Run from the repository root, e.g.
    python -m entity_resolution.run --data-dir data --out-dir output --cross-encoder
    python -m entity_resolution.run --data-dir data --out-dir output --dev 100000   # quick check
or use scripts/run_pipeline.py, which also checks the GPU / environment first.
"""
import argparse
import dataclasses
import gc
import json
import hashlib
import os
import shutil
import time

if __name__ == "__main__" and not __package__:
    import sys
    sys.exit("run as a module from the repository root: python -m entity_resolution.run "
             "--data-dir ... --out-dir ...  (or python scripts/run_pipeline.py)")

import numpy as np
import pandas as pd

from .blocking import blocking_report, generate_candidates, record_keys
from .emb_blocking import load_embeddings
from .config import Config
from .decide import Evaluator, apply_decision, one_to_one_mask, tune_decision
from .features import global_context, prepare_context, run_chunks, stage2_features
from .io_utils import fetch_raw, read_ground_truth, read_split, run_official_validator
from .metrics import fbeta_row
from .decoy import DECOY_VERSION, learn_decoy_words, pair_context, record_context
from .decide import _logit_shift
from .model import FeatureMatrix, Predictor, make_folds, train_row_sample, resolve_backend, train_stage
from .normalize import NORM_VERSION, TRANSLIT_MAP, build_profiles, learn_translit_map
from .progress import bar

T0 = time.time()


def log(msg):
    print(f"[{(time.time() - T0) / 60:6.1f} min] {msg}", flush=True)


def hardware():
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    except (ValueError, OSError, AttributeError):
        ram = float("nan")
    return os.cpu_count() or 1, ram


def load_profiles(data_dir, split, work_dir, n_jobs):
    path = os.path.join(work_dir, f"{split}_profiles_v{NORM_VERSION}.pkl")
    if os.path.exists(path):
        log(f"{split}: normalised profiles from cache {path}")
        return pd.read_pickle(path)
    log(f"{split}: reading TSVs")
    rec = read_split(data_dir, split)
    log(f"{split}: normalising {len(rec):,} records with {n_jobs} processes")
    prof = build_profiles(rec, n_jobs=n_jobs)
    del rec
    prof.to_pickle(path)
    return prof


def load_translit_map(data_dir, work_dir, log=print):
    """Learned once from the TRAINING ground truth, then reused for train and test."""
    path = os.path.join(work_dir, "translit_map.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            tmap = json.load(f)
        log(f"transliteration map from cache ({len(tmap):,} tokens)")
    else:
        log("learning transliteration map from training pairs")
        rec = read_split(data_dir, "train")
        s1_ids = rec.loc[rec["src"] == 1, "entity_id"].tolist()
        tmap = learn_translit_map(rec, read_ground_truth(data_dir, s1_ids), log=log)
        del rec
        with open(path, "w", encoding="utf-8") as f:
            json.dump(tmap, f, ensure_ascii=False, indent=0)
    TRANSLIT_MAP.clear()
    TRANSLIT_MAP.update(tmap)


def candidate_pairs(prof, s1_rows, cfg, n_jobs, work_dir, tag):
    sig = (f"v4norm{NORM_VERSION}_{tag}_n{len(s1_rows)}_k{cfg.k_forward}-{cfg.k_name}-{cfg.k_addr}_r{cfg.k_reverse}_m{int(cfg.max_key_pairs)}"
           f"_s{cfg.min_block_score}_c{int(cfg.block_within_country)}"
           + (f"_e{cfg.k_emb}-{cfg.min_emb_score}" if cfg.use_emb else ""))
    path = os.path.join(work_dir, f"pairs_{sig}.npz")
    if os.path.exists(path):
        log(f"    candidate pairs from cache {path}")
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    emb = None
    if cfg.use_emb:
        split = "test" if tag.startswith("test") else "train"
        emb = load_embeddings(prof, cfg, cfg.device, work_dir, split, NORM_VERSION, log=log)
    pairs = generate_candidates(prof, s1_rows, cfg, n_jobs=n_jobs, log=log, emb=emb)
    del emb
    np.savez(path, **pairs)
    return pairs


# ----------------------------------------------------------------------------- checkpoints
# Every long step saves its result under work/ckpt_<hash of the settings it depends on>/, so a
# run that crashes restarts from the last finished step (delete the folder to force a redo).
def ckpt_dir(work, *parts):
    h = hashlib.md5(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:10]
    return os.path.join(work, f"ckpt_{h}")


def ckpt_done(path):
    return os.path.exists(os.path.join(path, "DONE"))


def ckpt_mark(path):
    open(os.path.join(path, "DONE"), "w").close()


def save_arrays(path, arrays, log=print):
    """np.save each array into path/ (skipped if the disk has < 2x their size free)."""
    size = sum(v.nbytes for v in arrays.values())
    os.makedirs(path, exist_ok=True)
    if shutil.disk_usage(path).free < 2 * size:
        log(f"    checkpoint skipped: {size / 2 ** 30:.0f} GB needed, disk too full ({path})")
        return
    for k, v in arrays.items():
        np.save(os.path.join(path, f"{k}.npy"), v)
    ckpt_mark(path)
    log(f"    checkpoint saved: {os.path.basename(path)} ({size / 2 ** 30:.1f} GB)")


def load_arrays(path):
    return {f[:-4]: np.load(os.path.join(path, f)) for f in sorted(os.listdir(path))
            if f.endswith(".npy")}


def add_decoy(sctx, pairs, gctx, path, tag):
    """Adds the label-free decoy features (decoy.py) to gctx and the decoy-word table to sctx
    (used by the pair-feature workers). path: checkpoint folder (None = no checkpoint)."""
    record_context(sctx, log=log)
    if path and ckpt_done(path):
        gctx.update(load_arrays(path))
        with open(os.path.join(path, "decoy_words.json")) as f:
            sctx["decoy_words"] = json.load(f)
        log(f"{tag}: decoy features from checkpoint")
        return
    log(f"{tag}: label-free decoy features")
    dctx = pair_context(sctx, pairs, gctx, log=log)
    table = learn_decoy_words(sctx, pairs, gctx, log=log)
    gctx.update(dctx)
    sctx["decoy_words"] = table
    if path:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "decoy_words.json"), "w") as f:
            json.dump(table, f)
        save_arrays(path, dctx, log=log)


def calibrate_unseen(code, pairs, src, p, params, cfg, countries, s1_rows, sel, unseen):
    """Selection where each unseen country (no training labels, e.g. France) gets a logit
    bias on its probabilities so its matched-S1 share equals that of the labelled countries."""
    c_s1 = countries[s1_rows]
    c_pair = c_s1[code]
    matched = np.zeros(len(s1_rows), bool)
    matched[code[sel]] = True
    seen = ~np.isin(c_s1, unseen)
    target = matched[seen].mean()
    out = sel.copy()
    for c in unseen:
        rows = np.where(c_pair == c)[0]
        n_c = int((c_s1 == c).sum())
        codes_c = code[rows]

        def share(bias):
            s = apply_decision(codes_c, pairs["i2"][rows], src[pairs["i2"][rows]],
                               _logit_shift(p[rows], bias).astype(np.float32), params, cfg)
            return len(np.unique(codes_c[s])) / max(1, n_c), s

        lo, hi = -4.0, 4.0
        base, _ = share(0.0)
        for _ in range(14):                     # matched share rises with the bias
            mid = 0.5 * (lo + hi)
            if share(mid)[0] > target:
                hi = mid
            else:
                lo = mid
        b = 0.5 * (lo + hi)
        sh, s = share(b)
        out[rows] = s
        log(f"calibrated {c}: matched share {base:.4f} -> {sh:.4f} (labelled countries "
            f"{target:.4f}) with logit bias {b:+.3f} -> matching_results_calibrated.tsv")
    return out


def label_pairs(pairs, truth, ids):
    s_list = [s for s, ms in truth.items() for _ in ms]
    m_list = [m for ms in truth.values() for m in ms]
    index = pd.Index(ids)
    si, mi = index.get_indexer(s_list), index.get_indexer(m_list)
    ok = (si >= 0) & (mi >= 0)
    n = len(ids)
    gt_keys = si[ok].astype(np.int64) * n + mi[ok]
    return np.isin(pairs["i1"] * n + pairs["i2"], gt_keys)


def write_sorted_lists(path, s1_rows, pair_i1, pair_other_ids, ids, col_name):
    """pair_i1 sorted ascending; one output line per S1 row, in s1_rows order."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    lo = np.searchsorted(pair_i1, s1_rows, side="left")
    hi = np.searchsorted(pair_i1, s1_rows, side="right")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"source1_entity_id\t{col_name}\n")
        for r, a, b in bar(zip(s1_rows, lo, hi), f"write {os.path.basename(path)}",
                           total=len(s1_rows), unit="S1"):
            f.write(ids[r] + "\t" + ",".join(dict.fromkeys(pair_other_ids[a:b])) + "\n")


def dump_blocking_misses(path, data_dir, missed, prof, limit=3000, log=print):
    """Sample of true pairs that blocking did NOT find, with the reason signals needed to fix
    blocking: same country label? how many blocking keys do the two records share?"""
    missed = sorted(missed)
    if not missed:
        return {}
    rng = np.random.RandomState(0)
    pick = [missed[i] for i in rng.choice(len(missed), min(limit, len(missed)), replace=False)]
    index = pd.Index(prof["entity_id"].values)
    core, addr, alias = prof["name_core"].values, prof["addr_norm"].values, prof["name_alias"].values
    post, ck = prof["addr_postal"].values, prof["country_key"].values
    raw = fetch_raw(data_dir, "train", {x for pr in pick for x in pr})
    rows = []
    for s, m in pick:
        a, b = index.get_indexer([s, m])
        row = {"source1_entity_id": s, "match_id": m}
        for tag, x in (("s1", s), ("match", m)):
            row[f"{tag}_name"] = raw.at[x, "business_name"] if x in raw.index else ""
            row[f"{tag}_address"] = raw.at[x, "business_address"] if x in raw.index else ""
            row[f"{tag}_country"] = raw.at[x, "country"] if x in raw.index else ""
        if a < 0 or b < 0:
            row.update(same_country="", shared_keys=-1, shared_examples="ID NOT IN SOURCE FILES")
        else:
            ka = set(record_keys(core[a], alias[a], addr[a], post[a]))
            kb = set(record_keys(core[b], alias[b], addr[b], post[b]))
            sh = sorted(ka & kb)
            row.update(same_country=bool(ck[a] == ck[b]), shared_keys=len(sh),
                       shared_examples=" ".join(sh[:10]),
                       s1_normalised=f"{core[a]} ; {addr[a]}",
                       match_normalised=f"{core[b]} ; {addr[b]}")
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(path, sep="\t", index=False)
    sc = df["same_country"].astype(str)
    summ = {"missed_pairs_total": len(missed), "sampled": len(df),
            "share_different_country": float((sc == "False").mean()),
            "share_no_shared_key": float((df["shared_keys"] == 0).mean()),
            "share_id_missing": float((df["shared_keys"] < 0).mean())}
    log(f"    blocking misses: {summ}")
    return summ


def dump_errors(path, data_dir, sel, code, i2, p, y, samp_ids, truth, ids, per, limit=5000):
    bad = np.where(per < 1.0)[0]
    bad = bad[np.argsort(per[bad], kind="stable")][:limit]
    rows_by_code = {}
    for r in np.where(np.isin(code, bad))[0]:
        rows_by_code.setdefault(code[r], []).append(r)
    need = set(samp_ids[bad])
    for c in bar(bad, "collect error ids", unit="S1", leave=False):
        need |= truth.get(samp_ids[c], set())
        need |= {ids[i2[r]] for r in rows_by_code.get(c, [])}
    raw = fetch_raw(data_dir, "train", need)

    def fmt(x, prob):
        if x in raw.index:
            return f"{x} | {raw.at[x, 'business_name']} | {raw.at[x, 'business_address']} | p={prob}"
        return f"{x} | p={prob}"

    out = []
    for c in bad:
        s = samp_ids[c]
        rws = rows_by_code.get(c, [])
        pmap = {ids[i2[r]]: f"{p[r]:.3f}" for r in rws}
        pred = {ids[i2[r]] for r in rws if sel[r]}
        true = truth.get(s, set())
        if not true:
            kind = "singleton_false_merge"
        elif not pred:
            kind = "missed_all"
        elif not (pred & true):
            kind = "wrong_match"
        elif pred - true:
            kind = "extra_match"
        else:
            kind = "partial_miss"
        out.append({
            "loss": 1.0 - fbeta_row(pred, true), "type": kind, "source1_entity_id": s,
            "name": raw.at[s, "business_name"] if s in raw.index else "",
            "address": raw.at[s, "business_address"] if s in raw.index else "",
            "country": raw.at[s, "country"] if s in raw.index else "",
            "false_pos": " || ".join(fmt(x, pmap.get(x, "-")) for x in sorted(pred - true)),
            "false_neg": " || ".join(fmt(x, pmap.get(x, "NOT IN CANDIDATES"))
                                     for x in sorted(true - pred)),
            "true_pos": " || ".join(fmt(x, pmap.get(x, "-")) for x in sorted(pred & true)),
        })
    err = pd.DataFrame(out)
    err.to_csv(path, sep="\t", index=False)
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="folder with train/ and test/")
    ap.add_argument("--out-dir", required=True, help="where the two submission TSVs go")
    ap.add_argument("--work-dir", default=None, help="cache + artifacts (default: <out-dir>/../work)")
    ap.add_argument("--n-jobs", type=int, default=0)
    ap.add_argument("--dev", type=int, default=0,
                    help="quick mode: use only N train S1 entities and skip the test set")
    ap.add_argument("--train-sample", type=int, default=None)
    ap.add_argument("--seeds", default=None, help="comma separated, e.g. 42,7,2026")
    ap.add_argument("--folds", type=int, default=None, help="CV folds for the boosters (default 3)")
    ap.add_argument("--max-train-rows", type=int, default=None,
                    help="GPU memory budget: rows per booster training (0 = all)")
    ap.add_argument("--k-forward", type=int, default=None)
    ap.add_argument("--no-country-block", action="store_true")
    ap.add_argument("--k-reverse", type=int, default=None)
    ap.add_argument("--k-name", type=int, default=None)
    ap.add_argument("--k-addr", type=int, default=None)
    ap.add_argument("--max-key-pairs", type=float, default=None)
    ap.add_argument("--min-block-score", type=float, default=None)
    ap.add_argument("--blocking-only", action="store_true",
                    help="stop after the blocking report + blocking_misses.tsv (fast tuning loop)")
    ap.add_argument("--device", default=None, choices=["auto", "cuda", "cpu"])
    ap.add_argument("--backend", default=None, choices=["auto", "xgb", "lgbm"])
    ap.add_argument("--holdout", type=float, default=None,
                    help="share of train S1 entities kept out of all training for the reported "
                         "score (default 0.10; 0 = out-of-fold evaluation on everything)")
    ap.add_argument("--ce-max-len", type=int, default=None)
    ap.add_argument("--ce-topn", type=int, default=None)
    ap.add_argument("--ce-epochs", type=int, default=None)
    ap.add_argument("--ce-batch", type=int, default=None)
    ap.add_argument("--ce-lr", type=float, default=None)
    ap.add_argument("--ce-max-train-pairs", type=int, default=None)
    ap.add_argument("--no-emb", action="store_true", help="skip the GPU embedding passes")
    ap.add_argument("--no-decoy", action="store_true", help="skip the label-free decoy features")
    ap.add_argument("--no-calibrate", action="store_true",
                    help="do not write matching_results_calibrated.tsv for unseen countries")
    ap.add_argument("--k-emb", type=int, default=None)
    ap.add_argument("--cross-encoder", action="store_true",
                    help="fine-tune the multilingual cross-encoder on the GPU (used in the final run)")
    ap.add_argument("--ce-model", default=None, help="HF model id or local folder")
    ap.add_argument("--validator", default=None,
                    help="path to utils/validate_submission.py (default: <data-dir>/../utils/...)")
    args = ap.parse_args()

    cfg = Config()
    if args.seeds:
        cfg.seeds = tuple(int(x) for x in args.seeds.split(","))
    if args.train_sample is not None:
        cfg.train_sample_s1 = args.train_sample
    if args.folds:
        cfg.n_folds = args.folds
    if args.max_train_rows is not None:
        cfg.max_train_rows = args.max_train_rows
    if args.k_forward:
        cfg.k_forward = args.k_forward
    if args.no_country_block:
        cfg.block_within_country = False
    if args.k_reverse is not None:
        cfg.k_reverse = args.k_reverse
    if args.k_name is not None:
        cfg.k_name = args.k_name
    if args.k_addr is not None:
        cfg.k_addr = args.k_addr
    if args.max_key_pairs is not None:
        cfg.max_key_pairs = args.max_key_pairs
    if args.min_block_score is not None:
        cfg.min_block_score = args.min_block_score
    if args.device:
        cfg.device = args.device
    if args.backend:
        cfg.backend = args.backend
    if args.ce_model:
        cfg.ce_model = args.ce_model
    if args.holdout is not None:
        cfg.holdout_frac = args.holdout
    for a_, c_ in (("ce_max_len", "ce_max_len"), ("ce_topn", "ce_topn"), ("ce_epochs", "ce_epochs"),
                   ("ce_batch", "ce_batch"), ("ce_lr", "ce_lr"),
                   ("ce_max_train_pairs", "ce_max_train_pairs")):
        if getattr(args, a_) is not None:
            setattr(cfg, c_, getattr(args, a_))
    if args.no_emb:
        cfg.use_emb = False
    if args.no_decoy:
        cfg.use_decoy = False
    if args.no_calibrate:
        cfg.calibrate_unseen = False
    if args.k_emb is not None:
        cfg.k_emb = args.k_emb
    ncpu, ram = hardware()
    n_jobs = args.n_jobs or cfg.n_jobs or ncpu
    work = args.work_dir or os.path.join(os.path.dirname(os.path.abspath(args.out_dir)), "work")
    art = os.path.join(work, "artifacts")
    os.makedirs(art, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    log(f"machine: {ncpu} CPUs, {ram:.0f} GB RAM | using {n_jobs} processes | work dir {work}")
    resolve_backend(cfg, log=log)
    if args.cross_encoder and cfg.device != "cuda" and args.device != "cpu":
        log("WARNING: --cross-encoder needs a working CUDA PyTorch (torch.cuda.is_available() is "
            "False) -> cross-encoder skipped. Use --device cpu to force it (very slow).")
        args.cross_encoder = False
    report = {"config": {k: (list(v) if isinstance(v, tuple) else v)
                         for k, v in cfg.__dict__.items()},
              "machine": {"cpus": ncpu, "ram_gb": ram}, "dev": args.dev}
    rng = np.random.RandomState(0)

    # ================================================================= TRAIN
    load_translit_map(args.data_dir, work, log=log)
    prof = load_profiles(args.data_dir, "train", work, n_jobs)
    ids = prof["entity_id"].values
    src = prof["src"].values
    s1_all = np.where(src == 1)[0]
    log(f"train: {len(s1_all):,} S1, {int((src == 2).sum()):,} S2, {int((src == 3).sum()):,} S3")
    truth = read_ground_truth(args.data_dir, list(ids[s1_all]))
    s1_query = np.sort(rng.choice(s1_all, size=args.dev, replace=False)) \
        if args.dev and args.dev < len(s1_all) else s1_all
    log(f"train: blocking for {len(s1_query):,} S1 entities")
    dev_tag = "train" + (f"dev{args.dev}" if args.dev else "")
    pairs = candidate_pairs(prof, s1_query, cfg, n_jobs, work, dev_tag)
    y_all = label_pairs(pairs, truth, ids)
    n_all = len(pairs["i1"])
    ck_root = ckpt_dir(work, dev_tag, n_all, NORM_VERSION, cfg.use_emb, cfg.k_emb, cfg.k_forward,
                       cfg.k_name, cfg.k_addr, cfg.k_reverse, cfg.max_key_pairs)
    os.makedirs(ck_root, exist_ok=True)
    log(f"checkpoints: {ck_root}")
    brep_path = os.path.join(ck_root, "blocking_report.json")
    if os.path.exists(brep_path):
        with open(brep_path) as f:
            brep, missed = json.load(f), None
        log("train: blocking report from checkpoint")
    else:
        log("train: blocking report (recall per pass / depth; can take a few minutes)")
        brep, missed = blocking_report(pairs, truth, list(ids[s1_query]), ids,
                                       int((src != 1).sum()), src, cfg.k_forward)
        with open(brep_path, "w") as f:
            json.dump(brep, f, default=float)
    report["blocking_train"] = brep
    log(f"train blocking: pair recall {brep['pair_recall']:.4f} (forward only "
        f"{brep['pair_recall_forward_only']:.4f}) | oracle macro-F0.5 ceiling "
        f"{brep['oracle_macro_f05']:.4f} | {brep['avg_candidates_per_s1']:.1f} candidates/S1")
    log("    recall by pass: " + ", ".join(f"{k[12:]}={v:.4f}" for k, v in brep.items()
                                            if k.startswith("recall_pass_")))
    log("    recall by forward depth: " + ", ".join(
        f"top{k.split('top')[1].split('_')[0]}={v:.4f}" for k, v in brep.items()
        if k.startswith("recall_forward_top")))
    if missed is not None:
        report["blocking_misses"] = dump_blocking_misses(
            os.path.join(art, "blocking_misses.tsv"), args.data_dir, missed, prof, log=log)
    if args.blocking_only:
        with open(os.path.join(art, "report.json"), "w") as f:
            json.dump(report, f, indent=2, default=float)
        log(f"blocking-only run finished. See {art}/blocking_misses.tsv and report.json")
        return

    sctx = prepare_context(prof, n_jobs, log=log)
    gctx_dir = os.path.join(ck_root, "train_gctx")
    if ckpt_done(gctx_dir):
        gctx = load_arrays(gctx_dir)
        log(f"train: global context from checkpoint ({len(gctx)} features)")
    else:
        gctx = global_context(sctx, pairs, log=log)
        save_arrays(gctx_dir, gctx, log=log)
    if cfg.use_decoy:
        add_decoy(sctx, pairs, gctx, os.path.join(ck_root, f"train_decoy_v{DECOY_VERSION}"),
                  "train")
    val_ent = np.zeros(len(s1_query), dtype=bool)        # holdout entities (index into s1_query)
    pool = s1_query
    if cfg.holdout_frac > 0:
        n_val = int(round(cfg.holdout_frac * len(s1_query)))
        val_ent[np.random.RandomState(2026).choice(len(s1_query), n_val, replace=False)] = True
        pool = s1_query[~val_ent]
        log(f"HOLDOUT: {int(val_ent.sum()):,} S1 entities ({cfg.holdout_frac:.0%}) kept out of ALL "
            f"training; models train on the other {len(pool):,}")
    n_samp = cfg.train_sample_s1 or len(pool)
    samp = np.sort(rng.choice(pool, size=min(n_samp, len(pool)), replace=False))
    in_samp = np.isin(pairs["i1"], samp)
    idx_s, idx_o = np.where(in_samp)[0], np.where(~in_samp)[0]
    sp = {k: v[idx_s] for k, v in pairs.items()}
    y_all = y_all.astype(int)
    y = y_all[idx_s]
    folds, fid = make_folds(sp["i1"], cfg.n_folds)
    keep, weight = (None, None)
    if cfg.max_train_rows and len(idx_s) > cfg.max_train_rows:
        keep, weight = train_row_sample(y, gctx["qn"][idx_s].astype(np.float32) >= cfg.hard_neg_qn,
                                        cfg.max_train_rows, log=log, tag="train")
    s1_dir = os.path.join(ckpt_dir(ck_root, "stage1", len(samp), cfg.holdout_frac, cfg.seeds,
                                   cfg.backend, cfg.xgb_params, cfg.xgb_max_bin, cfg.lgb_params,
                                   cfg.n_folds, cfg.monotone, cfg.num_boost_round,
                                   cfg.early_stopping_rounds, cfg.use_decoy and DECOY_VERSION,
                                   cfg.max_train_rows, cfg.hard_neg_qn,
                                   "features_v2"), "stage1")
    if ckpt_done(s1_dir):
        log(f"train: stage 1 from checkpoint {s1_dir}")
        p1 = np.load(os.path.join(s1_dir, "p1.npy"))
        pred1 = Predictor.load(os.path.join(s1_dir, "model"), cfg.device)
        imp1 = pd.read_csv(os.path.join(s1_dir, "importance.tsv"), sep="\t", index_col=0).iloc[:, 0]
    else:
        log(f"train: features for {len(samp):,} training S1 entities ({len(idx_s):,} pairs, "
            f"{y.sum():,} positives)")
        X, _ = run_chunks(sctx, sp, {k: v[idx_s] for k, v in gctx.items()}, cfg, n_jobs,
                          keep=True, log=log)
        log(f"train: stage 1 ({cfg.backend}) on {X.shape[1]} features, {cfg.n_folds}-fold "
            f"grouped CV")
        if cfg.backend == "xgb":                  # one float32 copy only (saves ~40 GB)
            X = FeatureMatrix(X)
        # the global context (30-40 GB) is not needed while the boosters train: drop it and
        # reload it from its checkpoint afterwards (less RAM = less risk of the OOM killer)
        dec_dir = os.path.join(ck_root, f"train_decoy_v{DECOY_VERSION}")
        can_reload = ckpt_done(gctx_dir) and (not cfg.use_decoy or ckpt_done(dec_dir))
        if can_reload:
            del gctx
        gc.collect()
        p1_s, pred1, imp1 = train_stage(X, y, folds, cfg, log=log, tag="stage1",
                                        ckpt=os.path.join(s1_dir, "folds"), keep=keep,
                                        weight=weight)
        del X
        gc.collect()
        if can_reload:
            gctx = load_arrays(gctx_dir)
            if cfg.use_decoy:
                gctx.update(load_arrays(dec_dir))
            log("    global context reloaded from checkpoint")

        # Held-out stage-1 probability for EVERY train pair: out-of-fold for the training
        # entities, predictions of the sample-trained model for all other entities (never
        # trained on). This gives stage 2, one-to-one assignment and the evaluation the full
        # competition between S1 entities that the test set will have.
        p1 = np.empty(n_all, dtype=np.float32)
        p1[idx_s] = p1_s
        if len(idx_o):
            log(f"train: stage-1 scoring of {len(idx_o):,} pairs of the other S1 entities")
            _, p1[idx_o] = run_chunks(sctx, {k: v[idx_o] for k, v in pairs.items()},
                                      {k: v[idx_o] for k, v in gctx.items()}, cfg, n_jobs,
                                      predict_fn=pred1.predict, keep=False, log=log)
        os.makedirs(s1_dir, exist_ok=True)
        np.save(os.path.join(s1_dir, "p1.npy"), p1)
        pred1.save(os.path.join(s1_dir, "model"))
        imp1.rename("gain").to_csv(os.path.join(s1_dir, "importance.tsv"), sep="\t")
        ckpt_mark(s1_dir)
        log(f"    checkpoint saved: stage 1 ({s1_dir})")
    src2 = src[pairs["i2"]]

    ce, ce_models, ce_key = None, None, None
    if args.cross_encoder:
        from .cross_encoder import (cross_fit_train, load_models, save_models, score_test,
                                    select_for_ce, top_up)
        # cache: a rerun after a later crash skips the (hours-long) cross-encoder training
        # the cross-encoder reads only text, so its cache does not depend on stage 1: after a
        # stage-1 change the models are reused and only newly selected pairs get scored
        ce_key = (cfg.ce_model, cfg.ce_max_len, cfg.ce_topn, cfg.ce_min_p, cfg.ce_max_train_pairs,
                  cfg.ce_epochs, cfg.ce_lr, cfg.ce_batch)
        ce_dir = os.path.join(ckpt_dir(ck_root, "ce", len(samp), cfg.holdout_frac, *ce_key),
                              "cross_encoder")
        cached = load_models(cfg, cfg.device, ce_dir, log=log)
        if cached is not None:
            ce, ce_models = cached
            log(f"train: cross-encoder scores + models from cache {ce_dir}")
            sel_ce = select_for_ce(pairs["i1"], p1, cfg.ce_topn, cfg.ce_min_p)
            half = np.searchsorted(samp, sp["i1"]) % 2
            ce_s, n_s = top_up(ce_models, sctx, sp["i1"], sp["i2"], ce[idx_s], sel_ce[idx_s], half)
            ce[idx_s] = ce_s
            ce_o, n_o = top_up(ce_models, sctx, pairs["i1"][idx_o], pairs["i2"][idx_o],
                               ce[idx_o], sel_ce[idx_o])
            ce[idx_o] = ce_o
            log(f"    cross-encoder top-up: {n_s + n_o:,} newly selected pairs scored")
            if n_s + n_o:
                np.save(os.path.join(ce_dir, "ce.npy"), ce)    # next resume needs no top-up
        else:
            sel_ce = select_for_ce(pairs["i1"], p1, cfg.ce_topn, cfg.ce_min_p)
            log(f"train: cross-encoder {cfg.ce_model} on {cfg.device}, {int(sel_ce.sum()):,} pairs "
                f"selected (cross-fitted on 2 halves of the training entities)")
            ce = np.full(n_all, np.nan, dtype=np.float32)
            half = np.searchsorted(samp, sp["i1"]) % 2
            ce[idx_s], ce_models = cross_fit_train(sctx, sp["i1"], sp["i2"], y, sel_ce[idx_s],
                                                   half, cfg, cfg.device, log=log)
            if len(idx_o):
                ce[idx_o] = score_test(ce_models, sctx, pairs["i1"][idx_o], pairs["i2"][idx_o],
                                       sel_ce[idx_o])
            save_models(ce_models, ce, ce_dir)
        # the models stay on the CPU until the test set; free PyTorch's GPU cache for XGBoost
        for m_ in ce_models:
            m_.offload()

    s2_dir = os.path.join(ckpt_dir(s1_dir, "stage2", ce_key if ce is not None else "no-ce"),
                          "stage2")
    if ckpt_done(s2_dir):
        log(f"train: stage 2 from checkpoint {s2_dir}")
        p2 = np.load(os.path.join(s2_dir, "p2.npy"))
        pred2 = Predictor.load(os.path.join(s2_dir, "model"), cfg.device)
        imp2 = pd.read_csv(os.path.join(s2_dir, "importance.tsv"), sep="\t", index_col=0).iloc[:, 0]
        del gctx
    else:
        log("train: stage 2 (stacking on stage-1 probability" +
            (" + cross-encoder" if ce is not None else "") + ", full competition)")
        X2 = stage2_features(pairs["i1"], pairs["i2"], src2, p1, gctx, ce)
        del gctx
        p2_s, pred2, imp2 = train_stage(X2.iloc[idx_s].reset_index(drop=True), y, folds, cfg,
                                        log=log, tag="stage2", ckpt=os.path.join(s2_dir, "folds"),
                                        keep=keep, weight=weight)
        p2 = np.empty(n_all, dtype=np.float32)
        p2[idx_s] = p2_s
        if len(idx_o):
            p2[idx_o] = pred2.predict(X2.iloc[idx_o])
        del X2
        gc.collect()
        os.makedirs(s2_dir, exist_ok=True)
        np.save(os.path.join(s2_dir, "p2.npy"), p2)
        pred2.save(os.path.join(s2_dir, "model"))
        imp2.rename("gain").to_csv(os.path.join(s2_dir, "importance.tsv"), sep="\t")
        ckpt_mark(s2_dir)
        log(f"    checkpoint saved: stage 2 ({s2_dir})")
    pd.concat([imp1.rename("stage1_gain"), imp2.rename("stage2_gain")], axis=1).to_csv(
        os.path.join(art, "feature_importance.tsv"), sep="\t")

    # evaluation over ALL queried S1 entities: every probability used here is held-out
    code = np.searchsorted(s1_query, pairs["i1"])
    ntrue = np.array([len(truth.get(s, ())) for s in ids[s1_query]])
    variants = {"stage1": p1, "stage2": p2, "blend": 0.5 * (p1 + p2)}
    tune_ent = ~val_ent                                   # holdout never used for any choice
    tune_code = np.cumsum(tune_ent) - 1
    no121 = dataclasses.replace(cfg, one_to_one=False)
    best_name, params = None, None
    for name, pv in variants.items():
        log(f"decision tuning on {name} held-out probabilities"
            + (" (training entities only)" if val_ent.any() else ""))
        if val_ent.any():
            # one-to-one over ALL pairs (full competition), then tune on training entities
            rows = np.where((one_to_one_mask(pairs["i2"], pv) if cfg.one_to_one else True)
                            & tune_ent[code])[0]
            prm, rep = tune_decision(tune_code[code[rows]], pairs["i2"][rows], src2[rows], pv[rows],
                                     y_all[rows], ntrue[tune_ent], no121, log=log)
        else:
            prm, rep = tune_decision(code, pairs["i2"], src2, pv, y_all, ntrue, cfg, log=log)
        report[f"decision_search_{name}"] = rep
        if params is None or prm["score"] > params["score"] + 1e-9:
            best_name, params = name, prm
    params["probabilities"] = best_name
    oof = variants[best_name]
    sel = apply_decision(code, pairs["i2"], src2, oof, params, cfg)
    per = Evaluator(code, y_all, ntrue, cfg.beta).per_entity(sel)
    countries = prof["country"].values[s1_query]
    train_countries = set(map(str, pd.unique(countries)))
    report["decision"] = params
    report["oof_macro_f05"] = float(per.mean())
    report["oof_macro_f05_by_country"] = {str(c): float(per[countries == c].mean())
                                          for c in pd.unique(countries)}
    report["oof_entities"] = int(len(s1_query))
    report["train_sample_entities"] = int(len(samp))
    log(f"held-out macro-F0.5 = {per.mean():.5f} on {len(s1_query):,} S1 ({best_name} + "
        f"{params['method']}) by country {report['oof_macro_f05_by_country']}")
    if val_ent.any():
        hv = per[val_ent]
        report["holdout_frac"] = cfg.holdout_frac
        report["holdout_entities"] = int(val_ent.sum())
        report["holdout_macro_f05"] = float(hv.mean())
        report["holdout_macro_f05_by_country"] = {
            str(c): float(hv[countries[val_ent] == c].mean()) for c in pd.unique(countries[val_ent])}
        report["train_entities_oof_macro_f05"] = float(per[~val_ent].mean())
        log(f"HOLDOUT macro-F0.5 = {hv.mean():.5f} on {int(val_ent.sum()):,} never-trained S1 "
            f"(training entities, out-of-fold: {per[~val_ent].mean():.5f}) by country "
            f"{report['holdout_macro_f05_by_country']}")
    np.savez(os.path.join(art, "oof_all.npz"), i1=pairs["i1"], i2=pairs["i2"], p1=p1, p2=p2,
             y=y_all, ce=ce if ce is not None else np.zeros(0))
    log("writing error worklist")
    err = dump_errors(os.path.join(art, "oof_errors.tsv"), args.data_dir, sel, code, pairs["i2"],
                      oof, y_all, ids[s1_query], truth, ids, per)
    if len(err):
        report["oof_error_breakdown"] = {k: {"rows": int(len(g)),
                                             "total_loss": float(g["loss"].sum())}
                                         for k, g in err.groupby("type")}
    with open(os.path.join(art, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    del prof, pairs, sctx, sp, truth, y_all, p1, p2, oof, ce
    gc.collect()

    if args.dev:
        log(f"dev run finished (test skipped). Report: {art}/report.json")
        return

    # ================================================================= TEST
    prof = load_profiles(args.data_dir, "test", work, n_jobs)
    ids = prof["entity_id"].values
    src = prof["src"].values
    s1_rows = np.where(src == 1)[0]
    log(f"test: {len(s1_rows):,} S1, {int((src == 2).sum()):,} S2, {int((src == 3).sum()):,} S3 "
        f"| countries {prof.loc[src == 1, 'country'].value_counts().to_dict()}")
    pairs = candidate_pairs(prof, s1_rows, cfg, n_jobs, work, "test")
    sctx = prepare_context(prof, n_jobs, log=log)
    gctx = global_context(sctx, pairs, log=log)
    if cfg.use_decoy:
        add_decoy(sctx, pairs, gctx, None, "test")
    q1_path = os.path.join(s1_dir, f"test_q1_{len(pairs['i1'])}.npy")
    if os.path.exists(q1_path):
        q1 = np.load(q1_path)
        log("test: stage-1 probabilities from checkpoint")
    else:
        log(f"test: scoring {len(pairs['i1']):,} candidate pairs (stage 1)")
        _, q1 = run_chunks(sctx, pairs, gctx, cfg, n_jobs, predict_fn=pred1.predict, keep=False,
                           log=log)
        np.save(q1_path, q1)
    src2_te = src[pairs["i2"]]
    ce_te = None
    if ce_models:
        from .cross_encoder import score_test, select_for_ce, top_up
        sel_ce = select_for_ce(pairs["i1"], q1, cfg.ce_topn, cfg.ce_min_p)
        cte_path = os.path.join(ce_dir, f"test_ce_{len(pairs['i1'])}.npy")
        if os.path.exists(cte_path):
            ce_te = np.load(cte_path)
            ce_te, n_new = top_up(ce_models, sctx, pairs["i1"], pairs["i2"], ce_te, sel_ce)
            log(f"test: cross-encoder scores from checkpoint (+{n_new:,} newly selected)")
            np.save(cte_path, ce_te)
        else:
            log(f"test: cross-encoder on {int(sel_ce.sum()):,} selected pairs")
            ce_te = score_test(ce_models, sctx, pairs["i1"], pairs["i2"], sel_ce)
            np.save(cte_path, ce_te)
        for m_ in ce_models:
            m_.release()
    log("test: stage 2")
    q2 = pred2.predict(stage2_features(pairs["i1"], pairs["i2"], src2_te, q1, gctx, ce_te))
    del gctx
    p = {"stage1": q1, "stage2": q2, "blend": 0.5 * (q1 + q2)}[best_name]
    np.savez(os.path.join(work, "test_probabilities.npz"), i1=pairs["i1"], i2=pairs["i2"],
             p1=q1, p2=q2, ce=ce_te if ce_te is not None else np.zeros(0))
    code_te = np.searchsorted(s1_rows, pairs["i1"])
    sel = apply_decision(code_te, pairs["i2"], src[pairs["i2"]], p, params, cfg)

    match_path = os.path.join(args.out_dir, "matching_results.tsv")
    cand_path = os.path.join(args.out_dir, "candidate_pairs.tsv")
    log("writing matching_results.tsv and candidate_pairs.tsv")
    write_sorted_lists(match_path, s1_rows, pairs["i1"][sel], ids[pairs["i2"][sel]], ids,
                       "matched_entity_ids")
    write_sorted_lists(cand_path, s1_rows, pairs["i1"], ids[pairs["i2"]], ids,
                       "candidate_entity_ids")
    matched = np.unique(pairs["i1"][sel])
    countries = prof["country"].values
    unseen = sorted(set(map(str, pd.unique(countries[s1_rows]))) - train_countries)
    if cfg.calibrate_unseen and unseen:
        cal_path = os.path.join(args.out_dir, "matching_results_calibrated.tsv")
        sel_c = calibrate_unseen(code_te, pairs, src, p, params, cfg, countries, s1_rows, sel,
                                 unseen)
        write_sorted_lists(cal_path, s1_rows, pairs["i1"][sel_c], ids[pairs["i2"][sel_c]], ids,
                           "matched_entity_ids")
        report["calibrated_file"] = {"countries": unseen, "path": cal_path}
    report["blocking_test"] = {
        "candidate_pairs": int(len(pairs["i1"])),
        "avg_candidates_per_s1": len(pairs["i1"]) / max(1, len(s1_rows)),
        "reduction_ratio": 1.0 - len(pairs["i1"]) / max(1, len(s1_rows) * int((src != 1).sum())),
    }
    n_match = np.bincount(np.searchsorted(s1_rows, pairs["i1"][sel]), minlength=len(s1_rows))
    for c in pd.unique(countries[s1_rows]):
        m = countries[s1_rows] == c
        log(f"test {c!s:>8}: {int(m.sum()):,} S1 | matches per S1 {n_match[m].mean():.2f} | "
            f"no-match share {(n_match[m] == 0).mean():.3f}   (France has no training labels: "
            f"compare it with US / India)")
    report["test_prediction_summary"] = {
        "s1_entities": int(len(s1_rows)),
        "with_matches": int(len(matched)),
        "matched_share_by_country": {
            str(c): float(np.isin(s1_rows[countries[s1_rows] == c], matched).mean())
            for c in pd.unique(countries[s1_rows])},
    }
    validator = args.validator or os.path.join(
        os.path.dirname(os.path.abspath(args.data_dir)), "utils", "validate_submission.py")
    log("official validator:")
    report["validator_exit_code"] = run_official_validator(
        validator, match_path, cand_path, os.path.join(args.data_dir, "test"), log=log)
    with open(os.path.join(art, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    log(f"done. outputs: {args.out_dir} | report: {art}/report.json")


if __name__ == "__main__":
    main()
