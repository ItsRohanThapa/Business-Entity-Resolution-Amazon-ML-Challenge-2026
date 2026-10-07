"""Reading the challenge TSV files, writing the two output files and checking them
against the submission rules (a local mirror of utils/validate_submission.py)."""
import csv
import os
import re

import pandas as pd

from .progress import bar

ID_RE = re.compile(r"^S[123]-")
SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    first = df.columns[0]
    if len(df) and not df[first].map(lambda x: bool(ID_RE.match(str(x)))).all():
        # a stray quote character can merge lines under default quoting -> re-read raw
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                         quoting=csv.QUOTE_NONE)
    return df


def read_split(data_dir, split):
    frames = []
    for k in (1, 2, 3):
        path = os.path.join(data_dir, split, f"{split}_source{k}.tsv")
        df = read_tsv(path)
        missing = [c for c in SOURCE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        df = df[SOURCE_COLS].copy()
        df["src"] = k
        frames.append(df)
    rec = pd.concat(frames, ignore_index=True)
    dup = rec["entity_id"].duplicated()
    if dup.any():
        raise ValueError(f"{split}: {dup.sum()} duplicated entity ids across source files")
    return rec


def read_ground_truth(data_dir, s1_ids):
    gt = read_tsv(os.path.join(data_dir, "train", "train_ground_truth.tsv"))
    truth = {s: set() for s in s1_ids}
    for s1, m in bar(zip(gt["source1_entity_id"], gt["matched_entity_ids"]), "ground truth",
                     total=len(gt), unit="S1"):
        truth[s1] = {x.strip() for x in str(m).split(",") if x.strip()}
    return truth


def write_lists(path, s1_ids, row_s1, row_ids, col_name):
    """s1_ids: all S1 ids in output order; row_s1 / row_ids: parallel arrays of
    (S1 id, S2/S3 id) pairs to list (any order, duplicates removed)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    lists = {}
    for s, x in zip(row_s1, row_ids):
        lists.setdefault(s, {})[x] = None          # dict keeps order, drops duplicates
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"source1_entity_id\t{col_name}\n")
        for s in s1_ids:
            f.write(s + "\t" + ",".join(lists.get(s, ())) + "\n")


def run_official_validator(validator, match_path, cand_path, test_dir, log=print):
    """Runs utils/validate_submission.py (stdlib only) if it exists."""
    import subprocess
    import sys
    if not validator or not os.path.isfile(validator):
        log("official validator not found - run utils/validate_submission.py yourself")
        return None
    cmd = [sys.executable, validator, "--matching", match_path, "--candidate", cand_path,
           "--test-dir", test_dir]
    res = subprocess.run(cmd, capture_output=True, text=True)
    for line in (res.stdout + res.stderr).strip().splitlines():
        log("    | " + line)
    return res.returncode


def fetch_raw(data_dir, split, ids, chunksize=1_000_000):
    """Raw name/address of a small set of ids (for error analysis), streamed from the TSVs."""
    ids = set(ids)
    out = []
    for k in (1, 2, 3):
        path = os.path.join(data_dir, split, f"{split}_source{k}.tsv")
        for ch in bar(pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                                  chunksize=chunksize), f"scan {os.path.basename(path)}",
                      unit="chunk", leave=False):
            out.append(ch[ch["entity_id"].isin(ids)])
    df = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=SOURCE_COLS)
    return df.set_index("entity_id")
