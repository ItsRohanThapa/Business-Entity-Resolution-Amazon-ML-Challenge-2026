"""Central configuration. Every tunable knob of the pipeline lives here.

Defaults reproduce the documented final run (holdout macro-F0.5 0.98908); the command-line
flags of run.py override them.
"""
from dataclasses import dataclass, field


@dataclass
class Config:
    # ---------------- compute -------------------------------------------------------
    n_jobs: int = 0                     # worker processes, 0 = all CPU cores
    feature_chunk_pairs: int = 250_000  # pairs per feature chunk (memory per worker)
    device: str = "auto"                # auto | cuda | cpu
    backend: str = "auto"               # auto (xgb on GPU, else lgbm) | xgb | lgbm

    # ---------------- blocking (see blocking.py) ------------------------------------
    k_forward: int = 20                 # ALL-keys pass: top-k per S1 entity, per source
    k_name: int = 10                    # NAME-keys pass: top-k per S1 entity, per source
    k_addr: int = 15                    # ADDRESS-keys pass: top-k per S1 entity, per source
    k_reverse: int = 3                  # every S2/S3 record proposes its top-k S1 entities
    max_key_pairs: float = 2e6          # drop a key if (#S1 with it) x (#S2/S3 with it) > this
    min_block_score: float = 0.05
    block_chunk_products: float = 2e7   # work budget per sparse-product chunk
    block_within_country: bool = True   # only compare records with the same country label
    # GPU embedding passes (emb_blocking.py): multilingual-e5 on normalised AND original text
    use_emb: bool = True
    emb_model: str = "intfloat/multilingual-e5-small"  # MIT licence, 118M parameters
    emb_batch: int = 512
    k_emb: int = 20                     # EMBEDDING passes: top-k per S1 entity, per source, per view
    min_emb_score: float = 0.0
    # label-free decoy signals (decoy.py): house-number shift / support, twins, decoy words
    use_decoy: bool = True
    # countries in test but not in train (France): also write matching_results_calibrated.tsv
    # with a per-country logit bias so their matched-S1 share equals the labelled countries'
    calibrate_unseen: bool = True

    # ---------------- gradient boosting -----------------------------------------------
    train_sample_s1: int = 600_000      # S1 entities used for training (0 = all non-holdout).
                                        # 0 needs >100 GB of GPU memory in XGBoost
    holdout_frac: float = 0.10          # share of train S1 entities never trained on or tuned on;
                                        # the reported score is measured on them
    n_folds: int = 3                    # entity-grouped CV folds
    seeds: tuple = (42,)                # booster seeds averaged, e.g. (42, 7, 2026)
    monotone: bool = True
    num_boost_round: int = 1500         # upper bound; early stopping picks the rounds
    early_stopping_rounds: int = 30
    lgb_params: dict = field(default_factory=lambda: dict(
        objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=50,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1))
    xgb_params: dict = field(default_factory=lambda: dict(
        objective="binary:logistic", eval_metric="logloss", learning_rate=0.05, max_depth=8,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0))
    xgb_max_bin: int = 256
    # GPU memory budget: at most this many rows per booster training (0 = all). XGBoost needs
    # ~1.8 KB of GPU memory per row with ~150 features -> 12M rows ~ 25 GB (fits a 32 GB GPU).
    # All positives + look-alike negatives are kept; easy negatives are sampled and weighted.
    max_train_rows: int = 12_000_000    # ~12M rows for 32 GB free GPU memory, ~40M for 80 GB
    hard_neg_qn: float = 0.80           # look-alike (hard) negative: quick name similarity >= this

    # ---------------- GPU cross-encoder (--cross-encoder) ------------------------------
    ce_model: str = "intfloat/multilingual-e5-base"    # MIT licence, 278M parameters
    ce_max_len: int = 128
    ce_topn: int = 10                   # per S1: score the top-n candidates by stage-1 p ...
    ce_min_p: float = 0.005             # ... plus every candidate with p1 >= this
    ce_max_train_pairs: int = 6_000_000 # per cross-fitting half
    ce_epochs: int = 1
    ce_lr: float = 2e-5
    ce_batch: int = 128
    ce_infer_batch: int = 2048

    # ---------------- decision --------------------------------------------------------
    beta: float = 0.5
    one_to_one: bool = True             # an S2/S3 record belongs to at most one S1 entity
    threshold_grid: tuple = tuple(round(0.20 + 0.025 * i, 3) for i in range(31))
    expf_bias_grid: tuple = (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)
    expf_min_p: float = 0.02
