"""Experiment configuration.

Single source of truth for every knob the paper needs to report. Anything a
reviewer could ask "what value did you use?" about should live here, not be
buried as a literal in a function body.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal

from .privacy import DPConfig


@dataclass
class DataConfig:
    """Where the data is and how it gets turned into a matrix."""

    # Which dataset spec drives loading, labelling and column exclusion.
    # "ciciot" preserves the original pipeline byte-for-byte; the clinical
    # domains route through the split-then-fit path instead.
    domain: Literal["ciciot", "wids", "diabetes"] = "ciciot"

    # Directory holding the CIC-IoT-2023 CSV parts. On Kaggle this is the
    # mounted dataset path; locally it's wherever the subsample was written.
    data_dir: Path = Path("data/cic-iot-2023")

    # Which files inside data_dir to read. Clinical sets are a single CSV.
    file_glob: str = "*.csv"

    # Explicit target column. None lets the spec locate it.
    label_column: str | None = None

    # How train/val/test are carved. See fedlgbm.splits for what each measures:
    # "random" and "group" isolate the cost of federation, "site_holdout"
    # measures cross-site generalisation.
    split_mode: Literal["random", "group", "site_holdout"] = "random"

    # Per-row entity id (patient) and site id (hospital). None = spec default.
    group_column: str | None = None
    site_column: str | None = None

    # Sites smaller than this are never drawn as held-out test sites and are
    # pooled rather than kept as standalone federated clients.
    min_rows_per_site: int = 200

    # --- preprocessing (clinical path only) ---
    # 0.95 not 0.5: at 0.5 the APACHE lab panel, 40% of the columns and where
    # the signal lives, is discarded wholesale.
    missing_drop_threshold: float = 0.95
    # An unordered lab reflects clinical judgement, so keep the trace.
    add_missing_indicators: bool = True
    rare_level_min_count: int = 50
    # Above this many levels a categorical is ordinal-coded rather than
    # one-hot; apache_3j_diagnosis has 399 levels.
    categorical_max_cardinality: int = 20

    # Cap on rows read, for fast local iteration. None = read everything.
    max_rows: int | None = 200_000

    # Number of CSV part files to read. None = all of them.
    max_files: int | None = None

    # "binary"    -> benign vs malicious (the manuscript's framing)
    # "category"  -> 8 coarse attack families
    # "fine"      -> all 34 original labels
    label_mode: Literal["binary", "category", "fine"] = "binary"

    test_size: float = 0.2
    val_size: float = 0.1
    scaler: Literal["standard", "minmax", "none"] = "standard"

    # Cap per fine-grained class before splitting. CIC-IoT-2023 is extremely
    # imbalanced (DDoS floods dominate); without this the benign class is
    # numerically invisible and every model looks great by predicting one class.
    max_per_class: int | None = 50_000


@dataclass
class PartitionConfig:
    """How the training set is split across federated clients.

    Every scheme but "natural" simulates the split. "natural" uses a real site
    identifier carried through from the data (a hospital, in the clinical
    setting), which makes client count an output rather than an input and
    removes the need to justify a simulated heterogeneity parameter.
    """

    n_clients: int = 15
    scheme: Literal["dirichlet", "iid", "shard", "natural"] = "dirichlet"

    # Lower alpha = more skewed / more non-IID. 0.5 is the standard
    # "moderately non-IID" setting in the FL literature.
    dirichlet_alpha: float = 0.5

    shards_per_client: int = 2
    min_samples_per_client: int = 50


@dataclass
class MLPConfig:
    """The on-device feature extractor."""

    hidden_dims: tuple[int, ...] = (128, 64)
    embedding_dim: int = 32
    dropout: float = 0.2
    activation: Literal["relu", "gelu"] = "relu"
    batch_norm: bool = True
    # LayerNorm instead of BatchNorm. Required for user-level DP on the update
    # channel: BatchNorm's running statistics are buffers, not parameters, and
    # perturbing them with DP noise can drive a variance negative. LayerNorm has
    # no running statistics, so every aggregated tensor is a trainable parameter.
    layer_norm: bool = False

    # Weight on an auxiliary reconstruction loss (embedding -> raw features),
    # so the total local objective is `CE + reconstruction_weight * MSE`.
    #
    # Trained on cross-entropy alone, the encoder is free to collapse the
    # embedding onto whatever its own softmax layer finds separable, which
    # discards the axis-aligned threshold structure a tree ensemble splits on.
    # Forcing the embedding to also reconstruct its input keeps it
    # information-preserving and therefore useful to LightGBM.
    #
    # PRIVACY TRADE-OFF: an embedding that reconstructs well is by construction
    # easier to invert. Raising this is expected to raise measured inversion
    # leakage, and the paper must report both ends of that curve.
    # 0.0 disables the decoder entirely (the original architecture).
    reconstruction_weight: float = 0.0
    decoder_hidden: int = 64


@dataclass
class FedConfig:
    """Federated training loop."""

    n_rounds: int = 30
    local_epochs: int = 2
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    optimizer: Literal["adam", "sgd"] = "adam"

    # Fraction of clients sampled per round. 1.0 = full participation.
    client_fraction: float = 1.0

    # Stop if the global validation loss hasn't improved in this many rounds.
    # None disables early stopping (report the full curve for the paper).
    patience: int | None = None


@dataclass
class LGBMConfig:
    """Central LightGBM classifier trained on pooled embeddings."""

    n_estimators: int = 500
    learning_rate: float = 0.05
    num_leaves: int = 31
    max_depth: int = -1
    min_child_samples: int = 20
    subsample: float = 0.8
    subsample_freq: int = 1
    colsample_bytree: float = 0.8

    # Regularisation terms — these are gamma and lambda in Eq. 5 of the paper,
    # so they must be reported.
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0

    early_stopping_rounds: int | None = 50
    class_weight: Literal["balanced"] | None = "balanced"


@dataclass
class ExperimentConfig:
    """Top-level config passed to every runner."""

    name: str = "fed_mlp_lgbm"
    seed: int = 42
    output_dir: Path = Path("outputs")
    n_jobs: int = -1
    device: Literal["cpu", "cuda", "auto"] = "auto"

    data: DataConfig = field(default_factory=DataConfig)
    partition: PartitionConfig = field(default_factory=PartitionConfig)
    mlp: MLPConfig = field(default_factory=MLPConfig)
    fed: FedConfig = field(default_factory=FedConfig)
    lgbm: LGBMConfig = field(default_factory=LGBMConfig)
    dp: DPConfig = field(default_factory=DPConfig)

    # Seeds for the multi-seed protocol. A single seed is not publishable.
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46)

    # Attacks train an extra decoder network; skip inside large sweeps.
    skip_privacy_attacks: bool = False

    # Write every model's per-record test probabilities to predictions.npz, so
    # reliability diagrams and complete-case comparisons can be drawn later
    # without re-running anything.
    save_predictions: bool = False

    def to_dict(self) -> dict:
        return _jsonify(asdict(self))

    def save(self, path: Path | str) -> None:
        """Persist alongside results so every run is reproducible."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def _jsonify(obj):
    """dataclasses.asdict leaves Paths and tuples that json can't handle."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def smoke_config() -> ExperimentConfig:
    """Tiny config for tests and local sanity checks — seconds, not hours."""
    cfg = ExperimentConfig(name="smoke")
    cfg.data.max_rows = 5_000
    cfg.data.max_files = 1
    cfg.data.max_per_class = 2_000
    cfg.partition.n_clients = 4
    cfg.fed.n_rounds = 3
    cfg.fed.local_epochs = 1
    cfg.lgbm.n_estimators = 50
    cfg.lgbm.early_stopping_rounds = 10
    return cfg
