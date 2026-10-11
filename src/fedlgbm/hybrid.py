"""The proposed Federated Hybrid MLP-LightGBM model.

Pipeline:
  1. Train the MLP encoder across clients with FedAvg (raw data stays local).
  2. Freeze the resulting global encoder and push it to every client.
  3. Each client embeds its own rows locally and sends only embeddings up.
  4. The server pools embeddings and trains one central LightGBM (Eq. 4-5).
  5. Inference: encode with the global MLP, classify with the global LightGBM.

PRIVACY NOTE — this must be stated plainly in the paper. Raw feature vectors
never leave the client, but *embeddings* do. An embedding is a learned lossy
projection, not an anonymisation primitive, and embedding-inversion attacks
against exactly this pattern are well documented. The defensible claim is
"reduced exposure relative to centralised training", not "privacy-preserving"
without qualification. `embedding_noise_std` provides an optional Gaussian
perturbation to support a utility-vs-noise ablation; it is a mitigation knob,
not a calibrated differential-privacy guarantee, and must not be reported as one.
"""

from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import torch

from .config import ExperimentConfig
from .fedavg import FedAvgTrainer
from .mlp import MLPClassifier, extract_embeddings, resolve_device
from .partition import Partition
from .privacy import calibrate_noise_multiplier, privatize_embeddings

logger = logging.getLogger(__name__)


def compute_class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    """Inverse-frequency weights, so rare attack classes aren't ignored."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = len(y) / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


class FederatedHybridMLPLightGBM:
    """Global MLP encoder (FedAvg) + central LightGBM classifier."""

    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.encoder: MLPClassifier | None = None
        self.classifier: lgb.LGBMClassifier | None = None
        self.fed_history: list[dict] = []
        self.n_classes: int = 0
        self.ledger = None
        self._embedding_sigma: float = 0.0
        self._rng = np.random.default_rng(cfg.seed)

    def _resolve_embedding_sigma(self) -> float:
        """Noise multiplier for the embedding channel.

        Each record's embedding is released exactly once, so the accounting is
        a single Gaussian release at q=1 rather than a composition over rounds.
        """
        dp = self.cfg.dp
        if not (dp.enabled and dp.protect_embeddings):
            return 0.0
        if dp.embedding_noise_multiplier is not None:
            return dp.embedding_noise_multiplier
        if dp.embedding_target_epsilon is None:
            raise ValueError(
                "Embedding DP is on but neither embedding_noise_multiplier nor "
                "embedding_target_epsilon is set"
            )
        sigma = calibrate_noise_multiplier(
            target_epsilon=dp.embedding_target_epsilon,
            delta=dp.embedding_delta,
            q=1.0,
            steps=1,
        )
        dp.embedding_noise_multiplier = sigma
        return sigma

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        partition: Partition,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_classes: int,
    ) -> "FederatedHybridMLPLightGBM":
        self.n_classes = n_classes
        class_weights = compute_class_weights(y_train, n_classes).to(self.device)

        # --- Stage 1: federated encoder training -------------------------
        logger.info(
            "Stage 1: FedAvg encoder over %d clients for %d rounds",
            len(partition), self.cfg.fed.n_rounds,
        )
        trainer = FedAvgTrainer(
            input_dim=X_train.shape[1],
            n_classes=n_classes,
            mlp_cfg=self.cfg.mlp,
            fed_cfg=self.cfg.fed,
            device=self.device,
            seed=self.cfg.seed,
            dp_cfg=self.cfg.dp,
        )
        self.encoder = trainer.fit(
            X_train, y_train, partition.client_indices, X_val, y_val, class_weights
        )
        self.fed_history = trainer.history
        self.ledger = trainer.ledger
        self._embedding_sigma = self._resolve_embedding_sigma()

        # --- Stage 2: local embedding extraction -------------------------
        # Done per-client to mirror the deployed system, where each device
        # encodes only its own rows with the shared global encoder.
        logger.info("Stage 2: extracting embeddings client-side")
        embeddings, labels = [], []
        for client_id, idx in enumerate(partition.client_indices):
            if len(idx) == 0:
                continue
            embeddings.append(
                self._maybe_perturb(
                    extract_embeddings(self.encoder, X_train[idx], self.device)
                )
            )
            labels.append(y_train[idx])

        Z_train = np.concatenate(embeddings, axis=0)
        z_labels = np.concatenate(labels, axis=0)
        logger.info("Pooled embeddings at server: %s", Z_train.shape)

        # --- Stage 3: central LightGBM (Eq. 4-5) -------------------------
        logger.info("Stage 3: training central LightGBM on pooled embeddings")
        Z_val = self._maybe_perturb(
            extract_embeddings(self.encoder, X_val, self.device)
        )
        self.classifier = self._make_lgbm()

        callbacks = []
        if self.cfg.lgbm.early_stopping_rounds:
            callbacks.append(
                lgb.early_stopping(self.cfg.lgbm.early_stopping_rounds, verbose=False)
            )
        callbacks.append(lgb.log_evaluation(period=0))

        self.classifier.fit(
            Z_train, z_labels,
            eval_set=[(Z_val, y_val)],
            eval_metric="multi_logloss" if n_classes > 2 else "binary_logloss",
            callbacks=callbacks,
        )
        return self

    def _maybe_perturb(self, Z: np.ndarray) -> np.ndarray:
        """Apply record-level DP to embeddings before they leave the client."""
        if self._embedding_sigma <= 0:
            return Z
        Z_private = privatize_embeddings(
            Z,
            clip_norm=self.cfg.dp.embedding_clip_norm,
            noise_multiplier=self._embedding_sigma,
            rng=self._rng,
        )
        # Account once, not once per call: each record's embedding is released
        # a single time. Composing per-batch would inflate the reported epsilon
        # for what is really one release per record.
        if self.ledger is not None and self.ledger.embedding_accountant.steps == 0:
            self.ledger.record_embedding_release(
                q=1.0, noise_multiplier=self._embedding_sigma
            )
        return Z_private

    def privacy_summary(self) -> dict:
        """Per-channel privacy guarantees for the results table."""
        if self.ledger is None:
            return {"enabled": False, "guarantee": "Model not yet fitted"}
        return self.ledger.summary()

    def _make_lgbm(self) -> lgb.LGBMClassifier:
        c = self.cfg.lgbm
        return lgb.LGBMClassifier(
            n_estimators=c.n_estimators,
            learning_rate=c.learning_rate,
            num_leaves=c.num_leaves,
            max_depth=c.max_depth,
            min_child_samples=c.min_child_samples,
            subsample=c.subsample,
            subsample_freq=c.subsample_freq,
            colsample_bytree=c.colsample_bytree,
            reg_alpha=c.reg_alpha,
            reg_lambda=c.reg_lambda,
            class_weight=c.class_weight,
            n_jobs=self.cfg.n_jobs,
            random_state=self.cfg.seed,
            verbose=-1,
        )

    def _require_fitted(self) -> None:
        if self.encoder is None or self.classifier is None:
            raise RuntimeError("Model is not fitted — call fit() first")

    def embed(self, X: np.ndarray, *, private: bool = True) -> np.ndarray:
        """Client-side embedding, perturbed exactly as it is when released.

        `private=True` is the default because in deployment a client applies the
        record-level mechanism before uploading, at inference no less than
        during training. Returning clean embeddings here would misreport the
        system in both directions at once: the central classifier would be
        trained on noised vectors and scored on clean ones, inventing a
        train/test mismatch that looks like a cost of privacy, while an attacker
        would be handed a cleaner artifact than the protocol ever releases,
        making the mechanism look more effective than it is.

        `private=False` exists only to measure the mechanism itself — the gap
        between the two is the effect of the noise, not of the encoder.
        """
        self._require_fitted()
        Z = extract_embeddings(self.encoder, X, self.device)
        return self._maybe_perturb(Z) if private else Z

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._require_fitted()
        return self.classifier.predict(self.embed(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._require_fitted()
        return self.classifier.predict_proba(self.embed(X))
