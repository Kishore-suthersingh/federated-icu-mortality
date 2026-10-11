"""Baselines, all trained and evaluated on the identical split.

This module exists to answer Reviewer 2's comment #2 directly: every number in
the comparison table is produced here, on our own data, under one split — not
copied from an unrelated smart-grid paper.

Ablation ladder, chosen so each row isolates one factor:
  centralized_lgbm        - no MLP, no FL          (is the MLP earning its keep?)
  centralized_mlp_lgbm    - MLP, no FL             (upper bound on the hybrid)
  fedavg_mlp_only         - MLP + FL, no LightGBM  (is LightGBM earning its keep?)
  random_forest           - classic tree ensemble
  logistic_regression     - linear reference
  isolation_forest        - unsupervised anomaly detection
  lstm                    - sequence model from the manuscript's comparison
"""

from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .config import ExperimentConfig
from .fedavg import FedAvgTrainer, centralized_baseline
from .hybrid import compute_class_weights
from .mlp import (
    evaluate,
    extract_embeddings,
    make_loader,
    predict_proba,
    resolve_device,
)

logger = logging.getLogger(__name__)


def _lgbm_from_cfg(cfg: ExperimentConfig) -> lgb.LGBMClassifier:
    c = cfg.lgbm
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
        n_jobs=cfg.n_jobs,
        random_state=cfg.seed,
        verbose=-1,
    )


def _fit_lgbm(model, X_tr, y_tr, X_val, y_val, cfg, n_classes):
    callbacks = [lgb.log_evaluation(period=0)]
    if cfg.lgbm.early_stopping_rounds:
        callbacks.insert(
            0, lgb.early_stopping(cfg.lgbm.early_stopping_rounds, verbose=False)
        )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="multi_logloss" if n_classes > 2 else "binary_logloss",
        callbacks=callbacks,
    )
    return model


def centralized_lgbm(cfg, X_tr, y_tr, X_val, y_val, X_test, n_classes):
    """LightGBM on raw features. If this matches the hybrid, the MLP adds nothing."""
    model = _fit_lgbm(_lgbm_from_cfg(cfg), X_tr, y_tr, X_val, y_val, cfg, n_classes)
    return model.predict(X_test), model.predict_proba(X_test)


def centralized_mlp_lgbm(cfg, X_tr, y_tr, X_val, y_val, X_test, n_classes):
    """The hybrid without federation — the accuracy ceiling FedAvg is measured against."""
    device = resolve_device(cfg.device)
    weights = compute_class_weights(y_tr, n_classes).to(device)

    encoder, _ = centralized_baseline(
        X_tr, y_tr, X_val, y_val,
        input_dim=X_tr.shape[1], n_classes=n_classes,
        mlp_cfg=cfg.mlp, fed_cfg=cfg.fed, device=device,
        seed=cfg.seed, class_weights=weights,
    )

    Z_tr = extract_embeddings(encoder, X_tr, device)
    Z_val = extract_embeddings(encoder, X_val, device)
    Z_test = extract_embeddings(encoder, X_test, device)

    model = _fit_lgbm(_lgbm_from_cfg(cfg), Z_tr, y_tr, Z_val, y_val, cfg, n_classes)
    return model.predict(Z_test), model.predict_proba(Z_test)


def stacked_mlp_lgbm(cfg, X_tr, y_tr, X_val, y_val, X_test, n_classes, n_folds=3):
    """MLP *stacked* with LightGBM instead of cascaded into it.

    The cascade (`centralized_mlp_lgbm`) puts the MLP between the data and
    LightGBM, so anything the encoder discards is lost for good — measurably a
    net negative on this data. Stacking instead appends the MLP's class
    probabilities to the raw features, so LightGBM keeps everything it already
    had and the MLP can only contribute. It supplies the smooth multi-feature
    interactions trees approximate poorly; the trees keep their axis-aligned
    splits on the untouched raw columns.

    Training-set probabilities are generated out-of-fold. In-sample MLP
    probabilities are overconfident on rows the MLP memorised, and LightGBM
    would learn to trust a signal that is far weaker at test time — the classic
    stacking leak, which inflates validation and collapses on test.

    NOTE — this is a *centralised* model. LightGBM consumes raw features at the
    server, so it carries none of the federated architecture's privacy
    properties and must never be reported as if it did.
    """
    from sklearn.model_selection import StratifiedKFold

    device = resolve_device(cfg.device)
    weights = compute_class_weights(y_tr, n_classes).to(device)

    def train_mlp(Xa, ya):
        model, _ = centralized_baseline(
            Xa, ya, X_val, y_val,
            input_dim=Xa.shape[1], n_classes=n_classes,
            mlp_cfg=cfg.mlp, fed_cfg=cfg.fed, device=device,
            seed=cfg.seed, class_weights=weights,
        )
        return model

    logger.info("Stacking: generating %d-fold out-of-fold MLP probabilities", n_folds)
    oof = np.zeros((len(X_tr), n_classes), dtype=np.float32)
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cfg.seed)
    for fold, (fit_idx, hold_idx) in enumerate(folds.split(X_tr, y_tr), start=1):
        logger.info("  fold %d/%d", fold, n_folds)
        fold_model = train_mlp(X_tr[fit_idx], y_tr[fit_idx])
        oof[hold_idx] = predict_proba(fold_model, X_tr[hold_idx], device)

    # A model fit on all training data supplies probabilities for val/test,
    # which are out-of-sample for those rows by construction.
    logger.info("Stacking: fitting full MLP for val/test probabilities")
    full = train_mlp(X_tr, y_tr)

    Z_tr = np.hstack([X_tr, oof])
    Z_val = np.hstack([X_val, predict_proba(full, X_val, device)])
    Z_test = np.hstack([X_test, predict_proba(full, X_test, device)])

    model = _fit_lgbm(_lgbm_from_cfg(cfg), Z_tr, y_tr, Z_val, y_val, cfg, n_classes)
    return model.predict(Z_test), model.predict_proba(Z_test)


def fedavg_mlp_only(cfg, X_tr, y_tr, client_indices, X_val, y_val, X_test, n_classes):
    """FedAvg MLP with its softmax head — isolates LightGBM's contribution."""
    device = resolve_device(cfg.device)
    weights = compute_class_weights(y_tr, n_classes).to(device)

    trainer = FedAvgTrainer(
        input_dim=X_tr.shape[1], n_classes=n_classes,
        mlp_cfg=cfg.mlp, fed_cfg=cfg.fed, device=device, seed=cfg.seed,
    )
    model = trainer.fit(X_tr, y_tr, client_indices, X_val, y_val, weights)

    proba = predict_proba(model, X_test, device)
    return proba.argmax(axis=1), proba


def random_forest(cfg, X_tr, y_tr, X_test):
    model = RandomForestClassifier(
        n_estimators=200, n_jobs=cfg.n_jobs,
        random_state=cfg.seed, class_weight="balanced",
    )
    model.fit(X_tr, y_tr)
    return model.predict(X_test), model.predict_proba(X_test)


def logistic_regression(cfg, X_tr, y_tr, X_test):
    model = LogisticRegression(
        max_iter=1000, n_jobs=cfg.n_jobs,
        random_state=cfg.seed, class_weight="balanced",
    )
    model.fit(X_tr, y_tr)
    return model.predict(X_test), model.predict_proba(X_test)


def isolation_forest(cfg, X_tr, y_tr, X_test):
    """Unsupervised anomaly detection — binary framing only.

    Fitted on benign traffic only, which is how it is meant to be used: it
    learns "normal" and flags departures from it.
    """
    benign = X_tr[y_tr == 0]
    if len(benign) < 10:
        raise ValueError("Too few benign samples to fit Isolation Forest")

    contamination = float(np.clip((y_tr != 0).mean(), 0.001, 0.5))
    model = IsolationForest(
        n_estimators=200, contamination=contamination,
        random_state=cfg.seed, n_jobs=cfg.n_jobs,
    )
    model.fit(benign)

    # IsolationForest returns +1 for inlier, -1 for outlier; map to 0/1.
    y_pred = (model.predict(X_test) == -1).astype(np.int64)

    # Higher score = more anomalous, for ROC.
    scores = -model.score_samples(X_test)
    lo, hi = scores.min(), scores.max()
    normalised = (scores - lo) / (hi - lo) if hi > lo else np.zeros_like(scores)
    return y_pred, np.column_stack([1 - normalised, normalised])


class _LSTM(nn.Module):
    """LSTM over the feature vector treated as a length-T sequence.

    Flow records are independent, so there is no true temporal axis here. The
    features are reshaped into pseudo-timesteps purely so the architecture the
    manuscript compares against can be evaluated on the same data. This
    framing must be stated in the paper — it is a limitation of the comparison,
    not a claim that the data is sequential.
    """

    def __init__(self, n_features: int, n_classes: int, hidden: int = 64, steps: int = 8):
        super().__init__()
        self.steps = steps
        self.chunk = int(np.ceil(n_features / steps))
        self.padded = self.chunk * steps
        self.n_features = n_features
        self.lstm = nn.LSTM(self.chunk, hidden, batch_first=True, num_layers=1)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padded > self.n_features:
            x = nn.functional.pad(x, (0, self.padded - self.n_features))
        out, _ = self.lstm(x.view(x.size(0), self.steps, self.chunk))
        return self.head(out[:, -1, :])


def lstm(cfg, X_tr, y_tr, X_val, y_val, X_test, n_classes):
    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)

    model = _LSTM(X_tr.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.fed.lr)
    criterion = nn.CrossEntropyLoss(
        weight=compute_class_weights(y_tr, n_classes).to(device)
    )
    loader = make_loader(X_tr, y_tr, cfg.fed.batch_size, shuffle=True)

    epochs = cfg.fed.n_rounds * cfg.fed.local_epochs
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            criterion(model(xb), yb).backward()
            opt.step()
        if (epoch + 1) % 10 == 0:
            logger.info("LSTM epoch %d/%d | %s", epoch + 1, epochs,
                        evaluate(model, X_val, y_val, device))

    proba = predict_proba(model, X_test, device)
    return proba.argmax(axis=1), proba
