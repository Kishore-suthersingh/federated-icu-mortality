"""Empirical privacy evaluation.

A DP epsilon is a worst-case bound against an unbounded adversary. Reviewers of
a privacy paper increasingly want the complementary evidence: what does a
*concrete* attacker actually recover? These attacks supply that, and they are
what justify the architecture's privacy claim rather than merely asserting it.

Two attacks, matched to the two leakage channels in `privacy.py`:

  1. Embedding inversion (channel 2). The adversary is an honest-but-curious
     server holding uploaded embeddings. It trains a decoder to map embeddings
     back to raw flow features. Success is measured against the trivial
     baseline of predicting the training mean — a decoder that cannot beat the
     mean has learned nothing, which is the outcome the defence wants.

  2. Membership inference (both channels). Given a record and model access, did
     this record appear in training? Reported as AUC, where 0.5 is no leakage.

Both are run against the model with and without DP, so the paper can show
epsilon translating into a measured reduction in attack success.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, roc_auc_score

from .mlp import make_loader, resolve_device

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------
# Attack 1: embedding inversion
# --------------------------------------------------------------------

class _Decoder(nn.Module):
    """Adversary's reconstruction network: embedding -> raw features.

    Deliberately given more capacity than the encoder. Underpowering the
    attacker would understate leakage and flatter the defence.
    """

    def __init__(self, embedding_dim: int, output_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def embedding_inversion_attack(
    Z_train: np.ndarray,
    X_train: np.ndarray,
    Z_test: np.ndarray,
    X_test: np.ndarray,
    *,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: torch.device | None = None,
    seed: int = 0,
) -> dict:
    """Train a decoder to reconstruct raw features from embeddings.

    The adversary is assumed to hold an auxiliary set of (embedding, raw)
    pairs — the strongest realistic assumption, and the standard one in the
    inversion literature. Returns reconstruction quality on held-out records.
    """
    device = device or resolve_device("auto")
    torch.manual_seed(seed)

    decoder = _Decoder(Z_train.shape[1], X_train.shape[1]).to(device)
    opt = torch.optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.MSELoss()

    loader = make_loader(Z_train, np.zeros(len(Z_train), dtype=np.int64),
                         batch_size, shuffle=False)
    targets = torch.from_numpy(np.ascontiguousarray(X_train, dtype=np.float32))

    for epoch in range(epochs):
        decoder.train()
        for i, (zb, _) in enumerate(loader):
            start = i * batch_size
            xb = targets[start : start + len(zb)].to(device)
            opt.zero_grad()
            criterion(decoder(zb.to(device)), xb).backward()
            opt.step()

    decoder.eval()
    with torch.no_grad():
        reconstructed = decoder(
            torch.from_numpy(np.ascontiguousarray(Z_test, dtype=np.float32)).to(device)
        ).cpu().numpy()

    # The honest yardstick: an attacker who ignores the embedding entirely and
    # always predicts the training mean. Beating it is the definition of leakage.
    baseline = np.tile(X_train.mean(axis=0), (len(X_test), 1))

    mse_attack = float(np.mean((reconstructed - X_test) ** 2))
    mse_baseline = float(np.mean((baseline - X_test) ** 2))

    return {
        "reconstruction_mse": mse_attack,
        "mean_baseline_mse": mse_baseline,
        # >0 means the attacker beat the trivial baseline; <=0 means no leakage.
        "leakage_ratio": float(1.0 - mse_attack / (mse_baseline + 1e-12)),
        "r2_attack": float(r2_score(X_test, reconstructed, multioutput="variance_weighted")),
        "r2_baseline": float(r2_score(X_test, baseline, multioutput="variance_weighted")),
        "interpretation": (
            "leakage_ratio is the fraction of reconstruction error the attacker "
            "removes relative to predicting the training mean. ~0 means the "
            "embedding reveals essentially nothing beyond population statistics; "
            "values near 1 mean near-perfect inversion of raw features."
        ),
    }


# --------------------------------------------------------------------
# Attack 2: membership inference
# --------------------------------------------------------------------

def membership_inference_attack(
    model,
    X_members: np.ndarray,
    y_members: np.ndarray,
    X_nonmembers: np.ndarray,
    y_nonmembers: np.ndarray,
) -> dict:
    """Loss-threshold membership inference (Yeom et al., 2018).

    Models are typically more confident on data they trained on. The attacker
    scores each record by the model's confidence in its true label and
    thresholds. AUC is reported because it is threshold-free.

    AUC ~ 0.5 means the model does not betray membership. Meaningfully above
    0.5 means it does, regardless of what the DP bound promises.
    """
    def confidence(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        proba = model.predict_proba(X)
        return proba[np.arange(len(y)), y]

    conf_members = confidence(X_members, y_members)
    conf_nonmembers = confidence(X_nonmembers, y_nonmembers)

    scores = np.concatenate([conf_members, conf_nonmembers])
    labels = np.concatenate([
        np.ones(len(conf_members), dtype=np.int64),
        np.zeros(len(conf_nonmembers), dtype=np.int64),
    ])

    auc = float(roc_auc_score(labels, scores))
    return {
        "attack_auc": auc,
        # Symmetric: an AUC of 0.3 leaks as much as 0.7, just inverted.
        "advantage_over_chance": float(abs(auc - 0.5) * 2.0),
        "mean_confidence_members": float(conf_members.mean()),
        "mean_confidence_nonmembers": float(conf_nonmembers.mean()),
        "confidence_gap": float(conf_members.mean() - conf_nonmembers.mean()),
        "interpretation": (
            "AUC 0.5 = no membership leakage. Values meaningfully above 0.5 "
            "indicate the model reveals whether a record was in training."
        ),
    }


# --------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------

def evaluate_privacy(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    max_samples: int = 5000,
    seed: int = 0,
    device: torch.device | None = None,
) -> dict:
    """Run both attacks against a fitted hybrid model."""
    rng = np.random.default_rng(seed)

    def subsample(X, y, n):
        if len(X) <= n:
            return X, y
        idx = rng.choice(len(X), n, replace=False)
        return X[idx], y[idx]

    Xtr, ytr = subsample(X_train, y_train, max_samples)
    Xte, yte = subsample(X_test, y_test, max_samples)

    results: dict = {}

    try:
        results["embedding_inversion"] = embedding_inversion_attack(
            model.embed(Xtr), Xtr, model.embed(Xte), Xte, seed=seed, device=device
        )
        logger.info(
            "Embedding inversion: leakage_ratio=%.4f (0 = none, 1 = full)",
            results["embedding_inversion"]["leakage_ratio"],
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Embedding inversion attack failed: %s", exc)
        results["embedding_inversion"] = {"error": str(exc)}

    try:
        # Members must come from data the model actually trained on; the test
        # split supplies non-members. Equal sizes keep AUC unbiased.
        n = min(len(Xtr), len(Xte))
        results["membership_inference"] = membership_inference_attack(
            model, Xtr[:n], ytr[:n], Xte[:n], yte[:n]
        )
        logger.info(
            "Membership inference: AUC=%.4f (0.5 = no leakage)",
            results["membership_inference"]["attack_auc"],
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Membership inference attack failed: %s", exc)
        results["membership_inference"] = {"error": str(exc)}

    return results
