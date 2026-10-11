"""The DP mechanism must apply to every path that releases an embedding.

These exist because a bug here is invisible in the results and corrupts the
privacy-utility curve in both directions at once: the classifier trains on
noised vectors and scores on clean ones, which manufactures a utility drop that
looks like the cost of privacy, while the inversion attack is handed a cleaner
artifact than the protocol ever releases, which makes the mechanism look more
effective than it is. Both errors flatter the method, so neither would look
suspicious in a results table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fedlgbm import ExperimentConfig, config, partition  # noqa: E402
from fedlgbm.hybrid import FederatedHybridMLPLightGBM  # noqa: E402


def _fitted(protect: bool, seed: int = 0):
    rng = np.random.default_rng(seed)
    n, d = 900, 12
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (X[:, 0] + 0.4 * rng.normal(size=n) > 0).astype(np.int64)

    cfg = ExperimentConfig()
    cfg.device = "cpu"
    cfg.fed.n_rounds = 2
    cfg.fed.local_epochs = 1
    cfg.mlp.embedding_dim = 8
    cfg.mlp.hidden_dims = (16,)
    cfg.lgbm.n_estimators = 20
    cfg.lgbm.early_stopping_rounds = None
    cfg.lgbm.class_weight = None

    cfg.dp.enabled = protect
    cfg.dp.protect_embeddings = protect
    cfg.dp.embedding_noise_multiplier = 1.0
    cfg.dp.embedding_target_epsilon = None
    cfg.dp.noise_multiplier = 0.0
    cfg.dp.target_epsilon = None
    cfg.dp.adaptive_clipping = False
    cfg.dp.clip_norm = float("inf")

    split = int(0.8 * n)
    pcfg = config.PartitionConfig()
    pcfg.scheme = "iid"
    pcfg.n_clients = 4
    parts = partition.partition(y[:split], pcfg, ["neg", "pos"], seed=seed)

    model = FederatedHybridMLPLightGBM(cfg)
    model.fit(
        X[:split], y[:split], parts,
        X[split:], y[split:], n_classes=2,
    )
    return model, X[split:]


def test_embed_applies_noise_when_dp_is_on() -> None:
    """The default release path must not hand out clean embeddings."""
    model, X = _fitted(protect=True)
    released = model.embed(X)
    clean = model.embed(X, private=False)
    assert not np.allclose(released, clean), (
        "embed() returned unperturbed embeddings while embedding DP was enabled"
    )


def test_embed_is_a_noop_when_dp_is_off() -> None:
    model, X = _fitted(protect=False)
    assert np.allclose(model.embed(X), model.embed(X, private=False))


def test_repeated_release_draws_fresh_noise() -> None:
    """Two releases must differ, or the noise is a fixed offset an attacker
    could average away across queries."""
    model, X = _fitted(protect=True)
    assert not np.allclose(model.embed(X), model.embed(X))


def test_predict_path_uses_the_released_embedding() -> None:
    """Scoring must go through the same mechanism training did.

    Otherwise LightGBM is fitted on noised vectors and scored on clean ones, and
    the resulting mismatch is misread as the utility cost of privacy.
    """
    model, X = _fitted(protect=True)
    calls: list[bool] = []
    original = model.embed

    def spy(arr, *, private=True):
        calls.append(private)
        return original(arr, private=private)

    model.embed = spy  # type: ignore[method-assign]
    model.predict_proba(X)
    assert calls and all(calls), "predict_proba bypassed the DP release path"


def test_noise_scale_responds_to_the_multiplier() -> None:
    quiet, X = _fitted(protect=True, seed=1)
    quiet_delta = np.abs(quiet.embed(X) - quiet.embed(X, private=False)).mean()

    loud, X2 = _fitted(protect=True, seed=1)
    loud._embedding_sigma *= 5.0
    loud_delta = np.abs(loud.embed(X2) - loud.embed(X2, private=False)).mean()

    assert loud_delta > quiet_delta
