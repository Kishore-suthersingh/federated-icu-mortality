"""User-level DP must not perturb normalization running statistics.

Regression tests for the fault behind the withdrawn user-level DP result:
DP-FedAvg noise added to BatchNorm running variances drove some negative, and
inference then returned NaN while every weight stayed small.
"""

import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fedlgbm.config import FedConfig, MLPConfig  # noqa: E402
from fedlgbm.fedavg import FedAvgTrainer  # noqa: E402
from fedlgbm.mlp import MLPClassifier  # noqa: E402
from fedlgbm.privacy import DPConfig, aggregate_dp  # noqa: E402


def _dp(eps: float = 10.0) -> DPConfig:
    return DPConfig(enabled=True, target_epsilon=eps, protect_embeddings=False)


def test_noised_batchnorm_statistics_break_inference():
    # The failure itself: one noised aggregation with zero client updates.
    torch.manual_seed(0)
    model = MLPClassifier(40, 2, MLPConfig(hidden_dims=(32, 16), embedding_dim=8))
    state = OrderedDict((k, v.clone()) for k, v in model.state_dict().items())
    zero = [OrderedDict((k, torch.zeros_like(v) if torch.is_floating_point(v) else v.clone())
                        for k, v in state.items()) for _ in range(5)]
    model.load_state_dict(aggregate_dp(state, zero, clip_norm=21.0, noise_multiplier=2.9,
                                       generator=torch.Generator().manual_seed(0)))
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(16, 40))
    assert not torch.isfinite(out).all()


def test_trainer_refuses_dp_noise_on_batchnorm():
    with pytest.raises(ValueError, match="running statistics"):
        FedAvgTrainer(40, 2, MLPConfig(hidden_dims=(32, 16), embedding_dim=8),
                      FedConfig(n_rounds=2), torch.device("cpu"), dp_cfg=_dp())


def test_layernorm_model_has_no_float_buffers():
    model = MLPClassifier(40, 2, MLPConfig(hidden_dims=(32, 16), embedding_dim=8,
                                           layer_norm=True))
    assert not [n for n, b in model.named_buffers() if torch.is_floating_point(b)]


def test_dp_fedavg_with_layernorm_stays_finite():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 40)).astype(np.float32)
    y = (X[:, 0] + rng.normal(scale=0.5, size=600) > 0).astype(np.int64)
    clients = np.array_split(np.arange(500), 10)
    trainer = FedAvgTrainer(
        40, 2, MLPConfig(hidden_dims=(32, 16), embedding_dim=8, layer_norm=True),
        FedConfig(n_rounds=3, local_epochs=1, batch_size=64), torch.device("cpu"),
        dp_cfg=_dp(eps=1.0),
    )
    model = trainer.fit(X[:500], y[:500], clients, X[500:], y[500:])
    model.eval()
    with torch.no_grad():
        emb = model.embed(torch.from_numpy(X[500:]))
    assert torch.isfinite(emb).all()
