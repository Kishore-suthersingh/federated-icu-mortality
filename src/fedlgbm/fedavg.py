"""Federated Averaging.

Implements Eq. 6 of the manuscript:

    w_{t+1} = sum_k (n_k / n) * w_k^{t}

where w_k is client k's locally-updated parameter vector, n_k is client k's
sample count and n is the total across participating clients.

This is a faithful single-process simulation: each round, every selected client
starts from the current global weights, trains locally on data that never
leaves its partition, and returns only parameters. No client ever reads
another client's rows.
"""

from __future__ import annotations

import copy
import logging
import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from .config import FedConfig, MLPConfig
from .mlp import MLPClassifier, evaluate, train_local
from .privacy import (
    AdaptiveClipper,
    DPConfig,
    PrivacyLedger,
    aggregate_dp,
    calibrate_noise_multiplier,
    clip_update,
    compute_update,
)

logger = logging.getLogger(__name__)


def average_weights(
    state_dicts: list[OrderedDict], n_samples: list[int]
) -> OrderedDict:
    """Sample-count-weighted average of client parameters (Eq. 6)."""
    if not state_dicts:
        raise ValueError("Cannot aggregate an empty set of client updates")

    total = float(sum(n_samples))
    weights = [n / total for n in n_samples]

    averaged = OrderedDict()
    for key in state_dicts[0]:
        stacked = state_dicts[0][key]
        # Integer buffers (BatchNorm's num_batches_tracked) can't be averaged
        # into a float; carry the first client's value instead.
        if not torch.is_floating_point(stacked):
            averaged[key] = stacked.clone()
            continue
        acc = torch.zeros_like(stacked, dtype=torch.float32)
        for sd, w in zip(state_dicts, weights):
            acc += sd[key].to(torch.float32) * w
        averaged[key] = acc.to(stacked.dtype)

    return averaged


class FedAvgTrainer:
    """Runs the FedAvg communication loop over simulated clients."""

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        mlp_cfg: MLPConfig,
        fed_cfg: FedConfig,
        device: torch.device,
        seed: int = 42,
        dp_cfg: DPConfig | None = None,
    ) -> None:
        torch.manual_seed(seed)
        self.global_model = MLPClassifier(input_dim, n_classes, mlp_cfg)
        self.mlp_cfg = mlp_cfg
        self.fed_cfg = fed_cfg
        self.device = device
        self.n_classes = n_classes
        self.rng = np.random.default_rng(seed)
        self.history: list[dict] = []

        self.dp_cfg = dp_cfg or DPConfig(enabled=False)
        self.ledger = PrivacyLedger(self.dp_cfg)

        self.torch_generator = torch.Generator(device="cpu").manual_seed(seed)
        self._noise_multiplier = self._resolve_noise_multiplier()

        self.clipper = (
            AdaptiveClipper(
                initial=self.dp_cfg.clip_norm,
                target_quantile=self.dp_cfg.clip_quantile,
                lr=self.dp_cfg.clip_lr,
                noise=self.dp_cfg.clip_quantile_noise,
                seed=seed,
            )
            if self.dp_cfg.enabled and self.dp_cfg.adaptive_clipping
            else None
        )
        self.ledger.clipper = self.clipper
        self._n_clients = 0
        self._divergence_threshold = 1e6

    def _resolve_noise_multiplier(self) -> float:
        """Derive sigma from the target epsilon, if DP is on."""
        if not self.dp_cfg.enabled:
            return 0.0
        if self.dp_cfg.noise_multiplier is not None:
            return self.dp_cfg.noise_multiplier
        if self.dp_cfg.target_epsilon is None:
            raise ValueError(
                "DP is enabled but neither noise_multiplier nor target_epsilon is set"
            )

        sigma = calibrate_noise_multiplier(
            target_epsilon=self.dp_cfg.target_epsilon,
            delta=self.dp_cfg.delta,
            q=self.fed_cfg.client_fraction,
            steps=self.fed_cfg.n_rounds,
        )
        # Write it back so the ledger and saved config report the real value.
        self.dp_cfg.noise_multiplier = sigma
        return sigma

    def _update_norms(self, updates: list[OrderedDict]) -> list[float]:
        return [
            math.sqrt(sum(
                float(v.pow(2).sum()) for v in u.values() if torch.is_floating_point(v)
            ))
            for u in updates
        ]

    def _resolve_clip_norm(self, updates: list[OrderedDict], rnd: int) -> float:
        """Clip bound for this round: adaptive if enabled, else the fixed value."""
        if self.clipper is None:
            return self.dp_cfg.clip_norm

        norms = self._update_norms(updates)
        if rnd == 1:
            # Warm-start rather than crawling up from an arbitrary constant.
            return self.clipper.warm_start(norms)
        return self.clipper.update(norms)

    def dp_feasibility(self, clients_per_round: int, clip_norm: float) -> dict:
        """Per-parameter noise the aggregate actually receives.

        DP-FedAvg adds N(0, (sigma*C)^2) to the *sum* of client updates and
        then divides by m, so each parameter of the averaged model carries
        noise of standard deviation sigma*C/m. Because that shrinks only as
        1/m, user-level DP is fundamentally a large-cohort technique: with a
        handful of clients the noise swamps the signal no matter how the clip
        bound is tuned. Production DP-FedAvg deployments use thousands of
        clients for exactly this reason.
        """
        noise_std = self._noise_multiplier * clip_norm / max(clients_per_round, 1)
        return {
            "noise_std_per_parameter": noise_std,
            "clients_per_round": clients_per_round,
            "clip_norm": clip_norm,
            "noise_multiplier": self._noise_multiplier,
        }

    def _guard_divergence(self, rnd: int) -> None:
        """Fail loudly if DP noise has destroyed the model.

        Silent NaN weights would otherwise propagate into embeddings and fill
        the results table with meaningless numbers. Explosion is caught as well
        as NaN, because a model with weights of magnitude 1e6 is already dead
        even though every value is still technically finite.
        """
        for name, param in self.global_model.state_dict().items():
            if not torch.is_floating_point(param):
                continue
            if not torch.isfinite(param).all():
                raise RuntimeError(
                    f"Round {rnd}: global model diverged to NaN/Inf at {name!r}. "
                    f"{self._divergence_advice()}"
                )
            if float(param.abs().max()) > self._divergence_threshold:
                raise RuntimeError(
                    f"Round {rnd}: global model exploded at {name!r} "
                    f"(max |w| = {float(param.abs().max()):.3e}). "
                    f"{self._divergence_advice()}"
                )

    def _divergence_advice(self) -> str:
        clients = max(1, int(round(self.fed_cfg.client_fraction * self._n_clients)))
        feasibility = self.dp_feasibility(clients, self.dp_cfg.clip_norm
                                          if self.clipper is None
                                          else self.clipper.clip_norm)
        return (
            f"DP noise per parameter is {feasibility['noise_std_per_parameter']:.3e} "
            f"with only {clients} client(s) per round. This noise falls as 1/m, "
            "so the usual remedy is more clients, not more tuning: user-level "
            "DP-FedAvg needs hundreds to thousands of clients to be usable. "
            "Alternatively raise target_epsilon, lower n_rounds, or disable DP "
            "on channel 1 and rely on embedding DP alone."
        )

    def _select_clients(self, n_clients: int) -> np.ndarray:
        """Sample the participating subset for this round."""
        n_selected = max(1, int(round(self.fed_cfg.client_fraction * n_clients)))
        if n_selected >= n_clients:
            return np.arange(n_clients)
        return self.rng.choice(n_clients, size=n_selected, replace=False)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        client_indices: list[np.ndarray],
        X_val: np.ndarray,
        y_val: np.ndarray,
        class_weights: torch.Tensor | None = None,
    ) -> MLPClassifier:
        """Run the full federated training loop and return the global model."""
        n_clients = len(client_indices)
        self._n_clients = n_clients
        best_loss, best_state, rounds_without_improvement = float("inf"), None, 0

        if self.dp_cfg.enabled:
            clients_per_round = max(
                1, int(round(self.fed_cfg.client_fraction * n_clients))
            )
            if clients_per_round < 50:
                logger.warning(
                    "DP enabled with only %d client(s) per round. Aggregate "
                    "noise scales as 1/m, so user-level DP typically needs "
                    "hundreds of clients; expect severe utility loss or "
                    "divergence at this scale.",
                    clients_per_round,
                )

        for rnd in range(1, self.fed_cfg.n_rounds + 1):
            selected = self._select_clients(n_clients)
            # Snapshot on CPU: training moves the model to the accelerator, and
            # client updates are collected on CPU, so a GPU-resident snapshot
            # would make the delta a cross-device subtraction.
            global_state = OrderedDict(
                (k, v.detach().cpu().clone())
                for k, v in self.global_model.state_dict().items()
            )

            client_states, client_sizes, client_losses = [], [], []
            client_updates: list[OrderedDict] = []

            for client_id in selected:
                idx = client_indices[client_id]
                if len(idx) < 2:
                    continue

                # Each client starts from the current global model.
                local_model = copy.deepcopy(self.global_model)
                local_model.load_state_dict(copy.deepcopy(global_state))

                stats = train_local(
                    local_model,
                    X_train[idx],
                    y_train[idx],
                    epochs=self.fed_cfg.local_epochs,
                    batch_size=self.fed_cfg.batch_size,
                    lr=self.fed_cfg.lr,
                    weight_decay=self.fed_cfg.weight_decay,
                    optimizer=self.fed_cfg.optimizer,
                    device=self.device,
                    class_weights=class_weights,
                )

                # Only parameters cross the boundary — never raw rows.
                local_state = OrderedDict(
                    (k, v.detach().cpu()) for k, v in local_model.state_dict().items()
                )
                client_states.append(local_state)
                client_sizes.append(stats["n_samples"])
                client_losses.append(stats["mean_loss"])

                if self.dp_cfg.enabled:
                    # Collect the raw delta; the clip bound for this round is
                    # decided once all norms are known (see below).
                    client_updates.append(compute_update(global_state, local_state))

            if not client_states:
                raise RuntimeError(
                    f"Round {rnd}: no client had enough samples to train. "
                    "Lower n_clients or raise min_samples_per_client."
                )

            if self.dp_cfg.enabled:
                clip_norm = self._resolve_clip_norm(client_updates, rnd)

                clipped = []
                for delta in client_updates:
                    bounded, raw_norm = clip_update(delta, clip_norm)
                    clipped.append(bounded)
                    self.ledger.clip_norms_observed.append(raw_norm)

                # Uniform average + calibrated noise. Sample-count weighting is
                # dropped deliberately; see aggregate_dp's docstring.
                self.global_model.load_state_dict(
                    aggregate_dp(
                        global_state, clipped,
                        clip_norm=clip_norm,
                        noise_multiplier=self._noise_multiplier,
                        generator=self.torch_generator,
                    )
                )
                self.ledger.record_round(
                    q=self.fed_cfg.client_fraction,
                    noise_multiplier=self._noise_multiplier,
                )
                self._guard_divergence(rnd)
            else:
                self.global_model.load_state_dict(
                    average_weights(client_states, client_sizes)
                )

            val = evaluate(self.global_model, X_val, y_val, self.device)
            record = {
                "round": rnd,
                "n_participating_clients": len(client_states),
                "mean_client_train_loss": float(np.mean(client_losses)),
                "val_loss": val["loss"],
                "val_accuracy": val["accuracy"],
            }
            if "roc_auc" in val:
                record["val_roc_auc"] = val["roc_auc"]
            if self.dp_cfg.enabled:
                # Cumulative spend, so the paper can plot utility against the
                # privacy budget actually consumed at each round.
                record["epsilon_spent"] = self.ledger.update_accountant.get_epsilon(
                    self.dp_cfg.delta
                )
            self.history.append(record)
            logger.info(
                "Round %3d/%d | clients %2d | train loss %.4f | val loss %.4f | val acc %.4f",
                rnd, self.fed_cfg.n_rounds, len(client_states),
                record["mean_client_train_loss"], val["loss"], val["accuracy"],
            )

            # Under DP, selecting the best round by validation loss is itself a
            # data-dependent choice whose privacy cost is not in the accountant.
            # Keep the final model instead, so the reported epsilon is honest.
            if self.dp_cfg.enabled:
                continue

            if val["loss"] < best_loss - 1e-5:
                best_loss = val["loss"]
                best_state = copy.deepcopy(self.global_model.state_dict())
                rounds_without_improvement = 0
            else:
                rounds_without_improvement += 1
                if (
                    self.fed_cfg.patience is not None
                    and rounds_without_improvement >= self.fed_cfg.patience
                ):
                    logger.info("Early stopping at round %d", rnd)
                    break

        # Restore the best-validation global model rather than the last one.
        if best_state is not None:
            self.global_model.load_state_dict(best_state)

        return self.global_model


def centralized_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    n_classes: int,
    mlp_cfg: MLPConfig,
    fed_cfg: FedConfig,
    device: torch.device,
    seed: int = 42,
    class_weights: torch.Tensor | None = None,
) -> tuple[MLPClassifier, list[dict]]:
    """Train the same architecture on pooled data — the centralized upper bound.

    Reported alongside the federated result to quantify the accuracy cost of
    keeping data on-device, which is the honest way to frame the privacy
    trade-off.
    """
    torch.manual_seed(seed)
    model = MLPClassifier(input_dim, n_classes, mlp_cfg)
    history = []

    # Match total gradient work to the federated run for a fair comparison.
    total_epochs = fed_cfg.n_rounds * fed_cfg.local_epochs
    for epoch in range(1, total_epochs + 1):
        stats = train_local(
            model, X_train, y_train,
            epochs=1,
            batch_size=fed_cfg.batch_size,
            lr=fed_cfg.lr,
            weight_decay=fed_cfg.weight_decay,
            optimizer=fed_cfg.optimizer,
            device=device,
            class_weights=class_weights,
        )
        val = evaluate(model, X_val, y_val, device)
        history.append({
            "round": epoch,
            "mean_client_train_loss": stats["mean_loss"],
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
        })

    return model, history
