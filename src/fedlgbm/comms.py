"""Communication cost accounting.

The manuscript claims bandwidth efficiency as a benefit of the architecture but
never quantifies it. Any FL paper is expected to report communication cost, and
for this hybrid the accounting is more interesting than usual: there are two
distinct uploads, and the embedding upload scales with *data size* rather than
model size, which partially offsets the saving from federating the encoder.

Being straight about this is better than the usual hand-wave. If the embedding
channel dominates, the honest conclusion is that the architecture trades
bandwidth for the ability to use a central tree model — a real trade-off worth
reporting, not a flaw to hide.
"""

from __future__ import annotations

import numpy as np
import torch.nn as nn

BYTES_PER_FLOAT32 = 4


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def encoder_parameter_count(model: nn.Module) -> int:
    """Parameters actually transmitted each round.

    Only the encoder is federated in the hybrid; the classification head exists
    to generate a local training signal and is discarded before LightGBM.
    """
    encoder = getattr(model, "encoder", model)
    return sum(p.numel() for p in encoder.parameters())


def analyse(
    model: nn.Module,
    *,
    n_clients: int,
    n_rounds: int,
    client_fraction: float,
    n_train_samples: int,
    n_raw_features: int,
    embedding_dim: int,
    bytes_per_value: int = BYTES_PER_FLOAT32,
) -> dict:
    """Full communication breakdown for one federated run."""
    params_total = count_parameters(model)
    params_sent = encoder_parameter_count(model)
    clients_per_round = max(1, int(round(client_fraction * n_clients)))

    # Channel 1: parameters, both directions, every round.
    bytes_per_client_round = params_sent * bytes_per_value
    upload_per_round = bytes_per_client_round * clients_per_round
    download_per_round = bytes_per_client_round * clients_per_round
    channel_1_total = (upload_per_round + download_per_round) * n_rounds

    # Channel 2: embeddings, once, after the encoder converges.
    channel_2_total = n_train_samples * embedding_dim * bytes_per_value

    # The counterfactual: centralised training, where raw data is shipped once.
    raw_upload = n_train_samples * n_raw_features * bytes_per_value

    total = channel_1_total + channel_2_total

    return {
        "model": {
            "total_parameters": int(params_total),
            "transmitted_parameters_encoder_only": int(params_sent),
            "bytes_per_model_transfer": int(bytes_per_client_round),
        },
        "channel_1_model_updates": {
            "clients_per_round": clients_per_round,
            "rounds": n_rounds,
            "upload_bytes_per_round": int(upload_per_round),
            "download_bytes_per_round": int(download_per_round),
            "total_bytes": int(channel_1_total),
            "total_mb": round(channel_1_total / 1e6, 3),
        },
        "channel_2_embeddings": {
            "n_samples": int(n_train_samples),
            "embedding_dim": int(embedding_dim),
            "total_bytes": int(channel_2_total),
            "total_mb": round(channel_2_total / 1e6, 3),
            "note": (
                "Scales with dataset size, not model size. On large datasets "
                "this channel dominates and is the architecture's main "
                "communication cost."
            ),
        },
        "comparison_to_centralized": {
            "raw_data_upload_bytes": int(raw_upload),
            "raw_data_upload_mb": round(raw_upload / 1e6, 3),
            "federated_total_bytes": int(total),
            "federated_total_mb": round(total / 1e6, 3),
            "compression_ratio": round(raw_upload / max(total, 1), 3),
            "saves_bandwidth": bool(total < raw_upload),
            "embedding_compression_vs_raw": round(
                n_raw_features / max(embedding_dim, 1), 2
            ),
        },
        "interpretation": (
            "compression_ratio > 1 means the federated scheme moves less data "
            "than shipping raw records once. Because channel 2 scales with "
            "dataset size, the ratio approaches n_raw_features/embedding_dim "
            "on large datasets rather than growing without bound."
        ),
    }


def convergence_efficiency(
    history: list[dict],
    target: float,
    metric: str | None = None,
) -> dict:
    """Rounds (hence bytes) needed to reach a target score.

    A method that reaches the same quality in fewer rounds is cheaper to
    deploy, which matters more than final quality on constrained devices.

    `metric` defaults to ROC-AUC when the training history carries it and falls
    back to accuracy otherwise. That default is not cosmetic: on a rare binary
    outcome an accuracy target is satisfied by the majority-class predictor, so
    an accuracy-based reading reports convergence in round 1 for a model that
    has learned nothing. The first WiDS run did exactly that — "0.90 accuracy in
    1 round" against a base rate of 0.914.
    """
    # fedavg writes "val_roc_auc"; callers passing a raw eval dict may use the
    # bare name. Accept either spelling, both when detecting and when reading.
    def _lookup(record: dict, name: str):
        for candidate in (f"val_{name}", name):
            if candidate in record:
                return record[candidate]
        return None

    if metric is None:
        has_auc = any(_lookup(r, "roc_auc") is not None for r in history)
        metric = "roc_auc" if has_auc else "accuracy"

    def _value(record: dict) -> float:
        value = _lookup(record, metric)
        return float(value) if value is not None else 0.0

    for record in history:
        if _value(record) >= target:
            return {
                "metric": metric,
                "target": target,
                "rounds_to_target": record["round"],
                "reached": True,
            }
    best = max((_value(r) for r in history), default=0.0)
    return {
        "metric": metric,
        "target": target,
        "rounds_to_target": None,
        "reached": False,
        "best_achieved": float(best),
    }
