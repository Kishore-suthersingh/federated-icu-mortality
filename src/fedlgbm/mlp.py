"""The on-device MLP feature extractor.

Implements Eq. 1-3 of the manuscript: each hidden layer applies a linear
transform followed by a non-linearity. The network is split into an *encoder*
(which produces the embedding shipped to the server) and a lightweight
*classification head* (used only to generate a training signal locally, and for
the FedAvg-MLP-only baseline).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from .config import MLPConfig


class MLPEncoder(nn.Module):
    """Feature extractor: input features -> embedding vector.

    The embedding is what leaves the device in this architecture, so its
    dimensionality is a privacy-relevant parameter, not just a capacity knob.
    """

    def __init__(self, input_dim: int, cfg: MLPConfig) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = cfg.embedding_dim

        activation = nn.ReLU if cfg.activation == "relu" else nn.GELU

        layers: list[nn.Module] = []
        prev = input_dim
        for hidden in cfg.hidden_dims:
            layers.append(nn.Linear(prev, hidden))
            if getattr(cfg, "layer_norm", False):
                layers.append(nn.LayerNorm(hidden))
            elif cfg.batch_norm:
                layers.append(nn.BatchNorm1d(hidden))
            layers.append(activation())
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            prev = hidden

        # Final projection to the embedding space. No activation here: the
        # embedding feeds a tree model, which is scale-invariant but benefits
        # from an unsquashed, full-range representation.
        layers.append(nn.Linear(prev, cfg.embedding_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPClassifier(nn.Module):
    """Encoder + linear softmax head, plus an optional reconstruction decoder.

    The head exists so the encoder can be trained with a supervised local loss
    during federated rounds. In the hybrid model the head is discarded after
    training and LightGBM classifies the embeddings instead.

    When `cfg.reconstruction_weight > 0` an auxiliary decoder maps the embedding
    back to the input features. It is a training-time regulariser only: like the
    head, it is discarded at inference and never used to produce predictions.
    Its purpose is to stop the embedding collapsing onto whatever the softmax
    head finds convenient, which is what costs LightGBM accuracy.
    """

    def __init__(self, input_dim: int, n_classes: int, cfg: MLPConfig) -> None:
        super().__init__()
        self.encoder = MLPEncoder(input_dim, cfg)
        self.head = nn.Linear(cfg.embedding_dim, n_classes)

        self.reconstruction_weight = float(cfg.reconstruction_weight)
        self.decoder: nn.Module | None = None
        if self.reconstruction_weight > 0:
            activation = nn.ReLU if cfg.activation == "relu" else nn.GELU
            self.decoder = nn.Sequential(
                nn.Linear(cfg.embedding_dim, cfg.decoder_hidden),
                activation(),
                nn.Linear(cfg.decoder_hidden, input_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def reconstruct(self, z: torch.Tensor) -> torch.Tensor:
        if self.decoder is None:
            raise RuntimeError("No decoder: reconstruction_weight is 0")
        return self.decoder(z)


def resolve_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64)),
    )
    # drop_last avoids a BatchNorm crash on a trailing batch of size 1.
    drop_last = shuffle and len(dataset) > batch_size
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )


def train_local(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    optimizer: str,
    device: torch.device,
    class_weights: torch.Tensor | None = None,
) -> dict:
    """Train one client's local model in place. Returns training statistics."""
    model.to(device).train()

    opt = (
        torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        if optimizer == "adam"
        else torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9
        )
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    loader = make_loader(X, y, batch_size, shuffle=True)

    # Optional auxiliary reconstruction term. Only MLPClassifier carries a
    # decoder; every other model reaching this function trains on CE alone.
    recon_weight = float(getattr(model, "reconstruction_weight", 0.0))
    use_recon = recon_weight > 0 and getattr(model, "decoder", None) is not None
    mse = nn.MSELoss()

    total_loss, total_ce, total_recon, n_batches = 0.0, 0.0, 0.0, 0
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()

            if use_recon:
                # Encode once and reuse for both heads rather than running the
                # encoder twice per batch.
                z = model.encoder(xb)
                ce = criterion(model.head(z), yb)
                recon = mse(model.decoder(z), xb)
                loss = ce + recon_weight * recon
                total_ce += ce.item()
                total_recon += recon.item()
            else:
                loss = criterion(model(xb), yb)
                total_ce += loss.item()

            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1

    stats = {
        "n_samples": int(len(y)),
        "mean_loss": total_loss / max(n_batches, 1),
    }
    if use_recon:
        # Reported separately so the two terms can be traded off knowingly
        # rather than by tuning one opaque number.
        stats["mean_ce"] = total_ce / max(n_batches, 1)
        stats["mean_recon"] = total_recon / max(n_batches, 1)
    return stats


@torch.no_grad()
def evaluate(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> dict:
    """Loss, accuracy, and — for binary targets — ROC-AUC on a held-out set.

    ROC-AUC is included because accuracy is close to useless under heavy class
    imbalance: on an 8.6%-prevalence outcome, predicting the majority class for
    every record scores 0.914, so an accuracy trace cannot distinguish a model
    that has learned something from one that has learned nothing.
    """
    model.to(device).eval()
    criterion = nn.CrossEntropyLoss()
    loader = make_loader(X, y, batch_size, shuffle=False)

    total_loss, correct, total = 0.0, 0, 0
    positive_scores: list[np.ndarray] = []
    binary = int(np.unique(y).size) <= 2

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total_loss += criterion(logits, yb).item() * len(yb)
        correct += (logits.argmax(dim=1) == yb).sum().item()
        total += len(yb)
        if binary and logits.shape[1] >= 2:
            probs = torch.softmax(logits, dim=1)[:, 1]
            positive_scores.append(probs.detach().cpu().numpy())

    out = {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
    }
    if positive_scores:
        scores = np.concatenate(positive_scores)
        try:
            out["roc_auc"] = float(roc_auc_score(y, scores))
        except ValueError:
            # A split containing a single class has no defined ROC-AUC.
            pass
    return out


@torch.no_grad()
def extract_embeddings(
    model: MLPClassifier,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    """Run the frozen encoder to produce the features LightGBM will consume."""
    model.to(device).eval()
    outputs = []
    for start in range(0, len(X), batch_size):
        batch = torch.from_numpy(
            np.ascontiguousarray(X[start : start + batch_size], dtype=np.float32)
        ).to(device)
        outputs.append(model.embed(batch).cpu().numpy())

    Z = np.concatenate(outputs, axis=0)

    # A diverged encoder (typically from excessive DP noise) yields non-finite
    # embeddings. Refuse to return them: downstream LightGBM, t-SNE and the
    # attack models would each fail in their own confusing way, far from the
    # actual cause.
    if not np.isfinite(Z).all():
        n_bad = int((~np.isfinite(Z)).any(axis=1).sum())
        raise ValueError(
            f"Encoder produced non-finite embeddings for {n_bad}/{len(Z)} "
            "records. The encoder has diverged — check DP noise settings."
        )
    return Z


@torch.no_grad()
def predict_proba(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    """Softmax probabilities, for the FedAvg-MLP-only baseline."""
    model.to(device).eval()
    outputs = []
    for start in range(0, len(X), batch_size):
        batch = torch.from_numpy(
            np.ascontiguousarray(X[start : start + batch_size], dtype=np.float32)
        ).to(device)
        outputs.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    return np.concatenate(outputs, axis=0)
