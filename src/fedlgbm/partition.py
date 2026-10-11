"""Split the training set across simulated federated clients.

Reviewer 1 asked explicitly: how many clients, and IID or non-IID? This module
is the answer, and it emits the per-client class distribution so the paper can
show the partition rather than assert it.
"""

from __future__ import annotations

import logging

import numpy as np

from .config import PartitionConfig

logger = logging.getLogger(__name__)


class Partition:
    """Client index assignments plus reportable distribution statistics."""

    def __init__(
        self,
        client_indices: list[np.ndarray],
        y: np.ndarray,
        class_names: list[str],
        scheme: str,
        alpha: float | None,
    ) -> None:
        self.client_indices = client_indices
        self.y = y
        self.class_names = class_names
        self.scheme = scheme
        self.alpha = alpha

    def __len__(self) -> int:
        return len(self.client_indices)

    @property
    def sizes(self) -> list[int]:
        return [len(idx) for idx in self.client_indices]

    def class_matrix(self) -> np.ndarray:
        """(n_clients, n_classes) count matrix — the partitioning heatmap."""
        matrix = np.zeros((len(self), len(self.class_names)), dtype=np.int64)
        for i, idx in enumerate(self.client_indices):
            values, counts = np.unique(self.y[idx], return_counts=True)
            matrix[i, values] = counts
        return matrix

    def summary(self) -> dict:
        matrix = self.class_matrix()
        return {
            "scheme": self.scheme,
            "dirichlet_alpha": self.alpha,
            "n_clients": len(self),
            "client_sizes": self.sizes,
            "client_class_counts": {
                f"client_{i}": {
                    self.class_names[c]: int(matrix[i, c])
                    for c in range(matrix.shape[1])
                    if matrix[i, c] > 0
                }
                for i in range(len(self))
            },
            # 0 = every client holds one class, 1 = every client matches global.
            "mean_label_skew": float(_mean_label_skew(matrix)),
        }


def _mean_label_skew(matrix: np.ndarray) -> float:
    """Mean normalised entropy of each client's class distribution."""
    n_classes = matrix.shape[1]
    if n_classes <= 1:
        return 1.0
    skews = []
    for row in matrix:
        total = row.sum()
        if total == 0:
            continue
        p = row / total
        p = p[p > 0]
        entropy = -(p * np.log(p)).sum()
        skews.append(entropy / np.log(n_classes))
    return float(np.mean(skews)) if skews else 0.0


def _dirichlet(
    y: np.ndarray, n_clients: int, alpha: float, min_samples: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """Standard Dirichlet label-skew partitioning (Hsu et al., 2019).

    For each class, draw a Dirichlet(alpha) vector over clients and split that
    class's samples accordingly. Small alpha concentrates each class on a few
    clients, producing realistic non-IID skew.
    """
    n_classes = int(y.max()) + 1

    # Retry until no client is starved; pathological draws happen at low alpha.
    for attempt in range(100):
        buckets: list[list[np.ndarray]] = [[] for _ in range(n_clients)]
        for c in range(n_classes):
            idx_c = np.where(y == c)[0]
            rng.shuffle(idx_c)
            proportions = rng.dirichlet(np.repeat(alpha, n_clients))
            cuts = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
            for client_id, chunk in enumerate(np.split(idx_c, cuts)):
                if len(chunk):
                    buckets[client_id].append(chunk)

        client_indices = [
            np.concatenate(b) if b else np.array([], dtype=np.int64) for b in buckets
        ]
        if min(len(idx) for idx in client_indices) >= min_samples:
            return client_indices

    logger.warning(
        "Dirichlet partitioning could not satisfy min_samples_per_client=%d "
        "after %d attempts; returning last draw. Consider raising alpha or "
        "lowering n_clients.",
        min_samples, attempt + 1,
    )
    return client_indices


def _iid(y: np.ndarray, n_clients: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Uniform random split — the IID control condition."""
    idx = np.arange(len(y))
    rng.shuffle(idx)
    return [np.sort(chunk) for chunk in np.array_split(idx, n_clients)]


def _shard(
    y: np.ndarray, n_clients: int, shards_per_client: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """Sort-and-shard partitioning (McMahan et al., 2017).

    Sort by label, cut into n_clients * shards_per_client contiguous shards, and
    deal shards out at random. With 2 shards per client this is the pathological
    non-IID setting from the original FedAvg paper.
    """
    order = np.argsort(y, kind="stable")
    n_shards = n_clients * shards_per_client
    shards = np.array_split(order, n_shards)
    shard_ids = rng.permutation(n_shards)
    return [
        np.sort(np.concatenate([shards[s] for s in shard_ids[i::n_clients]]))
        for i in range(n_clients)
    ]


def _natural(
    groups: np.ndarray, min_samples: int
) -> list[np.ndarray]:
    """One client per real site, pooling sites too small to train alone.

    Sites below `min_samples` are merged into a single additional client rather
    than dropped: discarding them would silently change the training population
    and overstate the amount of data the federation has.
    """
    unique = np.unique(groups)
    clients, small = [], []
    for site in unique:
        idx = np.where(groups == site)[0]
        (clients if len(idx) >= min_samples else small).append(idx)

    if small:
        clients.append(np.sort(np.concatenate(small)))
        logger.info(
            "Pooled %d site(s) below %d rows into one client of %d rows",
            len(small), min_samples, len(clients[-1]),
        )
    return [np.sort(c) for c in clients]


def partition(
    y: np.ndarray,
    cfg: PartitionConfig,
    class_names: list[str],
    seed: int = 42,
    groups: np.ndarray | None = None,
) -> Partition:
    """Dispatch to the configured partitioning scheme.

    `groups` carries a real per-row site identifier and is required only by the
    "natural" scheme, where client count is an output rather than an input.
    """
    rng = np.random.default_rng(seed)

    if cfg.scheme == "iid":
        indices = _iid(y, cfg.n_clients, rng)
    elif cfg.scheme == "shard":
        indices = _shard(y, cfg.n_clients, cfg.shards_per_client, rng)
    elif cfg.scheme == "dirichlet":
        indices = _dirichlet(
            y, cfg.n_clients, cfg.dirichlet_alpha, cfg.min_samples_per_client, rng
        )
    elif cfg.scheme == "natural":
        if groups is None:
            raise ValueError(
                "scheme='natural' requires a per-row site vector; set "
                "DataConfig.site_column so the pipeline carries one through"
            )
        if len(groups) != len(y):
            raise ValueError(
                f"site vector has {len(groups)} rows but y has {len(y)}"
            )
        indices = _natural(groups, cfg.min_samples_per_client)
    else:
        raise ValueError(f"Unknown partition scheme: {cfg.scheme}")

    result = Partition(
        indices, y, class_names, cfg.scheme,
        cfg.dirichlet_alpha if cfg.scheme == "dirichlet" else None,
    )
    logger.info(
        "Partitioned %d samples across %d clients (%s), sizes %s, skew %.3f",
        len(y), len(result), cfg.scheme, result.sizes, _mean_label_skew(result.class_matrix()),
    )
    return result
