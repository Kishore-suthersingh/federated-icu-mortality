"""Invariants for multi-site splitting and the natural client partition.

The split mode determines what the reported numbers mean, so these tests pin
the *semantics* as well as the mechanics — including the deliberate asymmetry
that sites span splits under `random` but must not under `site_holdout`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fedlgbm import config, partition  # noqa: E402
from fedlgbm.splits import make_split  # noqa: E402


def _multisite(n_sites: int = 40, seed: int = 0):
    """Sites of varied size and varied event rate, as real hospitals are."""
    rng = np.random.default_rng(seed)
    sites, y, groups = [], [], []
    patient = 0
    for s in range(n_sites):
        n = int(rng.integers(50, 600))
        rate = float(rng.uniform(0.01, 0.30))
        for _ in range(n):
            sites.append(s)
            y.append(int(rng.random() < rate))
            groups.append(patient)
            patient += 1
    return np.array(y), np.array(sites), np.array(groups)


# -- split modes ------------------------------------------------------------


def test_random_split_preserves_prevalence() -> None:
    y, sites, _ = _multisite()
    r = make_split(y, mode="random", sites=sites, seed=1)
    overall = y.mean()
    for name in ("train", "val", "test"):
        assert abs(r.diagnostics[f"prevalence_{name}"] - overall) < 0.015


def test_random_split_is_an_exact_cover() -> None:
    y, sites, _ = _multisite()
    r = make_split(y, mode="random", sites=sites, seed=1)
    combined = np.sort(np.concatenate([r.train_idx, r.val_idx, r.test_idx]))
    assert np.array_equal(combined, np.arange(len(y)))


def test_sites_do_span_splits_under_random() -> None:
    """Deliberate: `random` measures within-site deployment.

    If a later change makes sites disjoint here, it has silently switched the
    question the headline table answers.
    """
    y, sites, _ = _multisite()
    r = make_split(y, mode="random", sites=sites, seed=1)
    assert r.diagnostics["n_sites_spanning_splits"] > 0


def test_site_holdout_keeps_sites_disjoint() -> None:
    y, sites, _ = _multisite()
    r = make_split(y, mode="site_holdout", sites=sites, seed=1,
                   min_rows_per_site=100)
    assert r.diagnostics["n_sites_spanning_splits"] == 0


def test_site_holdout_val_and_test_sites_are_disjoint() -> None:
    y, sites, _ = _multisite()
    r = make_split(y, mode="site_holdout", sites=sites, seed=1,
                   min_rows_per_site=100)
    assert not (set(r.diagnostics["held_out_test_sites"])
                & set(r.diagnostics["held_out_val_sites"]))


def test_site_holdout_is_an_exact_cover() -> None:
    y, sites, _ = _multisite()
    r = make_split(y, mode="site_holdout", sites=sites, seed=1,
                   min_rows_per_site=100)
    combined = np.sort(np.concatenate([r.train_idx, r.val_idx, r.test_idx]))
    assert np.array_equal(combined, np.arange(len(y)))


def test_group_split_keeps_groups_disjoint() -> None:
    y, sites, _ = _multisite()
    # Force real recurrence: several rows per patient.
    groups = np.arange(len(y)) // 4
    r = make_split(y, mode="group", groups=groups, seed=1)
    assert r.diagnostics["n_groups_spanning_splits"] == 0


def test_group_split_with_unique_groups_still_works() -> None:
    """The WiDS case: patient_id is row-unique, so grouping is a no-op."""
    y, _, groups = _multisite()
    r = make_split(y, mode="group", groups=groups, seed=1)
    assert r.diagnostics["n_groups_spanning_splits"] == 0
    assert len(r.train_idx) + len(r.val_idx) + len(r.test_idx) == len(y)


def test_split_is_deterministic_given_seed() -> None:
    y, sites, _ = _multisite()
    a = make_split(y, mode="site_holdout", sites=sites, seed=7)
    b = make_split(y, mode="site_holdout", sites=sites, seed=7)
    assert np.array_equal(a.test_idx, b.test_idx)


def test_different_seeds_give_different_splits() -> None:
    y, sites, _ = _multisite()
    a = make_split(y, mode="site_holdout", sites=sites, seed=7)
    b = make_split(y, mode="site_holdout", sites=sites, seed=8)
    assert not np.array_equal(a.test_idx, b.test_idx)


def test_unknown_mode_raises() -> None:
    y, _, _ = _multisite()
    with pytest.raises(ValueError, match="Unknown split mode"):
        make_split(y, mode="nonsense")


def test_group_mode_requires_groups() -> None:
    y, _, _ = _multisite()
    with pytest.raises(ValueError, match="requires a groups vector"):
        make_split(y, mode="group")


# -- natural partition ------------------------------------------------------


def _partition_natural(y, sites, min_samples=100):
    cfg = config.PartitionConfig()
    cfg.scheme = "natural"
    cfg.min_samples_per_client = min_samples
    return partition.partition(y, cfg, ["neg", "pos"], seed=1, groups=sites)


def test_natural_partition_is_a_disjoint_cover() -> None:
    y, sites, _ = _multisite()
    p = _partition_natural(y, sites)
    combined = np.sort(np.concatenate(p.client_indices))
    assert np.array_equal(combined, np.arange(len(y)))


def test_natural_clients_correspond_to_real_sites() -> None:
    """Each full-size client must contain exactly one site's rows."""
    y, sites, _ = _multisite()
    p = _partition_natural(y, sites, min_samples=1)
    for idx in p.client_indices:
        assert len(np.unique(sites[idx])) == 1


def test_small_sites_are_pooled_not_dropped() -> None:
    y, sites, _ = _multisite()
    p = _partition_natural(y, sites, min_samples=400)
    assert sum(len(c) for c in p.client_indices) == len(y)


def test_natural_requires_a_site_vector() -> None:
    y, _, _ = _multisite()
    cfg = config.PartitionConfig()
    cfg.scheme = "natural"
    with pytest.raises(ValueError, match="requires a per-row site vector"):
        partition.partition(y, cfg, ["neg", "pos"], seed=1)


def test_natural_rejects_mismatched_site_vector() -> None:
    y, sites, _ = _multisite()
    cfg = config.PartitionConfig()
    cfg.scheme = "natural"
    with pytest.raises(ValueError, match="rows but y has"):
        partition.partition(y, cfg, ["neg", "pos"], seed=1, groups=sites[:-5])


def test_natural_partition_exposes_real_label_skew() -> None:
    """Real sites include single-class clients; that is the point.

    WiDS has 10 hospitals with zero deaths — a pathology no Dirichlet draw
    produces reliably, and one the federated optimiser must survive.
    """
    y, sites, _ = _multisite()
    y = y.copy()
    y[sites == 0] = 0  # a site that has never observed the positive class
    p = _partition_natural(y, sites, min_samples=1)
    single_class = [c for c in p.client_indices if len(np.unique(y[c])) == 1]
    assert single_class
