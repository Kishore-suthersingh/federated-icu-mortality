"""Split strategies for multi-site tabular data.

Three modes, answering different questions. The mode is a reported experimental
condition, not an implementation detail, because the choice changes what the
numbers mean:

``random``
    Stratified random split. Sites appear in both train and test, patients do
    not (on data where a patient contributes one row, these coincide). This
    isolates *the cost of federation*: federated and centralized models see
    identically distributed test data, so any utility gap is attributable to
    FedAvg and DP rather than to distribution shift.

``group``
    As above but no group (patient) may span splits. Required wherever one
    entity contributes several rows — Diabetes-130's ``patient_nbr`` recurs,
    WiDS's ``patient_id`` does not.

``site_holdout``
    Entire sites are held out. This measures *what federation buys*: exposure to
    case mix from many institutions is the whole argument for multi-site
    training, and generalisation to an unseen hospital is the only condition
    where that benefit can appear.

Neither of the first two is "more correct" than the third. Splitting hospitals
across train and test is not leakage under ``random``/``group``; it is the
deployment story (each hospital's future patients). Under ``site_holdout`` it
would be definitional leakage. Report both.

Note that AUPRC is not comparable across modes: held-out hospitals have a
different base rate, so `prevalence` is emitted alongside every split.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

logger = logging.getLogger(__name__)


@dataclass
class SplitResult:
    """Index arrays plus the diagnostics that go into results.json."""

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        sizes = len(self.train_idx) + len(self.val_idx) + len(self.test_idx)
        overlap = (
            set(self.train_idx.tolist())
            & set(self.val_idx.tolist())
            & set(self.test_idx.tolist())
        )
        if overlap:
            raise AssertionError("splits overlap")
        self.diagnostics.setdefault("n_train", int(len(self.train_idx)))
        self.diagnostics.setdefault("n_val", int(len(self.val_idx)))
        self.diagnostics.setdefault("n_test", int(len(self.test_idx)))
        self.diagnostics.setdefault("n_total", int(sizes))


def make_split(
    y: np.ndarray,
    *,
    mode: str = "random",
    groups: np.ndarray | None = None,
    sites: np.ndarray | None = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
    min_rows_per_site: int = 200,
) -> SplitResult:
    """Build a train/val/test split under the requested regime."""
    n = len(y)
    idx = np.arange(n)

    if mode == "random":
        result = _random(idx, y, test_size, val_size, seed)
    elif mode == "group":
        if groups is None:
            raise ValueError("mode='group' requires a groups vector")
        result = _group(idx, y, groups, test_size, val_size, seed)
    elif mode == "site_holdout":
        if sites is None:
            raise ValueError("mode='site_holdout' requires a sites vector")
        result = _site_holdout(idx, y, sites, test_size, val_size, seed,
                               min_rows_per_site)
    else:
        raise ValueError(f"Unknown split mode: {mode!r}")

    _add_diagnostics(result, y, groups, sites, mode)
    d = result.diagnostics
    logger.info(
        "Split (%s): %d train / %d val / %d test | prevalence %.4f/%.4f/%.4f",
        mode, d["n_train"], d["n_val"], d["n_test"],
        d["prevalence_train"], d["prevalence_val"], d["prevalence_test"],
    )
    return result


# -- strategies -----------------------------------------------------------


def _random(idx, y, test_size, val_size, seed) -> SplitResult:
    temp, test = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=y
    )
    val_fraction = val_size / (1.0 - test_size)
    train, val = train_test_split(
        temp, test_size=val_fraction, random_state=seed, stratify=y[temp]
    )
    return SplitResult(train, val, test)


def _group(idx, y, groups, test_size, val_size, seed) -> SplitResult:
    """Stratified split that keeps every group wholly within one partition.

    StratifiedGroupKFold balances class counts across folds while respecting
    groups, which is the closest achievable approximation when stratification
    and grouping conflict.
    """
    n_splits = max(2, int(round(1.0 / test_size)))
    outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    temp_pos, test_pos = next(outer.split(idx.reshape(-1, 1), y, groups))
    temp, test = idx[temp_pos], idx[test_pos]

    val_fraction = val_size / (1.0 - test_size)
    inner_splits = max(2, int(round(1.0 / val_fraction)))
    inner = StratifiedGroupKFold(
        n_splits=inner_splits, shuffle=True, random_state=seed
    )
    train_pos, val_pos = next(
        inner.split(temp.reshape(-1, 1), y[temp], groups[temp])
    )
    return SplitResult(temp[train_pos], temp[val_pos], test)


def _site_holdout(idx, y, sites, test_size, val_size, seed, min_rows_per_site) -> SplitResult:
    """Hold out whole sites, stratified by site size and site event rate.

    Drawing sites uniformly at random shifts test prevalence by several absolute
    points, because sites vary enormously in both size and outcome rate. So
    sites are binned on (size tercile x event-rate tercile) and drawn from
    within each bin, which keeps the held-out population representative.
    """
    rng = np.random.default_rng(seed)
    unique = np.unique(sites)

    size = np.array([(sites == s).sum() for s in unique])
    rate = np.array([y[sites == s].mean() for s in unique])

    # Sites too small to be a meaningful client stay in the pool but are never
    # chosen as held-out test sites, where they would add noise without adding
    # coverage.
    eligible = size >= min_rows_per_site

    def _tercile(values: np.ndarray) -> np.ndarray:
        edges = np.quantile(values, [1 / 3, 2 / 3])
        return np.digitize(values, edges)

    strata = _tercile(size) * 3 + _tercile(rate)

    target = test_size * len(y)
    chosen: list = []
    taken = 0
    # Round-robin across strata so no single bin dominates the holdout.
    order = rng.permutation(np.unique(strata))
    pools = {
        s: list(rng.permutation(np.where((strata == s) & eligible)[0]))
        for s in order
    }
    while taken < target and any(pools.values()):
        for s in order:
            if not pools[s]:
                continue
            j = pools[s].pop()
            chosen.append(unique[j])
            taken += size[j]
            if taken >= target:
                break

    test_sites = set(chosen.tolist() if isinstance(chosen, np.ndarray) else chosen)
    test_mask = np.isin(sites, list(test_sites))
    test = idx[test_mask]
    remaining = idx[~test_mask]

    # Validation must also be whole held-out sites, disjoint from test —
    # otherwise early stopping is tuned on the wrong distribution.
    rem_sites = np.unique(sites[remaining])
    rem_size = np.array([(sites[remaining] == s).sum() for s in rem_sites])
    val_target = val_size * len(y)
    shuffled = rng.permutation(len(rem_sites))
    val_sites: list = []
    taken = 0
    for j in shuffled:
        if taken >= val_target:
            break
        val_sites.append(rem_sites[j])
        taken += rem_size[j]

    val_mask = np.isin(sites, val_sites)
    val = idx[val_mask & ~test_mask]
    train = idx[~val_mask & ~test_mask]

    return SplitResult(
        train, val, test,
        diagnostics={
            "held_out_test_sites": sorted(int(s) for s in test_sites),
            "held_out_val_sites": sorted(int(s) for s in val_sites),
            "n_sites_excluded_from_holdout_pool": int((~eligible).sum()),
        },
    )


# -- diagnostics ----------------------------------------------------------


def _add_diagnostics(result, y, groups, sites, mode) -> None:
    d = result.diagnostics
    d["mode"] = mode
    for name, part in (("train", result.train_idx), ("val", result.val_idx),
                       ("test", result.test_idx)):
        d[f"prevalence_{name}"] = float(y[part].mean()) if len(part) else float("nan")

    def _spanning(vector: np.ndarray) -> int:
        if vector is None:
            return 0
        tr, va, te = (set(vector[p].tolist()) for p in
                      (result.train_idx, result.val_idx, result.test_idx))
        return len((tr & te) | (tr & va) | (va & te))

    if groups is not None:
        d["n_groups"] = int(len(np.unique(groups)))
        d["n_groups_spanning_splits"] = _spanning(groups)
    if sites is not None:
        d["n_sites"] = int(len(np.unique(sites)))
        d["n_sites_spanning_splits"] = _spanning(sites)
        d["n_sites_train"] = int(len(np.unique(sites[result.train_idx])))
