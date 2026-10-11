"""Ablation sweeps and the multi-seed experimental protocol.

Each sweep here answers a question a Q1 reviewer will ask and the current
manuscript cannot:

  scalability      Does it hold up as clients grow? (5 -> 50)
  non_iid          How much does label skew cost? (alpha 0.1 -> 100)
  privacy_utility  What does epsilon actually cost in F1?
  embedding_dim    How small can the uploaded representation be?
  local_epochs     Does more local work reduce rounds, or cause client drift?
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import replace
from pathlib import Path

from .config import ExperimentConfig
from .experiments import run_all
from .stats import aggregate_seeds, compare_models, summary_table

logger = logging.getLogger(__name__)


def run_multi_seed(
    cfg: ExperimentConfig,
    seeds: tuple[int, ...] | None = None,
    skip: set[str] | None = None,
    reference: str = "fed_hybrid_mlp_lgbm",
) -> dict:
    """Run the full experiment once per seed and aggregate.

    This is the protocol every headline number in the paper should come from.

    `reference` is the model every other model is paired against. It must be a
    model the run actually produces: skipping it yields a summary whose
    significance blocks are nothing but an error string, which is only
    discovered after every seed has already been paid for.
    """
    seeds = seeds or cfg.seeds
    if skip and reference in skip:
        raise ValueError(
            f"reference model {reference!r} is in skip={sorted(skip)}, so no "
            "paired test could be computed. Pass reference= a model that runs."
        )
    runs, per_seed_dirs = [], []

    for seed in seeds:
        logger.info("=" * 70)
        logger.info("SEED %d  (%d of %d)", seed, seeds.index(seed) + 1, len(seeds))

        seed_cfg = copy.deepcopy(cfg)
        seed_cfg.seed = seed
        seed_cfg.name = f"{cfg.name}/seed_{seed}"

        result = run_all(seed_cfg, skip=skip)

        # Fail on the first seed rather than the last. run_all guards each model
        # so one failure cannot abort the sweep, which is right for baselines but
        # wrong for the reference: without it there is no paired test, and the
        # only symptom is an error string in a summary produced hours later.
        ref_result = result.get("models", {}).get(reference, {})
        if "error" in ref_result:
            raise RuntimeError(
                f"reference model {reference!r} failed on seed {seed} — aborting "
                f"rather than running {len(seeds) - 1} more seeds that cannot "
                f"produce a paired test. Underlying error: {ref_result['error']}"
            )

        runs.append(result)
        per_seed_dirs.append(seed_cfg.name)

    aggregated = aggregate_seeds(runs)
    significance = compare_models(runs, reference=reference, metric="f1_macro")

    output_dir = Path(cfg.output_dir) / cfg.name
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "seeds": list(seeds),
        "per_seed_dirs": per_seed_dirs,
        "reference": reference,
        "aggregated": aggregated,
        "significance_f1_macro": significance,
        "significance_accuracy": compare_models(
            runs, reference=reference, metric="accuracy"
        ),
    }
    (output_dir / "multi_seed_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    table = summary_table(aggregated)
    (output_dir / "multi_seed_table.md").write_text(table, encoding="utf-8")
    logger.info("\n%s", table)

    return summary


def _sweep(
    cfg: ExperimentConfig,
    name: str,
    values: list,
    apply_value,
    seeds: tuple[int, ...] | None,
    skip: set[str] | None,
) -> dict:
    """Shared sweep driver: vary one parameter, hold everything else fixed."""
    seeds = seeds or (cfg.seed,)
    results = {}

    for value in values:
        logger.info("=" * 70)
        logger.info("SWEEP %s = %s", name, value)

        sweep_cfg = copy.deepcopy(cfg)
        sweep_cfg.name = f"{cfg.name}/{name}_{value}"
        apply_value(sweep_cfg, value)

        try:
            if len(seeds) > 1:
                results[str(value)] = run_multi_seed(sweep_cfg, seeds, skip)
            else:
                sweep_cfg.seed = seeds[0]
                results[str(value)] = run_all(sweep_cfg, skip=skip)
        except Exception as exc:  # noqa: BLE001 — one bad point shouldn't kill the sweep
            logger.exception("Sweep point %s=%s failed: %s", name, value, exc)
            results[str(value)] = {"error": str(exc)}

    output_dir = Path(cfg.output_dir) / cfg.name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"sweep_{name}.json").write_text(
        json.dumps({"parameter": name, "values": values, "results": results}, indent=2),
        encoding="utf-8",
    )
    return results


def sweep_scalability(cfg, client_counts=(5, 10, 15, 25, 50), seeds=None, skip=None):
    """Does the method survive more clients? Smaller shards, more aggregation."""
    def apply(c, value):
        c.partition.n_clients = value
        # Guard: many clients over a fixed dataset can starve each one.
        c.partition.min_samples_per_client = max(
            10, min(50, len(range(value)) and 1000 // value)
        )
    return _sweep(cfg, "n_clients", list(client_counts), apply, seeds, skip)


def sweep_non_iid(cfg, alphas=(0.1, 0.3, 0.5, 1.0, 10.0, 100.0), seeds=None, skip=None):
    """Robustness to label skew. alpha->0 is pathological, alpha->inf is IID."""
    def apply(c, value):
        c.partition.scheme = "dirichlet"
        c.partition.dirichlet_alpha = value
        c.partition.min_samples_per_client = 10  # low alpha starves clients
    return _sweep(cfg, "alpha", list(alphas), apply, seeds, skip)


def sweep_privacy_utility(
    cfg,
    epsilons=(0.5, 1.0, 3.0, 5.0, 10.0),
    seeds=None,
    skip=None,
    channels: str = "both",
):
    """The privacy-utility curve. Includes a no-DP reference point.

    `channels` selects which release is protected, and the distinction matters
    more than it first appears, because the two have very different economics:

    "updates"     user-level DP-FedAvg on the model-update channel. Aggregate
                  noise scales as 1/m in the number of clients, so with a
                  cohort of real hospitals — 82 here, not the thousands assumed
                  by cross-device FL — the noise needed for a meaningful budget
                  can swamp the signal entirely.
    "embeddings"  record-level DP on the uploaded representations. The
                  population is the 64k training records rather than the 82
                  sites, so the same budget costs far less utility.
    "both"        the full guarantee, and the composition of the two.

    Sweeping them separately answers the question a reviewer actually asks:
    not "is DP expensive?" but "which channel can you afford to protect?"
    """
    if channels not in {"both", "updates", "embeddings"}:
        raise ValueError(
            f"channels must be 'both', 'updates' or 'embeddings', got {channels!r}"
        )

    def apply(c, value):
        if value is None:
            c.dp.enabled = False
            c.dp.protect_embeddings = False
            return
        c.dp.enabled = True
        c.dp.protect_embeddings = channels in {"both", "embeddings"}
        # Under "embeddings" the update channel must carry no noise, otherwise
        # the point measures the composition rather than the channel.
        if channels == "embeddings":
            c.dp.clip_norm = float("inf")
            c.dp.adaptive_clipping = False
            c.dp.noise_multiplier = 0.0
            c.dp.target_epsilon = None
        else:
            c.dp.target_epsilon = value
            c.dp.noise_multiplier = None
        c.dp.embedding_target_epsilon = value if c.dp.protect_embeddings else None
        c.dp.embedding_noise_multiplier = None

    return _sweep(cfg, "epsilon", [None, *epsilons], apply, seeds, skip)


def sweep_embedding_dim(cfg, dims=(8, 16, 32, 64, 128), seeds=None, skip=None):
    """How compact can the uploaded representation be before accuracy drops?

    Directly relevant to both bandwidth and privacy: a smaller embedding is
    cheaper to send and carries less invertible information.
    """
    def apply(c, value):
        c.mlp.embedding_dim = value
    return _sweep(cfg, "embedding_dim", list(dims), apply, seeds, skip)


def sweep_reconstruction(cfg, weights=(0.0, 0.1, 0.5, 1.0, 2.0), seeds=None, skip=None):
    """How much does forcing the embedding to stay information-preserving help?

    0.0 is the original cross-entropy-only encoder. Higher weights buy accuracy
    by keeping the embedding faithful to its input — and, for exactly the same
    reason, are expected to make it easier to invert. Report utility and
    inversion leakage together; this sweep is the source of both.
    """
    def apply(c, value):
        c.mlp.reconstruction_weight = value
    return _sweep(cfg, "recon_weight", list(weights), apply, seeds, skip)


def sweep_local_epochs(cfg, epochs=(1, 2, 5, 10), seeds=None, skip=None):
    """More local computation per round trades communication for client drift."""
    def apply(c, value):
        c.fed.local_epochs = value
    return _sweep(cfg, "local_epochs", list(epochs), apply, seeds, skip)


SWEEPS = {
    "scalability": sweep_scalability,
    "non_iid": sweep_non_iid,
    "privacy_utility": sweep_privacy_utility,
    "embedding_dim": sweep_embedding_dim,
    "local_epochs": sweep_local_epochs,
    "reconstruction": sweep_reconstruction,
}


def run_sweeps(
    cfg: ExperimentConfig,
    which: list[str] | None = None,
    seeds: tuple[int, ...] | None = None,
    skip: set[str] | None = None,
) -> dict:
    """Run the named sweeps (all of them by default)."""
    which = which or list(SWEEPS)
    # Baselines are irrelevant inside a sweep of the proposed model's settings.
    skip = skip or {
        "centralized_lgbm", "centralized_mlp_lgbm", "fedavg_mlp_only",
        "random_forest", "logistic_regression", "isolation_forest", "lstm",
    }

    out = {}
    for name in which:
        if name not in SWEEPS:
            logger.warning("Unknown sweep %r; known: %s", name, list(SWEEPS))
            continue
        out[name] = SWEEPS[name](cfg, seeds=seeds, skip=skip)
    return out
