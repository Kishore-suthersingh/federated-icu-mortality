"""End-to-end experiment runner.

`run_all` produces every number and figure the revised manuscript needs, in one
reproducible pass, and writes them to `output_dir` as JSON + PNG + a markdown
comparison table.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from . import attacks, baselines, comms, data, metrics, partition
from .config import ExperimentConfig
from .hybrid import FederatedHybridMLPLightGBM

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s",
        datefmt="%H:%M:%S",
    )


def run_all(cfg: ExperimentConfig, skip: set[str] | None = None) -> dict:
    """Run the proposed model plus every baseline on one identical split."""
    skip = skip or set()
    output_dir = Path(cfg.output_dir) / cfg.name
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(output_dir / "config.json")

    np.random.seed(cfg.seed)

    # --- Data --------------------------------------------------------
    logger.info("=" * 70)
    logger.info("Preparing data")
    ds = data.prepare(cfg.data, seed=cfg.seed)

    # Under scheme="natural" the clients are real sites carried through from the
    # data, so client count is an output rather than a configured input.
    parts = partition.partition(
        ds.y_train, cfg.partition, ds.class_names,
        seed=cfg.seed, groups=ds.train_sites,
    )
    if cfg.partition.scheme == "natural":
        cfg.partition.n_clients = len(parts)

    results: dict = {
        "config": cfg.to_dict(),
        "dataset": ds.summary(),
        "partition": parts.summary(),
        "models": {},
        "timings_sec": {},
    }

    def record(name: str, y_pred, y_proba, elapsed: float) -> None:
        results["models"][name] = metrics.compute(
            ds.y_test, y_pred, y_proba, ds.class_names
        )
        results["timings_sec"][name] = round(elapsed, 2)
        logger.info("%-26s %s", name, metrics.headline(results["models"][name]))

    def guard(name: str, fn) -> None:
        """Run one model; a failure logs and continues rather than killing the run."""
        if name in skip:
            logger.info("Skipping %s", name)
            return
        logger.info("-" * 70)
        logger.info("Running: %s", name)
        start = time.time()
        try:
            y_pred, y_proba = fn()
            record(name, y_pred, y_proba, time.time() - start)
        except Exception as exc:  # noqa: BLE001 — one bad baseline shouldn't abort
            logger.exception("%s failed: %s", name, exc)
            results["models"][name] = {"error": str(exc)}

    # --- Proposed model ----------------------------------------------
    hybrid_model = None
    if "fed_hybrid_mlp_lgbm" not in skip:
        logger.info("-" * 70)
        logger.info("Running: fed_hybrid_mlp_lgbm  (PROPOSED)")
        start = time.time()
        hybrid_model = FederatedHybridMLPLightGBM(cfg).fit(
            ds.X_train, ds.y_train, parts, ds.X_val, ds.y_val, ds.n_classes
        )
        record(
            "fed_hybrid_mlp_lgbm",
            hybrid_model.predict(ds.X_test),
            hybrid_model.predict_proba(ds.X_test),
            time.time() - start,
        )
        results["fed_history"] = hybrid_model.fed_history

        # --- Privacy: formal guarantee + measured attack success ------
        results["privacy"] = hybrid_model.privacy_summary()

        if not cfg.skip_privacy_attacks:
            logger.info("Evaluating empirical privacy (inversion + membership)")
            results["privacy_attacks"] = attacks.evaluate_privacy(
                hybrid_model,
                ds.X_train, ds.y_train, ds.X_test, ds.y_test,
                seed=cfg.seed, device=hybrid_model.device,
            )

        # --- Communication cost --------------------------------------
        results["communication"] = comms.analyse(
            hybrid_model.encoder,
            # len(parts), not the configured value: under "natural" they differ
            # and the configured one would misreport the transmitted volume.
            n_clients=len(parts),
            n_rounds=len(hybrid_model.fed_history),
            client_fraction=cfg.fed.client_fraction,
            n_train_samples=len(ds.y_train),
            n_raw_features=ds.n_features,
            embedding_dim=cfg.mlp.embedding_dim,
        )
        # Target is metric-appropriate: 0.90 accuracy is below the majority-class
        # rate on a rare binary outcome and would be "reached" in round 1.
        binary = ds.n_classes == 2
        results["convergence_efficiency"] = comms.convergence_efficiency(
            hybrid_model.fed_history,
            target=0.85 if binary else 0.90,
            metric="roc_auc" if binary else "accuracy",
        )

    # --- Baselines ----------------------------------------------------
    guard("centralized_lgbm", lambda: baselines.centralized_lgbm(
        cfg, ds.X_train, ds.y_train, ds.X_val, ds.y_val, ds.X_test, ds.n_classes))

    guard("centralized_mlp_lgbm", lambda: baselines.centralized_mlp_lgbm(
        cfg, ds.X_train, ds.y_train, ds.X_val, ds.y_val, ds.X_test, ds.n_classes))

    guard("stacked_mlp_lgbm", lambda: baselines.stacked_mlp_lgbm(
        cfg, ds.X_train, ds.y_train, ds.X_val, ds.y_val, ds.X_test, ds.n_classes))

    guard("fedavg_mlp_only", lambda: baselines.fedavg_mlp_only(
        cfg, ds.X_train, ds.y_train, parts.client_indices,
        ds.X_val, ds.y_val, ds.X_test, ds.n_classes))

    guard("random_forest", lambda: baselines.random_forest(
        cfg, ds.X_train, ds.y_train, ds.X_test))

    guard("logistic_regression", lambda: baselines.logistic_regression(
        cfg, ds.X_train, ds.y_train, ds.X_test))

    if ds.n_classes == 2:
        guard("isolation_forest", lambda: baselines.isolation_forest(
            cfg, ds.X_train, ds.y_train, ds.X_test))

    guard("lstm", lambda: baselines.lstm(
        cfg, ds.X_train, ds.y_train, ds.X_val, ds.y_val, ds.X_test, ds.n_classes))

    # --- Incumbent score ----------------------------------------------
    # A published clinical risk model shipped with the data (APACHE IVa on
    # WiDS). It is the standard of care rather than a model we trained, so it is
    # the comparison that decides whether a federated model is worth deploying —
    # a more meaningful bar than any baseline we fit ourselves.
    for name, splits in getattr(ds, "baseline_scores", {}).items():
        scores = splits.get("test")
        if scores is None or ds.n_classes != 2:
            continue
        finite = np.isfinite(scores)
        if finite.sum() < 100:
            logger.warning("Incumbent score %s has too few values; skipping", name)
            continue
        filled = np.where(finite, scores, np.nanmedian(scores[finite]))
        proba = np.column_stack([1.0 - filled, filled])
        results["models"][f"incumbent_{name}"] = metrics.compute(
            ds.y_test, (filled >= 0.5).astype(np.int64), proba, ds.class_names
        )
        results["models"][f"incumbent_{name}"]["coverage"] = float(finite.mean())

    # --- Persist ------------------------------------------------------
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    ok = {k: v for k, v in results["models"].items() if "error" not in v}
    if ok:
        table = metrics.comparison_table(ok)
        (output_dir / "comparison_table.md").write_text(table, encoding="utf-8")
        logger.info("\n%s", table)

    # --- Figures ------------------------------------------------------
    try:
        from . import figures
        figures.generate_all(results, ds, parts, hybrid_model, output_dir / "figures")
    except Exception as exc:  # noqa: BLE001 — never lose results to a plotting bug
        logger.exception("Figure generation failed: %s", exc)

    logger.info("=" * 70)
    logger.info("Done. Results written to %s", output_dir)
    return results
