"""Find the LightGBM ceiling on raw features.

The hybrid can never beat plain LightGBM on raw features — the encoder can at
best stop destroying information, not create it. So before tuning the encoder,
establish how high that ceiling actually is. Whatever this script reports is
the number the hybrid is chasing.

Loads the data once and reuses the identical split across every config, so the
comparison is controlled and the whole grid costs one data prep.

    python scripts/tune_lgbm.py --data-dir data/ciciot2023 --max-rows -1 \
        --label-mode category
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fedlgbm import setup_logging  # noqa: E402
from fedlgbm import data, metrics  # noqa: E402
from fedlgbm.config import DataConfig  # noqa: E402

# Each entry isolates one hypothesis about why RF is beating LightGBM.
GRID: dict[str, dict] = {
    # Exactly what the committed run used — the number to beat.
    "current": dict(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        class_weight="balanced",
    ),
    # RF wins accuracy by leaning on majority classes. Does dropping the
    # balancing let LightGBM optimise the metric it is actually scored on?
    "unweighted": dict(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        class_weight=None,
    ),
    # Is 31 leaves simply too little capacity for 8 classes x 39 features?
    "wide_balanced": dict(
        n_estimators=1000, learning_rate=0.05, num_leaves=255,
        class_weight="balanced",
    ),
    "wide_unweighted": dict(
        n_estimators=1000, learning_rate=0.05, num_leaves=255,
        class_weight=None,
    ),
    # Capacity plus a slower, longer fit — the usual strong tabular setting.
    "deep_unweighted": dict(
        n_estimators=2000, learning_rate=0.03, num_leaves=127,
        min_child_samples=10, class_weight=None,
    ),
    # Middle ground: keep some rare-class pressure without full balancing.
    "sqrt_balanced": dict(
        n_estimators=1000, learning_rate=0.05, num_leaves=255,
        class_weight="balanced", min_child_samples=50,
    ),
}

DEFAULTS = dict(
    max_depth=-1, min_child_samples=20, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=1.0,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--label-mode", choices=["binary", "category", "fine"],
                   default="category")
    p.add_argument("--max-rows", type=int, default=-1, help="-1 for all")
    p.add_argument("--max-per-class", type=int, default=None, help="-1 for no cap")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="outputs/lgbm_ceiling.json")
    p.add_argument("--skip-rf", action="store_true")
    args = p.parse_args()

    setup_logging()

    dcfg = DataConfig(data_dir=Path(args.data_dir), label_mode=args.label_mode)
    dcfg.max_rows = None if args.max_rows < 0 else args.max_rows
    if args.max_per_class is not None:
        dcfg.max_per_class = None if args.max_per_class < 0 else args.max_per_class

    print(f"Preparing data from {args.data_dir} ...", flush=True)
    ds = data.prepare(dcfg, seed=args.seed)
    print(f"train={ds.X_train.shape} val={ds.X_val.shape} test={ds.X_test.shape} "
          f"classes={ds.n_classes}", flush=True)

    results: dict[str, dict] = {}

    for name, overrides in GRID.items():
        params = {**DEFAULTS, **overrides}
        print(f"\n=== {name}: {overrides}", flush=True)
        start = time.time()

        model = lgb.LGBMClassifier(
            **params, n_jobs=-1, random_state=args.seed, verbose=-1
        )
        model.fit(
            ds.X_train, ds.y_train,
            eval_set=[(ds.X_val, ds.y_val)],
            eval_metric="multi_logloss" if ds.n_classes > 2 else "binary_logloss",
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(period=0)],
        )

        m = metrics.compute(
            ds.y_test, model.predict(ds.X_test),
            model.predict_proba(ds.X_test), ds.class_names,
        )
        m["seconds"] = round(time.time() - start, 1)
        m["params"] = {k: v for k, v in overrides.items()}
        m["best_iteration"] = getattr(model, "best_iteration_", None)
        results[name] = m
        print(f"{name:18s} acc={m['accuracy']:.4f} bal_acc={m['balanced_accuracy']:.4f} "
              f"F1_macro={m['f1_macro']:.4f} MCC={m['mcc']:.4f} "
              f"({m['seconds']}s, best_iter={m['best_iteration']})", flush=True)

    # The number we are trying to beat, on this exact split.
    if not args.skip_rf:
        print("\n=== random_forest (reference)", flush=True)
        start = time.time()
        rf = RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=args.seed,
            class_weight="balanced",
        )
        rf.fit(ds.X_train, ds.y_train)
        m = metrics.compute(
            ds.y_test, rf.predict(ds.X_test),
            rf.predict_proba(ds.X_test), ds.class_names,
        )
        m["seconds"] = round(time.time() - start, 1)
        results["random_forest"] = m
        print(f"{'random_forest':18s} acc={m['accuracy']:.4f} "
              f"bal_acc={m['balanced_accuracy']:.4f} "
              f"F1_macro={m['f1_macro']:.4f} MCC={m['mcc']:.4f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n| Config | Accuracy | Bal. Acc | F1 (macro) | MCC |")
    print("|---|---|---|---|---|")
    for name, m in sorted(results.items(), key=lambda kv: -kv[1]["accuracy"]):
        print(f"| {name} | {m['accuracy']:.4f} | {m['balanced_accuracy']:.4f} "
              f"| {m['f1_macro']:.4f} | {m['mcc']:.4f} |")
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
