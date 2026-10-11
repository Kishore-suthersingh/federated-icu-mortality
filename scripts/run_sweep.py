"""CLI entry point for ablation sweeps and the multi-seed protocol.

One parameter varies, everything else is held fixed, and every point writes a
full `results.json` under `outputs/<name>/<param>_<value>/`.

Examples
--------
Does a wider embedding recover the accuracy the encoder loses?
    python scripts/run_sweep.py --sweep embedding_dim \
        --data-dir data/ciciot2023 --max-rows -1 \
        --label-mode category --clients 15 --rounds 30 --local-epochs 2 \
        --name sweep_embed --values 16 32 64 128 \
        --skip centralized_lgbm fedavg_mlp_only random_forest \
               logistic_regression lstm

Multi-seed confidence intervals on the headline table:
    python scripts/run_sweep.py --sweep multi_seed --data-dir data/ciciot2023 ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fedlgbm import setup_logging  # noqa: E402
from fedlgbm.ablations import SWEEPS, run_multi_seed  # noqa: E402
from run_experiment import build_config  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--sweep", required=True,
                   choices=[*SWEEPS, "multi_seed"],
                   help="Which sweep to run")
    p.add_argument("--values", nargs="*", default=None,
                   help="Override the sweep's default points (numeric)")
    p.add_argument("--seeds", nargs="*", type=int, default=None,
                   help="Seeds per point. >1 gives CIs but multiplies runtime.")

    # Same experiment knobs as run_experiment.py, so a sweep is reproducible
    # from the same command line as the single run it extends.
    p.add_argument("--data-dir", required=True)
    p.add_argument("--name", default=None)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None,
                   help="Torch device; 'cpu' forces CPU even when a GPU is present")

    p.add_argument("--label-mode", choices=["binary", "category", "fine"],
                   default="binary")
    p.add_argument("--max-rows", type=int, default=None, help="-1 for all")
    p.add_argument("--max-files", type=int, default=None, help="-1 for all")
    p.add_argument("--max-per-class", type=int, default=None, help="-1 for no cap")

    p.add_argument("--clients", type=int, default=None)
    p.add_argument("--partition", choices=["dirichlet", "iid", "shard"],
                   default="dirichlet")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--client-fraction", type=float, default=1.0)

    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--local-epochs", type=int, default=None)
    p.add_argument("--embedding-dim", type=int, default=None)
    p.add_argument("--recon-weight", type=float, default=None)
    p.add_argument("--hidden-dims", nargs="*", type=int, default=None)
    p.add_argument("--lgbm-class-weight", choices=["balanced", "none"], default=None)
    p.add_argument("--lgbm-leaves", type=int, default=None)
    p.add_argument("--lgbm-estimators", type=int, default=None)

    p.add_argument("--skip", nargs="*", default=[],
                   help="Model names to skip at every sweep point")
    p.add_argument("--reference", default="fed_hybrid_mlp_lgbm",
                   help="multi_seed only: model the paired tests compare "
                        "everything else against. Must not be in --skip.")
    p.add_argument("--privacy-attacks", action="store_true",
                   help="Run inversion/membership attacks at each point "
                        "(off by default — they train an extra decoder)")

    args = p.parse_args()
    setup_logging()

    cfg = build_config(args)
    cfg.skip_privacy_attacks = not args.privacy_attacks
    skip = set(args.skip)
    seeds = tuple(args.seeds) if args.seeds else None

    if args.sweep == "multi_seed":
        run_multi_seed(cfg, seeds=seeds or cfg.seeds, skip=skip,
                       reference=args.reference)
        return

    fn = SWEEPS[args.sweep]
    if args.values:
        # Each sweep names its points differently (dims, alphas, weights, ...),
        # so bind positionally rather than guessing the keyword. Keep whole
        # numbers as ints so output dirs read `embedding_dim_64`, not `_64.0`.
        points = [
            int(v) if float(v).is_integer() else float(v) for v in args.values
        ]
        fn(cfg, points, seeds=seeds, skip=skip)
    else:
        fn(cfg, seeds=seeds, skip=skip)


if __name__ == "__main__":
    main()
