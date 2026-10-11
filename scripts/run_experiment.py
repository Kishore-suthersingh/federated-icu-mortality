"""CLI entry point for a full experiment run.

Examples
--------
Smoke test on synthetic data (seconds):
    python scripts/run_experiment.py --data-dir data/synthetic --smoke

Full run on real CIC-IoT-2023:
    python scripts/run_experiment.py --data-dir data/cic-iot-2023 \
        --clients 15 --rounds 30 --label-mode binary --name fed_binary
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fedlgbm import ExperimentConfig, run_all, setup_logging, smoke_config  # noqa: E402


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    cfg = smoke_config() if args.smoke else ExperimentConfig()
    cfg.name = args.name or cfg.name
    cfg.seed = args.seed
    cfg.output_dir = Path(args.output_dir)
    if getattr(args, "device", None):
        cfg.device = args.device

    cfg.data.data_dir = Path(args.data_dir)
    cfg.data.label_mode = args.label_mode
    if args.max_rows is not None:
        cfg.data.max_rows = None if args.max_rows < 0 else args.max_rows
    if args.max_files is not None:
        cfg.data.max_files = None if args.max_files < 0 else args.max_files
    if args.max_per_class is not None:
        cfg.data.max_per_class = None if args.max_per_class < 0 else args.max_per_class

    if args.clients is not None:
        cfg.partition.n_clients = args.clients
    cfg.partition.scheme = args.partition
    cfg.partition.dirichlet_alpha = args.alpha

    if args.rounds is not None:
        cfg.fed.n_rounds = args.rounds
    if args.local_epochs is not None:
        cfg.fed.local_epochs = args.local_epochs
    cfg.fed.client_fraction = args.client_fraction

    if args.embedding_dim is not None:
        cfg.mlp.embedding_dim = args.embedding_dim
    if getattr(args, "recon_weight", None) is not None:
        cfg.mlp.reconstruction_weight = args.recon_weight
    if getattr(args, "hidden_dims", None):
        cfg.mlp.hidden_dims = tuple(args.hidden_dims)
    if getattr(args, "lgbm_class_weight", None) is not None:
        # "none" on the command line means "do not reweight" — distinct from
        # the flag being absent, which leaves the config default alone.
        cfg.lgbm.class_weight = (
            None if args.lgbm_class_weight == "none" else args.lgbm_class_weight
        )
    if getattr(args, "lgbm_leaves", None) is not None:
        cfg.lgbm.num_leaves = args.lgbm_leaves
    if getattr(args, "lgbm_estimators", None) is not None:
        cfg.lgbm.n_estimators = args.lgbm_estimators

    return cfg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True,
                   help="Directory of CIC-IoT-2023 CSV parts")
    p.add_argument("--name", default=None, help="Run name (output subdirectory)")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny/fast config for pipeline verification")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None,
                   help="Torch device. 'auto' picks CUDA when present; 'cpu' "
                        "forces CPU even on a machine with a working GPU.")

    p.add_argument("--label-mode", choices=["binary", "category", "fine"],
                   default="binary")
    p.add_argument("--max-rows", type=int, default=None, help="-1 for all")
    p.add_argument("--max-files", type=int, default=None, help="-1 for all")
    p.add_argument("--max-per-class", type=int, default=None, help="-1 for no cap")

    p.add_argument("--clients", type=int, default=None)
    p.add_argument("--partition", choices=["dirichlet", "iid", "shard"],
                   default="dirichlet")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Dirichlet concentration; lower = more non-IID")
    p.add_argument("--client-fraction", type=float, default=1.0)

    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--local-epochs", type=int, default=None)
    p.add_argument("--embedding-dim", type=int, default=None)
    p.add_argument("--recon-weight", type=float, default=None,
                   help="Weight on the encoder's auxiliary reconstruction loss. "
                        "0 disables the decoder (original architecture).")
    p.add_argument("--hidden-dims", nargs="*", type=int, default=None,
                   help="Encoder hidden layer widths, e.g. --hidden-dims 256 128")

    p.add_argument("--lgbm-class-weight", choices=["balanced", "none"], default=None,
                   help="'balanced' favours rare classes and costs raw accuracy; "
                        "'none' optimises accuracy directly")
    p.add_argument("--lgbm-leaves", type=int, default=None)
    p.add_argument("--lgbm-estimators", type=int, default=None)

    p.add_argument("--skip", nargs="*", default=[],
                   help="Model names to skip, e.g. --skip lstm isolation_forest")

    args = p.parse_args()
    setup_logging()
    run_all(build_config(args), skip=set(args.skip))


if __name__ == "__main__":
    main()
