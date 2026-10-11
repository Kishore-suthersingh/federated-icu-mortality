"""Federated Hybrid MLP-LightGBM for IoT cybersecurity threat detection.

Reference implementation of the architecture described in the manuscript:
an MLP feature extractor trained across clients with FedAvg, feeding a central
LightGBM classifier.
"""

from .ablations import run_multi_seed, run_sweeps
from .attacks import evaluate_privacy
from .config import (
    DataConfig,
    ExperimentConfig,
    FedConfig,
    LGBMConfig,
    MLPConfig,
    PartitionConfig,
    smoke_config,
)
from .experiments import run_all, setup_logging
from .hybrid import FederatedHybridMLPLightGBM
from .privacy import DPConfig, RDPAccountant, calibrate_noise_multiplier
from .stats import aggregate_seeds, compare_models, summary_table

__version__ = "0.2.0"

__all__ = [
    # config
    "DataConfig",
    "DPConfig",
    "ExperimentConfig",
    "FedConfig",
    "LGBMConfig",
    "MLPConfig",
    "PartitionConfig",
    "smoke_config",
    # model
    "FederatedHybridMLPLightGBM",
    # running
    "run_all",
    "run_multi_seed",
    "run_sweeps",
    "setup_logging",
    # privacy
    "RDPAccountant",
    "calibrate_noise_multiplier",
    "evaluate_privacy",
    # statistics
    "aggregate_seeds",
    "compare_models",
    "summary_table",
]
