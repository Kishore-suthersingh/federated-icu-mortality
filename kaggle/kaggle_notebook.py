"""Kaggle notebook driver — paste as a single cell, or attach as a script.

Setup on Kaggle
---------------
1. New Notebook, accelerator = **None (CPU)**.
   GPU is not useful here: the MLP is tiny and LightGBM is CPU-bound.
   Leaving it off preserves your 30h/week GPU quota.
2. Add Data -> search "CIC IoT 2023" -> attach. Note the mount path under
   /kaggle/input/, then set DATA_DIR below.
3. Add Data -> Your Datasets -> attach the `fedlgbm-src` dataset created by
   scripts/push_to_kaggle.py.
4. Settings -> Persistence -> Files only (so outputs survive).
5. Run All. Expect roughly 1.5-3 h for the full config on 4 CPU cores.

Everything written to /kaggle/working/outputs is downloadable afterwards.
"""

import sys
from pathlib import Path

# --- locate the attached source dataset ------------------------------
SRC_CANDIDATES = [
    Path("/kaggle/input/fedlgbm-src/src"),
    Path("/kaggle/input/fedlgbm-src"),
    Path("src"),
]
for candidate in SRC_CANDIDATES:
    if (candidate / "fedlgbm").exists():
        sys.path.insert(0, str(candidate))
        print(f"Using source at: {candidate}")
        break
else:
    raise RuntimeError(
        "Could not find the fedlgbm package. Attach the fedlgbm-src dataset, "
        f"or adjust SRC_CANDIDATES. Tried: {SRC_CANDIDATES}"
    )

import lightgbm  # noqa: E402
import torch  # noqa: E402

from fedlgbm import ExperimentConfig, run_all, setup_logging  # noqa: E402

setup_logging()
print(f"torch {torch.__version__} | lightgbm {lightgbm.__version__} | "
      f"cuda={torch.cuda.is_available()}")

# --- point this at the attached CIC-IoT-2023 mount -------------------
DATA_DIR = Path("/kaggle/input/cic-iot-2023")
if not DATA_DIR.exists():
    found = sorted(Path("/kaggle/input").glob("*"))
    raise FileNotFoundError(
        f"{DATA_DIR} not found. Attached datasets: {[p.name for p in found]}\n"
        "Set DATA_DIR to the correct mount path."
    )

# --- configuration ---------------------------------------------------
cfg = ExperimentConfig(name="fed_binary_full")
cfg.output_dir = Path("/kaggle/working/outputs")
cfg.n_jobs = 4                       # Kaggle CPU notebooks give 4 cores
cfg.device = "auto"

cfg.data.data_dir = DATA_DIR
cfg.data.label_mode = "binary"       # switch to "category" for the 8-class run
cfg.data.max_rows = 2_000_000        # None reads all ~46M rows (will not fit)
cfg.data.max_files = None
cfg.data.max_per_class = 100_000     # tame the DDoS skew

cfg.partition.n_clients = 15
cfg.partition.scheme = "dirichlet"
cfg.partition.dirichlet_alpha = 0.5

cfg.fed.n_rounds = 30
cfg.fed.local_epochs = 2
cfg.fed.batch_size = 512
cfg.fed.client_fraction = 1.0

cfg.mlp.hidden_dims = (128, 64)
cfg.mlp.embedding_dim = 32

cfg.lgbm.n_estimators = 500

results = run_all(cfg)

print("\n" + "=" * 70)
print(Path(cfg.output_dir, cfg.name, "comparison_table.md").read_text())
