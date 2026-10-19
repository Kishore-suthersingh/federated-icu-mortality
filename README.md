# Federated hybrid MLP–LightGBM: cost, leakage and communication of cross-silo federated learning

Code for *Cross-Silo Federated Learning for ICU Mortality Prediction Across 147
Hospitals (82 Site-Level Clients): Cost of Federation, Residual Leakage and
Communication*.

A multilayer-perceptron encoder is trained across hospitals with federated
averaging. Each client then uploads 64-dimensional embeddings of its own
records, and a central LightGBM classifier is trained on them. The repository
measures three things about this design on an observed (not simulated) hospital
partition:

| Question | Where |
|---|---|
| What does federation cost against pooling the data? | within-site split (`splits.py`, `split_mode="random"`) |
| Does it generalise to unseen hospitals? | cross-site split (`split_mode="site_holdout"`) |
| What do the uploaded channels leak, and what does DP remove? | `attacks.py`, `privacy.py`, `ablations.sweep_privacy_utility` |

## Data

The GOSSIS cohort as released for the WiDS Datathon 2020
(<https://www.kaggle.com/c/widsdatathon2020>). You must accept the competition
rules to download it. The data are not redistributed here. Place
`training_v2.csv` under `data/wids/`.

`src/fedlgbm/datasets/` also contains loaders for CIC-IoT-2023 and
Diabetes-130-US Hospitals. They are used by the tests and not by the paper's
results.

## Layout

```
src/fedlgbm/
  fedavg.py       FedAvg trainer (sample-weighted), optional DP-FedAvg
  hybrid.py       federated encoder -> client-side embeddings -> central LightGBM
  partition.py    "natural" partition: one client per hospital, small sites pooled
  splits.py       within-site and site-holdout splits
  preprocess.py   preprocessing fitted on the training partition only
  privacy.py      Gaussian mechanisms, RDP accountant, adaptive clipping
  attacks.py      embedding-inversion and membership-inference attacks
  comms.py        communication accounting
  baselines.py    centralized LightGBM, logistic regression, random forest, ...
  ablations.py    multi-seed protocol and parameter sweeps
notebooks/        Kaggle notebooks that produced the paper's runs
scripts/          CLI runners and figure scripts
tests/            pytest suite
```

## Reproducing the paper

The results were produced on Kaggle (CPU) by these notebooks:

| Notebook | Produces |
|---|---|
| `notebooks/kaggle_wids.ipynb` | single-seed run of both regimes, including attacks and the centralized-cascade and stacking ablations |
| `notebooks/kaggle_wids_multiseed.ipynb` | five-seed (42–46) runs of both regimes: discrimination, calibration and paired tests |
| `notebooks/kaggle_wids_privacy.ipynb` | single-seed record-level and user-level DP sweeps with attacks |
| `notebooks/kaggle_wids_revision.ipynb` | revision analyses: five-seed attacks with per-feature reconstruction error and per-record predictions, the five-seed DP sweep, and the client-pooling sensitivity (thresholds 2, 50, 100) |

Figures come from the stored outputs:

```
python scripts/make_figures.py --artefacts <outputs dir> --out figures/
python scripts/make_revision_figures.py --artefacts <outputs dir> --out figures/revision
```

The revision analyses (comparison with APACHE IVa on valid scores, reliability
diagrams, five-seed attacks and DP sweep, per-feature leakage, pooling
sensitivity) are summarised from the revision notebook's outputs, with a check
that the rerun reproduces the original per-seed AUROCs:

```
python scripts/analyze_revision.py --revision <revision outputs dir>     --original <multiseed outputs dir> --out figures/revision
```

Two evaluation details matter. APACHE IVa probabilities are missing for some
stays and stored as -1 for others; compare against it only on valid scores in
[0, 1]. The two diagnosis-code columns are ordinal-coded and unscaled, so they
dominate the pooled leakage ratio; report it per feature or without them.

Results are not bit-identical across devices or software stacks. Keep every run
that feeds one aggregate on the same stack.

## Known issue

When user-level DP is enabled, `privacy.aggregate_dp` adds noise to every
floating-point entry of the encoder state. That includes the batch-normalisation
running variances, which can go negative and make inference return non-finite
values. The paper withdraws its user-level DP result for this reason. A correct
evaluation must exclude normalisation statistics from the perturbed update.

## Setup

```
pip install -r requirements.txt
pytest
```
