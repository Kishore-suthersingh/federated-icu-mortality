"""Fetch a real CIC-IoT-2023 subsample and write it as CSV shards.

PROVENANCE — read this before citing any number produced from this data.

The canonical source is http://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/
(UNB Canadian Institute for Cybersecurity). That host is unreachable from some
networks, so this script falls back to a HuggingFace mirror that republishes a
1,342,371-row subsample under CC-BY-4.0.

The mirror differs from the canonical release in three ways that MUST be stated
in the paper's dataset section rather than glossed over:

  1. It is a subsample, not the full ~46.7M-row release. Benign is capped at
     200k and most attack classes at 50k, so the extreme class imbalance of the
     original is already partially corrected. Reported per-class scores are
     therefore optimistic relative to the untouched distribution.
  2. It carries 39 numeric features, not 46. Absent: flow_duration, Duration,
     Srate, Drate, urg_count, Magnitue, Radius, Covariance, Weight. The last
     four are deterministic functions of Min/Max/AVG/Std/Tot size, so dropping
     them is defensible; the rate/duration columns are a genuine loss. It adds
     Time_To_Live, which the original feature list lacks.
  3. It contains 32 of the 34 classes. DDoS-PSHACK_Flood and DDoS-RSTFINFlood
     are absent.

Use this to obtain real measurements now. Before submission, re-run against
the canonical CSVs (`--source local --path <dir>`) so the paper describes the
dataset the reviewers will recognise.

The mirror also ships two *derived* label columns alongside the fine-grained
`Label`: `attack_class` (the 8 coarse families) and `label` (binary). Both are
dropped here. Leaving them in place would hand the classifier the answer as a
feature and produce a meaningless ~1.00 F1 — the single most likely way to
manufacture an accidentally fraudulent result in this project.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

MIRROR = "https://huggingface.co/datasets/lacg030175/CIC-IoT-2023-raw/resolve/main"
PARTS = ("random/train-00000-of-00001.parquet", "random/test-00000-of-00001.parquet")

# Derived label columns that must never reach the feature matrix.
LEAKAGE_COLUMNS = ("attack_class", "label")
LABEL_COLUMN = "Label"


def fetch_mirror(cache: Path) -> pd.DataFrame:
    cache.mkdir(parents=True, exist_ok=True)
    frames = []
    for part in PARTS:
        local = cache / Path(part).name
        url = f"{MIRROR}/{part}"
        if not local.exists():
            print(f"downloading {url}")
            # pandas reads the URL directly via fsspec/requests.
            frame = pd.read_parquet(url)
            frame.to_parquet(local)
        else:
            print(f"using cached {local}")
            frame = pd.read_parquet(local)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_local(path: Path) -> pd.DataFrame:
    files = sorted(path.rglob("*.csv"))
    if not files:
        raise SystemExit(f"no CSV files under {path}")
    print(f"reading {len(files)} canonical CSV file(s)")
    return pd.concat((pd.read_csv(f, low_memory=False) for f in files),
                     ignore_index=True)


def strip_derived_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Remove coarse/binary label columns so only the fine label remains.

    The pipeline derives both of those itself from the fine label, so nothing
    is lost — but if they survive into the features, every metric is invalid.
    """
    present = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    if present:
        print(f"dropping derived label columns (leakage risk): {present}")
        df = df.drop(columns=present)

    if LABEL_COLUMN not in df.columns:
        raise SystemExit(
            f"expected a {LABEL_COLUMN!r} column, found {list(df.columns)[-5:]}")

    # Any *other* column whose name looks like a label is also suspect.
    suspicious = [
        c for c in df.columns
        if c != LABEL_COLUMN
        and str(c).strip().lower() in {"label", "labels", "class", "attack"}
    ]
    if suspicious:
        raise SystemExit(f"unexpected extra label-like columns: {suspicious}")
    return df


def write_shards(df: pd.DataFrame, out: Path, n_shards: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for existing in out.glob("*.csv"):
        existing.unlink()

    # Shuffle before sharding: the mirror is grouped by class, so contiguous
    # shards would each hold one attack type and `--max-files` would silently
    # select a single-class subset.
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    rows_per = -(-len(df) // n_shards)
    for i in range(n_shards):
        chunk = df.iloc[i * rows_per : (i + 1) * rows_per]
        if chunk.empty:
            break
        path = out / f"part-{i:05d}.csv"
        chunk.to_csv(path, index=False)
        print(f"wrote {path.name}: {len(chunk):,} rows")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=("mirror", "local"), default="mirror")
    ap.add_argument("--path", type=Path,
                    help="directory of canonical CSVs (--source local)")
    ap.add_argument("--out", type=Path, default=Path("data/ciciot2023"))
    ap.add_argument("--cache", type=Path, default=Path("data/.cache"))
    ap.add_argument("--shards", type=int, default=12)
    args = ap.parse_args()

    if args.source == "local":
        if not args.path:
            raise SystemExit("--source local requires --path")
        df = load_local(args.path)
    else:
        print(__doc__.split("Use this to obtain")[0])
        df = fetch_mirror(args.cache)

    print(f"\nloaded {len(df):,} rows x {df.shape[1]} columns")
    df = strip_derived_labels(df)

    counts = df[LABEL_COLUMN].value_counts()
    print(f"{len(counts)} classes, {df.shape[1] - 1} features")
    print(f"largest: {counts.index[0]} = {counts.iloc[0]:,}")
    print(f"smallest: {counts.index[-1]} = {counts.iloc[-1]:,}")

    write_shards(df, args.out, args.shards)
    print(f"\nready: point DataConfig.data_dir at {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
