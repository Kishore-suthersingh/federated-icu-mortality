"""Generate CIC-IoT-2023-shaped synthetic CSVs.

FOR PIPELINE TESTING ONLY. This exists so the code can be verified end-to-end
without the 13 GB download. Numbers produced from this data are meaningless and
must never appear in the paper — that conflation is precisely what got the
manuscript rejected.

Mimics the real schema: 46 numeric flow features + a `label` column carrying
fine-grained CIC-IoT-2023 class names, spread across several part files, with
realistic class imbalance, missing values, infinities and duplicate rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    "flow_duration", "Header_Length", "Protocol Type", "Duration", "Rate",
    "Srate", "Drate", "fin_flag_number", "syn_flag_number", "rst_flag_number",
    "psh_flag_number", "ack_flag_number", "ece_flag_number", "cwr_flag_number",
    "ack_count", "syn_count", "fin_count", "urg_count", "rst_count",
    "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP",
    "DHCP", "ARP", "ICMP", "IPv", "LLC", "Tot sum", "Min", "Max", "AVG",
    "Std", "Tot size", "IAT", "Number", "Magnitue", "Radius", "Covariance",
    "Variance", "Weight",
]

# (label, relative frequency) — mirrors the real dataset's heavy DDoS skew.
LABELS = [
    ("BenignTraffic", 0.10),
    ("DDoS-ICMP_Flood", 0.18), ("DDoS-UDP_Flood", 0.14),
    ("DDoS-TCP_Flood", 0.12), ("DDoS-SYN_Flood", 0.10),
    ("DoS-UDP_Flood", 0.09), ("DoS-TCP_Flood", 0.07), ("DoS-SYN_Flood", 0.05),
    ("Mirai-greeth_flood", 0.05), ("Mirai-udpplain", 0.04),
    ("Recon-PortScan", 0.02), ("Recon-HostDiscovery", 0.015),
    ("MITM-ArpSpoofing", 0.012), ("DNS_Spoofing", 0.010),
    ("DictionaryBruteForce", 0.006), ("XSS", 0.003),
    ("SqlInjection", 0.002), ("Backdoor_Malware", 0.002),
]


def generate(n_rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = [name for name, _ in LABELS]
    probs = np.array([p for _, p in LABELS])
    probs = probs / probs.sum()

    labels = rng.choice(names, size=n_rows, p=probs)

    # Give each class its own mean vector so the classes are genuinely
    # separable — otherwise every model scores at chance and the smoke test
    # can't distinguish "pipeline works" from "pipeline broken".
    offsets = {name: rng.normal(0, 2.5, size=len(FEATURES)) for name in names}

    X = rng.normal(0, 1, size=(n_rows, len(FEATURES)))
    for i, label in enumerate(labels):
        X[i] += offsets[label]

    # Positive-valued, heavy-tailed columns, as real flow features are.
    for j, feature in enumerate(FEATURES):
        if feature in {"flow_duration", "Duration", "Rate", "Srate", "IAT"}:
            X[:, j] = np.abs(X[:, j]) * 100
        elif feature.endswith("_number") or feature in {
            "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC",
            "TCP", "UDP", "DHCP", "ARP", "ICMP", "IPv", "LLC",
        }:
            X[:, j] = (X[:, j] > 0).astype(float)

    df = pd.DataFrame(X, columns=FEATURES)
    df["label"] = labels

    # Inject the messiness the cleaning code must survive.
    n_missing = max(1, n_rows // 100)
    for col in rng.choice(FEATURES, size=5, replace=False):
        df.loc[rng.choice(n_rows, n_missing, replace=False), col] = np.nan

    df.loc[rng.choice(n_rows, max(1, n_rows // 500), replace=False), "Rate"] = np.inf
    df["Drate"] = 0.0  # a constant column, which cleaning should drop

    n_dupes = max(1, n_rows // 50)
    dupes = df.iloc[rng.choice(n_rows, n_dupes, replace=False)]
    return pd.concat([df, dupes], ignore_index=True).sample(
        frac=1, random_state=seed
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--rows", type=int, default=60_000, help="total rows")
    parser.add_argument("--files", type=int, default=3, help="CSV part files")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    per_file = args.rows // args.files

    for i in range(args.files):
        df = generate(per_file, args.seed + i)
        path = args.out / f"part-{i:05d}-synthetic.csv"
        df.to_csv(path, index=False)
        print(f"Wrote {path}  ({len(df)} rows x {df.shape[1]} cols)")

    print(f"\nSynthetic data in {args.out}")
    print("WARNING: for pipeline testing only — never report these numbers.")


if __name__ == "__main__":
    main()
