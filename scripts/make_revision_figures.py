"""Redraw the figures changed in the major revision, from the measured artefacts.

Six figures change (4 and 6 only in wording):

  fig1  the pooled small-site client was drawn as "Hospital 1"; it is labelled
        as what it is, and the inversion cross-reference points at §4.4.
  fig2  the pooled client is marked, since it is the largest client and is not
        federated internally.
  fig7  communication for shortened schedules, computed from the measured
        per-round volume, because the 30-round total overstates what a
        convergence-tuned schedule would send.
  fig8  calibration over all five seeds. Per-record predictions were not
        retained, so binned reliability diagrams cannot be drawn; the per-seed
        slope and ECE can, and they replace the single-seed fitted lines.

Run against the directory holding the downloaded Kaggle kernel outputs:

    python scripts/make_revision_figures.py --artefacts <dir> --out figures/

<dir> must contain out_fedlgbm-wids-multiseed/outputs/wids_ms_{within,cross}_site/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_figures as mf  # noqa: E402  (shared style, palette and helpers)

plt = mf.plt
INK, BLUE, RED, GREY, LIGHT = mf.INK, mf.BLUE, mf.RED, mf.GREY, mf.LIGHT
SEEDS = range(42, 47)


def load(root: Path) -> dict:
    ms = root / "out_fedlgbm-wids-multiseed" / "outputs"
    return {
        regime: {
            s: json.loads((ms / f"wids_ms_{regime}_site" / f"seed_{s}" / "results.json")
                          .read_text())
            for s in SEEDS
        }
        for regime in ("within", "cross")
    }


# --------------------------------------------------------------------------
# Figure 1 — architecture, with the pooled client labelled correctly
# --------------------------------------------------------------------------
def fig_architecture(data: dict, path: Path) -> None:
    part = data["within"][42]["partition"]
    sizes = part["client_sizes"]
    pooled = sizes[-1]                      # the pooled client is appended last
    real = sorted(sizes[:-1], reverse=True)
    n_pooled_sites = 147 - len(real)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.3)
    ax.axis("off")

    def box(x, y, w, h, text, *, fc="white", ec=INK, fs=8, lw=0.9, ls="-"):
        ax.add_patch(mf.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                       facecolor=fc, edgecolor=ec, linewidth=lw,
                                       linestyle=ls))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2, *, color=INK, lw=0.9, scale=9):
        ax.add_patch(mf.FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                        color=color, linewidth=lw,
                                        mutation_scale=scale, shrinkA=0, shrinkB=0))

    ax.add_patch(mf.FancyBboxPatch((0.4, 5.35), 9.2, 1.85, boxstyle="round,pad=0.02",
                                   facecolor="#fbfbfd", edgecolor=GREY, linewidth=0.9))
    ax.text(5.0, 7.02, "coordinating server — honest-but-curious: holds weights, the "
            "global encoder and embeddings, never a client's raw record",
            ha="center", va="center", fontsize=7.4, color=GREY)
    box(0.8, 5.55, 2.5, 0.95, "FedAvg aggregation\n(size-weighted)", fc=LIGHT)
    box(3.75, 5.55, 2.5, 0.95, "global encoder\n393→256→128→64")
    box(6.7, 5.55, 2.6, 0.95, "LightGBM head\n1,000 trees, 255 leaves")
    arrow(3.3, 6.03, 3.75, 6.03)
    arrow(6.25, 6.03, 6.7, 6.03)

    for x in (2.6, 6.4):
        arrow(x, 3.5, x, 5.55, color=BLUE, lw=1.3, scale=11)
    ax.text(2.85, 4.62, f"channel 1: encoder weights\n{len(sizes)} clients × 30 rounds "
            "→ 2,810 MB", fontsize=7.4, color=BLUE, va="center")
    ax.text(6.65, 4.62, "channel 2: 64-d embeddings\nof 64,198 records → 16 MB",
            fontsize=7.4, color=BLUE, va="center")

    ax.plot([0.2, 9.8], [3.95, 3.95], color=RED, linewidth=1.5, linestyle=(0, (6, 3)))
    ax.text(9.8, 4.06, "institutional trust boundary", ha="right", va="bottom",
            fontsize=7.8, color=RED, style="italic")
    ax.text(0.2, 3.68, f"{len(sizes)} clients from\n147 hospitals (seed 42)",
            fontsize=7.2, color=GREY, va="top")
    ax.text(9.8, 3.68, "measured (adversary with in-distribution\nauxiliary records): "
            "diagnosis codes recovered\nalmost exactly, other features R² ≈ 0.21",
            fontsize=6.8, color=RED, ha="right", va="top", style="italic")

    ax.plot([1.0, 9.0], [3.0, 3.0], color=INK, linewidth=0.8)
    for x in (2.6, 6.4):
        ax.plot([x, x], [3.0, 3.5], color=INK, linewidth=0.8)

    labels = [f"Hospital 1\nn = {real[0]:,}", f"Hospital 2\nn = {real[1]:,}", "…",
              f"Hospital {len(real)}\nn = {real[-1]:,}",
              f"pooled client\n{n_pooled_sites} sites < 200 each\nn = {pooled:,}"]
    for i, label in enumerate(labels):
        x = 0.35 + i * 1.9
        is_pool = i == len(labels) - 1
        box(x, 0.35, 1.55, 0.95, label, fs=6.9 if is_pool else 7.2,
            ls=(0, (3, 2)) if is_pool else "-")
        box(x, 1.75, 1.55, 0.8, "local MLP\nencoder", fs=7.2, fc=LIGHT)
        arrow(x + 0.775, 1.3, x + 0.775, 1.75, lw=0.8, scale=8)
        ax.plot([x + 0.775, x + 0.775], [2.55, 3.0], color=INK, linewidth=0.8)
    ax.text(5.0, 0.05, "patient records remain at the site that holds them, except "
            "that the pooled client assumes an intermediary holding its sites' records",
            ha="center", fontsize=6.9, color=GREY)

    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 2 — cohort, with the pooled client marked
# --------------------------------------------------------------------------
def fig_cohort(data: dict, path: Path) -> None:
    part = data["within"][42]["partition"]
    sizes = np.array(part["client_sizes"])
    counts = [part["client_class_counts"][f"client_{i}"] for i in range(len(sizes))]
    rates = np.array([c.get("died", 0) / sum(c.values()) for c in counts])
    pooled_idx = len(sizes) - 1

    order = np.argsort(-sizes)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))

    ax = axes[0]
    colors = [RED if i == pooled_idx else BLUE for i in order]
    ax.bar(np.arange(len(sizes)), sizes[order], color=colors, width=1.0, linewidth=0)
    ax.set_yscale("log")
    ax.set_xlabel("client (sorted by size)")
    ax.set_ylabel("training records")
    ax.set_title("(a) client sizes", loc="left")
    real = np.sort(sizes[:-1])[::-1]
    ax.annotate(f"pooled client: {147 - len(real)} sites,\n{sizes[pooled_idx]:,} records",
                xy=(0.6, sizes[pooled_idx]), xytext=(12, sizes[pooled_idx] * 0.95),
                fontsize=7, va="center", color=RED,
                arrowprops=dict(arrowstyle="-", lw=0.6, color=RED))
    ax.annotate(f"largest hospital {real[0]:,}", xy=(1.4, real[0]),
                xytext=(14, real[0] * 0.55), fontsize=7, va="center",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=GREY))
    ax.annotate(f"smallest {real[-1]:,}", xy=(len(sizes) - 1.5, real[-1] * 1.05),
                xytext=(len(sizes) - 38, real[-1] * 1.5), fontsize=7, va="center",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=GREY))
    ax.set_ylim(real[-1] * 0.82, sizes.max() * 1.6)

    ax = axes[1]
    mask = np.arange(len(sizes)) != pooled_idx
    ax.scatter(sizes[mask], rates[mask] * 100, s=13, facecolor="none",
               edgecolor=BLUE, linewidth=0.8, label="single hospital")
    ax.scatter(sizes[pooled_idx], rates[pooled_idx] * 100, s=34, marker="D",
               facecolor="none", edgecolor=RED, linewidth=1.0, label="pooled client")
    ax.set_xscale("log")
    ax.set_xlabel("training records at client")
    ax.set_ylabel("in-hospital mortality (%)")
    ax.set_title("(b) event rate varies by client", loc="left")
    pooled_rate = 100 * np.average(rates, weights=sizes)
    ax.axhline(pooled_rate, color=INK, linewidth=0.8, linestyle=(0, (5, 3)))
    ax.text(ax.get_xlim()[1], pooled_rate, f" overall {pooled_rate:.1f}%", va="center",
            ha="right", fontsize=7, backgroundcolor="white")
    ax.legend(loc="upper right", fontsize=6.8, handletextpad=0.2)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 7 — communication by schedule length
# --------------------------------------------------------------------------
def communication_by_rounds(res: dict, rounds) -> dict:
    comm = res["communication"]
    ch1 = comm["channel_1_model_updates"]
    per_round = (ch1["upload_bytes_per_round"] + ch1["download_bytes_per_round"]) / 1e6
    embed = comm["channel_2_embeddings"]["total_bytes"] / 1e6
    raw = comm["comparison_to_centralized"]["raw_data_upload_bytes"] / 1e6
    return {
        "per_round_mb": per_round, "embed_mb": embed, "raw_mb": raw,
        "totals": {r: r * per_round + embed for r in rounds},
    }


def fig_communication(data: dict, path: Path) -> None:
    rounds = (1, 3, 5, 10, 30)
    c = communication_by_rounds(data["within"][42], rounds)

    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    xs = np.arange(len(rounds) + 1)
    heights = [c["raw_mb"]] + [c["totals"][r] for r in rounds]
    colors = [GREY] + [BLUE] * len(rounds)
    ax.bar(xs, heights, color=colors, width=0.62, linewidth=0)
    for x, h in zip(xs, heights):
        label = f"{h:,.0f} MB" if x == 0 else f"{h:,.0f} MB\n{h / c['raw_mb']:.1f}×"
        ax.text(x, h * 1.18, label, ha="center", fontsize=7.2)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(["raw records\n(centralized,\none upload)"]
                       + [f"{r} round{'s' if r > 1 else ''}" for r in rounds])
    ax.set_ylabel("data transmitted (MB, log scale)")
    ax.set_ylim(40, max(heights) * 6)
    ax.text(3.0, max(heights) * 3.1,
            f"federated = {c['per_round_mb']:.1f} MB per round (weights, both directions)"
            f" + {c['embed_mb']:.1f} MB embeddings once",
            ha="center", fontsize=7.0, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 8 — calibration over five seeds
# --------------------------------------------------------------------------
CAL_MODELS = [
    ("fed_hybrid_mlp_lgbm", "Federated hybrid", BLUE, "o"),
    ("centralized_lgbm", "Centralized LightGBM", INK, "s"),
    ("random_forest", "Random forest*", GREY, "^"),
    ("logistic_regression", "Logistic regression*", GREY, "v"),
    ("fedavg_mlp_only", "FedAvg MLP head†", LIGHT, "P"),
    ("incumbent_apache_4a_hospital_death_prob", "APACHE IVa", RED, "D"),
]


def fig_calibration(data: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    offsets = {"within": -0.17, "cross": 0.17}
    rng = np.random.default_rng(0)
    for ax, metric, title, ref in (
        (axes[0], "calibration_slope", "(a) calibration slope (1 = ideal)", 1.0),
        (axes[1], "ece_10bin", "(b) expected calibration error", None),
    ):
        for row, (model, _, color, marker) in enumerate(CAL_MODELS):
            for regime, off in offsets.items():
                vals = [data[regime][s]["models"][model][metric] for s in SEEDS]
                y = row + off + rng.uniform(-0.04, 0.04, len(vals))
                ax.scatter(vals, y, s=14, marker=marker,
                           facecolor=color if regime == "within" else "none",
                           edgecolor=color if color != LIGHT else GREY, linewidth=0.8)
                ax.plot([np.mean(vals)] * 2, [row + off - 0.12, row + off + 0.12],
                        color=INK, linewidth=1.0)
        if ref is not None:
            ax.axvline(ref, color=GREY, linewidth=0.7, linestyle=(0, (4, 3)))
        ax.set_yticks(range(len(CAL_MODELS)))
        ax.set_yticklabels([m[1] for m in CAL_MODELS] if ax is axes[0] else [])
        ax.invert_yaxis()
        ax.set_title(title, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", linewidth=0.4, color="#ececec")
        ax.set_axisbelow(True)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("ECE, 10 equal-width bins (log scale)")
    axes[0].set_xlabel("slope of logistic recalibration")
    axes[0].scatter([], [], s=14, marker="o", facecolor=GREY, edgecolor=GREY,
                    label="within-site (filled)")
    axes[0].scatter([], [], s=14, marker="o", facecolor="none", edgecolor=GREY,
                    label="cross-site (open)")
    axes[0].legend(loc="lower right", fontsize=6.6, handletextpad=0.1)
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figures 4 and 6 — redrawn only for the "reference score" wording
# --------------------------------------------------------------------------
def fig_forest(root: Path, path: Path) -> None:
    ms = root / "out_fedlgbm-wids-multiseed" / "outputs"
    mf.fig_forest({r: json.loads((ms / f"wids_ms_{r}_site/multi_seed_summary.json").read_text())
                   for r in ("within", "cross")}, path)


def fig_privacy(root: Path, path: Path) -> None:
    sweep = json.loads((root / "out_fedlgbm-wids-privacy/outputs/wids_privacy_embeddings/"
                        "sweep_epsilon.json").read_text())["results"]
    rows = []
    for key, res in sweep.items():
        rows.append({
            "channel": "none" if key == "None" else "embeddings",
            "eps": None if key == "None" else float(key),
            "auroc": res["models"]["fed_hybrid_mlp_lgbm"]["roc_auc"],
            "leakage": res["privacy_attacks"]["embedding_inversion"]["leakage_ratio"],
        })
    mf.fig_privacy({"privacy": rows, "seed42_within": sweep["None"]}, path)


FIGURES = {
    "fig1_architecture.png": fig_architecture,
    "fig2_cohort.png": fig_cohort,
    "fig7_communication.png": fig_communication,
    "fig8_calibration.png": fig_calibration,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artefacts", required=True, type=Path)
    ap.add_argument("--out", default=Path("figures"), type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = load(args.artefacts)
    for name, fn in FIGURES.items():
        target = args.out / name
        fn(data, target)
        print(f"wrote {target}")
    for name, fn in (("fig4_forest.png", fig_forest), ("fig6_privacy_utility.png", fig_privacy)):
        fn(args.artefacts, args.out / name)
        print(f"wrote {args.out / name}")
    c = communication_by_rounds(data["within"][42], (1, 3, 5, 10, 30))
    print(json.dumps({k: (round(v, 2) if isinstance(v, float) else
                          {r: round(t, 1) for r, t in v.items()}) for k, v in c.items()}))


if __name__ == "__main__":
    main()
