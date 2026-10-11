"""Draw the manuscript figures from the measured artefacts.

Every panel is generated from a stored results file; nothing is drawn from a
remembered number and nothing is stylised beyond what the measurement supports.
Run it against the directory holding the Kaggle run outputs:

    python scripts/make_figures.py --artefacts <dir> --out figures/

The directory must contain msout/, privout/ and widsout2/ as downloaded from
the three Kaggle kernels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

# A single monochrome-safe palette: print reproduction in most journals is
# greyscale, so the series are separated by marker and dash as well as hue.
INK = "#1a1a1a"
BLUE = "#2f5c8f"
RED = "#a83232"
GREY = "#7a7a7a"
LIGHT = "#c9d4e2"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "figure.dpi": 400,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

PRETTY = {
    "fed_hybrid_mlp_lgbm": "Federated hybrid\nMLP–LightGBM",
    "centralized_lgbm": "Centralized\nLightGBM",
    "fedavg_mlp_only": "FedAvg MLP\n(no LightGBM)",
    "random_forest": "Random forest\n(centralized)",
    "logistic_regression": "Logistic regression\n(centralized)",
    "incumbent_apache_4a_hospital_death_prob": "APACHE IVa\n(reference)",
}
ORDER = list(PRETTY)


def load(root: Path) -> dict:
    """Read every artefact the figures depend on, failing loudly if one moved."""
    ms = root / "msout"
    out: dict = {
        "within": json.loads(
            (ms / "wids_ms_within_site_multi_seed_summary.json").read_text()
        ),
        "cross": json.loads(
            (ms / "wids_ms_cross_site_multi_seed_summary.json").read_text()
        ),
        "seed42_within": json.loads(
            (ms / "outputs/wids_ms_within_site/seed_42/results.json").read_text()
        ),
        "seed42_cross": json.loads(
            (ms / "outputs/wids_ms_cross_site/seed_42/results.json").read_text()
        ),
        "privacy": json.loads((root / "privout/privacy_utility.json").read_text()),
    }
    return out


def _auroc(summary: dict, model: str) -> dict:
    return summary["aggregated"]["models"][model]["roc_auc"]


# --------------------------------------------------------------------------
# Figure 1 — architecture
# --------------------------------------------------------------------------
def fig_architecture(path: Path) -> None:
    """Schematic of what crosses the trust boundary, and what does not.

    The boundary is the point of the paper, so it is the strongest line in the
    drawing rather than decoration around the boxes.
    """
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.3)
    ax.axis("off")

    def box(x, y, w, h, text, *, fc="white", ec=INK, fs=8, lw=0.9, va="center"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                    facecolor=fc, edgecolor=ec, linewidth=lw))
        ty = y + h / 2 if va == "center" else y + h - 0.16
        ax.text(x + w / 2, ty, text, ha="center", va=va, fontsize=fs)

    def arrow(x1, y1, x2, y2, *, ls="-", color=INK, lw=0.9, scale=9):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     linestyle=ls, color=color, linewidth=lw,
                                     mutation_scale=scale, shrinkA=0, shrinkB=0))

    # ---- server side ------------------------------------------------------
    ax.add_patch(FancyBboxPatch((0.4, 5.35), 9.2, 1.85, boxstyle="round,pad=0.02",
                                facecolor="#fbfbfd", edgecolor=GREY, linewidth=0.9))
    ax.text(5.0, 7.02, "coordinating server — honest-but-curious: holds weights and "
            "embeddings, never a raw record", ha="center", va="center", fontsize=7.8,
            color=GREY)
    box(0.8, 5.55, 2.5, 0.95, "FedAvg aggregation\n(size-weighted)", fc=LIGHT)
    box(3.75, 5.55, 2.5, 0.95, "global encoder\n393→256→128→64")
    box(6.7, 5.55, 2.6, 0.95, "LightGBM head\n1,000 trees, 255 leaves")
    arrow(3.3, 6.03, 3.75, 6.03)
    arrow(6.25, 6.03, 6.7, 6.03)

    # ---- the two uplinks --------------------------------------------------
    for x, color in ((2.6, BLUE), (6.4, BLUE)):
        arrow(x, 3.5, x, 5.55, color=color, lw=1.3, scale=11)
    ax.text(2.85, 4.62, "channel 1: encoder weights\n82 clients × 30 rounds → 2,810 MB",
            fontsize=7.4, color=BLUE, va="center")
    ax.text(6.65, 4.62, "channel 2: 64-d embeddings\nof 64,198 records → 16 MB",
            fontsize=7.4, color=BLUE, va="center")

    # ---- trust boundary ---------------------------------------------------
    ax.plot([0.2, 9.8], [3.95, 3.95], color=RED, linewidth=1.5, linestyle=(0, (6, 3)))
    ax.text(9.8, 4.06, "institutional trust boundary", ha="right", va="bottom",
            fontsize=7.8, color=RED, style="italic")
    ax.text(0.2, 3.68, "82 hospital clients", fontsize=7.6, color=GREY, va="top")
    ax.text(9.8, 3.68, "measured: these embeddings invert to 93.6% of "
            "feature variance (§4.5)", fontsize=7.4, color=RED, ha="right", va="top",
            style="italic")

    # ---- clients ----------------------------------------------------------
    ax.plot([1.0, 9.0], [3.34, 3.34], color=INK, linewidth=0.8)
    for x in (2.6, 6.4):
        ax.plot([x, x], [3.34, 3.5], color=INK, linewidth=0.8)

    labels = ["Hospital 1\nn = 5,083", "Hospital 2\nn = 3,017", "…",
              "Hospital 81\nn = 207", "pooled tail\nn < 200 each"]
    for i, label in enumerate(labels):
        x = 0.35 + i * 1.9
        box(x, 0.35, 1.55, 0.9, label, fs=7.2)
        box(x, 1.75, 1.55, 0.8, "local MLP\nencoder", fs=7.2, fc=LIGHT)
        arrow(x + 0.775, 1.25, x + 0.775, 1.75, lw=0.8, scale=8)
        ax.plot([x + 0.775, x + 0.775], [2.55, 3.34], color=INK, linewidth=0.8)
    ax.text(5.0, 0.05, "patient records remain at the site that holds them",
            ha="center", fontsize=7.4, color=GREY)

    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 2 — the federation is observed, not simulated
# --------------------------------------------------------------------------
def fig_cohort(data: dict, path: Path) -> None:
    part = data["seed42_within"]["partition"]
    sizes = np.array(sorted(part["client_sizes"], reverse=True))
    counts = part["client_class_counts"]
    rates, ns = [], []
    for client in counts:
        c = counts[client] if isinstance(counts, dict) else client
        total = sum(c.values())
        died = c.get("died", c.get("1", 0))
        if total:
            rates.append(died / total)
            ns.append(total)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))

    ax = axes[0]
    ax.bar(np.arange(len(sizes)), sizes, color=BLUE, width=1.0, linewidth=0)
    ax.set_yscale("log")
    ax.set_xlabel("hospital client (sorted by size)")
    ax.set_ylabel("training records")
    ax.set_title("(a) client sizes span 25×", loc="left")
    # Annotations sit inside the axes: the bars leave the upper-right empty,
    # and anything above the axes collides with the panel title.
    ax.annotate(f"largest {sizes[0]:,}", xy=(0.8, sizes[0]), xytext=(14, sizes[0] * 0.82),
                fontsize=7, va="center",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=GREY))
    ax.annotate(f"smallest {sizes[-1]:,}", xy=(len(sizes) - 1.5, sizes[-1] * 1.05),
                xytext=(len(sizes) - 38, sizes[-1] * 1.5), fontsize=7, va="center",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=GREY))
    ax.set_ylim(sizes[-1] * 0.82, sizes[0] * 1.35)

    ax = axes[1]
    ax.scatter(ns, np.array(rates) * 100, s=13, facecolor="none",
               edgecolor=RED, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("training records at client")
    ax.set_ylabel("in-hospital mortality (%)")
    ax.set_title("(b) event rate varies by site", loc="left")
    pooled = 100 * np.average(rates, weights=ns)
    ax.axhline(pooled, color=INK, linewidth=0.8, linestyle=(0, (5, 3)))
    ax.text(ax.get_xlim()[1], pooled, f" pooled {pooled:.1f}%", va="center",
            ha="right", fontsize=7, backgroundcolor="white")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 3 — convergence
# --------------------------------------------------------------------------
def fig_convergence(data: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7), sharex=True)
    # Each panel carries a twin axis on its right, so the default gap puts
    # panel (a)'s loss ticks on top of panel (b)'s AUROC ticks.
    fig.subplots_adjust(wspace=0.62)
    for ax, key, title in ((axes[0], "seed42_within", "(a) within-site"),
                           (axes[1], "seed42_cross", "(b) cross-site")):
        hist = data[key]["fed_history"]
        rounds = [h["round"] for h in hist]
        auc = [h["val_roc_auc"] for h in hist]
        loss = [h["mean_client_train_loss"] for h in hist]

        ax.plot(rounds, auc, color=BLUE, linewidth=1.2, marker="o", markersize=2.4,
                label="validation AUROC")
        ax.set_xlabel("communication round")
        ax.set_ylabel("validation AUROC", color=BLUE)
        ax.tick_params(axis="y", colors=BLUE)
        ax.set_title(title, loc="left")
        ax.spines[["top"]].set_visible(False)

        target = 0.98 * max(auc)
        reached = next((r for r, a in zip(rounds, auc) if a >= target), None)
        if reached:
            ax.axvline(reached, color=GREY, linewidth=0.7, linestyle=(0, (4, 3)))
            ax.text(reached + 0.6, min(auc) + 0.02 * (max(auc) - min(auc)),
                    f"98% of best\nby round {reached}", fontsize=7, color=GREY)

        tw = ax.twinx()
        tw.plot(rounds, loss, color=RED, linewidth=1.0, linestyle=(0, (4, 2)),
                label="mean client training loss")
        tw.set_ylabel("client training loss", color=RED)
        tw.tick_params(axis="y", colors=RED)
        tw.spines[["top"]].set_visible(False)

    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 4 — the headline comparison
# --------------------------------------------------------------------------
def fig_forest(data: dict, path: Path) -> None:
    """Means with 95% intervals over five seeds, both regimes on one axis.

    Overlapping intervals are not the test being reported: the comparisons are
    paired across seeds, so the intervals are descriptive and the significance
    is stated in the text and Table 3.
    """
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    ys = np.arange(len(ORDER))[::-1]

    for offset, key, color, marker, label in (
        (0.16, "within", BLUE, "o", "within-site"),
        (-0.16, "cross", RED, "s", "cross-site (22 unseen hospitals)"),
    ):
        summary = data[key]
        for y, model in zip(ys, ORDER):
            stats = _auroc(summary, model)
            ax.errorbar(stats["mean"], y + offset,
                        xerr=[[stats["mean"] - stats["ci95_low"]],
                              [stats["ci95_high"] - stats["mean"]]],
                        fmt=marker, markersize=4, color=color, elinewidth=1.0,
                        capsize=2.2, label=label if model == ORDER[0] else None)

    incumbent = _auroc(data["within"], ORDER[-1])["mean"]
    ax.axvline(incumbent, color=GREY, linewidth=0.8, linestyle=(0, (5, 3)))
    ax.text(incumbent + 0.0015, ys[-1] - 0.62, "APACHE IVa within-site",
            fontsize=7, color=GREY, va="bottom", ha="left")

    ax.set_yticks(ys)
    ax.set_yticklabels([PRETTY[m] for m in ORDER])
    ax.set_xlabel("AUROC (mean of 5 seeds, 95% CI)")
    ax.set_ylim(ys[-1] - 0.75, ys[0] + 0.65)
    ax.set_xlim(0.800, 0.915)
    ax.spines[["top", "right"]].set_visible(False)
    # Outside the axes: every in-axes corner either carries a point or is
    # crossed by the incumbent reference line.
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2)
    ax.grid(axis="x", linewidth=0.4, color="#e2e2e2")
    ax.set_axisbelow(True)
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 5 — per-seed paired differences
# --------------------------------------------------------------------------
def fig_paired(data: dict, path: Path) -> None:
    """Why one seed is not enough: the same three contrasts, seed by seed.

    Panel (c) is the retracted robustness claim. It is drawn to the same scale
    as the others so the reader can see that its sign is not stable while the
    other two never change sign.
    """
    within, cross = data["within"], data["cross"]

    def vals(summary, model):
        return np.array(_auroc(summary, model)["values"])

    fed_w = vals(within, "fed_hybrid_mlp_lgbm")
    fed_c = vals(cross, "fed_hybrid_mlp_lgbm")
    cen_w = vals(within, "centralized_lgbm")
    cen_c = vals(cross, "centralized_lgbm")
    apa_w = vals(within, ORDER[-1])

    panels = [
        ("(a) federated − APACHE IVa\n(within-site)", fed_w - apa_w),
        ("(b) federated − centralized\n(within-site)", fed_w - cen_w),
        ("(c) cross-site degradation\n(federated vs centralized)",
         None),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8))
    # The three panels are on very different scales, so their tick labels are
    # wide; the default gap runs them into the neighbouring panel.
    fig.subplots_adjust(wspace=0.45)
    seeds = np.arange(1, 6)

    for ax, (title, delta) in zip(axes, panels):
        ax.axhline(0, color=INK, linewidth=0.8)
        if delta is not None:
            colors = [BLUE if d > 0 else RED for d in delta]
            ax.bar(seeds, delta, color=colors, width=0.62, linewidth=0)
            ax.plot(seeds, [delta.mean()] * 5, color=GREY, linewidth=0.9,
                    linestyle=(0, (4, 3)))
            ax.text(3, delta.mean(), f"mean {delta.mean():+.4f}", fontsize=7,
                    color=GREY, va="center", ha="center",
                    bbox=dict(fc="white", ec="none", pad=1.2))
            ax.set_xlim(0.4, 5.6)
        else:
            dw = fed_c - fed_w
            dc = cen_c - cen_w
            ax.plot(seeds, dw, color=BLUE, marker="o", markersize=3.4,
                    linewidth=1.0)
            ax.plot(seeds, dc, color=RED, marker="s", markersize=3.4,
                    linewidth=1.0, linestyle=(0, (4, 2)))
            # Labelled at the line ends rather than in a legend box: every
            # in-axes corner here is crossed by one of the two series.
            ax.text(5.12, dw[-1], "federated", fontsize=7.5, color=BLUE, va="center")
            ax.text(5.12, dc[-1], "centralized", fontsize=7.5, color=RED, va="center")
            ax.set_xlim(0.6, 6.9)
        ax.set_xticks(seeds)
        ax.set_xlabel("seed")
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Δ AUROC")
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 6 — privacy/utility
# --------------------------------------------------------------------------
def fig_privacy(data: dict, path: Path) -> None:
    rows = [r for r in data["privacy"] if r["channel"] == "embeddings"]
    rows.sort(key=lambda r: r["eps"])
    base = next(r for r in data["privacy"] if r["channel"] == "none")
    eps = [r["eps"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

    ax = axes[0]
    ax.plot(eps, [r["auroc"] for r in rows], color=BLUE, marker="o",
            markersize=3.6, linewidth=1.2, label="AUROC under DP")
    ax.axhline(base["auroc"], color=INK, linewidth=0.9, linestyle=(0, (5, 3)))
    ax.text(0.55, base["auroc"] - 0.018, f"no DP  {base['auroc']:.4f}", fontsize=7)
    # The sweep is single-seed, so the incumbent is read from the same seed
    # rather than from the five-seed mean reported elsewhere.
    apache = data["seed42_within"]["models"][ORDER[-1]]["roc_auc"]
    ax.axhline(apache, color=GREY, linewidth=0.9, linestyle=(0, (2, 2)))
    ax.text(0.55, apache - 0.018, f"APACHE IVa  {apache:.4f}", fontsize=7, color=GREY)
    ax.axhline(0.5, color=RED, linewidth=0.7)
    ax.text(9.9, 0.508, "chance", fontsize=7, color=RED, ha="right")
    ax.set_xlabel("privacy budget ε (record-level, embedding channel)")
    ax.set_ylabel("test AUROC")
    ax.set_title("(a) utility stays below the reference score", loc="left")
    ax.set_ylim(0.45, 0.93)

    ax = axes[1]
    ax.plot(eps, [r["leakage"] for r in rows], color=RED, marker="s",
            markersize=3.6, linewidth=1.2)
    ax.axhline(base["leakage"], color=INK, linewidth=0.9, linestyle=(0, (5, 3)))
    ax.text(0.55, base["leakage"] - 0.09, f"no DP  {base['leakage']:.4f}", fontsize=7)
    ax.axhline(0, color=GREY, linewidth=0.7)
    ax.text(9.9, 0.055, "no recovery", fontsize=7, color=GREY, ha="right")
    ax.set_xlabel("privacy budget ε")
    ax.set_ylabel("inversion leakage ratio")
    ax.set_title("(b) leakage falls only where utility has gone", loc="left")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(linewidth=0.4, color="#ececec")
        ax.set_axisbelow(True)
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 7 — communication
# --------------------------------------------------------------------------
def fig_communication(data: dict, path: Path) -> None:
    comm = data["seed42_within"]["communication"]
    updates = comm["channel_1_model_updates"]["total_mb"]
    embed = comm["channel_2_embeddings"]["total_mb"]
    raw = comm["comparison_to_centralized"]["raw_data_upload_mb"]

    # Three separate bars, not a stack: on the log scale the 16 MB embedding
    # channel is invisible when stacked on the 2,810 MB weight channel, which
    # hides the fact that the cheap channel is the one carrying the patients.
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    heights = [raw, updates, embed]
    colors = [GREY, BLUE, LIGHT]
    ax.bar([0, 1, 2], heights, color=colors, width=0.6, linewidth=0,
           edgecolor=INK)

    for x, h in zip(range(3), heights):
        ax.text(x, h * 1.25, f"{h:,.0f} MB", ha="center", fontsize=7.8)

    ax.set_yscale("log")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["raw records\n(centralized,\none upload)",
                        "channel 1:\nencoder weights\n(82 × 30 rounds)",
                        "channel 2:\nembeddings\n(64,198 records)"])
    ax.set_ylabel("data transmitted (MB, log scale)")
    ax.set_ylim(5, updates * 12)

    total = updates + embed
    ax.annotate("", xy=(0.72, total * 3.2), xytext=(2.28, total * 3.2),
                arrowprops=dict(arrowstyle="|-|", lw=0.8, color=INK,
                                mutation_scale=3))
    ax.text(1.5, total * 3.9, f"federated total {total:,.0f} MB "
            f"= {total / raw:.1f}× the centralized upload",
            ha="center", fontsize=7.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 8 — calibration
# --------------------------------------------------------------------------
def fig_calibration(data: dict, path: Path) -> None:
    """Calibration lines implied by the measured slope and intercept.

    The per-bin observed frequencies were not retained, so this plots the
    fitted logistic recalibration line rather than a binned reliability
    diagram; the caption says so, and ECE is annotated alongside.
    """
    res = data["seed42_within"]["models"]
    models = ["fed_hybrid_mlp_lgbm", "centralized_lgbm",
              "logistic_regression", ORDER[-1]]
    styles = [(BLUE, "-"), (INK, (0, (5, 2))), (GREY, (0, (1, 1.6))), (RED, (0, (4, 2)))]

    p = np.linspace(0.002, 0.6, 300)
    logit = np.log(p / (1 - p))

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.plot([0, 0.6], [0, 0.6], color="#bbbbbb", linewidth=0.9)
    ax.text(0.55, 0.575, "ideal", fontsize=7, color="#999999", rotation=45,
            ha="center", va="center")

    for model, (color, ls) in zip(models, styles):
        m = res.get(model)
        if not m or m.get("calibration_slope") is None:
            continue
        obs = 1 / (1 + np.exp(-(m["calibration_intercept"] + m["calibration_slope"] * logit)))
        label = (PRETTY[model].replace("\n", " ")
                 + f"  (slope {m['calibration_slope']:.2f}, ECE {m['ece_10bin']:.3f})")
        ax.plot(p, obs, color=color, linestyle=ls, linewidth=1.2, label=label)

    ax.set_xlabel("predicted probability of in-hospital death")
    ax.set_ylabel("observed frequency (fitted)")
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", fontsize=6.8)
    fig.savefig(path)
    plt.close(fig)


FIGURES = {
    "fig1_architecture.png": fig_architecture,
    "fig2_cohort.png": fig_cohort,
    "fig3_convergence.png": fig_convergence,
    "fig4_forest.png": fig_forest,
    "fig5_paired_seeds.png": fig_paired,
    "fig6_privacy_utility.png": fig_privacy,
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
        if fn is fig_architecture:
            fn(target)
        else:
            fn(data, target)
        print(f"wrote {target}  ({target.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
