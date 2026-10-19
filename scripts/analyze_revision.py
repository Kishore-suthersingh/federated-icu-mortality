"""Summarise the revision run (notebooks/kaggle_wids_revision.ipynb).

Reads the downloaded kernel outputs and writes:

  <out>/revision_facts.json          every number the revised manuscript quotes
  <out>/figS5_reliability.png        binned reliability diagrams, five seeds pooled
  <out>/figS6_feature_leakage.png    per-feature reconstruction R², within-site

    python scripts/analyze_revision.py --revision <dir with outputs/rev_*> \
        --original <dir with out_fedlgbm-wids-multiseed/> --out figures/revision

The reproduction check compares every model's per-seed AUROC in the rerun with
the paper's original multi-seed run; the manuscript only uses the new runs if
they reproduce the old numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_figures as mf  # noqa: E402

plt = mf.plt
REF = "fed_hybrid_mlp_lgbm"
APACHE = "apache_4a_hospital_death_prob"
APACHE_MODEL = f"incumbent_{APACHE}"


def mean_ci(v) -> dict:
    v = np.asarray(v, dtype=float)
    if len(v) < 2:
        return {"mean": float(v.mean()), "lo": float("nan"), "hi": float("nan"), "n": len(v)}
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return {"mean": float(v.mean()), "lo": float(v.mean() - h), "hi": float(v.mean() + h),
            "n": len(v)}


def paired(a, b) -> dict:
    d = np.asarray(a) - np.asarray(b)
    out = mean_ci(d)
    out["better"] = int((d > 0).sum())
    out["p"] = float(stats.ttest_1samp(d, 0).pvalue) if len(d) > 1 else float("nan")
    return out


def seeds_in(run: Path) -> dict[int, Path]:
    return {int(p.name.split("_")[1]): p for p in sorted(run.glob("seed_*"))
            if (p / "results.json").exists()}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


# --------------------------------------------------------------------------
def reproduction(rev: Path, orig: Path) -> dict:
    out = {}
    for regime in ("within", "cross"):
        new = seeds_in(rev / f"rev_A_{regime}_site")
        old_root = orig / "outputs" / f"wids_ms_{regime}_site"
        diffs = {}
        for seed, d in new.items():
            old_file = old_root / f"seed_{seed}" / "results.json"
            if not old_file.exists():
                continue
            nm, om = load(d / "results.json")["models"], load(old_file)["models"]
            for m in om:
                if m in nm and "roc_auc" in nm[m] and "roc_auc" in om[m]:
                    diffs.setdefault(m, []).append(abs(nm[m]["roc_auc"] - om[m]["roc_auc"]))
        out[regime] = {m: float(max(v)) for m, v in diffs.items()}
    out["max_abs_auroc_diff"] = max(v for r in ("within", "cross") for v in out[r].values())
    return out


# --------------------------------------------------------------------------
def valid_score(s: np.ndarray) -> np.ndarray:
    """APACHE IVa probabilities that are real: finite and inside [0, 1].

    The public release encodes some unavailable scores as -1, which the paper's
    evaluation scored as a (lowest-possible) risk.
    """
    return np.isfinite(s) & (s >= 0) & (s <= 1)


def complete_case(rev: Path) -> dict:
    out = {}
    for regime in ("within", "cross"):
        rows = {"hyb_all": [], "apa_all": [], "hyb_cc": [], "apa_cc": [],
                "hyb_cc_ap": [], "apa_cc_ap": [], "coverage": [], "n_missing": [],
                "n_negative": []}
        for seed, d in seeds_in(rev / f"rev_A_{regime}_site").items():
            z = np.load(d / "predictions.npz")
            y = z["y_test"]
            hyb = z[f"proba__{REF}"][:, 1]
            raw = z[f"raw_score__{APACHE}"]
            ok = valid_score(raw)
            filled = np.where(np.isfinite(raw), raw, np.nanmedian(raw[np.isfinite(raw)]))
            rows["hyb_all"].append(roc_auc_score(y, hyb))
            rows["apa_all"].append(roc_auc_score(y, filled))
            rows["hyb_cc"].append(roc_auc_score(y[ok], hyb[ok]))
            rows["apa_cc"].append(roc_auc_score(y[ok], raw[ok]))
            rows["hyb_cc_ap"].append(average_precision_score(y[ok], hyb[ok]))
            rows["apa_cc_ap"].append(average_precision_score(y[ok], raw[ok]))
            rows["coverage"].append(float(ok.mean()))
            rows["n_missing"].append(int((~np.isfinite(raw)).sum()))
            rows["n_negative"].append(int((np.isfinite(raw) & (raw < 0)).sum()))
        out[regime] = {
            "auroc_hybrid_cc": mean_ci(rows["hyb_cc"]),
            "auroc_apache_cc": mean_ci(rows["apa_cc"]),
            "diff_cc": paired(rows["hyb_cc"], rows["apa_cc"]),
            "diff_cc_auprc": paired(rows["hyb_cc_ap"], rows["apa_cc_ap"]),
            "diff_all": paired(rows["hyb_all"], rows["apa_all"]),
            "coverage": [min(rows["coverage"]), max(rows["coverage"])],
            "n_missing": [min(rows["n_missing"]), max(rows["n_missing"])],
            "n_negative": [min(rows["n_negative"]), max(rows["n_negative"])],
        }
    return out


# --------------------------------------------------------------------------
def attacks(rev: Path) -> dict:
    out = {}
    for regime in ("within", "cross"):
        leak, mia, gap = [], [], []
        for seed, d in seeds_in(rev / f"rev_A_{regime}_site").items():
            pa = load(d / "results.json").get("privacy_attacks", {})
            if "embedding_inversion" in pa and "leakage_ratio" in pa["embedding_inversion"]:
                leak.append(pa["embedding_inversion"]["leakage_ratio"])
            if "membership_inference" in pa and "attack_auc" in pa["membership_inference"]:
                mia.append(pa["membership_inference"]["attack_auc"])
                gap.append(pa["membership_inference"]["confidence_gap"])
        out[regime] = {"leakage": mean_ci(leak), "mia_auc": mean_ci(mia),
                       "confidence_gap": mean_ci(gap), "leakage_per_seed": leak,
                       "mia_per_seed": mia}
    return out


def feature_group(name: str) -> str:
    if name.endswith("__isna"):
        return "missingness indicator"
    if "=" in name:
        return "one-hot category"
    if name.endswith("__code"):
        return "diagnosis code"
    return "numeric"


def per_feature(rev: Path) -> dict:
    r2s, names = [], None
    for seed, d in seeds_in(rev / "rev_A_within_site").items():
        inv = load(d / "results.json")["privacy_attacks"]["embedding_inversion"]
        att = np.asarray(inv["per_feature_mse_attack"])
        base = np.asarray(inv["per_feature_mse_baseline"])
        with np.errstate(divide="ignore", invalid="ignore"):
            r2s.append(np.where(base > 1e-8, 1 - att / base, np.nan))
        names = inv["feature_names"]
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            r2 = np.nanmean(np.vstack(r2s), axis=0)  # NaN only for constant features
    groups = np.array([feature_group(n) for n in names])
    keep = np.isfinite(r2)
    summary = {}
    for g in ("numeric", "missingness indicator", "one-hot category", "diagnosis code"):
        m = keep & (groups == g)
        if m.any():
            summary[g] = {"n": int(m.sum()), "median_r2": float(np.median(r2[m])),
                          "frac_above_0.9": float((r2[m] > 0.9).mean()),
                          "frac_above_0.5": float((r2[m] > 0.5).mean())}
    num = [(names[i], float(r2[i])) for i in np.where(keep & (groups == "numeric"))[0]]
    num.sort(key=lambda t: -t[1])
    return {"by_group": summary, "top_numeric": num[:10], "bottom_numeric": num[-5:],
            "n_seeds": len(r2s), "r2": r2.tolist(), "names": names, "groups": groups.tolist()}


# --------------------------------------------------------------------------
def dp_sweep(rev: Path) -> list[dict]:
    rows = []
    root = rev / "rev_B_dp_embeddings"
    for key in ("None", "10.0", "5.0", "3.0", "1.0", "0.5"):
        runs = seeds_in(root / f"epsilon_{key}")
        au, ap, lk, mi, apa, sig = [], [], [], [], [], None
        for seed, d in runs.items():
            r = load(d / "results.json")
            if REF not in r["models"] or "roc_auc" not in r["models"][REF]:
                continue
            au.append(r["models"][REF]["roc_auc"])
            ap.append(r["models"][REF]["average_precision"])
            apa.append(r["models"][APACHE_MODEL]["roc_auc"])
            pa = r.get("privacy_attacks", {})
            lk.append(pa["embedding_inversion"]["leakage_ratio"])
            mi.append(pa["membership_inference"]["attack_auc"])
            sig = r["privacy"].get("channel_2_embeddings", {}).get("noise_multiplier", sig)
        rows.append({"eps": key, "sigma": sig, "auroc": mean_ci(au), "auprc": mean_ci(ap),
                     "leakage": mean_ci(lk), "mia": mean_ci(mi), "apache": mean_ci(apa),
                     "below_apache_all_seeds": bool(au) and all(a < b for a, b in zip(au, apa)),
                     "n": len(au)})
    return rows


# --------------------------------------------------------------------------
def sensitivity(rev: Path) -> list[dict]:
    rows = []
    ref = {s: load(d / "results.json") for s, d in seeds_in(rev / "rev_A_within_site").items()}
    ref_x = {s: load(d / "results.json") for s, d in seeds_in(rev / "rev_A_cross_site").items()}
    configs = [("within", 2), ("within", 50), ("within", 100), ("within", 200), ("cross", 2),
               ("cross", 200)]
    for regime, thr in configs:
        if thr == 200:
            runs = ref if regime == "within" else ref_x
        else:
            runs = {s: load(d / "results.json")
                    for s, d in seeds_in(rev / f"rev_C_{regime}_site_min{thr}").items()}
        base = ref if regime == "within" else ref_x
        seeds = sorted(set(runs) & set(base))
        fed = [runs[s]["models"][REF]["roc_auc"] for s in seeds]
        cen = [runs[s]["models"]["centralized_lgbm"]["roc_auc"] for s in seeds]
        apa = [runs[s]["models"][APACHE_MODEL]["roc_auc"] for s in seeds]
        fed200 = [base[s]["models"][REF]["roc_auc"] for s in seeds]
        nclients = [runs[s]["partition"]["n_clients"] for s in seeds]
        largest = [max(runs[s]["partition"]["client_sizes"]) for s in seeds]
        rows.append({
            "regime": regime, "threshold": thr, "n_clients": [min(nclients), max(nclients)],
            "largest_client": [min(largest), max(largest)],
            "fed_auroc": mean_ci(fed), "vs_centralized": paired(fed, cen),
            "vs_apache": paired(fed, apa), "vs_threshold_200": paired(fed, fed200),
            "n": len(seeds),
        })
    return rows


# --------------------------------------------------------------------------
REL_MODELS = [(REF, "Federated hybrid", mf.BLUE, "o"),
              ("centralized_lgbm", "Centralized LightGBM", mf.INK, "s"),
              ("logistic_regression", "Logistic regression*", mf.GREY, "v"),
              (APACHE, "APACHE IVa (valid scores)", mf.RED, "D")]


def reliability(rev: Path, path: Path, bins: int = 10) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    out = {}
    for ax, regime, title in ((axes[0], "within", "(a) within-site"),
                              (axes[1], "cross", "(b) cross-site")):
        ys, preds = [], {m[0]: [] for m in REL_MODELS}
        masks = []
        for seed, d in seeds_in(rev / f"rev_A_{regime}_site").items():
            z = np.load(d / "predictions.npz")
            ys.append(z["y_test"])
            raw = z[f"raw_score__{APACHE}"]
            masks.append(valid_score(raw))
            for m, *_ in REL_MODELS:
                preds[m].append(raw if m == APACHE else z[f"proba__{m}"][:, 1])
        y = np.concatenate(ys)
        valid = np.concatenate(masks)
        ax.plot([0, 1], [0, 1], color="#bbbbbb", linewidth=0.9)
        out[regime] = {}
        for m, label, color, marker in REL_MODELS:
            p = np.concatenate(preds[m])
            yy = y
            if m == APACHE:
                p, yy = p[valid], y[valid]
            # Equal-frequency bins: every point carries the same number of stays.
            edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
            idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
            mp, ob, lo, hi = [], [], [], []
            for b in range(len(edges) - 1):
                sel = idx == b
                n = int(sel.sum())
                if n == 0:
                    continue
                k = int(yy[sel].sum())
                # Wilson 95% interval for the observed event rate.
                ph, z_ = k / n, 1.96
                den = 1 + z_ ** 2 / n
                c = (ph + z_ ** 2 / (2 * n)) / den
                hw = z_ * np.sqrt(ph * (1 - ph) / n + z_ ** 2 / (4 * n ** 2)) / den
                mp.append(float(p[sel].mean())); ob.append(ph)
                lo.append(max(0.0, c - hw)); hi.append(min(1.0, c + hw))
            mp, ob = np.array(mp), np.array(ob)
            yerr = np.clip([ob - np.array(lo), np.array(hi) - ob], 0, None)
            ax.errorbar(mp, ob, yerr=yerr, fmt=marker,
                        markersize=3.2, color=color, linewidth=0.9, elinewidth=0.6,
                        capsize=1.5, label=label, linestyle="-")
            out[regime][m] = {"mean_pred": mp.tolist(), "observed": ob.tolist()}
        ax.set_xlim(0, 0.85); ax.set_ylim(0, 0.85)
        ax.set_xlabel("mean predicted risk in decile")
        ax.set_title(title, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(linewidth=0.4, color="#ececec"); ax.set_axisbelow(True)
    axes[0].set_ylabel("observed mortality (Wilson 95% CI)")
    axes[0].legend(loc="upper left", fontsize=6.6)
    fig.savefig(path)
    plt.close(fig)
    return out


def feature_figure(pf: dict, path: Path, floor: float = -0.5) -> None:
    """Per-feature R² by feature type, five seeds averaged.

    Rare one-hot categories have near-zero variance, so their R² is unstable
    and can be hugely negative; values below `floor` are drawn at the floor
    and counted in the label rather than stretching the axis.
    """
    r2 = np.asarray(pf["r2"], dtype=float)
    groups = np.asarray(pf["groups"])
    order = ["numeric", "missingness indicator", "one-hot category", "diagnosis code"]
    colors = [mf.BLUE, mf.RED, mf.GREY, mf.INK]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    rng = np.random.default_rng(0)
    labels = []
    for k, (g, c) in enumerate(zip(order, colors), start=1):
        v = r2[(groups == g) & np.isfinite(r2)]
        clipped = int((v < floor).sum())
        shown = np.clip(v, floor, None)
        ax.scatter(k + rng.uniform(-0.18, 0.18, len(shown)), shown, s=7, color=c,
                   alpha=0.55, linewidth=0)
        ax.plot([k - 0.28, k + 0.28], [np.median(v)] * 2, color=mf.INK, linewidth=1.4)
        labels.append(f"{g}\n(n = {len(v)}" + (f"; {clipped} < {floor:g}" if clipped else "")
                      + ")")
    ax.set_xticks(range(1, len(order) + 1)); ax.set_xticklabels(labels, fontsize=7.2)
    ax.set_ylabel("per-feature reconstruction R²")
    ax.axhline(0, color=mf.GREY, linewidth=0.7)
    ax.set_ylim(floor - 0.05, 1.02)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linewidth=0.4, color="#ececec"); ax.set_axisbelow(True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--revision", required=True, type=Path,
                    help="directory containing rev_A_*, rev_B_*, rev_C_*")
    ap.add_argument("--original", required=True, type=Path,
                    help="downloaded out_fedlgbm-wids-multiseed directory")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    facts = {
        "reproduction": reproduction(args.revision, args.original),
        "complete_case": complete_case(args.revision),
        "attacks": attacks(args.revision),
        "per_feature": per_feature(args.revision),
        "dp": dp_sweep(args.revision),
        "sensitivity": sensitivity(args.revision),
    }
    facts["reliability"] = reliability(args.revision, args.out / "figS5_reliability.png")
    feature_figure(facts["per_feature"], args.out / "figS6_feature_leakage.png")
    (args.out / "revision_facts.json").write_text(json.dumps(facts, indent=1))

    short = {k: v for k, v in facts.items() if k not in ("per_feature", "reliability")}
    short["per_feature"] = {k: facts["per_feature"][k]
                            for k in ("by_group", "top_numeric", "bottom_numeric", "n_seeds")}
    print(json.dumps(short, indent=1, default=str))


if __name__ == "__main__":
    main()
