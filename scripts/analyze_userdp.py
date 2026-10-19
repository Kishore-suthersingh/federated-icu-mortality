"""Summarise the corrected user-level DP sweep (notebooks/kaggle_wids_userdp.ipynb).

For every schedule length (5, 10, 30 rounds) and privacy budget, reports the
five-seed mean and 95% CI of AUROC, AUPRC, the inversion leakage ratio (with and
without the two diagnosis-code columns) and the membership-inference AUC, next
to the no-DP LayerNorm reference and APACHE IVa on the same seeds.

    python scripts/analyze_userdp.py --runs <dir containing userdp_rounds*> \
        --out figures/revision/userdp_facts.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

REF = "fed_hybrid_mlp_lgbm"
APACHE = "incumbent_apache_4a_hospital_death_prob"


def mean_ci(v) -> dict | None:
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return None
    if len(v) == 1:
        return {"mean": float(v[0]), "lo": float("nan"), "hi": float("nan"), "n": 1}
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return {"mean": float(v.mean()), "lo": float(v.mean() - h), "hi": float(v.mean() + h),
            "n": len(v)}


def leakage_without_codes(inv: dict) -> float | None:
    if "per_feature_mse_attack" not in inv:
        return None
    a = np.asarray(inv["per_feature_mse_attack"])
    b = np.asarray(inv["per_feature_mse_baseline"])
    code = np.array([n.endswith("__code") for n in inv["feature_names"]])
    return float(1 - a[~code].sum() / b[~code].sum())


def summarise(runs: Path) -> dict:
    out = {}
    for rdir in sorted(runs.glob("userdp_rounds*")):
        rounds = int(rdir.name.replace("userdp_rounds", ""))
        rows = []
        for key in ("None", "10.0", "5.0", "3.0", "1.0", "0.5"):
            point = rdir / f"epsilon_{key}"
            au, ap, lk, lk2, mia, apa, clip, errors, eps_acc = ([] for _ in range(9))
            sig = None
            for seed_dir in sorted(point.glob("seed_*")):
                f = seed_dir / "results.json"
                if not f.exists():
                    errors.append(seed_dir.name)
                    continue
                r = json.loads(f.read_text())
                m = r["models"].get(REF, {})
                if "roc_auc" not in m:
                    errors.append(m.get("error", seed_dir.name))
                    continue
                au.append(m["roc_auc"]); ap.append(m["average_precision"])
                apa.append(r["models"][APACHE]["roc_auc"])
                pa = r.get("privacy_attacks", {})
                inv = pa.get("embedding_inversion", {})
                lk.append(inv.get("leakage_ratio")); lk2.append(leakage_without_codes(inv))
                mia.append(pa.get("membership_inference", {}).get("attack_auc"))
                ch1 = r.get("privacy", {}).get("channel_1_model_updates", {})
                sig = ch1.get("noise_multiplier", sig)
                eps_acc.append(ch1.get("epsilon"))
                clip.append(ch1.get("observed_update_norms", {}).get("final_clip_norm"))
            summary = point / "multi_seed_summary.json"
            if not au and summary.exists():
                errors.append("summary present, no per-seed model results")
            per_seed = {sd.name: json.loads((sd / "results.json").read_text())["models"]
                        .get(REF, {}).get("roc_auc")
                        for sd in sorted(point.glob("seed_*")) if (sd / "results.json").exists()}
            rows.append({
                "eps": key, "sigma": sig, "n_ok": len(au), "errors": errors,
                "auroc_per_seed": per_seed,
                "accounted_eps": mean_ci(eps_acc),
                "auroc": mean_ci(au), "auprc": mean_ci(ap), "leakage": mean_ci(lk),
                "leakage_no_codes": mean_ci(lk2), "mia": mean_ci(mia), "apache": mean_ci(apa),
                "final_clip_norm": mean_ci(clip),
                "below_apache_all_seeds": bool(au) and all(x < y for x, y in zip(au, apa)),
            })
        base = rows[0]["auroc_per_seed"]
        for r in rows[1:]:
            seeds = sorted(set(base) & set(r["auroc_per_seed"]))
            d = [r["auroc_per_seed"][k] - base[k] for k in seeds
                 if r["auroc_per_seed"][k] is not None and base[k] is not None]
            r["cost_vs_no_dp"] = mean_ci(d)
        out[rounds] = rows
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    facts = summarise(args.runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(facts, indent=1))

    c = lambda d, k=4: "—" if d is None else (
        f"{d['mean']:.{k}f}" + ("" if d["n"] < 2 else f" [{d['lo']:.{k}f}, {d['hi']:.{k}f}]"))
    for rounds, rows in sorted(facts.items()):
        print(f"\n=== {rounds} rounds")
        for r in rows:
            print(f"eps {r['eps']:>4}  sigma {r['sigma'] if r['sigma'] is None else round(r['sigma'], 3)!s:>6}"
                  f"  n={r['n_ok']}  AUROC {c(r['auroc'])}  AUPRC {c(r['auprc'])}"
                  f"  L {c(r['leakage'], 3)}  L' {c(r['leakage_no_codes'], 3)}"
                  f"  MIA {c(r['mia'])}  APACHE {c(r['apache'])}"
                  f"  below-APACHE-all {r['below_apache_all_seeds']}"
                  + (f"  ERRORS {r['errors'][:2]}" if r["errors"] else ""))


if __name__ == "__main__":
    main()
