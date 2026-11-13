"""Pool all folds/seeds: per (source vendor, target) means over seeds, a pooled paired test over (fold, target)
units, and the per-patient table for figures.

python scripts/aggregate.py --results results --tags srcA_s0 srcA_s1 srcA_s2 srcB_s0 srcB_s1 --out results/aggregate
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
ap.add_argument("--tags", nargs="+", required=True)
ap.add_argument("--dev-tags", nargs="*", default=["srcA_s0"], help="folds used during design (reported separately)")
ap.add_argument("--evaldir", default="eval_wide")
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

KEY = ["Source (no adaptation)", "TENT fixed 1 step", "TENT fixed 3 steps", "EATA fixed 3 steps", "SAR fixed 3 steps",
       "TENT (low lr) 3 steps", "Histogram matching always", "Consensus ensemble always", "Best fixed on source val.",
       "Minimum-entropy selection", "Entropy gate (src-cal.)", "Fragmentation gate (src-cal.)",
       "Fragmentation tertile router (src-cal.)", "CARS steps-only (src, 1-3 TENT steps)", "CARS linear (ridge scorer)",
       "CARS-TTA (ours)", "[privileged] Frag. tertile router, target-cal.", "[privileged] Oracle, all candidates"]


def canon(m):
    return "Best fixed on source val." if m.startswith("Best fixed on source val.") else m


rows, per_rows = [], []
for tag in a.tags:
    p = os.path.join(a.results, tag, a.evaldir, "summary.json")
    if not os.path.exists(p):
        print("missing", p)
        continue
    s = json.load(open(p))
    src, seed = tag[3], int(tag.split("_s")[1])
    for tgt, tv in s["targets"].items():
        for m, r in tv["methods"].items():
            m = canon(m)
            rows.append({"tag": tag, "source": src, "seed": seed, "target": tgt, "n": tv["n"], "method": m,
                         "dev": tag in a.dev_tags, **{k: r.get(k) for k in ["dice", "hd95", "gain", "harm_rate", "help_rate",
                                                                            "cat_harm_002", "cat_harm_005", "cat_harm_010",
                                                                            "adapt_rate", "p_holm", "delta_ours_minus_method"]}})
        rq = tv["routing"]
        rows.append({"tag": tag, "source": src, "seed": seed, "target": tgt, "n": tv["n"], "method": "__routing__",
                     "dev": tag in a.dev_tags, **rq})
        pp = pd.read_csv(os.path.join(a.results, tag, a.evaldir, f"per_patient_{tgt}.csv"))
        pp = pp[[c for c in pp.columns if not c.endswith("[choice]")]]
        pp = pp.rename(columns={c: canon(c) for c in pp.columns})
        pp["tag"], pp["target"] = tag, tgt
        per_rows.append(pp)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(a.out, "all_rows.csv"), index=False)
per = pd.concat(per_rows, ignore_index=True)
per.to_csv(os.path.join(a.out, "per_patient_all.csv"), index=False)
met = df[df.method != "__routing__"]


def summarize(sub, name):
    g = sub.groupby(["source", "target", "method"])
    out = g.agg(dice=("dice", "mean"), dice_sd=("dice", "std"), hd95=("hd95", "mean"), gain=("gain", "mean"),
                gain_sd=("gain", "std"), harm=("harm_rate", "mean"), harm_sd=("harm_rate", "std"),
                cat5=("cat_harm_005", "mean"), adapt=("adapt_rate", "mean"), n_seeds=("seed", "nunique")).reset_index()
    out.to_csv(os.path.join(a.out, f"table_{name}.csv"), index=False)
    return out


conf = summarize(met[~met.dev], "confirmation")
allf = summarize(met, "all_folds")
# pooled over (fold, target) units: mean gain/harm per method and paired Wilcoxon of CARS vs each method on unit means
units = met.pivot_table(index=["tag", "target"], columns="method", values="dice")
gains = met.pivot_table(index=["tag", "target"], columns="method", values="gain")
harms = met.pivot_table(index=["tag", "target"], columns="method", values="harm_rate")
pooled = []
for subset, mask in [("confirmation", ~units.index.get_level_values(0).isin(a.dev_tags)), ("all", np.ones(len(units), bool))]:
    U, Gn, Hm = units[mask], gains[mask], harms[mask]
    for m in [k for k in KEY if k in U.columns]:
        d = U["CARS-TTA (ours)"] - U[m]
        ok = d.notna()
        p = float(wilcoxon(d[ok]).pvalue) if (ok.sum() > 1 and np.any(d[ok] != 0)) else float("nan")
        pooled.append({"subset": subset, "method": m, "units": int(ok.sum()), "mean_gain": float(Gn[m].mean()),
                       "mean_harm": float(Hm[m].mean()), "ours_minus_method": float(d[ok].mean()),
                       "ours_better_units": int((d[ok] > 0).sum()), "wilcoxon_p": p})
pooled = pd.DataFrame(pooled)
# Holm correction within each subset over the non-ours comparisons
pooled["p_holm"] = np.nan
for sub_name in pooled.subset.unique():
    m = (pooled.subset == sub_name) & pooled.wilcoxon_p.notna()
    ps = pooled.loc[m, "wilcoxon_p"].to_numpy(); order = np.argsort(ps); k = len(ps)
    adj = np.empty(k); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (k - r) * ps[i])); adj[i] = run
    pooled.loc[m, "p_holm"] = adj
pooled.to_csv(os.path.join(a.out, "pooled_units.csv"), index=False)
rout = df[df.method == "__routing__"]
rout.groupby("target")[["spearman_gain_pooled", "spearman_gain_within_patient", "auroc_beneficial", "brier_beneficial",
                        "ece_beneficial"]].agg(["mean", "std"]).to_csv(os.path.join(a.out, "routing_quality.csv"))
pd.set_option("display.width", 250)
print(pooled[pooled.subset == "confirmation"].round(4).to_string(index=False))
print(pooled[pooled.subset == "all"].round(4).to_string(index=False))
print(rout.groupby("target")[["spearman_gain_within_patient", "auroc_beneficial", "ece_beneficial"]].mean().round(3))
