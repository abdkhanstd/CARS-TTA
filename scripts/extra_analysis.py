"""Round-1 referee analyses on the confirmation folds (all label-free unless marked privileged):

1. extra selectors: consensus-agreement (AETTA-style accuracy proxy), max-confidence, cohort-level (one candidate per
   target site chosen from the unlabelled cohort by mean entropy, mean agreement, or CARS-TTA's mean predicted gain),
   and the privileged best fixed candidate per site;
2. harm at tolerances (Dice drop > t for t in 0, 0.005, 0.01, 0.02, 0.05) and the help rate;
3. paired differences of CARS-TTA vs key methods with three resampling schemes: units (fold x target), vendor pairs
   (seeds averaged), and a patient-clustered bootstrap (patients resampled within target, both seeds kept together);
4. vendor-pair level Wilcoxon (8 pairs).
python scripts/extra_analysis.py --out results/aggregate/extra
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
ap = argparse.ArgumentParser()
ap.add_argument("--tags", nargs="+", default=["srcA_s1", "srcA_s2", "srcB_s0", "srcB_s1"])
ap.add_argument("--out", default=os.path.join(RES, "aggregate", "extra"))
ap.add_argument("--n-boot", type=int, default=5000)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
rng = np.random.default_rng(0)
C = ["src", "hi1", "hi2", "hi3", "lo1", "lo2", "lo3", "g07", "g14", "hm", "dn", "ens"]
OURS = "CARS-TTA (ours)"

rows = []
for tag in a.tags:
    summ = json.load(open(os.path.join(RES, tag, "eval_wide", "summary.json")))
    tau = summ["selected"]["tau"]
    for spec in open(os.path.join(RES, tag, "wide_targets.txt")).read().split():
        tgt = spec.split("=")[0]
        te = pd.read_csv(os.path.join(RES, tag, f"cand_target_{tgt}.csv"))
        pp = pd.read_csv(os.path.join(RES, tag, "eval_wide", f"per_patient_{tgt}.csv")).set_index("pid")
        ro = pd.read_csv(os.path.join(RES, tag, "eval_wide", f"router_outputs_{tgt}.csv")).set_index("pid")
        te = te.set_index("pid").loc[pp.index]
        D = te[[f"dice__{c}" for c in C]].to_numpy()
        HD = te[[f"hd95__{c}" for c in C]].to_numpy()
        g = D - D[:, :1]
        n = len(te)
        sel = {}
        # per-patient label-free selectors
        agree = np.stack([te["fc__ens__dis_dice"]] + [te[f"fc__{c}__dis_cons_dice"] for c in C[1:-1]], 1)  # src..dn
        sel["Agreement selection (AETTA-style)"] = agree.argmax(1)
        msp = np.stack([te["fs_unc_msp0"]] + [te[f"fc__{c}__unc_msp"] for c in C[1:]], 1)
        sel["Max-confidence selection"] = msp.argmax(1)
        ent = np.stack([te["fs_unc_ent0"]] + [te[f"fc__{c}__unc_ent"] for c in C[1:]], 1)
        sel["Minimum-entropy selection"] = ent.argmin(1)
        # cohort-level (transductive, unlabelled target cohort)
        sel["Cohort min-entropy (one per site)"] = np.full(n, ent.mean(0).argmin())
        sel["Cohort agreement (one per site)"] = np.full(n, agree.mean(0).argmax())
        gp = np.stack([ro.loc[pp.index, f"gpred__{c}"] for c in C[1:]], 1).mean(0)
        sel["CARS-TTA cohort (one per site)"] = np.full(n, int(gp.argmax()) + 1 if gp.max() > tau else 0)
        sel[OURS] = np.array([C.index(c) for c in pp[OURS + " [choice]"]])
        sel["TENT fixed 3 steps"] = np.full(n, C.index("hi3"))
        sel["Consensus ensemble always"] = np.full(n, C.index("ens"))
        sel["[privileged] Best fixed per site"] = np.full(n, D.mean(0).argmax())
        sel["[privileged] Oracle, all candidates"] = D.argmax(1)
        for m, s in sel.items():
            for pid, gi, di, hi in zip(pp.index, g[np.arange(n), s], D[np.arange(n), s], HD[np.arange(n), s]):
                rows.append({"tag": tag, "source": tag[3], "target": tgt, "pid": pid, "method": m, "gain": gi,
                             "dice": di, "hd95": hi})
df = pd.DataFrame(rows)
df.to_csv(os.path.join(a.out, "per_patient_extra.csv"), index=False)
# sanity: CARS/TENT gains here must equal the main per-patient table
main = pd.read_csv(os.path.join(RES, "aggregate", "per_patient_all.csv"))
chk = main[main.tag.isin(a.tags)].assign(g=lambda d: d[OURS] - d["Source (no adaptation)"])
print("check CARS mean gain", round(chk.g.mean(), 5), round(df[df.method == OURS].gain.mean(), 5))

# ---- 1+2: unit-averaged gain and harm at tolerances
TOL = [0, 0.005, 0.01, 0.02, 0.05]
U = df.groupby(["method", "tag", "target"]).gain
agg = []
for m, gdf in df.groupby("method"):
    u = gdf.groupby(["tag", "target"]).gain
    r = {"method": m, "gain": u.mean().mean() * 100, "help": u.apply(lambda x: (x > 0).mean()).mean() * 100,
         "dice": gdf.groupby(["tag", "target"]).dice.mean().mean() * 100,
         "hd95": gdf.groupby(["tag", "target"]).hd95.apply(np.nanmean).mean(),
         "changed": u.apply(lambda x: (x != 0).mean()).mean() * 100}
    for t in TOL + [0.10]:
        r[f"harm>{t}"] = u.apply(lambda x: (x < -t).mean()).mean() * 100
    agg.append(r)
agg = pd.DataFrame(agg).sort_values("gain")
agg.to_csv(os.path.join(a.out, "selectors_tolerance.csv"), index=False)
pd.set_option("display.width", 250)
print(agg.round(2).to_string(index=False))

# ---- 3+4: paired differences vs CARS
wide = df.pivot_table(index=["tag", "source", "target", "pid"], columns="method", values="gain").reset_index()
wide["pair"] = wide.source + ">" + wide.target
per_main = main[main.tag.isin(a.tags)].copy()
for m in ["SAR fixed 3 steps", "Fragmentation gate (src-cal.)", "[privileged] Frag. tertile router, target-cal."]:
    per_main["g"] = per_main[m] - per_main["Source (no adaptation)"]
    wide = wide.merge(per_main[["tag", "target", "pid", "g"]].rename(columns={"g": m}), on=["tag", "target", "pid"])
comp = ["TENT fixed 3 steps", "SAR fixed 3 steps", "Minimum-entropy selection", "Fragmentation gate (src-cal.)",
        "Consensus ensemble always", "Agreement selection (AETTA-style)", "Max-confidence selection",
        "Cohort min-entropy (one per site)", "Cohort agreement (one per site)", "CARS-TTA cohort (one per site)",
        "[privileged] Frag. tertile router, target-cal.", "[privileged] Best fixed per site"]
out = []
for m in comp:
    d = wide[OURS] - wide[m]
    unit = d.groupby([wide.tag, wide.target]).mean()
    pair = d.groupby(wide.pair).mean()
    bu = [rng.choice(unit.values, len(unit)).mean() for _ in range(a.n_boot)]
    # patient-clustered bootstrap: within each vendor pair resample patients (both seeds of a patient together)
    pat = d.groupby([wide.pair, wide.pid]).mean().reset_index(name="d")
    groups = [g_.d.to_numpy() for _, g_ in pat.groupby("pair")]
    bc = []
    for _ in range(a.n_boot // 5):
        bc.append(np.mean([rng.choice(x, len(x)).mean() for x in groups]))
    p_unit = wilcoxon(unit.values).pvalue if np.any(unit.values != 0) else np.nan
    p_pair = wilcoxon(pair.values).pvalue if np.any(pair.values != 0) else np.nan
    out.append({"method": m, "diff": d.groupby([wide.tag, wide.target]).mean().mean() * 100,
                "unit_ci": tuple(np.round(np.quantile(bu, [0.025, 0.975]) * 100, 2)), "p_unit": p_unit,
                "pair_diff": pair.mean() * 100, "pairs_better": f"{int((pair > 0).sum())}/{len(pair)}", "p_pair": p_pair,
                "cluster_ci": tuple(np.round(np.quantile(bc, [0.025, 0.975]) * 100, 2))})
out = pd.DataFrame(out)
for col in ["p_unit", "p_pair"]:
    ps = out[col].to_numpy(); order = np.argsort(ps); k = len(ps); adj = np.empty(k); run = 0
    for r_, i in enumerate(order):
        run = max(run, min(1, (k - r_) * ps[i])); adj[i] = run
    out[col + "_holm"] = adj
out.to_csv(os.path.join(a.out, "paired_vs_cars.csv"), index=False)
print(out.round(4).to_string(index=False))
# per-source gains of the main selectors
print(df.groupby(["source", "method"]).gain.mean().unstack(0).mul(100).round(2).to_string())
