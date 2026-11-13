"""What does the router key on, how well does it flag its own harmful choices, and TENT at a fixed literature rate.

1. Spearman correlation of the predicted gain of each input correction with the shared (scan-level) features, on
   real targets and on held-out synthetic episodes; synthetic shift types with the largest true gamma-0.7 gain.
2. Among adapted patients: AUROC of the predicted gain of the selected candidate for "realised gain < -0.01".
3. TENT 3 steps at eta = 1e-3 (hi3 when eta_hi = 1e-3, lo3 when eta_hi = 1e-2) on the confirmation folds.
python scripts/mechanism_analysis.py
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
CONF = ["srcA_s1", "srcA_s2", "srcB_s0", "srcB_s1"]
OURS = "CARS-TTA (ours)"
FS = ["fs_unc_ent0", "fs_unc_msp0", "fs_unc_flip_agree", "fs_ood_maha", "fs_ood_img_mean", "fs_ood_img_p90",
      "fs_ood_img_skew", "fs_ood_img_kurt", "fs_ood_img_grad", "fs_morph_fg_frac0", "fs_dis_spread"]

real, syn = [], []
for t in CONF:
    for spec in open(os.path.join(RES, t, "wide_targets.txt")).read().split():
        n = spec.split("=")[0]
        te = pd.read_csv(os.path.join(RES, t, f"cand_target_{n}.csv")).set_index("pid")
        ro = pd.read_csv(os.path.join(RES, t, "eval_wide", f"router_outputs_{n}.csv")).set_index("pid")
        pp = pd.read_csv(os.path.join(RES, t, "eval_wide", f"per_patient_{n}.csv")).set_index("pid")
        d = te.loc[ro.index, FS].join(ro).join(pp[[OURS + " [choice]", OURS, "Source (no adaptation)"]])
        d["tag"], d["target"], d["source"] = t, n, t[3]
        real.append(d)
    va = pd.read_csv(os.path.join(RES, t, "cand_router_val.csv"))
    so = pd.read_csv(os.path.join(RES, t, "eval_synthval", "router_outputs_synthval.csv"))
    so.index = va.index
    d = va[FS + ["shift", "severity"]].join(so.drop(columns=["pid"]))
    d["tag"], d["source"] = t, t[3]
    syn.append(d)
real, syn = pd.concat(real), pd.concat(syn)

print("== 1. Spearman of predicted gain with scan features (pooled over confirmation folds)")
rows = []
for c in ["g07", "g14", "hm", "dn", "hi3"]:
    for f in FS:
        rows.append({"cand": c, "feature": f, "rho_real": spearmanr(real[f], real[f"gpred__{c}"])[0],
                     "rho_syn": spearmanr(syn[f], syn[f"gpred__{c}"])[0],
                     "rho_true_syn": spearmanr(syn[f], syn[f"gtrue__{c}"])[0],
                     "rho_true_real": spearmanr(real[f], real[f"gtrue__{c}"])[0]})
r = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(r[r.cand == "g07"].round(2).to_string(index=False))
print("-- true gamma-0.7 gain by synthetic shift type (contains op)")
for op in ["gamma", "contrast", "scale", "bias", "gauss_noise", "rician", "blur", "kspace_lowpass", "downsample", "lowfreq_amp"]:
    m = syn["shift"].fillna("none").str.contains(op)
    print(f"{op:15s} n={m.sum():4d} g07 true {syn.loc[m, 'gtrue__g07'].mean()*100:+.2f}  hm true {syn.loc[m, 'gtrue__hm'].mean()*100:+.2f}")
print("-- real targets: image mean and source entropy, patients routed to gamma 0.7 vs others (Philips source)")
pb = real[real.source == "B"]
for k, g in pb.groupby(pb[OURS + " [choice]"] == "g07"):
    print("g07" if k else "other", len(g), g[["fs_ood_img_mean", "fs_ood_img_p90", "fs_unc_ent0", "gtrue__g07"]].mean().round(4).to_dict())
r.to_csv(os.path.join(RES, "aggregate", "extra", "mechanism_spearman.csv"), index=False)

print("== 2. self-flagging of harmful choices (adapted patients)")
C = ["hi1", "hi2", "hi3", "lo1", "lo2", "lo3", "g07", "g14", "hm", "dn", "ens"]
ad = real[real[OURS + " [choice]"] != "src"].copy()
ad["gsel_pred"] = [row[f"gpred__{row[OURS + ' [choice]']}"] for _, row in ad.iterrows()]
ad["gsel_true"] = ad[OURS] - ad["Source (no adaptation)"]
for tol in [0.0, 0.01]:
    y = (ad.gsel_true < -tol).astype(int)
    print(f"tol {tol}: n_adapted={len(ad)} harmful={y.sum()} AUROC(-pred)={roc_auc_score(y, -ad.gsel_pred):.3f}")
print("spearman pred vs realised gain of selected:", round(spearmanr(ad.gsel_pred, ad.gsel_true)[0], 3))
ece = []
q = pd.qcut(ad.gsel_pred, 10, labels=False, duplicates="drop")
calib = ad.groupby(q)[["gsel_pred", "gsel_true"]].mean()
print("regression calibration error (mean |pred - realised| over deciles, Dice points):",
      round((calib.gsel_pred - calib.gsel_true).abs().mean() * 100, 3))

print("== 3. TENT 3 steps at eta = 1e-3")
for t in CONF:
    lr = float(open(os.path.join(RES, t, "lr.txt")).read())
    col = "hi3" if abs(lr - 1e-3) < 1e-9 else ("lo3" if abs(lr - 1e-2) < 1e-9 else None)
    s = json.load(open(os.path.join(RES, t, "eval_wide", "summary.json")))
    lab = {"hi3": "TENT fixed 3 steps", "lo3": "TENT (low lr) 3 steps"}[col]
    out = {n: (round(v["methods"][lab]["gain"] * 100, 2), round(v["methods"][lab]["harm_rate"] * 100, 1)) for n, v in s["targets"].items()}
    print(t, "eta_hi", lr, "->", col, out)
