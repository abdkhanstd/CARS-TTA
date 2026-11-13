"""Ablation table over three router seeds (seed 0: results/<tag>/ablations, seeds 1-2: results/seeds/<tag>).
Per variant: mean over units of the gain and harm, mean +- s.d. over seeds; Delta vs the full model with the same
seed and unit, seed-averaged per unit, 95% bootstrap CI over the 16 units.
python scripts/abl_seeds.py > results/aggregate/abl_rows.tex
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
CONF = ["srcA_s1", "srcA_s2", "srcB_s0", "srcB_s1"]
OURS = "CARS-TTA (ours)"
ABL = [("full", "Full \\cars{}"), None,
       ("A1_no_counterfactuals", "No shifts (clean episodes only)"), ("A2_intensity_only", "Intensity shifts only"),
       ("A3_frequency_only", "Frequency shifts only"), None,
       ("A6_no_uncertainty", "w/o uncertainty features"), ("A8b_no_disagreement", "w/o disagreement features"),
       ("A5_no_fragmentation", "w/o fragmentation features"), ("A7_no_morphology", "w/o morphology features"),
       ("A8_no_ood", "w/o OOD features"), ("A14_scan_features_only", "Scan-level information only"), None,
       ("A9_no_harm_term", "w/o harm term ($\\lambda_h{=}0$)"), None,
       ("A11_binary_consensus_gate", "Actions: consensus gate only"), ("A12_steps_only", "Actions: TENT steps only"),
       ("A12_one_step_only", "Actions: one TENT step only")]
rows = []
for tag in CONF:
    for item in ABL:
        if item is None:
            continue
        k = item[0]
        for sd, p in [(0, os.path.join(RES, tag, "ablations", k)), (1, os.path.join(RES, "seeds", tag, f"{k}_s1")),
                      (2, os.path.join(RES, "seeds", tag, f"{k}_s2"))]:
            f = os.path.join(p, "summary.json")
            if not os.path.exists(f):
                continue
            for tt, tv in json.load(open(f))["targets"].items():
                r = tv["methods"][OURS]
                rows.append({"v": k, "seed": sd, "tag": tag, "target": tt, "gain": r["gain"], "harm": r["harm_rate"]})
d = pd.DataFrame(rows)
full = d[d.v == "full"].set_index(["seed", "tag", "target"])
rng = np.random.default_rng(0)
out = []
for item in ABL:
    if item is None:
        out.append("\\midrule")
        continue
    k, lab = item
    e = d[d.v == k].set_index(["seed", "tag", "target"])
    nseed = e.reset_index().seed.nunique()
    per_seed = e.groupby(level=0)[["gain", "harm"]].mean() * 100
    g = f"{per_seed.gain.mean():+.2f}$\\pm${per_seed.gain.std(ddof=1) if nseed > 1 else 0:.2f}".replace("+-", "$-$")
    h = f"{per_seed.harm.mean():.1f}"
    if k == "full":
        out.append(f"{lab} & {g} & {h} & -- \\\\")
        continue
    j = e.join(full, rsuffix="_f", how="inner")
    delta = (j.gain - j.gain_f).groupby(level=[1, 2]).mean() * 100
    bs = [rng.choice(delta.values, len(delta)).mean() for _ in range(5000)]
    lo, hi = np.quantile(bs, [0.025, 0.975])
    fm = lambda v: f"{v:+.2f}".replace("-", "$-$")
    star = "" if lo <= 0 <= hi else "$^\\ast$"
    out.append(f"{lab} & {g.replace('-', '$-$') if g.startswith('-') else g} & {h} & {fm(delta.mean())} [{fm(lo)}, {fm(hi)}]{star} \\\\")
    print(f"% {k}: seeds={nseed} units={len(delta)}")
print("\n".join(out))
