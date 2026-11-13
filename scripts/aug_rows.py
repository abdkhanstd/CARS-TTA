"""Table I rows for the shift-augmented segmenter (results/aug/<tag>), confirmation folds.
Gains and harms are measured against the ORIGINAL source model's Dice of the same patient, so the rows are
directly comparable with the rest of Table I. p: unit-level Wilcoxon of CARS-TTA (original segmenter) minus the row,
Holm over the rows of this block.
python scripts/aug_rows.py > results/aggregate/aug_rows.tex
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AG = os.path.join(ROOT, "results", "aggregate")
CONF = ["srcA_s1", "srcA_s2", "srcB_s0", "srcB_s1"]
OURS = "CARS-TTA (ours)"
rng = np.random.default_rng(0)
per = pd.read_csv(os.path.join(AG, "per_patient_all.csv"))
per = per[per.tag.isin(CONF)]
recs = []
for tag in CONF:
    for tt in sorted(per[per.tag == tag].target.unique()):
        c = pd.read_csv(os.path.join(ROOT, "results", "aug", tag, f"cand_target_{tt}.csv"))
        c = (c[c.episode == 0] if "episode" in c else c).set_index("pid")
        o = per[(per.tag == tag) & (per.target == tt)].set_index("pid")
        pp = os.path.join(ROOT, "results", "aug", tag, "eval_wide", f"per_patient_{tt}.csv")
        ch = pd.read_csv(pp).set_index("pid")[OURS + " [choice]"] if os.path.exists(pp) else None
        for pid in o.index:
            base = o.loc[pid, "Source (no adaptation)"]
            sel = {"aug_src": "src", "aug_tent3": "hi3"}
            if ch is not None:
                sel["aug_cars"] = ch.loc[pid]
            for m, cc in sel.items():
                recs.append({"tag": tag, "target": tt, "pid": pid, "method": m, "dice": c.loc[pid, f"dice__{cc}"],
                             "hd95": c.loc[pid, f"hd95__{cc}"], "gain": c.loc[pid, f"dice__{cc}"] - base,
                             "chg": True})  # a retrained segmenter changes every prediction of the original model
            recs.append({"tag": tag, "target": tt, "pid": pid, "method": "ours", "dice": o.loc[pid, OURS],
                         "hd95": np.nan, "gain": o.loc[pid, OURS] - base, "chg": True})
d = pd.DataFrame(recs)
d.to_csv(os.path.join(AG, "aug_per_patient.csv"), index=False)
U = d.groupby(["method", "tag", "target"])
st = pd.DataFrame({"gain": U.gain.mean(), "dice": U.dice.mean(), "hd95": U.hd95.apply(np.nanmean) if d.hd95.notna().any() else np.nan,
                   "h0": U.gain.apply(lambda x: (x < 0).mean()), "h1": U.gain.apply(lambda x: (x < -0.01).mean()),
                   "h5": U.gain.apply(lambda x: (x < -0.05).mean()), "chg": U.chg.mean()})
LAB = [("aug_src", "Shift-augmented segmenter, no adaptation"), ("aug_tent3", "Shift-augmented segmenter + TENT, 3 steps"),
       ("aug_cars", "Shift-augmented segmenter + \\cars{}")]
LAB = [x for x in LAB if x[0] in st.index.get_level_values(0)]
ours = st.loc["ours"]
ps = {}
for m, _ in LAB:
    diff = (ours.dice - st.loc[m].dice)
    ps[m] = wilcoxon(diff).pvalue
order = sorted(ps, key=ps.get); holm, run = {}, 0
for r, k in enumerate(order):
    run = max(run, min(1, (len(order) - r) * ps[k])); holm[k] = run
print("\\midrule")
f = lambda v: f"{v:+.2f}".replace("-", "$-$")
for m, lab in LAB:
    s = st.loc[m]
    lo, hi = np.quantile([rng.choice(s.gain.values, len(s)).mean() for _ in range(5000)], [0.025, 0.975])
    better = int(((ours.dice - s.dice) > 0).sum())
    p = holm[m]; pf = "$<$0.001" if p < 0.001 else f"{p:.3f}"
    print(f"{lab}$^\\ddagger$ & {s.dice.mean()*100:.2f} & {s.hd95.mean():.2f} & {f(s.gain.mean()*100)} [{f(lo*100)}, {f(hi*100)}] & "
          f"{s.h0.mean()*100:.1f} & {s.h1.mean()*100:.1f} & {s.h5.mean()*100:.1f} & {s.chg.mean()*100:.0f} & {better}/16, {pf} \\\\")
src = st.reset_index(); src["v"] = src.tag.str[:4]
print("% by source vendor (gain vs original source, points):")
print("% " + src.groupby(["method", "v"]).gain.mean().mul(100).round(2).to_string().replace("\n", "\n% "))
