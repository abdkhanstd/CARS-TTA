"""Table I (confirmation folds): Dice, HD95, gain [95% CI over units], harm at 0 / 1 / 5 Dice points (unit means),
fraction of patients changed, units where CARS-TTA is better, unit-level Wilcoxon with Holm over all comparisons.
Merges the main per-patient table with the extra selectors of scripts/extra_analysis.py.
python scripts/table_main.py > results/aggregate/table_main.tex
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
rows = pd.read_csv(os.path.join(AG, "all_rows.csv"))
rows = rows[rows.tag.isin(CONF) & (rows.method != "__routing__")]
src = per["Source (no adaptation)"]
long = []
for m in [c for c in per.columns if c not in ("pid", "dice_source", "tag", "target")]:
    long.append(pd.DataFrame({"tag": per.tag, "target": per.target, "pid": per.pid, "method": m,
                              "gain": per[m] - src, "dice": per[m]}))
long = pd.concat(long)
hd_unit = rows.groupby(["method", "tag", "target"]).hd95.mean()
ex = pd.read_csv(os.path.join(AG, "extra", "per_patient_extra.csv"))
NEW = ["Agreement selection (AETTA-style)", "Max-confidence selection", "Cohort min-entropy (one per site)",
       "Cohort agreement (one per site)", "CARS-TTA cohort (one per site)", "[privileged] Best fixed per site"]
ex = ex[ex.method.isin(NEW)]
long = pd.concat([long, ex[["tag", "target", "pid", "method", "gain", "dice"]]])
hd_unit = pd.concat([hd_unit, ex.groupby(["method", "tag", "target"]).hd95.apply(np.nanmean)])

NAME = [("Source (no adaptation)", "Source model, no adaptation", "src"),
        ("TENT fixed 1 step", "TENT, 1 step \\cite{wang2021tent}", "fixed"),
        ("TENT fixed 3 steps", "TENT, 3 steps \\cite{wang2021tent}", "fixed"),
        ("EATA fixed 3 steps", "EATA, 3 steps, untuned \\cite{niu2022eata}", "fixed"),
        ("SAR fixed 3 steps", "SAR, 3 steps \\cite{niu2023sar}", "fixed"),
        ("Histogram matching always", "Histogram matching", "fixed"),
        ("Consensus ensemble always", "Consensus of all candidates", "fixed"),
        ("Best fixed on source val.", "Best candidate on source val.", "fixed"),
        ("Minimum-entropy selection", "Minimum-entropy selection", "select"),
        ("Max-confidence selection", "Max-confidence selection", "select"),
        ("Agreement selection (AETTA-style)", "Consensus-agreement selection \\cite{lee2024aetta}", "select"),
        ("Entropy gate (src-cal.)", "Entropy gate", "select"),
        ("Fragmentation gate (src-cal.)", "Fragmentation gate \\cite{wang2026fragmentation}", "select"),
        ("Fragmentation tertile router (src-cal.)", "Fragmentation router \\cite{wang2026fragmentation}", "select"),
        ("Cohort min-entropy (one per site)", "Cohort min-entropy, one per site$^\\ast$", "cohort"),
        ("Cohort agreement (one per site)", "Cohort agreement, one per site$^\\ast$", "cohort"),
        ("CARS-TTA cohort (one per site)", "\\cars{} cohort, one per site$^\\ast$", "cohort"),
        ("CARS steps-only (src, 1-3 TENT steps)", "\\cars{}, TENT steps only", "ours"),
        ("CARS linear (ridge scorer)", "\\cars{}, linear scorer", "ours"),
        (OURS, "\\textbf{\\cars{}}", "ours"),
        ("[privileged] Frag. tertile router, target-cal.", "Frag.\\ router, target labels$^\\dagger$", "priv"),
        ("[privileged] Best fixed per site", "Best candidate per site$^\\dagger$", "priv"),
        ("[privileged] Oracle, all candidates", "Oracle over candidates$^\\dagger$", "priv")]

U = long.groupby(["method", "tag", "target"])
stat = pd.DataFrame({"gain": U.gain.mean(), "dice": U.dice.mean(), "h0": U.gain.apply(lambda x: (x < 0).mean()),
                     "h1": U.gain.apply(lambda x: (x < -0.01).mean()), "h5": U.gain.apply(lambda x: (x < -0.05).mean()),
                     "chg": U.gain.apply(lambda x: (x != 0).mean())})
stat["hd95"] = hd_unit
ours = stat.loc[OURS]
ps, better = {}, {}
for m, _, g in NAME:
    if m == OURS:
        continue
    d = (ours.dice - stat.loc[m].dice).dropna()
    ps[m] = wilcoxon(d).pvalue if np.any(d != 0) else np.nan
    better[m] = f"{int((d > 0).sum())}/{len(d)}"
keys = [k for k in ps if np.isfinite(ps[k])]
order = sorted(keys, key=lambda k: ps[k])
holm, run = {}, 0
for r, k in enumerate(order):
    run = max(run, min(1, (len(order) - r) * ps[k])); holm[k] = run
last = None
for m, lab, g in NAME:
    s = stat.loc[m]
    if last is not None and g != last:
        print("\\midrule")
    last = g
    lo, hi = np.quantile([rng.choice(s.gain.values, len(s)).mean() for _ in range(5000)], [0.025, 0.975])
    p = holm.get(m, np.nan)
    pf = "--" if m == OURS else ("$<$0.001" if p < 0.001 else f"{p:.3f}")
    bt = "--" if m == OURS else better[m]
    f = lambda v: f"{v:+.2f}".replace("-", "$-$")
    print(f"{lab} & {s.dice.mean()*100:.2f} & {s.hd95.mean():.2f} & {f(s.gain.mean()*100)} [{f(lo*100)}, {f(hi*100)}] & "
          f"{s.h0.mean()*100:.1f} & {s.h1.mean()*100:.1f} & {s.h5.mean()*100:.1f} & {s.chg.mean()*100:.0f} & {bt}, {pf} \\\\")
print("% Holm family size:", len(order), "methods:", "; ".join(order))
