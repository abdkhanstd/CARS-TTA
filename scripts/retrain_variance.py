"""Per-patient Dice difference between the two source models of the same vendor (different split seeds) on the
shared target patients, as a reference for how much retraining alone changes individual predictions.
python scripts/retrain_variance.py   ->  results/aggregate/retrain_variance.csv
"""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AG = os.path.join(ROOT, "results", "aggregate")
per = pd.read_csv(os.path.join(AG, "per_patient_all.csv"))
rows = []
for a, b in [("srcA_s1", "srcA_s2"), ("srcB_s0", "srcB_s1")]:
    x = per[per.tag == a].set_index(["target", "pid"])["Source (no adaptation)"]
    y = per[per.tag == b].set_index(["target", "pid"])["Source (no adaptation)"]
    d = (y - x).dropna() * 100
    both = pd.concat([d, -d])
    rows.append({"source": a[:4], "model_a": a, "model_b": b, "n": len(d), "mean_abs_diff": d.abs().mean(),
                 "pct_diff_gt1": (d.abs() > 1).mean() * 100, "pct_drop_gt1_either_direction": (both < -1).mean() * 100,
                 "pct_drop_gt5": (both < -5).mean() * 100})
r = pd.DataFrame(rows)
r.to_csv(os.path.join(AG, "retrain_variance.csv"), index=False)
print(r.round(2).to_string(index=False))
