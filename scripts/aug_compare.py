"""Compare the shift-augmented segmenter (results/aug/<tag>) with the original one (results/<tag>) on the
confirmation folds: unit means of source Dice and TENT-3 Dice (Dice points)."""
import glob, os, json
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONF = ["srcA_s1", "srcA_s2", "srcB_s0", "srcB_s1"]
rows = []
for tag in CONF:
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "aug", tag, "cand_target_*.csv"))):
        t = os.path.basename(f)[12:-4]
        a = pd.read_csv(f); a = a[a.episode == 0] if "episode" in a else a
        o = pd.read_csv(os.path.join(ROOT, "results", tag, f"cand_target_{t}.csv")); o = o[o.episode == 0] if "episode" in o else o
        m = a.merge(o, on="pid", suffixes=("_a", "_o"))
        r = {"tag": tag, "target": t, "n": len(m)}
        for c in ["src", "hi3", "g07", "hm", "ens"]:
            r[f"orig_{c}"] = m[f"dice__{c}_o"].mean() * 100
            r[f"aug_{c}"] = m[f"dice__{c}_a"].mean() * 100
        r["aug_tent_harm"] = (m["dice__hi3_a"] < m["dice__src_a"]).mean() * 100
        r["aug_oracle"] = (m[[c for c in m if c.startswith("dice__") and c.endswith("_a") and c.count("_") == 3]].max(axis=1)).mean() * 100
        rows.append(r)
d = pd.DataFrame(rows)
os.makedirs(os.path.join(ROOT, "results", "aggregate"), exist_ok=True)
d.to_csv(os.path.join(ROOT, "results", "aggregate", "aug_compare.csv"), index=False)
pd.set_option("display.width", 250)
print(d.round(2))
print(d.drop(columns=["tag", "target"]).mean().round(2))
d["src"] = d.tag.str[:4]
print(d.groupby("src")[[c for c in d if c.startswith(("orig", "aug"))]].mean().round(2))
