"""Write a patient-level split CSV (pid, dataset, vendor, role).

Roles for source patients: seg_train / seg_val / router_train / router_val. Target patients: test.
python scripts/make_split.py --index data/cache/acdc/index.csv --source-vendors Siemens --out splits/dev_acdc.csv
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cars_tta.common import make_source_split  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--index", nargs="+", required=True)
ap.add_argument("--source-vendors", nargs="+", required=True)
ap.add_argument("--source-dataset", default=None)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
df = pd.concat([pd.read_csv(i) for i in a.index], ignore_index=True)
is_src = df.vendor.isin(a.source_vendors)
if a.source_dataset:
    is_src &= df.dataset == a.source_dataset
src = df[is_src]
roles = make_source_split(src.pid.tolist(), seed=a.seed, strata=src.group.fillna("").tolist())
df["role"] = [roles.get(p, "test") if s else "test" for p, s in zip(df.pid, is_src)]
assert not set(df.pid[df.role == "test"]) & set(df.pid[df.role != "test"])
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
df[["pid", "dataset", "vendor", "centre", "group", "role"]].to_csv(a.out, index=False)
print(df.groupby(["dataset", "vendor", "role"]).size())
