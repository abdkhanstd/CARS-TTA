"""Convert raw ACDC / M&Ms releases into per-patient caches and an index CSV.

python scripts/preprocess.py --dataset acdc --raw data/raw/acdc --out data/cache/acdc
python scripts/preprocess.py --dataset mnms --raw data/raw/mnms --info data/raw/mnms/<info>.csv --out data/cache/mnms
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cars_tta.data import preprocess_acdc, preprocess_mnms  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, choices=["acdc", "mnms"])
ap.add_argument("--raw", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--info", default=None)
a = ap.parse_args()
rows = preprocess_acdc(a.raw, a.out) if a.dataset == "acdc" else preprocess_mnms(a.raw, a.info, a.out)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(a.out, "index.csv"), index=False)
print(df.groupby(["vendor"]).size() if len(df) else "no cases")
print(len(df), "patients written to", a.out)
