"""Save one mid-ventricular slice under every synthetic shift (severity 0.8) for the counterfactual figure."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cars_tta import shifts as S  # noqa: E402
from cars_tta.data import load_patient  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="data/cache/acdc")
ap.add_argument("--pid", default="patient001")
ap.add_argument("--out", default="results/figdata/shift_examples.npz")
a = ap.parse_args()
pat = load_patient(a.cache, a.pid)
v, lab = pat.img["ED"], pat.lab["ED"]
z = int(np.argmax((lab > 0).sum((1, 2))))
out = {"clean": v[z], "label": lab[z]}
for n in S.ALL:
    rng = np.random.default_rng(3)
    out[n] = S.FUNCS[n](v.astype(np.float64), rng, 0.8)[z].astype(np.float32)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
np.savez_compressed(a.out, **out)
print("saved", a.out, list(out))
