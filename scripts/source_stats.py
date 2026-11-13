"""Source statistics from seg_train patients only: per-channel mean/std of the pooled bottleneck descriptor
(for the OOD features) and the mean intensity quantile function (template for histogram matching)."""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cars_tta.candidates import source_template  # noqa: E402
from cars_tta.common import patient_input  # noqa: E402
from cars_tta.data import load_patient  # noqa: E402
from cars_tta.unet import UNet2D  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--split", required=True)
ap.add_argument("--cache", nargs="+", default=["data/cache/mnms", "data/cache/acdc"])
ap.add_argument("--out", required=True)
a = ap.parse_args()
split = pd.read_csv(a.split)
model = UNet2D().cuda()
model.load_state_dict(torch.load(a.ckpt, weights_only=True))
model.eval()
D, vols = [], []
for pid in split.pid[split.role == "seg_train"]:
    c = next(c for c in a.cache if os.path.exists(os.path.join(c, f"{pid}.npz")))
    pat = load_patient(c, pid)
    vols += [pat.img[ph] for ph in pat.phases]
    x, _ = patient_input(pat)
    with torch.no_grad():
        b = torch.cat([model(x[i:i + 16].cuda(), return_bottleneck=True)[1].mean((2, 3)).cpu() for i in range(0, len(x), 16)])
    D.append(b.mean(0).numpy())
D = np.stack(D)
np.savez(a.out, mu=D.mean(0), sd=D.std(0) + 1e-6, template=source_template(vols))
print("source stats from", len(D), "patients ->", a.out)
