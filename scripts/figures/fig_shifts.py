"""Counterfactual acquisition shifts applied to one source (vendor A) slice, plus the four label-free input
corrections. Output: paper/figs/shifts.pdf (Times New Roman, one row per family)."""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from cars_tta import shifts as S  # noqa: E402
from cars_tta.candidates import correct  # noqa: E402
from cars_tta.data import load_patient  # noqa: E402

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 7, "mathtext.fontset": "stix"})
OUT = os.path.join(os.path.dirname(ROOT), "paper", "figs")
os.makedirs(OUT, exist_ok=True)
pat = load_patient(os.path.join(ROOT, "data", "cache", "mnms"), "A0S9V9")
v, lab = pat.img["ED"], pat.lab["ED"]
z = int(np.argmax((lab > 0).sum((1, 2))))
ys, xs = np.where(lab[z] > 0)
cy, cx = int(ys.mean()), int(xs.mean())
crop = (slice(max(cy - 70, 0), cy + 70), slice(max(cx - 70, 0), cx + 70))
NAMES = {"bias": "Bias field", "gamma": "Gamma", "contrast": "Contrast", "scale": "Scaling", "gauss_noise": "Gauss. noise",
         "rician": "Rician noise", "blur": "Blur", "kspace_lowpass": "k-space LP", "downsample": "Downsample",
         "lowfreq_amp": "LF amplitude"}
panels = [("Source scan", v[z])]
for n in S.ALL:
    rng = np.random.default_rng(3)
    panels.append((NAMES[n], S.FUNCS[n](v.astype(np.float64), rng, 1.0)[z]))
lo, hi = np.percentile(v[z], [0.5, 99.5])  # one display window (source scan) for every panel
fig, axes = plt.subplots(1, len(panels), figsize=(7.16, 0.95))
for ax, (t, im) in zip(axes, panels):
    ax.imshow(im[crop], cmap="gray", vmin=lo, vmax=hi)
    ax.set_title(t, fontsize=6.5, pad=2)
    ax.axis("off")
plt.subplots_adjust(left=0.002, right=0.998, top=0.84, bottom=0.0, wspace=0.03)
fig.savefig(os.path.join(OUT, "shifts.pdf"), bbox_inches="tight", pad_inches=0.01)
fig.savefig(os.path.join(OUT, "shifts.png"), dpi=200, bbox_inches="tight", pad_inches=0.01)
print("saved", OUT)
