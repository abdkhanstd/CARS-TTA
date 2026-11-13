"""Qualitative cases (Philips source, fold srcB_s0): image, ground truth, source prediction, TENT 3 steps, CARS-TTA choice.
Two wins and one failure. Output: paper/figs/qualitative.pdf"""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(os.path.dirname(ROOT), "paper", "figs")
R = os.path.join(ROOT, "results", "srcB_s0")
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 7, "mathtext.fontset": "stix", "pdf.fonttype": 42})
CMAP = ListedColormap([(0, 0, 0, 0), "#2a78d6", "#1baf7a", "#eb6834"])   # bg, LV, MYO, RV
CASES = [("A", "C6U4W8", "Philips$\\to$Siemens"), ("ACDC", "patient114", "Philips$\\to$ACDC"), ("A", "P3P9S5", "Philips$\\to$Siemens (failure)")]
NAME = {"g07": "gamma 0.7", "g14": "gamma 1.4", "hm": "hist. match", "dn": "denoise", "src": "source"}


def load(tgt, pid):
    for f in glob.glob(os.path.join(R, f"cand_target_{tgt}_shard*_masks.npz")):
        z = np.load(f)
        if f"{pid}__img" in z:
            return {k.split("__")[1]: z[k] for k in z.keys() if k.startswith(pid + "__")}
    raise KeyError(pid)


def dice_slice(p, g):
    return np.mean([2 * ((p == c) & (g == c)).sum() / max((p == c).sum() + (g == c).sum(), 1) for c in (1, 2, 3)])


fig, ax = plt.subplots(len(CASES), 5, figsize=(3.5, 2.25), gridspec_kw={"wspace": 0.03, "hspace": 0.2})
for r, (tgt, pid, title) in enumerate(CASES):
    pp = pd.read_csv(os.path.join(R, "eval_wide", f"per_patient_{tgt}.csv")).set_index("pid").loc[pid]
    choice = pp["CARS-TTA (ours) [choice]"]
    m = load(tgt, pid)
    img, gt = m["img"].astype(np.float32), m["gt"]
    diff = [abs(dice_slice(m[choice][z], gt[z]) - dice_slice(m["src"][z], gt[z])) if gt[z].any() else -1 for z in range(len(gt))]
    z = int(np.argmax(diff))
    ys, xs = np.where(gt[z] > 0)
    cy, cx = int(ys.mean()), int(xs.mean())
    sl = (slice(max(cy - 48, 0), cy + 48), slice(max(cx - 48, 0), cx + 48))
    panels = [("Image", None, None), ("Ground truth", gt, None), ("Source", m["src"], pp["Source (no adaptation)"]),
              ("TENT-3", m["hi3"], pp["TENT fixed 3 steps"]), (f"CARS: {NAME.get(choice, choice)}", m[choice], pp["CARS-TTA (ours)"])]
    for c, (lab, msk, d) in enumerate(panels):
        a = ax[r, c]
        a.imshow(img[z][sl], cmap="gray", vmin=np.percentile(img[z], 1), vmax=np.percentile(img[z], 99.5))
        if msk is not None:
            a.imshow(msk[z][sl], cmap=CMAP, vmin=0, vmax=3, alpha=0.55, interpolation="nearest")
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
        if r == 0:
            a.set_title(lab if c < 4 else "CARS-TTA", fontsize=6.5, pad=2)
        if d is not None:
            a.text(0.03, 0.03, f"{d * 100:.1f}", transform=a.transAxes, color="white", fontsize=5.8, va="bottom",
                   bbox=dict(facecolor="black", alpha=0.45, pad=0.6, lw=0))
        if c == 4:
            a.text(0.97, 0.97, NAME.get(choice, choice), transform=a.transAxes, color="white", fontsize=5.5, ha="right", va="top",
                   bbox=dict(facecolor="black", alpha=0.45, pad=0.6, lw=0))
    ax[r, 0].set_ylabel(title.replace(" (failure)", "\n(failure)"), fontsize=5.8, labelpad=2)
fig.savefig(os.path.join(OUT, "qualitative.pdf"), bbox_inches="tight", pad_inches=0.01)
fig.savefig(os.path.join(OUT, "qualitative.png"), bbox_inches="tight", pad_inches=0.01, dpi=300)
print("ok")
