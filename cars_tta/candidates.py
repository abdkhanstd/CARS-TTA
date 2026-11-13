"""Wide candidate set for one patient: the source prediction, two TENT trajectories (high and low learning rate,
1-3 steps each), four label-free input corrections, and the consensus (mean softmax of all of these). The flip prediction is
kept only as an uncertainty feature.

All candidates are produced without labels. Input corrections act on the [0,1]-scaled scan before the per-volume
z-score; the histogram-matching template is computed from source segmentation-training scans only.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from .common import patient_input

QS = np.linspace(0, 100, 101)
INPUT_CANDS = ["g07", "g14", "hm", "dn"]


def source_template(volumes):
    """Mean intensity quantile function of source volumes ([0,1] scale)."""
    return np.mean([np.percentile(v, QS) for v in volumes], 0)


def hist_match(v, template):
    q = np.percentile(v, QS)
    q = np.maximum.accumulate(q + np.arange(len(q)) * 1e-9)
    return np.interp(v, q, template).astype(np.float32)


def correct(v, name, template=None):
    if name == "g07":
        return np.clip(v, 0, None) ** 0.7
    if name == "g14":
        return np.clip(v, 0, None) ** 1.4
    if name == "hm":
        return hist_match(v, template)
    if name == "dn":
        return gaussian_filter(v, sigma=(0, 0.75, 0.75)).astype(np.float32)
    raise ValueError(name)


@torch.no_grad()
def predict(model, x, bs=16):
    return torch.cat([model(x[i:i + bs]).softmax(1).half().cpu() for i in range(0, len(x), bs)]).numpy()


def make_candidates(model, episode_hi, episode_lo, pat, img, template, dev, K=3, bs=16):
    """Returns ordered dict name -> (N,C,H,W) fp16 softmax, slice index, bottleneck descriptor, flip prob, cost stats."""
    x, idx = patient_input(pat, img_override=img)
    x = x.to(dev)
    cands, cost = {}, {}
    hi, bott, st_hi = episode_hi.run(x, K, return_bottleneck=True)
    cands["src"] = hi[0]
    for k in range(1, K + 1):
        cands[f"hi{k}"] = hi[k]
    lo, _, st_lo = episode_lo.run(x, K)
    for k in range(1, K + 1):
        cands[f"lo{k}"] = lo[k]
    for n in INPUT_CANDS:
        img_c = {ph: correct(img[ph], n, template) for ph in img}
        xc, _ = patient_input(pat, img_override=img_c)
        cands[n] = predict(episode_hi.model, xc.to(dev), bs)
    flip = predict(episode_hi.model, x.flip(-1), bs)[..., ::-1].copy()
    stack = np.stack([c.astype(np.float32) for c in cands.values()])
    cands["ens"] = stack.mean(0).astype(np.float16)  # consensus of all candidates (label-free ensemble)
    cost = {"tta_seconds": st_hi["seconds"] + st_lo["seconds"], "grad_steps": 2 * K}
    return cands, idx, bott, flip, cost


CAND_NAMES = ["src", "hi1", "hi2", "hi3", "lo1", "lo2", "lo3"] + INPUT_CANDS + ["ens"]  # flip ensemble dropped: it swaps LV/RV sides (source-val Dice 0.770 vs 0.805)
FAMILY = {"src": "source", "hi1": "tent_hi", "hi2": "tent_hi", "hi3": "tent_hi", "lo1": "tent_lo", "lo2": "tent_lo",
          "lo3": "tent_lo", "g07": "input", "g14": "input", "hm": "input", "dn": "input", "ens": "consensus"}
