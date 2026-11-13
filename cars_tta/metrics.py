"""Patient-level segmentation metrics (Dice, HD95 in mm)."""
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

CLASSES = {1: "LV", 2: "MYO", 3: "RV"}


def dice(a: np.ndarray, b: np.ndarray, c: int) -> float:
    x, y = a == c, b == c
    s = x.sum() + y.sum()
    if s == 0:
        return 1.0
    return float(2 * np.logical_and(x, y).sum() / s)


def mean_dice(pred, gt):
    return float(np.mean([dice(pred, gt, c) for c in CLASSES]))


def _surface(m):
    return m & ~binary_erosion(m, structure=np.ones((1, 3, 3)))


def hd95(pred, gt, c, spacing_zyx):
    p, g = pred == c, gt == c
    if p.sum() == 0 and g.sum() == 0:
        return 0.0
    if p.sum() == 0 or g.sum() == 0:
        return float("nan")
    sp, sg = _surface(p), _surface(g)
    dg = distance_transform_edt(~sg, sampling=spacing_zyx)
    dp = distance_transform_edt(~sp, sampling=spacing_zyx)
    d = np.concatenate([dg[sp], dp[sg]])
    return float(np.percentile(d, 95))


def patient_metrics(preds: dict, gts: dict, spacing_xyz) -> dict:
    """preds/gts: phase -> (S, h, w) label maps. Returns per-class Dice/HD95 averaged over phases plus means."""
    sp = (spacing_xyz[2], spacing_xyz[0], spacing_xyz[1])
    out = {}
    for c, n in CLASSES.items():
        out[f"dice_{n}"] = float(np.mean([dice(preds[p], gts[p], c) for p in gts]))
        h = [hd95(preds[p], gts[p], c, sp) for p in gts]
        h = [x for x in h if np.isfinite(x)]
        out[f"hd95_{n}"] = float(np.mean(h)) if h else float("nan")
    out["dice"] = float(np.mean([out[f"dice_{n}"] for n in CLASSES.values()]))
    hs = [out[f"hd95_{n}"] for n in CLASSES.values() if np.isfinite(out[f"hd95_{n}"])]
    out["hd95"] = float(np.mean(hs)) if hs else float("nan")
    return out


def grid_dice(preds: dict, labs: dict) -> float:
    """Fast patient Dice on the network grid (used for source-side oracle labels)."""
    return float(np.mean([mean_dice(preds[p], labs[p]) for p in labs]))
