"""Label-free, patient-level router features computed from a TTA trajectory.

Groups (used for the feature ablations):
  unc   - uncertainty: entropy, max-softmax, boundary entropy, flip consistency, their changes
  dis   - disagreement: mask Dice/IoU to step 0, changed-pixel fraction, step-to-step change
  frag  - fragmentation: connected components of the disagreement region, largest-component share
  morph - morphology: structure volume ratios, connected-component counts, centroid shifts, RV/LV ratio
  ood   - feature distance of the bottleneck descriptor to source statistics (diagonal Mahalanobis), image stats
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

MIN_REGION = 16


def _ent(p):
    return -(p * np.log(p + 1e-8)).sum(1)


def _cc(mask2d):
    lab, n = ndimage.label(mask2d)
    if n == 0:
        return 0, np.zeros(0)
    sizes = ndimage.sum(mask2d, lab, range(1, n + 1))
    return n, sizes


def _boundary(m):
    return m & ~ndimage.binary_erosion(m)


def trajectory_features(cands, flip_prob0=None, bott=None, src_stats=None, img=None, K=3):
    """cands: list of (N, C, H, W) softmax arrays (fp16 ok), candidate 0 = source model.
    Returns dict name -> float."""
    P = [c[:, :, :, :].astype(np.float32) for c in cands[: K + 1]]
    M = [p.argmax(1) for p in P]
    f = {}
    N = M[0].shape[0]
    fg0 = M[0] > 0
    fgarea = max(fg0.sum(), 1)
    ent0 = _ent(P[0])
    f["unc_ent0"] = float(ent0.mean())
    f["unc_ent0_fg"] = float(ent0[fg0].mean()) if fg0.any() else 0.0
    f["unc_msp0"] = float(P[0].max(1).mean())
    bnd = np.stack([_boundary(fg0[i]) for i in range(N)])
    f["unc_bnd0"] = float(ent0[bnd].mean()) if bnd.any() else 0.0
    if flip_prob0 is not None:
        mf = flip_prob0.argmax(1)
        both = (mf > 0) | fg0
        f["unc_flip_agree"] = float((mf == M[0])[both].mean()) if both.any() else 1.0
    vol0 = np.array([(M[0] == c).sum() for c in (1, 2, 3)], dtype=float)
    f["morph_rv_lv0"] = float(vol0[2] / max(vol0[0], 1))
    f["morph_myo_lv0"] = float(vol0[1] / max(vol0[0], 1))
    f["morph_fg_frac0"] = float(fg0.mean())
    cc0 = []
    for c in (1, 2, 3):
        cc0.append(np.mean([_cc(M[0][i] == c)[0] for i in range(N)]))
    f["morph_cc0"] = float(np.mean(cc0))
    for k in range(1, K + 1):
        Mk, Pk = M[k], P[k]
        entk = _ent(Pk)
        f[f"unc_ent{k}"] = float(entk.mean())
        f[f"unc_dent{k}"] = float(entk.mean() - ent0.mean())
        f[f"unc_msp{k}"] = float(Pk.max(1).mean())
        D = Mk != M[0]
        dsum = D.sum()
        f[f"dis_delta{k}"] = float(dsum / fgarea)
        f[f"dis_step{k}"] = float((Mk != M[k - 1]).sum() / fgarea)
        dices, ious = [], []
        for c in (1, 2, 3):
            a, b = M[0] == c, Mk == c
            s = a.sum() + b.sum()
            inter = np.logical_and(a, b).sum()
            dices.append(1.0 if s == 0 else 2 * inter / s)
            u = np.logical_or(a, b).sum()
            ious.append(1.0 if u == 0 else inter / u)
        f[f"dis_dice{k}"] = float(np.mean(dices))
        f[f"dis_iou{k}"] = float(np.mean(ious))
        f[f"dis_dmin{k}"] = float(np.min(dices))
        f[f"unc_entD{k}"] = float(entk[D].mean()) if dsum else 0.0
        # fragmentation of the disagreement region
        nreg, sizes_all = 0, []
        for i in range(N):
            n, s = _cc(D[i])
            s = s[s >= MIN_REGION]
            nreg += len(s)
            sizes_all.extend(s.tolist())
        f[f"frag_n{k}"] = float(nreg / N)
        f[f"frag_largest{k}"] = float(max(sizes_all) / max(dsum, 1)) if sizes_all else 0.0
        f[f"frag_meansize{k}"] = float(np.mean(sizes_all) / fgarea) if sizes_all else 0.0
        for c, nm in ((1, "lv"), (2, "myo"), (3, "rv")):
            Dc = (Mk == c) != (M[0] == c)
            f[f"frag_{nm}{k}"] = float(sum(len([x for x in _cc(Dc[i])[1] if x >= MIN_REGION]) for i in range(N)) / N)
        volk = np.array([(Mk == c).sum() for c in (1, 2, 3)], dtype=float)
        ratio = np.log((volk + 1) / (vol0 + 1))
        f[f"morph_dvol{k}"] = float(np.abs(ratio).mean())
        f[f"morph_dvol_max{k}"] = float(np.abs(ratio).max())
        cck = np.mean([np.mean([_cc(Mk[i] == c)[0] for i in range(N)]) for c in (1, 2, 3)])
        f[f"morph_dcc{k}"] = float(cck - f["morph_cc0"])
        cs = []
        for c in (1, 2, 3):
            a, b = np.argwhere(M[0] == c), np.argwhere(Mk == c)
            if len(a) and len(b):
                cs.append(np.linalg.norm(a.mean(0)[1:] - b.mean(0)[1:]))
        f[f"morph_dcent{k}"] = float(np.mean(cs)) if cs else 0.0
    if bott is not None and src_stats is not None:
        mu, sd = src_stats["mu"], src_stats["sd"]
        z = (bott.mean(0) - mu) / sd
        f["ood_maha"] = float(np.sqrt((z ** 2).mean()))
        f["ood_maha_max"] = float(np.abs(z).max())
    if img is not None:
        x = img.astype(np.float32)
        f["ood_img_skew"] = float(((x - x.mean()) ** 3).mean() / (x.std() ** 3 + 1e-6))
        f["ood_img_kurt"] = float(((x - x.mean()) ** 4).mean() / (x.std() ** 4 + 1e-6))
        f["ood_img_grad"] = float(np.abs(np.diff(x, axis=-1)).mean() / (x.std() + 1e-6))
    return f


GROUPS = ["unc", "dis", "frag", "morph", "ood"]


# ---------------------------------------------------------------------------------------------------------------
# Candidate-wise features (wide action set): shared features of the scan and the source prediction, plus features
# that compare one candidate prediction with the source prediction. The router scores each candidate separately.
# ---------------------------------------------------------------------------------------------------------------

def shared_features(P0, flip_prob0=None, bott=None, src_stats=None, img=None):
    P0 = P0.astype(np.float32)
    M0 = P0.argmax(1)
    N = M0.shape[0]
    fg0 = M0 > 0
    ent0 = _ent(P0)
    f = {"unc_ent0": float(ent0.mean()), "unc_ent0_fg": float(ent0[fg0].mean()) if fg0.any() else 0.0,
         "unc_msp0": float(P0.max(1).mean())}
    bnd = np.stack([_boundary(fg0[i]) for i in range(N)])
    f["unc_bnd0"] = float(ent0[bnd].mean()) if bnd.any() else 0.0
    if flip_prob0 is not None:
        mf = flip_prob0.argmax(1)
        both = (mf > 0) | fg0
        f["unc_flip_agree"] = float((mf == M0)[both].mean()) if both.any() else 1.0
    vol0 = np.array([(M0 == c).sum() for c in (1, 2, 3)], dtype=float)
    f["morph_rv_lv0"] = float(vol0[2] / max(vol0[0], 1))
    f["morph_myo_lv0"] = float(vol0[1] / max(vol0[0], 1))
    f["morph_fg_frac0"] = float(fg0.mean())
    f["morph_cc0"] = float(np.mean([np.mean([_cc(M0[i] == c)[0] for i in range(N)]) for c in (1, 2, 3)]))
    if bott is not None and src_stats is not None:
        z = (bott.mean(0) - src_stats["mu"]) / src_stats["sd"]
        f["ood_maha"] = float(np.sqrt((z ** 2).mean()))
        f["ood_maha_max"] = float(np.abs(z).max())
    if img is not None:
        x = img.astype(np.float32)
        f["ood_img_skew"] = float(((x - x.mean()) ** 3).mean() / (x.std() ** 3 + 1e-6))
        f["ood_img_kurt"] = float(((x - x.mean()) ** 4).mean() / (x.std() ** 4 + 1e-6))
        f["ood_img_grad"] = float(np.abs(np.diff(x, axis=-1)).mean() / (x.std() + 1e-6))
        f["ood_img_mean"] = float(x.mean())
        f["ood_img_p90"] = float(np.percentile(x, 90))
    return f


def _mean_dice(A, B):
    d = []
    for c in (1, 2, 3):
        a, b = A == c, B == c
        s = a.sum() + b.sum()
        d.append(1.0 if s == 0 else 2 * np.logical_and(a, b).sum() / s)
    return float(np.mean(d))


def consensus_spread(cands_list, Pc):
    """Shared feature: mean disagreement of the candidates with their consensus mask."""
    Mc = Pc.argmax(1)
    return {"dis_spread": float(np.mean([1 - _mean_dice(P.argmax(1), Mc) for P in cands_list]))}


def candidate_features(P0, Pj, Pc=None):
    """Label-free comparison of candidate j with the source prediction (same slices) and, if given, with the
    consensus prediction Pc of all candidates."""
    P0, Pj = P0.astype(np.float32), Pj.astype(np.float32)
    M0, Mj = P0.argmax(1), Pj.argmax(1)
    N = M0.shape[0]
    fgarea = max((M0 > 0).sum(), 1)
    ent0, entj = _ent(P0), _ent(Pj)
    f = {"unc_ent": float(entj.mean()), "unc_dent": float(entj.mean() - ent0.mean()),
         "unc_msp": float(Pj.max(1).mean()), "unc_dmsp": float(Pj.max(1).mean() - P0.max(1).mean())}
    fgj = Mj > 0
    f["unc_ent_fg"] = float(entj[fgj].mean()) if fgj.any() else 0.0
    D = Mj != M0
    dsum = D.sum()
    f["dis_delta"] = float(dsum / fgarea)
    f["dis_prob_l1"] = float(np.abs(Pj - P0).sum(1).mean())
    dices = []
    for c in (1, 2, 3):
        a, b = M0 == c, Mj == c
        s = a.sum() + b.sum()
        dices.append(1.0 if s == 0 else 2 * np.logical_and(a, b).sum() / s)
    f["dis_dice"] = float(np.mean(dices))
    f["dis_dmin"] = float(np.min(dices))
    f["unc_entD"] = float(entj[D].mean()) if dsum else 0.0
    f["unc_ent0D"] = float(ent0[D].mean()) if dsum else 0.0
    nreg, sizes_all = 0, []
    for i in range(N):
        _, s = _cc(D[i])
        s = s[s >= MIN_REGION]
        nreg += len(s)
        sizes_all.extend(s.tolist())
    f["frag_n"] = float(nreg / N)
    f["frag_largest"] = float(max(sizes_all) / max(dsum, 1)) if sizes_all else 0.0
    for c, nm in ((1, "lv"), (2, "myo"), (3, "rv")):
        Dc = (Mj == c) != (M0 == c)
        f[f"frag_{nm}"] = float(sum(len([x for x in _cc(Dc[i])[1] if x >= MIN_REGION]) for i in range(N)) / N)
    vol0 = np.array([(M0 == c).sum() for c in (1, 2, 3)], dtype=float)
    volj = np.array([(Mj == c).sum() for c in (1, 2, 3)], dtype=float)
    ratio = np.log((volj + 1) / (vol0 + 1))
    f["morph_dvol"] = float(np.abs(ratio).mean())
    f["morph_dvol_max"] = float(np.abs(ratio).max())
    f["morph_dvol_signed"] = float(ratio.mean())
    ccj = np.mean([np.mean([_cc(Mj[i] == c)[0] for i in range(N)]) for c in (1, 2, 3)])
    cc0 = np.mean([np.mean([_cc(M0[i] == c)[0] for i in range(N)]) for c in (1, 2, 3)])
    f["morph_cc"] = float(ccj)
    f["morph_dcc"] = float(ccj - cc0)
    f["morph_rv_lv"] = float(volj[2] / max(volj[0], 1))
    if Pc is not None:
        f["dis_cons_dice"] = _mean_dice(Mj, Pc.argmax(1))
        f["dis_cons_l1"] = float(np.abs(Pj - Pc.astype(np.float32)).sum(1).mean())
    return f


def select(names, drop=()):
    return [n for n in names if n.split("_")[0] not in drop]
