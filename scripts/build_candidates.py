"""Wide-candidate episodes: for each patient (optionally under synthetic shifts) produce all candidates, their quality
(grid Dice; original-grid Dice/HD95 with --orig-metrics) and the label-free router features.

Columns: q__<cand>, dice__<cand>, hd95__<cand>, fs_<shared feature>, fc__<cand>__<candidate feature>.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cars_tta import shifts as S  # noqa: E402
from cars_tta.candidates import CAND_NAMES, make_candidates  # noqa: E402
from cars_tta.common import seed_all, split_by_phase  # noqa: E402
from cars_tta.data import load_patient, to_original  # noqa: E402
from cars_tta.features import candidate_features, consensus_spread, shared_features  # noqa: E402
from cars_tta.metrics import grid_dice, patient_metrics  # noqa: E402
from cars_tta.tta import Episode  # noqa: E402
from cars_tta.unet import UNet2D  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--split", required=True)
ap.add_argument("--cache", nargs="+", default=["data/cache/mnms", "data/cache/acdc"])
ap.add_argument("--role", required=True)
ap.add_argument("--vendor", default=None)
ap.add_argument("--dataset", default=None)
ap.add_argument("--n-shifts", type=int, default=0)
ap.add_argument("--pool", default="all", choices=["all", "intensity", "frequency"])
ap.add_argument("--sev-max", type=float, default=1.0)
ap.add_argument("--lr-hi", type=float, required=True)
ap.add_argument("--lr-lo", type=float, required=True)
ap.add_argument("--method", default="tent")
ap.add_argument("--orig-metrics", action="store_true")
ap.add_argument("--src-stats", required=True)
ap.add_argument("--save-masks", type=int, default=0)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--shard", default="0/1")
ap.add_argument("--out", required=True)
a = ap.parse_args()
seed_all(a.seed)
dev = "cuda"
split = pd.read_csv(a.split)
sel = split.role == a.role
if a.vendor:
    sel &= split.vendor == a.vendor
if a.dataset:
    sel &= split.dataset == a.dataset
si, sn = map(int, a.shard.split("/"))
pids = split.pid[sel].tolist()[si::sn]


def find(pid):
    for c in a.cache:
        if os.path.exists(os.path.join(c, f"{pid}.npz")):
            return c
    raise FileNotFoundError(pid)


model = UNet2D().to(dev)
model.load_state_dict(torch.load(a.ckpt, map_location=dev, weights_only=True))
model.eval()
ep_hi = Episode(model, lr=a.lr_hi, method=a.method, bs=a.bs)
ep_lo = Episode(model, lr=a.lr_lo, method=a.method, bs=a.bs)
st = dict(np.load(a.src_stats))
template = st["template"]
pool = {"all": S.ALL, "intensity": S.INTENSITY, "frequency": S.FREQUENCY}[a.pool]
rng = np.random.default_rng([a.seed, si])
rows, masks = [], {}
t0 = time.time()
torch.cuda.reset_peak_memory_stats()
for pi, pid in enumerate(pids):
    pat = load_patient(find(pid), pid)
    plan = [([], 0.0)] + [S.sample_shift(rng, pool) for _ in range(a.n_shifts)]
    for ei, (names, sev) in enumerate(plan):
        sev = sev * a.sev_max
        img = {ph: (S.apply_shift(pat.img[ph], names, rng, sev) if names else pat.img[ph]) for ph in pat.phases}
        t1 = time.time()
        cands, idx, bott, flip, cost = make_candidates(model, ep_hi, ep_lo, pat, img, template, dev, bs=a.bs)
        t_cand = time.time() - t1
        imgs = np.concatenate([img[ph] for ph in pat.phases])
        row = {"pid": pid, "episode": ei, "shift": "+".join(names) if names else "none", "severity": sev,
               "n_slices": int(sum(b - a_ for a_, b in idx.values())), "cand_seconds": t_cand, **cost}
        t2 = time.time()
        row.update({f"fs_{k}": v for k, v in shared_features(cands["src"], flip, bott, st, imgs).items()})
        row.update({f"fs_{k}": v for k, v in consensus_spread([cands[c] for c in CAND_NAMES[:-1]], cands["ens"]).items()})
        for c in CAND_NAMES[1:]:
            row.update({f"fc__{c}__{k}": v for k, v in candidate_features(cands["src"], cands[c], cands["ens"]).items()})
        row["feature_seconds"] = time.time() - t2
        for c in CAND_NAMES:
            P = cands[c]
            row[f"q__{c}"] = grid_dice(split_by_phase(P.argmax(1).astype(np.uint8), idx), pat.lab)
            if a.orig_metrics:
                po = {ph: to_original(split_by_phase(P, idx)[ph].astype(np.float32), pat.geom[ph]) for ph in pat.phases}
                m = patient_metrics(po, pat.orig, pat.geom["ED"]["spacing"])
                for kk, vv in m.items():
                    row[f"{kk}__{c}"] = vv
            if a.save_masks and pi < a.save_masks and ei == 0:
                masks[f"{pid}__{c}"] = P.argmax(1).astype(np.uint8)
        if a.save_masks and pi < a.save_masks and ei == 0:
            masks[f"{pid}__img"] = imgs.astype(np.float16)
            masks[f"{pid}__gt"] = np.concatenate([pat.lab[ph] for ph in pat.phases])
        rows.append(row)
    print(f"[{pi + 1}/{len(pids)}] {pid} episodes={len(plan)} elapsed={time.time() - t0:.0f}s", flush=True)
df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
df.to_csv(a.out, index=False)
if masks:
    np.savez_compressed(a.out.replace(".csv", "_masks.npz"), **masks)
json.dump({"args": vars(a), "peak_gpu_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
           "total_seconds": time.time() - t0, "n_episodes": len(df)}, open(a.out.replace(".csv", "_meta.json"), "w"), indent=1)
print("wrote", a.out, len(df))
