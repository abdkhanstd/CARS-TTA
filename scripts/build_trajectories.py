"""Run K-step episodic TTA for a set of patients and store, per episode, the quality of every candidate and the
label-free router features.

Source mode (router data): patients of a source role, each with `--n-shifts` synthetic acquisition shifts
(plus the clean scan). Labels are used only to score candidates (q_k) for the router's training targets.
Target mode: real patients of a target role, no shift. Labels are used only for the evaluation columns,
which the router never reads.

python scripts/build_trajectories.py --ckpt checkpoints/dev_acdc_s0/best.pt --split splits/dev_acdc.csv \
    --role router_train --n-shifts 40 --out results/dev/traj_router_train.parquet
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
from cars_tta.common import patient_input, seed_all, split_by_phase  # noqa: E402
from cars_tta.data import load_patient, to_original  # noqa: E402
from cars_tta.features import trajectory_features  # noqa: E402
from cars_tta.metrics import grid_dice, patient_metrics  # noqa: E402
from cars_tta.tta import Episode  # noqa: E402
from cars_tta.unet import UNet2D  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--split", required=True)
ap.add_argument("--cache", nargs="+", default=["data/cache/acdc", "data/cache/mnms"])
ap.add_argument("--role", required=True)
ap.add_argument("--vendor", default=None, help="restrict to one vendor (target mode)")
ap.add_argument("--dataset", default=None)
ap.add_argument("--n-shifts", type=int, default=0)
ap.add_argument("--pool", default="all", choices=["all", "intensity", "frequency"])
ap.add_argument("--sev-max", type=float, default=1.0)
ap.add_argument("--K", type=int, default=5)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--method", default="tent")
ap.add_argument("--orig-metrics", action="store_true", help="also compute Dice/HD95 on the original grid")
ap.add_argument("--src-stats", default=None, help="npz with bottleneck mu/sd from seg_train patients")
ap.add_argument("--save-masks", type=int, default=0, help="save candidate masks for the first N episodes")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--no-features", action="store_true")
ap.add_argument("--shard", default="0/1", help="i/n: process every n-th patient starting at i")
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
pids = split.pid[sel].tolist()
_si, _sn = map(int, a.shard.split("/"))
pids = pids[_si::_sn]


def find(pid):
    for c in a.cache:
        if os.path.exists(os.path.join(c, f"{pid}.npz")):
            return c
    raise FileNotFoundError(pid)


model = UNet2D().to(dev)
model.load_state_dict(torch.load(a.ckpt, map_location=dev, weights_only=True))
model.eval()
ep = Episode(model, lr=a.lr, method=a.method, bs=a.bs)
src_stats = dict(np.load(a.src_stats)) if a.src_stats else None
pool = {"all": S.ALL, "intensity": S.INTENSITY, "frequency": S.FREQUENCY}[a.pool]
rng = np.random.default_rng([a.seed, _si])
rows, masks = [], {}
t_start = time.time()
torch.cuda.reset_peak_memory_stats()
for pi, pid in enumerate(pids):
    pat = load_patient(find(pid), pid)
    plan = [([], 0.0)] + [S.sample_shift(rng, pool) for _ in range(a.n_shifts)] if a.n_shifts else [([], 0.0)]
    for si, (names, sev) in enumerate(plan):
        sev = sev * a.sev_max
        img = {ph: (S.apply_shift(pat.img[ph], names, rng, sev) if names else pat.img[ph]) for ph in pat.phases}
        x, idx = patient_input(pat, img_override=img)
        x = x.to(dev)
        cands, bott, st = ep.run(x, a.K, return_bottleneck=True)
        with torch.no_grad():
            fl = torch.cat([model(x[i:i + 32].flip(-1)).softmax(1).flip(-1).half().cpu() for i in range(0, len(x), 32)]).numpy()
        imgs = np.concatenate([img[ph] for ph in pat.phases])
        feats = {} if a.no_features else trajectory_features(cands, flip_prob0=fl, bott=bott, src_stats=src_stats, img=imgs, K=a.K)
        row = {"pid": pid, "episode": si, "shift": "+".join(names) if names else "none", "severity": sev,
               "n_slices": int(x.shape[0]), "tta_seconds": st["seconds"]}
        for k, c in enumerate(cands):
            pr = split_by_phase(c.argmax(1).astype(np.uint8), idx)
            row[f"q{k}"] = grid_dice(pr, pat.lab)
            if a.orig_metrics:
                po = {ph: to_original(split_by_phase(c, idx)[ph].astype(np.float32), pat.geom[ph]) for ph in pat.phases}
                m = patient_metrics(po, pat.orig, pat.geom["ED"]["spacing"])
                for kk, vv in m.items():
                    row[f"{kk}_{k}"] = vv
            if a.save_masks and len(masks) < a.save_masks * (a.K + 1):
                masks[f"{pid}_{si}_k{k}"] = c.argmax(1).astype(np.uint8)
        if a.save_masks and f"{pid}_{si}_img" not in masks and len(masks) < a.save_masks * (a.K + 3):
            masks[f"{pid}_{si}_img"] = imgs.astype(np.float16)
            masks[f"{pid}_{si}_gt"] = np.concatenate([pat.lab[ph] for ph in pat.phases])
        row.update({f"f_{k}": v for k, v in feats.items()})
        rows.append(row)
    print(f"[{pi + 1}/{len(pids)}] {pid} episodes={len(plan)} elapsed={time.time() - t_start:.0f}s "
          f"q0={np.mean([r['q0'] for r in rows[-len(plan):]]):.3f} q3={np.mean([r['q3'] for r in rows[-len(plan):]]):.3f}", flush=True)
df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
df.to_csv(a.out, index=False)
if masks:
    np.savez_compressed(a.out.replace(".csv", "_masks.npz"), **masks)
json.dump({"args": vars(a), "peak_gpu_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
           "total_seconds": time.time() - t_start, "n_episodes": len(df)},
          open(a.out.replace(".csv", "_meta.json"), "w"), indent=1)
print("wrote", a.out, len(df))
