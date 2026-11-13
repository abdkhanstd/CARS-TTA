"""Train the source 2D U-Net on seg_train patients; early stopping on seg_val patient Dice.

python scripts/train_seg.py --cache data/cache/acdc --split splits/dev_acdc.csv --out checkpoints/dev_acdc --seed 0
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import affine_transform

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cars_tta.common import patient_input, seed_all, split_by_phase  # noqa: E402
from cars_tta.data import load_patient, zscore  # noqa: E402
from cars_tta import shifts as SH  # noqa: E402
from cars_tta.metrics import grid_dice  # noqa: E402
from cars_tta.unet import UNet2D  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True, nargs="+")
ap.add_argument("--split", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=120)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--patience", type=int, default=25)
ap.add_argument("--shift-aug", type=float, default=0.0,
                help="probability of applying a synthetic acquisition shift (same sampler as the router episodes) to the training volume")
a = ap.parse_args()
seed_all(a.seed)
os.makedirs(a.out, exist_ok=True)
split = pd.read_csv(a.split)


def find(pid):
    for c in a.cache:
        if os.path.exists(os.path.join(c, f"{pid}.npz")):
            return c
    raise FileNotFoundError(pid)


tr = [load_patient(find(p), p) for p in split.pid[split.role == "seg_train"]]
va = [load_patient(find(p), p) for p in split.pid[split.role == "seg_val"]]
X, Y, SRC = [], [], []
VOL = []  # raw [0,1] volumes for shift augmentation
for pat in tr:
    for ph in pat.phases:
        VOL.append(pat.img[ph])
        xz = zscore(pat.img[ph])
        for i in range(xz.shape[0]):
            if pat.lab[ph][i].any():
                X.append(xz[i]); Y.append(pat.lab[ph][i]); SRC.append((len(VOL) - 1, i))
X, Y = np.stack(X), np.stack(Y)
print(f"train slices {len(X)} from {len(tr)} patients; val patients {len(va)}", flush=True)
rng = np.random.default_rng(a.seed)


def augment(x, y):
    """Standard spatial augmentation plus mild intensity jitter (milder than the synthetic shift ranges)."""
    ang = np.deg2rad(rng.uniform(-15, 15)); sc = rng.uniform(0.85, 1.15)
    c, s = np.cos(ang) / sc, np.sin(ang) / sc
    M = np.array([[c, -s], [s, c]])
    ctr = np.array(x.shape) / 2
    off = ctr - M @ ctr + rng.uniform(-10, 10, 2)
    x = affine_transform(x, M, off, order=1, mode="constant", cval=float(x.min()))
    y = affine_transform(y, M, off, order=0, mode="constant", cval=0)
    if rng.uniform() < 0.3:
        x = x * rng.uniform(0.9, 1.1) + rng.uniform(-0.1, 0.1)
    if rng.uniform() < 0.15:
        x = x + rng.normal(0, 0.05, x.shape)
    return x.astype(np.float32), y


dev = "cuda"
model = UNet2D().to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
iters = len(X) // a.bs
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * iters)
scaler = torch.cuda.amp.GradScaler()


def dice_loss(logits, y):
    p = logits.softmax(1)
    oh = F.one_hot(y, 4).permute(0, 3, 1, 2).float()
    inter = (p * oh).sum((0, 2, 3)); den = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    return 1 - ((2 * inter + 1e-5) / (den + 1e-5))[1:].mean()


@torch.no_grad()
def evaluate(model, pats):
    model.eval(); ds = []
    for pat in pats:
        x, idx = patient_input(pat)
        pr = torch.cat([model(x[i:i + 32].to(dev)).argmax(1).cpu() for i in range(0, len(x), 32)]).numpy()
        ds.append(grid_dice(split_by_phase(pr, idx), pat.lab))
    return float(np.mean(ds))


best, bad, log = -1, 0, []
t0 = time.time()
for ep in range(a.epochs):
    model.train(); perm = rng.permutation(len(X)); tl = 0
    for it in range(iters):
        b = perm[it * a.bs:(it + 1) * a.bs]
        xs = []
        for i in b:
            if a.shift_aug > 0 and rng.uniform() < a.shift_aug:
                v, z = SRC[i]
                names, sev = SH.sample_shift(rng)
                xs.append(zscore(SH.apply_shift(VOL[v], names, rng, sev))[z])
            else:
                xs.append(X[i])
        xb, yb = zip(*[augment(xs[k], Y[i]) for k, i in enumerate(b)])
        xb = torch.from_numpy(np.stack(xb)[:, None]).to(dev); yb = torch.from_numpy(np.stack(yb).astype(np.int64)).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(xb)
        lg = lg.float()
        loss = F.cross_entropy(lg, yb) + dice_loss(lg, yb)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); tl += loss.item()
    vd = evaluate(model, va)
    log.append({"epoch": ep, "loss": tl / iters, "val_dice": vd, "time": time.time() - t0})
    print(json.dumps(log[-1]), flush=True)
    if vd > best:
        best, bad = vd, 0
        torch.save(model.state_dict(), os.path.join(a.out, "best.pt"))
    else:
        bad += 1
        if bad >= a.patience:
            break
json.dump({"best_val_dice": best, "log": log, "args": vars(a)}, open(os.path.join(a.out, "train_log.json"), "w"), indent=1)
print("best source-validation Dice", best)
