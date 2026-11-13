"""Episodic test-time adaptation that records the whole trajectory of candidate predictions.

For one patient: restore source weights, predict (candidate 0), then take K optimiser steps, saving the
prediction after every step (candidates 1..K). Only normalisation affine parameters are updated. The model
uses instance normalisation, so there are no running statistics that could leak between patients. After the
episode the source state is restored and the optimiser is discarded.

Objectives: 'tent' (mean pixel entropy), 'eata' (entropy filter + sample re-weighting + L2 anchor to source,
a Fisher-free EATA variant) and 'sar' (reliable-entropy filter + sharpness-aware update).
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from .unet import norm_affine_params


def entropy_map(logits):
    p = logits.softmax(1)
    return -(p * torch.log(p + 1e-8)).sum(1)  # (B, H, W)


class Episode:
    def __init__(self, model, lr=1e-3, method="tent", bs=32, eata_margin=0.4, eata_l2=1.0, sar_rho=0.05, sar_margin=0.4):
        self.model = model
        self.src_state = copy.deepcopy(model.state_dict())
        self.lr, self.method, self.bs = lr, method, bs
        self.eata_margin, self.eata_l2 = eata_margin * math.log(4), eata_l2
        self.sar_rho, self.sar_margin = sar_rho, sar_margin * math.log(4)

    def reset(self):
        self.model.load_state_dict(self.src_state)

    def _loss_on(self, x, params0=None):
        logits = self.model(x)
        ent = entropy_map(logits)                       # (B, H, W)
        slice_ent = ent.mean((1, 2))                     # (B,)
        if self.method == "tent" or self.method == "sar":
            if self.method == "sar":
                keep = slice_ent < self.sar_margin
                loss = (slice_ent * keep).sum() / keep.sum().clamp(min=1)
            else:
                loss = slice_ent.mean()
        elif self.method == "eata":
            keep = (slice_ent < self.eata_margin).float()
            w = keep * torch.exp(-(slice_ent.detach() - self.eata_margin))
            loss = (w * slice_ent).sum() / max(1.0, float(keep.sum()))
        else:
            raise ValueError(self.method)
        return loss, logits

    def run(self, x: torch.Tensor, K: int, return_bottleneck=False):
        """x: (N, 1, H, W) slices of one patient (all phases). Returns list of K+1 softmax arrays (N, C, H, W, fp16 numpy),
        optional bottleneck descriptor at step 0, and cost stats."""
        self.reset()
        model = self.model
        model.eval()  # instance norm: identical behaviour in train/eval
        params = norm_affine_params(model)
        for p in model.parameters():
            p.requires_grad_(False)
        for p in params:
            p.requires_grad_(True)
        opt = torch.optim.Adam(params, lr=self.lr, betas=(0.9, 0.999))
        params0 = [p.detach().clone() for p in params]
        cands, bott = [], None
        n = x.shape[0]
        torch.cuda.synchronize() if x.is_cuda else None
        t0 = time.time()
        for k in range(K + 1):
            last = k == K
            probs = []
            if not last:
                opt.zero_grad(set_to_none=True)
            for i in range(0, n, self.bs):
                xb = x[i:i + self.bs]
                w = xb.shape[0] / n
                if last:
                    with torch.no_grad():
                        lg = model(xb)
                    probs.append(lg.softmax(1).half().cpu())
                    continue
                if k == 0 and return_bottleneck:
                    with torch.no_grad():
                        _, b = model(xb, return_bottleneck=True)
                    bott = b.mean((2, 3)) if bott is None else torch.cat([bott, b.mean((2, 3))])
                loss, lg = self._loss_on(xb)
                probs.append(lg.detach().softmax(1).half().cpu())
                (loss * w).backward()
            cands.append(torch.cat(probs).numpy())
            if last:
                break
            if self.method == "eata" and self.eata_l2 > 0:
                for p, p0 in zip(params, params0):
                    p.grad.add_(self.eata_l2 * 2 * (p.detach() - p0))
            if self.method == "sar":
                # sharpness-aware: perturb along the gradient, recompute the gradient there, step from the original point
                grads = [p.grad.detach().clone() for p in params]
                gnorm = torch.sqrt(sum((g ** 2).sum() for g in grads)) + 1e-12
                eps = [self.sar_rho * g / gnorm for g in grads]
                with torch.no_grad():
                    for p, e in zip(params, eps):
                        p.add_(e)
                opt.zero_grad(set_to_none=True)
                for i in range(0, n, self.bs):
                    xb = x[i:i + self.bs]
                    loss, _ = self._loss_on(xb)
                    (loss * xb.shape[0] / n).backward()
                with torch.no_grad():
                    for p, e in zip(params, eps):
                        p.sub_(e)
            opt.step()
        torch.cuda.synchronize() if x.is_cuda else None
        stats = {"seconds": time.time() - t0, "steps_executed": K}
        for p in params:
            p.requires_grad_(False)
        self.reset()
        return cands, (bott.cpu().numpy() if bott is not None else None), stats


def assert_restored(model, src_state) -> bool:
    cur = model.state_dict()
    return all(torch.equal(cur[k], src_state[k]) for k in src_state)
