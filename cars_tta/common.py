"""Shared helpers: seeding, splits, building network inputs for a patient."""
from __future__ import annotations

import json
import os
import random

import numpy as np
import pandas as pd
import torch

from .data import load_patient, zscore

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def make_source_split(pids, seed=0, fracs=(0.60, 0.15, 0.15, 0.10), strata=None):
    """Patient-level split into seg_train / seg_val / router_train / router_val. Stratified by `strata` if given."""
    rng = np.random.default_rng(seed)
    pids = list(pids)
    groups = {}
    for i, p in enumerate(pids):
        groups.setdefault(strata[i] if strata is not None else 0, []).append(p)
    out = {}
    names = ["seg_train", "seg_val", "router_train", "router_val"]
    for g in sorted(groups):
        ps = list(groups[g])
        rng.shuffle(ps)
        n = len(ps)
        cuts = np.cumsum([int(round(f * n)) for f in fracs])[:-1]
        for name, chunk in zip(names, np.split(np.array(ps), cuts)):
            for p in chunk:
                out[p] = name
    return out


def patient_input(pat, phases=None, img_override=None):
    """Stack all phases' slices into one (N,1,H,W) tensor (per-phase z-score). Returns tensor and slice index per phase."""
    phases = phases or pat.phases
    xs, idx, s = [], {}, 0
    for ph in phases:
        v = img_override[ph] if img_override is not None else pat.img[ph]
        xs.append(zscore(v))
        idx[ph] = (s, s + v.shape[0])
        s += v.shape[0]
    return torch.from_numpy(np.concatenate(xs)[:, None]), idx


def split_by_phase(arr, idx):
    return {ph: arr[a:b] for ph, (a, b) in idx.items()}


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def read_split(path):
    return pd.read_csv(path)
