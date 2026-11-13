"""Dataset loading and preprocessing for ACDC and M&Ms cardiac MRI.

Every patient is converted once into an .npz cache file with two cardiac phases (ED, ES).
Unified label convention: 0 background, 1 LV cavity, 2 myocardium, 3 RV cavity.
Images are kept as [0,1]-scaled intensities (percentile clipping) on a 1.25 mm in-plane grid,
centre-cropped or padded to 256 x 256. The network input is a per-volume z-score of that image,
computed at run time, so synthetic acquisition shifts can be applied before normalisation.
Original-resolution labels and the geometry needed to map predictions back are stored too.
"""
from __future__ import annotations

import csv
import glob
import json
import os
from dataclasses import dataclass

import nibabel as nib
import numpy as np
from skimage.transform import resize

TARGET_SPACING = 1.25
CROP = 256

# raw label -> unified label
ACDC_MAP = {0: 0, 1: 3, 2: 2, 3: 1}   # ACDC: 1 RV, 2 MYO, 3 LV
MNMS_MAP = {0: 0, 1: 1, 2: 2, 3: 3}   # M&Ms: 1 LV, 2 MYO, 3 RV


def remap(lab: np.ndarray, mapping: dict) -> np.ndarray:
    out = np.zeros_like(lab, dtype=np.uint8)
    for k, v in mapping.items():
        out[lab == k] = v
    return out


def scale01(vol: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(vol, [0.5, 99.5])
    return np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def crop_or_pad(arr: np.ndarray, size: int = CROP):
    """Centre crop/pad the last two axes. Returns array and (pad/crop) record for inversion."""
    h, w = arr.shape[-2:]
    rec = {}
    out = arr
    for ax, n in ((-2, h), (-1, w)):
        if n >= size:
            s = (n - size) // 2
            idx = [slice(None)] * out.ndim
            idx[ax] = slice(s, s + size)
            out = out[tuple(idx)]
            rec[ax] = ("crop", s, n)
        else:
            b = (size - n) // 2
            pad = [(0, 0)] * out.ndim
            pad[ax] = (b, size - n - b)
            out = np.pad(out, pad)
            rec[ax] = ("pad", b, n)
    return out, rec


def uncrop(arr: np.ndarray, rec) -> np.ndarray:
    """Invert crop_or_pad on the last two axes (cropped-away regions become 0)."""
    out = arr
    for ax in (-2, -1):
        kind, s, n = rec[ax]
        if kind == "crop":
            pad = [(0, 0)] * out.ndim
            pad[ax] = (s, n - s - out.shape[ax])
            out = np.pad(out, pad)
        else:
            idx = [slice(None)] * out.ndim
            idx[ax] = slice(s, s + n)
            out = out[tuple(idx)]
    return out


def preprocess_volume(img: np.ndarray, lab: np.ndarray, spacing):
    """img, lab: (X, Y, Z) arrays in nibabel order. Returns slices-first arrays on the network grid."""
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)  # (Z, X, Y)
    lab = np.transpose(lab, (2, 0, 1))
    z, h, w = img.shape
    nh, nw = int(round(h * spacing[0] / TARGET_SPACING)), int(round(w * spacing[1] / TARGET_SPACING))
    img01 = scale01(img)
    img_r = resize(img01, (z, nh, nw), order=1, preserve_range=True, anti_aliasing=False).astype(np.float32)
    lab_r = resize(lab, (z, nh, nw), order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
    img_c, rec = crop_or_pad(img_r)
    lab_c, _ = crop_or_pad(lab_r)
    geom = {"orig_hw": [h, w], "resampled_hw": [nh, nw], "rec": {str(k): list(v) for k, v in rec.items()},
            "spacing": [float(s) for s in spacing]}
    return img_c, lab_c, lab, geom


def _load(path):
    nii = nib.load(path)
    return np.asarray(nii.dataobj), tuple(float(s) for s in nii.header.get_zooms()[:3])


def read_acdc_info(pdir: str) -> dict:
    info = {}
    with open(os.path.join(pdir, "Info.cfg")) as f:
        for line in f:
            if ":" in line:
                k, v = line.split(":", 1)
                info[k.strip()] = v.strip()
    return info


def preprocess_acdc(raw_root: str, out_root: str) -> list[dict]:
    """raw_root contains training/ and testing/ folders of the official ACDC release."""
    os.makedirs(out_root, exist_ok=True)
    rows = []
    for pdir in sorted(glob.glob(os.path.join(raw_root, "*", "patient*"))):
        if not os.path.isdir(pdir):
            continue
        pid = os.path.basename(pdir)
        info = read_acdc_info(pdir)
        phases = {"ED": int(info["ED"]), "ES": int(info["ES"])}
        rec = {"dataset": "acdc", "pid": pid, "vendor": "Siemens", "centre": "Dijon",
               "group": info.get("Group", ""), "split_official": os.path.basename(os.path.dirname(pdir))}
        save = {}
        geoms = {}
        for ph, fr in phases.items():
            ip = os.path.join(pdir, f"{pid}_frame{fr:02d}.nii.gz")
            lp = os.path.join(pdir, f"{pid}_frame{fr:02d}_gt.nii.gz")
            img, sp = _load(ip)
            lab, _ = _load(lp)
            lab = remap(np.rint(lab).astype(np.int64), ACDC_MAP)
            ic, lc, lo, g = preprocess_volume(img, lab, sp)
            save[f"img_{ph}"], save[f"lab_{ph}"], save[f"orig_{ph}"] = ic.astype(np.float16), lc, lo
            geoms[ph] = g
        np.savez_compressed(os.path.join(out_root, f"{pid}.npz"), **save, geom=json.dumps(geoms))
        rec["n_slices"] = int(save["img_ED"].shape[0])
        rec["spacing"] = geoms["ED"]["spacing"]
        rows.append(rec)
    return rows


def preprocess_mnms(raw_root: str, info_csv: str, out_root: str) -> list[dict]:
    """M&Ms 2020 release: <split>/[Labeled/]<code>/<code>_sa.nii.gz and _sa_gt.nii.gz (4D, labels at ED and ES)
    plus the official info CSV listing External code, Vendor, Centre, ED, ES. Unlabelled cases are skipped."""
    os.makedirs(out_root, exist_ok=True)
    meta = {}
    with open(info_csv, newline="") as f:
        for r in csv.DictReader(f):
            code = r.get("External code") or r.get("External Code") or r.get("Code")
            meta[code] = r
    rows = []
    for gt in sorted(glob.glob(os.path.join(raw_root, "**", "*_sa_gt.nii.gz"), recursive=True)):
        code = os.path.basename(gt).split("_")[0]
        if code not in meta:
            continue
        m = meta[code]
        img4, sp = _load(gt.replace("_sa_gt", "_sa"))
        lab4, _ = _load(gt)
        save, geoms = {}, {}
        for ph in ("ED", "ES"):
            t = int(m[ph])
            lab = remap(np.rint(lab4[..., t]).astype(np.int64), MNMS_MAP)
            if lab.max() == 0:
                break
            ic, lc, lo, g = preprocess_volume(img4[..., t], lab, sp)
            save[f"img_{ph}"], save[f"lab_{ph}"], save[f"orig_{ph}"] = ic.astype(np.float16), lc, lo
            geoms[ph] = g
        if len(geoms) < 2:
            continue
        np.savez_compressed(os.path.join(out_root, f"{code}.npz"), **save, geom=json.dumps(geoms))
        rows.append({"dataset": "mnms", "pid": code, "vendor": m["Vendor"], "vendor_name": m.get("VendorName", ""), "centre": m["Centre"],
                     "group": m.get("Pathology", ""), "split_official": gt.split(os.sep)[-3],
                     "n_slices": int(save["img_ED"].shape[0]), "spacing": geoms["ED"]["spacing"]})
    return rows


@dataclass
class Patient:
    pid: str
    img: dict      # phase -> (S, H, W) float32 in [0,1]
    lab: dict      # phase -> (S, H, W) uint8 network grid
    orig: dict     # phase -> (S, h, w) uint8 original grid
    geom: dict

    @property
    def phases(self):
        return list(self.img.keys())


def load_patient(cache_root: str, pid: str) -> Patient:
    z = np.load(os.path.join(cache_root, f"{pid}.npz"))
    geom = json.loads(str(z["geom"]))
    ph = [p for p in ("ED", "ES") if f"img_{p}" in z]
    return Patient(pid, {p: z[f"img_{p}"].astype(np.float32) for p in ph},
                   {p: z[f"lab_{p}"] for p in ph}, {p: z[f"orig_{p}"] for p in ph}, geom)


def zscore(vol: np.ndarray) -> np.ndarray:
    """Per-volume normalisation used as the network input (computed after any synthetic shift)."""
    m, s = float(vol.mean()), float(vol.std())
    return ((vol - m) / max(s, 1e-6)).astype(np.float32)


def to_original(prob: np.ndarray, geom_phase: dict) -> np.ndarray:
    """prob: (S, C, 256, 256) softmax on the network grid -> label map (S, h, w) on the original grid."""
    rec = {int(k): tuple(v) for k, v in geom_phase["rec"].items()}
    p = uncrop(prob, rec)
    s, c = p.shape[:2]
    h, w = geom_phase["orig_hw"]
    p = resize(p, (s, c, h, w), order=1, preserve_range=True, anti_aliasing=False)
    return p.argmax(1).astype(np.uint8)
