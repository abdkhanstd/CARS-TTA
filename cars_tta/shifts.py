"""Synthetic acquisition shifts (counterfactual scans) applied to a [0,1] volume of shape (S, H, W).

All shifts are intensity or resolution changes on a fixed grid, so the label map stays aligned.
Parameter ranges are declared here in advance and are never tuned on target data.
Families: 'intensity' (bias field, gamma, contrast, scaling, noise, rician) and
'frequency' (blur, k-space low-pass, down/up-sampling, low-frequency amplitude perturbation).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

INTENSITY = ["bias", "gamma", "contrast", "scale", "gauss_noise", "rician"]
FREQUENCY = ["blur", "kspace_lowpass", "downsample", "lowfreq_amp"]
ALL = INTENSITY + FREQUENCY


def _sev(rng, severity):
    return rng.uniform(0.5, 1.0) * severity


def bias(v, rng, s):
    S, H, W = v.shape
    g = rng.normal(size=(4, 4))
    yy, xx = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing="ij")
    field = np.zeros((H, W))
    for i in range(3):
        for j in range(3):
            field += g[i, j] * np.cos(np.pi * i * yy + rng.uniform(0, 6.28)) * np.cos(np.pi * j * xx + rng.uniform(0, 6.28))
    field = field / (np.abs(field).max() + 1e-6)
    return v * np.exp(0.6 * s * field)[None]


def gamma(v, rng, s):
    g = np.exp(rng.choice([-1, 1]) * 0.8 * s)
    return np.clip(v, 0, None) ** g


def contrast(v, rng, s):
    m = v.mean()
    c = np.exp(rng.choice([-1, 1]) * 0.7 * s)
    return (v - m) * c + m


def scale(v, rng, s):
    return v * np.exp(rng.uniform(-1, 1) * 0.7 * s) + rng.uniform(-0.1, 0.1) * s


def gauss_noise(v, rng, s):
    return v + rng.normal(0, 0.12 * s, v.shape)


def rician(v, rng, s):
    sd = 0.12 * s
    return np.sqrt((v + rng.normal(0, sd, v.shape)) ** 2 + rng.normal(0, sd, v.shape) ** 2)


def blur(v, rng, s):
    return gaussian_filter(v, sigma=(0, 2.0 * s + 0.3, 2.0 * s + 0.3))


def kspace_lowpass(v, rng, s):
    S, H, W = v.shape
    k = np.fft.fftshift(np.fft.fft2(v), axes=(-2, -1))
    frac = max(0.15, 1.0 - 0.6 * s)  # fraction of k-space kept
    yy, xx = np.meshgrid(np.arange(H) - H / 2, np.arange(W) - W / 2, indexing="ij")
    if rng.uniform() < 0.5:
        mask = (np.abs(yy) <= frac * H / 2) & (np.abs(xx) <= frac * W / 2)   # rectangular truncation (ringing)
    else:
        mask = np.exp(-(yy ** 2 + xx ** 2) / (2 * (frac * H / 3) ** 2))      # smooth apodisation
    return np.abs(np.fft.ifft2(np.fft.ifftshift(k * mask, axes=(-2, -1))))


def downsample(v, rng, s):
    f = 1.0 / (1.0 + 2.0 * s)
    out = np.empty_like(v)
    for i in range(v.shape[0]):
        lo = zoom(v[i], f, order=1)
        out[i] = zoom(lo, (v.shape[1] / lo.shape[0], v.shape[2] / lo.shape[1]), order=1)[: v.shape[1], : v.shape[2]]
    return out


def lowfreq_amp(v, rng, s):
    """Perturb the low-frequency Fourier amplitude (phase kept), a generic model of contrast/protocol change."""
    S, H, W = v.shape
    f = np.fft.fftshift(np.fft.fft2(v), axes=(-2, -1))
    amp, ph = np.abs(f), np.angle(f)
    r = max(2, int(0.06 * H))
    cy, cx = H // 2, W // 2
    noise = 1 + 0.6 * s * rng.uniform(-1, 1, size=(2 * r, 2 * r))
    noise[r, r] = 1.0  # keep DC
    amp[:, cy - r:cy + r, cx - r:cx + r] *= noise[None]
    return np.abs(np.fft.ifft2(np.fft.ifftshift(amp * np.exp(1j * ph), axes=(-2, -1))))


FUNCS = {f.__name__: f for f in [bias, gamma, contrast, scale, gauss_noise, rician, blur, kspace_lowpass, downsample, lowfreq_amp]}


def apply_shift(v: np.ndarray, names, rng: np.random.Generator, severity: float) -> np.ndarray:
    out = v.astype(np.float64)
    for n in names:
        out = FUNCS[n](out, rng, _sev(rng, severity))
    return np.clip(out, 0, None).astype(np.float32)


def sample_shift(rng: np.random.Generator, pool=ALL, max_ops=3):
    k = int(rng.integers(1, max_ops + 1))
    names = list(rng.choice(pool, size=min(k, len(pool)), replace=False))
    sev = float(rng.uniform(0.2, 1.5))
    return names, sev
