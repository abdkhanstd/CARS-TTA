"""Routers that choose how many TTA steps (0..A-1) to deploy for one patient, from label-free features.

CARS router: two-layer MLP with an action head and a gain head, trained on source counterfactual episodes with
  L = CE(action logits, oracle action) + lambda_h * sum_k p_k * max(0, -g_k) + lambda_g * MSE(gain head, g)
At deployment the action is argmax_k of the predicted gain (g_0 = 0), with a source-validated safety margin tau:
  if max_k pred_gain_k < tau, deploy action 0.
"""
from __future__ import annotations

import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression


def feature_columns(cols, K, drop_groups=()):
    out = []
    for c in cols:
        if not c.startswith("f_"):
            continue
        name = c[2:]
        if name.split("_")[0] in drop_groups:
            continue
        m = re.search(r"(\d+)$", name)
        if m and name.split("_")[0] in ("unc", "dis", "frag", "morph") and int(m.group(1)) > K:
            continue
        out.append(c)
    return out


def oracle_actions(Q: np.ndarray, eps=0.005):
    """Q: (n, A) qualities. Highest-quality action; prefer fewer steps unless the gain over them exceeds eps."""
    best = np.zeros(len(Q), dtype=int)
    for i, q in enumerate(Q):
        b = 0
        for k in range(1, Q.shape[1]):
            if q[k] > q[b] + eps:
                b = k
        best[i] = b
    return best


class Standardizer:
    def fit(self, X):
        self.mu = np.nanmean(X, 0)
        self.sd = np.nanstd(X, 0) + 1e-6
        return self

    def __call__(self, X):
        Z = (X - self.mu) / self.sd
        return np.clip(np.nan_to_num(Z), -6, 6).astype(np.float32)


class MLP(nn.Module):
    def __init__(self, d, A, h=64, p=0.1):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(p), nn.Linear(h, h), nn.GELU(), nn.Dropout(p))
        self.act = nn.Linear(h, A)
        self.gain = nn.Linear(h, A - 1)

    def forward(self, x):
        z = self.body(x)
        return self.act(z), self.gain(z)


class CARSRouter:
    def __init__(self, A=4, lam_h=1.0, lam_g=1.0, h=64, epochs=300, lr=3e-3, wd=1e-3, n_ens=5, seed=0, gain_scale=20.0):
        self.A, self.lam_h, self.lam_g, self.h = A, lam_h, lam_g, h
        self.epochs, self.lr, self.wd, self.n_ens, self.seed, self.gs = epochs, lr, wd, n_ens, seed, gain_scale
        self.tau = 0.0

    def fit(self, X, Q):
        self.std = Standardizer().fit(X)
        Z = torch.from_numpy(self.std(X))
        G = torch.from_numpy(((Q[:, 1:] - Q[:, :1]) * self.gs).astype(np.float32))
        y = torch.from_numpy(oracle_actions(Q))
        cnt = np.bincount(y.numpy(), minlength=self.A).astype(np.float32)
        w = torch.from_numpy((cnt.sum() / (self.A * np.maximum(cnt, 1))) ** 0.5)
        self.models = []
        for e in range(self.n_ens):
            torch.manual_seed(self.seed * 100 + e)
            m = MLP(Z.shape[1], self.A, self.h)
            opt = torch.optim.AdamW(m.parameters(), lr=self.lr, weight_decay=self.wd)
            for _ in range(self.epochs):
                m.train()
                idx = torch.randperm(len(Z))
                for b in idx.split(256):
                    lg, gp = m(Z[b])
                    p = lg.softmax(1)
                    harm = torch.cat([torch.zeros(len(b), 1), F.relu(-G[b])], 1)
                    loss = F.cross_entropy(lg, y[b], weight=w) + self.lam_h * (p * harm).sum(1).mean() \
                        + self.lam_g * F.smooth_l1_loss(gp, G[b])
                    opt.zero_grad(); loss.backward(); opt.step()
            m.eval()
            self.models.append(m)
        return self

    @torch.no_grad()
    def predict_parts(self, X):
        Z = torch.from_numpy(self.std(X))
        ps, gs = [], []
        for m in self.models:
            lg, gp = m(Z)
            ps.append(lg.softmax(1).numpy()); gs.append(gp.numpy() / self.gs)
        return np.mean(ps, 0), np.mean(gs, 0)

    def decide(self, X, tau=None, rule="gain"):
        tau = self.tau if tau is None else tau
        p, g = self.predict_parts(X)
        if rule == "prob":
            return p.argmax(1)
        best = g.argmax(1)
        return np.where(g.max(1) > tau, best + 1, 0)

    def p_beneficial(self, X):
        """Probability that the deployed adapted action beats the source model (for calibration metrics)."""
        p, _ = self.predict_parts(X)
        return 1 - p[:, 0]

    def n_params(self):
        return sum(p.numel() for p in self.models[0].parameters())


class LogRegRouter:
    def __init__(self, A=4, C=1.0):
        self.A, self.C = A, C

    def fit(self, X, Q):
        self.std = Standardizer().fit(X)
        y = oracle_actions(Q)
        self.clf = LogisticRegression(C=self.C, max_iter=3000, class_weight="balanced")
        self.clf.fit(self.std(X), y)
        return self

    def proba(self, X):
        P = np.zeros((len(X), self.A))
        P[:, self.clf.classes_] = self.clf.predict_proba(self.std(X))
        return P

    def decide(self, X, **_):
        return self.proba(X).argmax(1)

    def p_beneficial(self, X):
        return 1 - self.proba(X)[:, 0]


def choose_tau(router, X, Q, grid=np.linspace(-0.02, 0.03, 51)):
    """Pick the safety margin on SOURCE validation episodes: maximise mean realised gain."""
    best, bt = -1e9, 0.0
    for t in grid:
        a = router.decide(X, tau=t)
        v = (Q[np.arange(len(Q)), a] - Q[:, 0]).mean()
        if v > best + 1e-9:
            best, bt = v, t
    router.tau = float(bt)
    return router.tau


# ---------------------------------------------------------------------------------------------------------------
# Candidate-wise router for a wide action set (CARS-TTA final form).
# score_j = MLP([shared scan features, features of candidate j vs source, one-hot(j)]) estimates the Dice gain of
# candidate j over the source prediction. Training (source counterfactual episodes only):
#   L = SmoothL1(score_j, s*g_j) + lambda_h * sum_j p_j max(0, -g_j) + lambda_ce * CE(p, oracle)
# with p = softmax over [0 (source), score_1..score_J] / T. Deployment: deploy argmax_j score_j if its predicted
# gain exceeds the source-validated margin tau, else keep the source prediction.
# ---------------------------------------------------------------------------------------------------------------
from sklearn.linear_model import Ridge  # noqa: E402


def wide_arrays(df, cands, drop_groups=()):
    """Shared features (n, ds), candidate features (n, J, dc), gains (n, J) or None, feature names."""
    fs = [c for c in df.columns if c.startswith("fs_") and c[3:].split("_")[0] not in drop_groups]
    feat_names = sorted({c.split("__")[2] for c in df.columns if c.startswith("fc__")})
    feat_names = [f for f in feat_names if f.split("_")[0] not in drop_groups]
    S_ = df[fs].to_numpy(float) if fs else np.zeros((len(df), 0))
    C_ = np.stack([df[[f"fc__{c}__{f}" for f in feat_names]].to_numpy(float) for c in cands], 1)
    G = None
    if all(f"q__{c}" in df for c in cands):
        G = np.stack([df[f"q__{c}"].to_numpy() - df["q__src"].to_numpy() for c in cands], 1)
    return S_, C_, G, fs, feat_names


def _oracle_idx(G, eps=0.005):
    """0 = source, j+1 = candidate j. Ties within eps go to the source."""
    best = G.argmax(1)
    return np.where(G[np.arange(len(G)), best] > eps, best + 1, 0)


class WideRouter:
    def __init__(self, cands, lam_h=1.0, lam_ce=0.5, h=64, epochs=200, lr=3e-3, wd=1e-3, n_ens=5, seed=0,
                 gain_scale=20.0, temp=1.0, linear=False):
        self.cands, self.J = list(cands), len(cands)
        self.lam_h, self.lam_ce, self.h, self.epochs, self.lr, self.wd = lam_h, lam_ce, h, epochs, lr, wd
        self.n_ens, self.seed, self.gs, self.T, self.linear = n_ens, seed, gain_scale, temp, linear
        self.tau = 0.0

    def _inputs(self, S_, C_):
        n = len(S_)
        Zs = np.repeat(self.std_s(S_)[:, None, :], self.J, 1) if S_.shape[1] else np.zeros((n, self.J, 0), np.float32)
        Zc = (self.std_c(C_.reshape(-1, C_.shape[-1])).reshape(n, self.J, -1) if C_.shape[-1]
              else np.zeros((n, self.J, 0), np.float32))
        E = np.broadcast_to(np.eye(self.J, dtype=np.float32), (n, self.J, self.J))
        return np.concatenate([Zs, Zc, E], -1).astype(np.float32)

    def fit(self, S_, C_, G):
        self.std_s = Standardizer().fit(S_) if S_.shape[1] else None
        self.std_c = Standardizer().fit(C_.reshape(-1, C_.shape[-1])) if C_.shape[-1] else None
        X = self._inputs(S_, C_)
        if self.linear:
            self.models = [Ridge(alpha=10.0).fit(X.reshape(-1, X.shape[-1]), G.reshape(-1))]
            return self
        Xt = torch.from_numpy(X)
        Gt = torch.from_numpy((G * self.gs).astype(np.float32))
        Gr = torch.from_numpy(G.astype(np.float32))
        y = torch.from_numpy(_oracle_idx(G))
        self.models = []
        for e in range(self.n_ens):
            torch.manual_seed(self.seed * 100 + e)
            m = nn.Sequential(nn.Linear(X.shape[-1], self.h), nn.GELU(), nn.Dropout(0.1),
                              nn.Linear(self.h, self.h), nn.GELU(), nn.Dropout(0.1), nn.Linear(self.h, 1))
            opt = torch.optim.AdamW(m.parameters(), lr=self.lr, weight_decay=self.wd)
            for _ in range(self.epochs):
                m.train()
                for b in torch.randperm(len(Xt)).split(128):
                    s = m(Xt[b]).squeeze(-1)                                   # (B, J)
                    logits = torch.cat([torch.zeros(len(b), 1), s], 1) / self.T
                    p = logits.softmax(1)
                    harm = torch.cat([torch.zeros(len(b), 1), F.relu(-Gr[b])], 1)
                    loss = F.smooth_l1_loss(s, Gt[b]) + self.lam_h * self.gs * (p * harm).sum(1).mean()
                    if self.lam_ce > 0:
                        loss = loss + self.lam_ce * F.cross_entropy(logits, y[b])
                    opt.zero_grad(); loss.backward(); opt.step()
            m.eval()
            self.models.append(m)
        return self

    def predict_gain(self, S_, C_):
        X = self._inputs(S_, C_)
        if self.linear:
            return self.models[0].predict(X.reshape(-1, X.shape[-1])).reshape(len(X), self.J)
        with torch.no_grad():
            Xt = torch.from_numpy(X)
            return np.mean([m(Xt).squeeze(-1).numpy() for m in self.models], 0) / self.gs

    def decide(self, S_, C_, tau=None):
        """Returns index into [src] + cands (0 = source)."""
        tau = self.tau if tau is None else tau
        g = self.predict_gain(S_, C_)
        j = g.argmax(1)
        return np.where(g[np.arange(len(g)), j] > tau, j + 1, 0)

    def p_beneficial(self, S_, C_):
        """Softmax probability mass on non-source actions (temperature T), for calibration analysis."""
        g = self.predict_gain(S_, C_) * self.gs
        logits = np.concatenate([np.zeros((len(g), 1)), g], 1) / self.T
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        return 1 - p[:, 0]

    def n_params(self):
        return 0 if self.linear else sum(p.numel() for p in self.models[0].parameters())


def choose_tau_wide(router, S_, C_, G, grid=np.linspace(-0.01, 0.03, 41), harm_cap=None):
    """Source-validation margin. Without harm_cap: maximise mean realised gain. With harm_cap (fixed in advance):
    maximise mean realised gain among margins whose source-validation harm rate is <= harm_cap; if none qualifies,
    use the largest margin. Ties go to the larger (safer) margin."""
    gfull = np.concatenate([np.zeros((len(G), 1)), G], 1)
    best, bt = -1e9, float(grid[-1])
    for t in grid:
        g = gfull[np.arange(len(G)), router.decide(S_, C_, tau=t)]
        if harm_cap is not None and (g < 0).mean() > harm_cap:
            continue
        v = g.mean()
        if v > best + 1e-9 or (abs(v - best) <= 1e-9 and t > bt):
            best, bt = v, t
    router.tau = float(bt)
    return router.tau
