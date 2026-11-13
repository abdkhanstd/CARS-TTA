"""Main-paper result figures from the pooled per-patient table (results/aggregate) and per-fold eval outputs.

Outputs (paper/figs): results.pdf (a: gain vs harm, b: per source->target gain, c: synthetic vs real headroom,
d: chosen candidates) and analysis.pdf (e: tau trade-off, f: catastrophic harm, g: reliability, h: per-patient ECDF).
python scripts/figures/fig_results.py --tags srcA_s0 srcA_s1 srcA_s2 srcB_s0 srcB_s1 --dev srcA_s0
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(os.path.dirname(ROOT), "paper", "figs")
RES = os.path.join(ROOT, "results")
ap = argparse.ArgumentParser()
ap.add_argument("--tags", nargs="+", required=True)
ap.add_argument("--dev", nargs="*", default=["srcA_s0"])
a = ap.parse_args()
CONF = [t for t in a.tags if t not in a.dev]

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 7, "mathtext.fontset": "stix", "axes.linewidth": 0.5,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42})
C1, C2, C3, C4, C5, C6, C7, C8 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"
GREY, INK = "#9a9a96", "#3a3a38"
OURS, ORACLE = "CARS-TTA (ours)", "[privileged] Oracle, all candidates"
SH = {"A": "Sie", "B": "Phi", "C": "GE", "D": "Can", "ACDC": "ACDC"}
VND = {"A": "Siemens", "B": "Philips", "C": "GE", "D": "Canon", "ACDC": "ACDC"}

per = pd.read_csv(os.path.join(RES, "aggregate", "per_patient_all.csv"))
per["source"] = per.tag.str[3]
per = per[per.tag.isin(a.tags)]
pc = per[per.tag.isin(CONF)]
gain = lambda df, m: df[m] - df["Source (no adaptation)"]


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight", pad_inches=0.01)
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", pad_inches=0.01, dpi=300)
    plt.close(fig)


def unit_means(df, m):
    """mean gain and harm over (fold, target) units, then averaged (matches pooled_units.csv)."""
    g = gain(df, m)
    u = pd.DataFrame({"g": g, "h": (g < -0.01).astype(float), "tag": df.tag, "target": df.target}).groupby(["tag", "target"]).mean()
    return u.g.mean(), u.h.mean() * 100  # harm beyond 1 Dice point, in percent


# ---------------------------------------------------------------- results.pdf
fig, ax = plt.subplots(1, 4, figsize=(7.16, 1.72), gridspec_kw={"width_ratios": [1.15, 1.05, 0.7, 0.95], "wspace": 0.5})
# (a) gain vs harm over confirmation units. Methods that coincide share one label with a leader line.
pts = {m: unit_means(pc, m) for m in ["TENT fixed 1 step", "TENT fixed 3 steps", "EATA fixed 3 steps", "SAR fixed 3 steps",
                                      "Consensus ensemble always", "Histogram matching always", "Minimum-entropy selection",
                                      "Entropy gate (src-cal.)", "Fragmentation gate (src-cal.)", "Best fixed on source val.",
                                      "TENT (low lr) 3 steps", "Fragmentation tertile router (src-cal.)",
                                      "[privileged] Frag. tertile router, target-cal.", OURS]}
for m, (g, h) in pts.items():
    if m == OURS:
        ax[0].scatter(h, g * 100, s=28, c=C1, zorder=4, edgecolors="white", linewidths=0.6)
    else:
        priv = "privileged" in m
        ax[0].scatter(h, g * 100, s=11, c="white" if priv else GREY, edgecolors=GREY, linewidths=0.6, zorder=3)
lab = [(OURS, "CARS-TTA (ours)", (10.0, 1.62)),
       ("Minimum-entropy selection", "Min-entropy", (12.0, 1.27)),
       ("TENT fixed 3 steps", "TENT-3, SAR-3,\nfrag. gate/router$^\\dagger$", (12.0, 0.88)),
       ("TENT fixed 1 step", "TENT-1", (12.0, 0.45)),
       ("Consensus ensemble always", "Consensus, entropy gate", (8.0, 0.12)),
       ("Best fixed on source val.", "Best on src. val.", (17.0, -0.05)),
       ("Histogram matching always", "Hist. match", (26.0, -0.88)),
       ("EATA fixed 3 steps", "EATA-3", (8.0, -0.75))]
for m, t, xy in lab:
    g, h = pts[m]
    ax[0].annotate(t, (h, g * 100), xytext=xy, textcoords="data", fontsize=5.8, va="center",
                   color=C1 if m == OURS else INK, fontweight="bold" if m == OURS else "normal",
                   arrowprops=dict(arrowstyle="-", color="#c9c8c3", lw=0.4, shrinkA=0, shrinkB=2))
ax[0].axhline(0, color=GREY, lw=0.4, ls=":")
ax[0].set_xlabel("Patients with Dice drop > 1 point (%)")
ax[0].set_ylabel("Mean Dice gain (points)")
ax[0].set_xlim(-1, 40)
ax[0].set_ylim(-0.95, 1.75)
ax[0].set_title("(a) Gain vs. harm > 1 point", fontsize=7)

# (b) per source->target gain, confirmation folds
units = []
for s in "BA":
    for t in sorted(pc[pc.source == s].target.unique(), key=lambda x: "ABCDE".find(x) if x != "ACDC" else 9):
        d = pc[(pc.source == s) & (pc.target == t)]
        units.append((f"{SH[s]}$\\to${SH[t]}", *[gain(d, m).mean() * 100 for m in ["TENT fixed 3 steps", "Consensus ensemble always", OURS, ORACLE]]))
y = np.arange(len(units))[::-1]
for k, (col, lab, mk) in enumerate([(C2, "TENT-3", "s"), (C3, "Consensus", "^"), (C1, "CARS-TTA", "o")]):
    ax[1].scatter([u[1 + k] for u in units], y, s=12, marker=mk, c=col, label=lab, zorder=3, edgecolors="white", linewidths=0.4)
ax[1].scatter([u[4] for u in units], y, s=12, marker="D", facecolors="white", edgecolors=GREY, linewidths=0.6, label="Oracle$^\\dagger$", zorder=2)
for yy, u in zip(y, units):
    ax[1].plot([min(u[1:]), max(u[1:])], [yy, yy], color="#e4e3df", lw=0.6, zorder=1)
ax[1].set_yticks(y)
ax[1].set_yticklabels([u[0] for u in units], fontsize=6)
ax[1].axvline(0, color=GREY, lw=0.4, ls=":")
ax[1].set_xlabel("Mean Dice gain (points)")
ax[1].legend(fontsize=5.5, loc="lower right", handletextpad=0.1, borderaxespad=0.1, labelspacing=0.2)
ax[1].set_title("(b) Gain per vendor pair", fontsize=7)

# (c) fraction of oracle headroom captured: synthetic held-out episodes vs real targets
cap = {"Synthetic": [], "Real": []}
for t in CONF:
    p = os.path.join(RES, t, "eval_synthval", "summary.json")
    if os.path.exists(p):
        mm = json.load(open(p))["targets"]["synthval"]["methods"]
        o = [v for k, v in mm.items() if k.startswith("[privileged] Oracle, all")]
        if o and OURS in mm:
            cap["Synthetic"].append(mm[OURS]["gain"] / o[0]["gain"])
    d = per[per.tag == t]
    for tt in d.target.unique():
        e = d[d.target == tt]
        cap["Real"].append(gain(e, OURS).mean() / gain(e, ORACLE).mean())
for i, (k, v) in enumerate(cap.items()):
    if not v:
        continue
    ax[2].bar(i, np.mean(v) * 100, width=0.55, color=[C1, C4][i], zorder=2)
    ax[2].scatter(np.full(len(v), i) + np.random.default_rng(0).uniform(-0.15, 0.15, len(v)), np.array(v) * 100, s=5,
                  c=INK, zorder=3, linewidths=0)
    ax[2].text(i + 0.33, np.mean(v) * 100, f"{np.mean(v) * 100:.0f}%", ha="left", va="center", fontsize=6.5, color=INK)
ax[2].set_xticks(range(len(cap)))
ax[2].set_xticklabels(list(cap), fontsize=6.5)
ax[2].set_ylabel("Oracle headroom captured (%)")
ax[2].axhline(0, color=GREY, lw=0.4)
ax[2].set_xlim(-0.5, 1.85)
ax[2].set_title("(c) Synthetic-to-real gap", fontsize=7)

# (d) candidates chosen by CARS-TTA per source vendor (all folds, all targets)
grp = {"No adaptation": ["src"], "TENT steps": ["hi1", "hi2", "hi3", "lo1", "lo2", "lo3"],
       "Input correction": ["g07", "g14", "hm", "dn"], "Consensus": ["ens"]}
rows = []
for t in a.tags:
    for f in os.listdir(os.path.join(RES, t, "eval_wide")):
        if f.startswith("per_patient_"):
            d = pd.read_csv(os.path.join(RES, t, "eval_wide", f))
            rows.append(pd.DataFrame({"src": t[3], "c": d[OURS + " [choice]"]}))
ch = pd.concat(rows)
left = np.zeros(2)
for k, (gname, members) in enumerate(grp.items()):
    fr = np.array([ch[ch.src == s].c.isin(members).mean() for s in "AB"])
    if fr.sum() == 0:
        continue
    ax[3].barh([1, 0], fr, left=left, color=[GREY, C2, C4, C3][k], height=0.55, label=gname, edgecolor="white", linewidth=0.8)
    for yy, (l, f_) in enumerate(zip(left[::-1], fr[::-1])):
        if f_ > 0.08:
            ax[3].text(l + f_ / 2, yy, f"{f_ * 100:.0f}", ha="center", va="center", fontsize=6, color="white" if k != 2 else INK)
    left += fr
ax[3].set_yticks([1, 0])
ax[3].set_yticklabels(["Siemens\nsource", "Philips\nsource"], fontsize=6.5)
ax[3].set_xlim(0, 1)
ax[3].legend(fontsize=5.5, ncol=3, loc="upper center", bbox_to_anchor=(0.45, 1.0), handlelength=0.8, columnspacing=0.6,
             handletextpad=0.3)
ax[3].set_xlabel("Fraction of test patients", labelpad=1)
ax[3].set_ylim(-0.45, 1.85)
ax[3].set_title("(d) Candidates chosen by CARS-TTA", fontsize=7)
save(fig, "results")

# ---------------------------------------------------------------- analysis.pdf
fig, ax = plt.subplots(1, 4, figsize=(7.16, 1.6), gridspec_kw={"wspace": 0.45})
# (e) tau trade-off on real targets (mean over confirmation units) with the source-selected tau marked
for s, col in [("A", C1), ("B", C2)]:
    curves, chosen = [], []
    for t in [x for x in CONF if x[3] == s]:
        summ = json.load(open(os.path.join(RES, t, "eval_wide", "summary.json")))
        chosen.append(summ["selected"]["tau"])
        for tt, tv in summ["targets"].items():
            c = pd.read_csv(os.path.join(RES, t, "eval_wide", f"tau_curve_{tt}.csv"))
            c["gain"] = c.dice - tv["methods"]["Source (no adaptation)"]["dice"]
            curves.append(c.set_index("tau")[["gain", "harm"]])
    if not curves:
        continue
    m = pd.concat(curves).groupby(level=0).mean()
    ax[0].plot(m.harm * 100, m.gain * 100, color=col, lw=1.2, label=f"{VND[s]} source")
    tc = m.index[np.argmin(np.abs(m.index - np.mean(chosen)))]
    ax[0].scatter(m.loc[tc, "harm"] * 100, m.loc[tc, "gain"] * 100, s=16, c=col, edgecolors="white", zorder=4)
ax[0].set_xlabel("Patients with any Dice drop (%)")
ax[0].set_ylabel("Dice gain (points)")
ax[0].legend(fontsize=5.5, loc="lower right")
ax[0].set_title(r"(a) Sweep of $\tau$ (dot: source-chosen)", fontsize=7)

# (f) catastrophic harm (Dice drop beyond threshold), confirmation folds
ms = [("TENT fixed 3 steps", "TENT-3", C2), ("Minimum-entropy selection", "Min-entropy", C4),
      ("Consensus ensemble always", "Consensus", C3), (OURS, "CARS-TTA", C1)]
thr = [0.0, 0.01, 0.05]
for k, (m, lab, col) in enumerate(ms):
    g = gain(pc, m)
    unit = pd.DataFrame({"g": g, "tag": pc.tag, "target": pc.target}).groupby(["tag", "target"]).g
    v = [unit.apply(lambda x: (x < -t).mean()).mean() * 100 for t in thr]
    ax[1].bar(np.arange(3) + (k - 1.5) * 0.2, v, width=0.18, color=col, label=lab)
ax[1].set_xticks(range(3))
ax[1].set_xticklabels(["> 0", "> 1", "> 5"])
ax[1].set_yscale("symlog", linthresh=2)
ax[1].set_yticks([0, 1, 2, 5, 10, 20])
ax[1].set_yticklabels(["0", "1", "2", "5", "10", "20"])
ax[1].set_ylim(0, 90)
ax[1].set_xlabel("Dice drop vs. source (points)")
ax[1].set_ylabel("Patients (%)")
ax[1].legend(fontsize=5.3, handlelength=0.8, loc="upper right", ncol=2, columnspacing=0.6, handletextpad=0.3)
ax[1].set_title("(b) Harm at three tolerances", fontsize=7)

# (g) predicted vs realised gain over all (patient, candidate) pairs: held-out synthetic episodes vs real targets
from scipy.stats import spearmanr  # noqa: E402


def pred_true(paths):
    P, T = [], []
    for f in paths:
        d = pd.read_csv(f)
        for c in [x[7:] for x in d.columns if x.startswith("gpred__")]:
            P.append(d[f"gpred__{c}"].to_numpy()); T.append(d[f"gtrue__{c}"].to_numpy())
    return np.concatenate(P) if P else np.array([]), np.concatenate(T) if T else np.array([])


for lab, col, paths in [("Synthetic held-out", C4, [os.path.join(RES, t, "eval_synthval", "router_outputs_synthval.csv") for t in CONF]),
                        ("Real targets", C1, [os.path.join(RES, t, "eval_wide", f) for t in CONF
                                              for f in os.listdir(os.path.join(RES, t, "eval_wide")) if f.startswith("router_outputs_")])]:
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        continue
    P, T = pred_true(paths)
    q = np.quantile(P, np.linspace(0, 1, 11))
    b = np.clip(np.digitize(P, q[1:-1]), 0, 9)
    ax[2].plot([P[b == i].mean() * 100 for i in range(10)], [T[b == i].mean() * 100 for i in range(10)], color=col, lw=1.1,
               marker="o", ms=2.5, label=f"{lab} ($\\rho$={spearmanr(P, T)[0]:.2f})")
lim = ax[2].get_xlim()
ax[2].plot(lim, lim, color=GREY, lw=0.5, ls=":")
ax[2].axhline(0, color=GREY, lw=0.4)
ax[2].set_xlabel("Predicted gain (points, decile mean)")
ax[2].set_ylabel("Realised gain (points)")
ax[2].legend(fontsize=5.3, loc="upper left", handlelength=1.0)
ax[2].set_title("(c) Router gain prediction", fontsize=7)

# (h) per-patient gain ECDF (confirmation folds)
for m, lab, col in [("TENT fixed 3 steps", "TENT-3", C2), ("Consensus ensemble always", "Consensus", C3), (OURS, "CARS-TTA", C1)]:
    g = np.sort(gain(pc, m).to_numpy() * 100)
    ax[3].plot(g, np.arange(1, len(g) + 1) / len(g), color=col, lw=1.1, label=lab)
ax[3].axvline(0, color=GREY, lw=0.4, ls=":")
ax[3].set_xlim(-6, 10)
ax[3].set_xlabel("Per-patient Dice gain (points)")
ax[3].set_ylabel("Cumulative fraction")
ax[3].legend(fontsize=5.5, loc="lower right")
ax[3].set_title("(d) Distribution of gains", fontsize=7)
save(fig, "analysis")
print("ok", len(pc), "confirmation patient-target rows")
