"""Additional source vendors (GE = srcC_s0, Canon = srcD_s0, one seed each, design frozen).
Writes results/aggregate/newsrc_rows.tex (method rows) and newsrc_pairs.tex (per vendor pair, CARS vs TENT-3).
Needs results/aggregate/newsrc/per_patient_all.csv from
python scripts/aggregate.py --tags srcC_s0 srcD_s0 --dev-tags --out results/aggregate/newsrc
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AG = os.path.join(ROOT, "results", "aggregate")
per = pd.read_csv(os.path.join(AG, "newsrc", "per_patient_all.csv"))
SRC = "Source (no adaptation)"
OURS = "CARS-TTA (ours)"
NAME = [(SRC, "Source model, no adaptation"),
        ("TENT fixed 1 step", "TENT, 1 step"),
        ("TENT fixed 3 steps", "TENT, 3 steps"),
        ("EATA fixed 3 steps", "EATA, 3 steps, untuned"),
        ("SAR fixed 3 steps", "SAR, 3 steps"),
        ("Consensus ensemble always", "Consensus ensemble"),
        ("Best fixed on source val.", "Best fixed on source val."),
        ("Minimum-entropy selection", "Minimum-entropy selection"),
        ("Entropy gate (src-cal.)", "Entropy gate"),
        ("CARS steps-only (src, 1-3 TENT steps)", "\\cars{}, TENT steps only"),
        ("CARS linear (ridge scorer)", "\\cars{}, linear scorer"),
        (OURS, "\\cars{}"),
        ("[privileged] Oracle, all candidates", "Oracle$^\\dagger$")]
TAGNAME = {"srcC_s0": "GE", "srcD_s0": "Canon"}
TGNAME = {"A": "Sie", "B": "Phi", "C": "GE", "D": "Can", "ACDC": "ACDC"}


def units(m):
    g = (per[m] - per[SRC]) * 100
    d = pd.DataFrame({"tag": per.tag, "target": per.target, "g": g, "ch": per[m] != per[SRC]})
    return d.groupby(["tag", "target"]).agg(
        gain=("g", "mean"), h0=("g", lambda x: (x < 0).mean() * 100), h1=("g", lambda x: (x < -1).mean() * 100),
        h5=("g", lambda x: (x < -5).mean() * 100), n5=("g", lambda x: int((x < -5).sum())), ch=("ch", "mean"))


U = {m: units(m) for m, _ in NAME}
ours = U[OURS].gain
pv = {}
for m, _ in NAME:
    if m in (OURS, SRC) or m.startswith("[priv"):
        continue
    d = ours - U[m].gain
    pv[m] = (int((d > 0).sum()), wilcoxon(d).pvalue)
# Holm over the label-free comparisons
order = sorted(pv, key=lambda k: pv[k][1])
holm, run = {}, 0.0
for i, k in enumerate(order):
    run = max(run, min(1.0, (len(order) - i) * pv[k][1]))
    holm[k] = run

lines = []
for m, lab in NAME:
    u = U[m]
    bytag = u.gain.groupby(level=0).mean()
    cmp = "" if m not in pv else f"{pv[m][0]}/8, {holm[m]:.2f}"
    row = (f"{lab} & {bytag['srcC_s0']:+.2f} & {bytag['srcD_s0']:+.2f} & {u.gain.mean():+.2f} & {u.h0.mean():.1f} & "
           f"{u.h1.mean():.1f} & {u.h5.mean():.1f} & {int(u.n5.sum())} & {100 * u.ch.mean():.0f} & {cmp} \\\\")
    if m == OURS:
        lines.append("\\midrule")
    lines.append(row.replace("+0.00", "0.00").replace("-0.00", "0.00"))
    if m == OURS:
        lines.append("\\midrule")
open(os.path.join(AG, "newsrc_rows.tex"), "w", newline="\n").write("\n".join(lines) + "\n")

pl = []
for tag in ["srcC_s0", "srcD_s0"]:
    for tg in ["A", "B", "C", "D", "ACDC"]:
        if (tag, tg) not in U[OURS].index:
            continue
        o, t, orc = U[OURS].loc[(tag, tg)], U["TENT fixed 3 steps"].loc[(tag, tg)], U["[privileged] Oracle, all candidates"].loc[(tag, tg)]
        pl.append(f"{TAGNAME[tag]} $\\to$ {TGNAME[tg]} & {t.gain:+.2f} / {t.h1:.1f} & {o.gain:+.2f} / {o.h1:.1f} & {orc.gain:+.2f} \\\\")
open(os.path.join(AG, "newsrc_pairs.tex"), "w", newline="\n").write("\n".join(pl) + "\n")
print("\n".join(lines))
print("\n".join(pl))
