"""LaTeX table bodies for the paper from results/aggregate (+ per-fold ablations and costs).

python scripts/make_tables.py --tags srcA_s0 srcA_s1 srcA_s2 srcB_s0 srcB_s1 --dev srcA_s0 > results/aggregate/tables.tex
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
ap = argparse.ArgumentParser()
ap.add_argument("--tags", nargs="+", required=True)
ap.add_argument("--dev", nargs="*", default=["srcA_s0"])
a = ap.parse_args()
CONF = [t for t in a.tags if t not in a.dev]
rng = np.random.default_rng(0)
OURS = "CARS-TTA (ours)"
ORC = "[privileged] Oracle, all candidates"

rows = pd.read_csv(os.path.join(RES, "aggregate", "all_rows.csv"))
rows = rows[rows.method != "__routing__"]
pooled = pd.read_csv(os.path.join(RES, "aggregate", "pooled_units.csv"))
per = pd.read_csv(os.path.join(RES, "aggregate", "per_patient_all.csv"))


def boot_ci(x, n=5000):
    x = np.asarray(x, float)
    b = rng.choice(x, (n, len(x))).mean(1)
    return np.quantile(b, [0.025, 0.975])


def pfmt(p):
    if not np.isfinite(p):
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


# ---------------------------------------------------------------- Table: main pooled comparison (confirmation folds)
NAME = [("Source (no adaptation)", "Source model, no adaptation", ""),
        ("TENT fixed 1 step", "TENT, 1 step \\cite{wang2021tent}", "fixed"),
        ("TENT fixed 3 steps", "TENT, 3 steps \\cite{wang2021tent}", "fixed"),
        ("EATA fixed 3 steps", "EATA, 3 steps \\cite{niu2022eata}", "fixed"),
        ("SAR fixed 3 steps", "SAR, 3 steps \\cite{niu2023sar}", "fixed"),
        ("Histogram matching always", "Histogram matching", "fixed"),
        ("Consensus ensemble always", "Consensus of all candidates", "fixed"),
        ("Best fixed on source val.", "Best candidate on source val.", "fixed"),
        ("Minimum-entropy selection", "Minimum-entropy selection", "select"),
        ("Entropy gate (src-cal.)", "Entropy gate", "select"),
        ("Fragmentation gate (src-cal.)", "Fragmentation gate \\cite{wang2026fragmentation}", "select"),
        ("Fragmentation tertile router (src-cal.)", "Fragmentation router \\cite{wang2026fragmentation}", "select"),
        ("CARS steps-only (src, 1-3 TENT steps)", "\\cars{}, TENT steps only", "ours"),
        ("CARS linear (ridge scorer)", "\\cars{}, linear scorer", "ours"),
        (OURS, "\\textbf{\\cars{} (ours)}", "ours"),
        ("[privileged] Frag. tertile router, target-cal.", "Frag. router, target labels$^\\dagger$", "priv"),
        ("[privileged] Oracle, all candidates", "Oracle over candidates$^\\dagger$", "priv")]


def main_table(tags, subset):
    sub = rows[rows.tag.isin(tags)]
    U = sub.pivot_table(index=["tag", "target"], columns="method", values="dice")
    H = sub.pivot_table(index=["tag", "target"], columns="method", values="hd95")
    G = sub.pivot_table(index=["tag", "target"], columns="method", values="gain")
    R = sub.pivot_table(index=["tag", "target"], columns="method", values="harm_rate")
    C = sub.pivot_table(index=["tag", "target"], columns="method", values="cat_harm_005")
    pp = pooled[pooled.subset == subset].set_index("method")
    out, last = [], None
    for m, lab, grp in NAME:
        if m not in U:
            continue
        if last is not None and grp != last:
            out.append("\\midrule")
        last = grp
        lo, hi = boot_ci(G[m] * 100)
        p = pp.loc[m, "p_holm"] if m in pp.index and m != OURS else np.nan
        better = f"{int(pp.loc[m, 'ours_better_units'])}/{int(pp.loc[m, 'units'])}" if m in pp.index and m != OURS else "--"
        out.append(f"{lab} & {U[m].mean() * 100:.2f} & {H[m].mean():.2f} & {G[m].mean() * 100:+.2f} [{lo:+.2f}, {hi:+.2f}] & "
                   f"{R[m].mean() * 100:.1f} & {C[m].mean() * 100:.1f} & {better} & {pfmt(p)} \\\\")
    return "\n".join(out)


print("% ===== main table, confirmation folds:", CONF)
print(main_table(CONF, "confirmation"))
print("% ===== main table, all folds incl. development")
print(main_table(a.tags, "all"))

# ---------------------------------------------------------------- Table: per source/target (all folds, mean over seeds)
print("% ===== per source/target gain (x100) and harm (%), all folds")
VND = {"A": "Siemens", "B": "Philips", "C": "GE", "D": "Canon", "ACDC": "ACDC"}
for s in "AB":
    sub = rows[(rows.source == s) & rows.tag.isin(a.tags)]
    for t in sorted(sub.target.unique(), key=lambda x: "ABCD".find(x) if x != "ACDC" else 9):
        e = sub[sub.target == t].groupby("method")[["gain", "harm_rate"]].mean()
        cells = []
        for m in ["TENT fixed 3 steps", "Consensus ensemble always", "Minimum-entropy selection", OURS, "[privileged] Oracle, all candidates"]:
            cells.append(f"{e.loc[m, 'gain'] * 100:+.2f}/{e.loc[m, 'harm_rate'] * 100:.0f}" if m != "[privileged] Oracle, all candidates"
                         else f"{e.loc[m, 'gain'] * 100:+.2f}")
        n = int(sub[sub.target == t].n.iloc[0])
        print(f"{VND[s]} $\\to$ {VND[t]} & {n} & " + " & ".join(cells) + " \\\\")

# ---------------------------------------------------------------- Table: ablations (mean over folds that have them)
ABL = [("full", "Full \\cars{}"), ("A1_no_counterfactuals", "No synthetic shifts (clean episodes only)"),
       ("A2_intensity_only", "Intensity shifts only"), ("A3_frequency_only", "Frequency shifts only"),
       ("A13_mild_shifts", "Mild shifts (severity $\\le$0.5)"),
       ("A6_no_uncertainty", "w/o uncertainty features"), ("A8b_no_disagreement", "w/o disagreement features"),
       ("A5_no_fragmentation", "w/o fragmentation features"), ("A7_no_morphology", "w/o morphology features"),
       ("A8_no_ood", "w/o OOD features"), ("A14_scan_features_only", "Scan-level information only"),
       ("A9_no_harm_term", "w/o harm term ($\\lambda_h{=}0$)"), ("A11_binary_consensus_gate", "Actions: consensus gate only"),
       ("A12_steps_only", "Actions: TENT steps only"), ("A12_one_step_only", "Actions: one TENT step only")]
print("% ===== ablations: gain x100 / harm %, mean over folds; (n folds); conf folds only and all")
abl = {}
for t in a.tags:
    for k, _ in ABL:
        p = os.path.join(RES, t, "ablations", k, "summary.json")
        if os.path.exists(p):
            s = json.load(open(p))
            for tt, tv in s["targets"].items():
                r = tv["methods"][OURS]
                abl.setdefault(k, []).append({"tag": t, "target": tt, "gain": r["gain"], "harm": r["harm_rate"],
                                              "syn": None})
for subset, tags in [("conf", CONF), ("all", a.tags)]:
    print(f"% -- {subset}")
    full = pd.DataFrame(abl.get("full", []))
    for k, lab in ABL:
        if k not in abl:
            continue
        d = pd.DataFrame(abl[k])
        d = d[d.tag.isin(tags)]
        if not len(d):
            continue
        m = d.merge(full, on=["tag", "target"], suffixes=("", "_full"))
        delta = (m.gain - m.gain_full).mean() * 100
        print(f"{lab} & {d.gain.mean() * 100:+.2f} & {d.harm.mean() * 100:.1f} & {delta:+.2f} & {d.tag.nunique()} \\\\")

# ---------------------------------------------------------------- costs
print("% ===== costs per patient (seconds), peak GPU MB, router params")
cs = []
for t in a.tags:
    for f in glob.glob(os.path.join(RES, t, "cand_target_*.csv")):
        if "shard" in f:
            continue
        d = pd.read_csv(f, usecols=lambda c: c in ["cand_seconds", "tta_seconds", "feature_seconds", "grad_steps"])
        cs.append(d)
    for f in glob.glob(os.path.join(RES, t, "cand_target_*_meta.json")):
        pass
cs = pd.concat(cs)
peaks = []
for t in a.tags:
    for f in glob.glob(os.path.join(RES, t, "cand_target_*_meta.json")):
        mj = json.load(open(f))
        peaks += [s["peak_gpu_mb"] for s in mj.get("shards", []) if "peak_gpu_mb" in s]
print(cs.describe().round(3).to_string())
print("peak GPU MB median/max:", np.median(peaks) if peaks else None, max(peaks) if peaks else None)
summ = json.load(open(os.path.join(RES, CONF[0], "eval_wide", "summary.json")))["selected"]
print("router params", summ["router_params"], "features", summ["n_shared_features"], summ["n_cand_features"])
for t in a.tags:
    s = json.load(open(os.path.join(RES, t, "eval_wide", "summary.json")))["selected"]
    print(t, "lambda_h", s["lambda_h"], "tau", s["tau"], "best_fixed", s["best_fixed_srcval"], "n_train", s["n_train_episodes"],
          "n_val", s["n_val_episodes"])

# ---------------------------------------------------------------- supplement: per fold x target
print("% ===== SUPP per fold/target: Dice source, TENT-3, consensus, CARS, oracle; harm TENT-3 / CARS; HD95 source / CARS")
V = {"A": "Sie", "B": "Phi", "C": "GE", "D": "Can", "ACDC": "ACDC"}
for t in a.tags:
    sub = rows[rows.tag == t]
    for tt in sorted(sub.target.unique(), key=lambda x: "ABCD".find(x) if x != "ACDC" else 9):
        e = sub[sub.target == tt].set_index("method")
        d = lambda m, k="dice": e.loc[m, k]
        lab = f"{V[t[3]]} s{t.split('_s')[1]}{'*' if t in a.dev else ''}"
        print(f"{lab} & {V[tt]} & {int(e.n.iloc[0])} & {d('Source (no adaptation)')*100:.2f} & {d('TENT fixed 3 steps')*100:.2f} & "
              f"{d('Consensus ensemble always')*100:.2f} & {d(OURS)*100:.2f} & {d(ORC)*100:.2f} & "
              f"{d('TENT fixed 3 steps','harm_rate')*100:.0f} & {d(OURS,'harm_rate')*100:.0f} & "
              f"{d('Source (no adaptation)','hd95'):.1f} & {d(OURS,'hd95'):.1f} \\\\")
print("% ===== SUPP routing quality per target (all folds mean)")
rq = pd.read_csv(os.path.join(RES, "aggregate", "all_rows.csv"))
rq = rq[rq.method == "__routing__"]
for tt, g in rq.groupby("target"):
    print(f"{V[tt]} & {g.spearman_gain_within_patient.mean():.2f} & {g.spearman_gain_pooled.mean():.2f} & {g.auroc_beneficial.mean():.2f} & "
          f"{g.brier_beneficial.mean():.3f} & {g.ece_beneficial.mean():.3f} \\\\")
print("% ===== SUPP per fold settings")
for t in a.tags:
    s = json.load(open(os.path.join(RES, t, "eval_wide", "summary.json")))["selected"]
    lr = open(os.path.join(RES, t, "lr.txt")).read().strip()
    print(f"{V[t[3]]} s{t.split('_s')[1]} & {lr} & {s['lambda_h']:.0f} & {s['tau']:.2f} & {s['best_fixed_srcval']} & {s['n_train_episodes']} & {s['n_val_episodes']} \\\\")
