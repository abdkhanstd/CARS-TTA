"""Train routers on source counterfactual episodes, select every hyper-parameter on source validation episodes,
then apply all methods, frozen, to real target patients.

python scripts/route_eval.py --train results/X/traj_router_train.csv --val results/X/traj_router_val.csv \
   --target vendorB=results/X/traj_target_B.csv --extra eata=results/X/traj_target_B_eata.csv --out results/X/eval
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cars_tta.router import CARSRouter, LogRegRouter, choose_tau, feature_columns, oracle_actions  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--train", required=True)
ap.add_argument("--val", required=True)
ap.add_argument("--target", nargs="+", required=True, help="name=path")
ap.add_argument("--extra", nargs="*", default=[], help="objective=path pattern with {name}, for EATA/SAR fixed baselines")
ap.add_argument("--K", type=int, default=3)
ap.add_argument("--drop", nargs="*", default=[])
ap.add_argument("--lam-h", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
ap.add_argument("--rule", default="gain", choices=["gain", "prob"])
ap.add_argument("--binary", action="store_true", help="A11: actions {0, K} only")
ap.add_argument("--train-filter", default=None, help="A1-A3: 'clean' keeps only unshifted episodes")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n-boot", type=int, default=2000)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
rng = np.random.default_rng(a.seed)
tr, va = pd.read_csv(a.train), pd.read_csv(a.val)
if a.train_filter == "clean":
    tr = tr[tr["shift"] == "none"]
A = a.K + 1
acts = [0, a.K] if a.binary else list(range(A))
fcols = feature_columns(tr.columns, a.K, a.drop)


def Qmat(df, prefix="q"):
    return df[[f"{prefix}{k}" for k in acts]].to_numpy()


Xtr, Qtr, Xva, Qva = tr[fcols].to_numpy(float), Qmat(tr), va[fcols].to_numpy(float), Qmat(va)


def val_gain(router):
    d = router.decide(Xva)
    return float((Qva[np.arange(len(Qva)), d] - Qva[:, 0]).mean())


# ---- CARS: choose lambda_h on source validation, then tau on source validation
sel = []
for lh in a.lam_h:
    r = CARSRouter(A=len(acts), lam_h=lh, seed=a.seed).fit(Xtr, Qtr)
    if a.rule == "gain":
        choose_tau(r, Xva, Qva)
    sel.append((val_gain(r), lh, r))
sel.sort(key=lambda t: -t[0])
cars = sel[0][2]
cars_ce = [s for s in sel if s[1] == 0.0]
cars_ce = cars_ce[0][2] if cars_ce else None
# ---- logistic regression router: choose C on source validation
lr_sel = []
for C in [0.01, 0.1, 1.0, 10.0]:
    r = LogRegRouter(A=len(acts), C=C).fit(Xtr, Qtr)
    lr_sel.append((val_gain(r), C, r))
lr_sel.sort(key=lambda t: -t[0])
logreg = lr_sel[0][2]

# ---- source-calibrated simple baselines (all from source validation episodes)
Qva_full = va[[f"q{k}" for k in range(6) if f"q{k}" in va]].to_numpy()
k_fixed = int(np.argmax(Qva_full.mean(0)))
ent_val = va["f_unc_ent0"].to_numpy()


def best_threshold(score, gains, higher_adapts):
    best, bt = -1e9, None
    for t in np.quantile(score, np.linspace(0, 1, 101)):
        m = score > t if higher_adapts else score < t
        v = np.where(m, gains, 0).mean()
        if v > best:
            best, bt = v, t
    return bt


g3_va = va[f"q{a.K}"].to_numpy() - va["q0"].to_numpy()
t_ent = best_threshold(ent_val, g3_va, higher_adapts=True)
mu_n, sd_n = va["f_frag_n1"].mean(), va["f_frag_n1"].std() + 1e-9
mu_d, sd_d = va["f_dis_delta1"].mean(), va["f_dis_delta1"].std() + 1e-9


def frag_risk(df):
    return 0.5 * (df["f_frag_n1"] - mu_n) / sd_n + 0.5 * (df["f_dis_delta1"] - mu_d) / sd_d


r_va = frag_risk(va).to_numpy()
cut_lo, cut_hi = np.quantile(r_va, [1 / 3, 2 / 3])
t_frag = best_threshold(r_va, g3_va, higher_adapts=False)
summary = {"selected": {"lambda_h": sel[0][1], "tau": cars.tau, "val_gain_per_lambda": {str(s[1]): s[0] for s in sel},
                        "logreg_C": lr_sel[0][1], "k_fixed_srcval": k_fixed, "t_entropy": float(t_ent),
                        "frag_cuts": [float(cut_lo), float(cut_hi)], "t_frag": float(t_frag),
                        "n_features": len(fcols), "router_params": cars.n_params(),
                        "n_train_episodes": len(tr), "n_val_episodes": len(va),
                        "oracle_action_hist_train": np.bincount(oracle_actions(Qtr), minlength=len(acts)).tolist()},
           "targets": {}}


def frag_tertile_actions(r, lo, hi, a_low=3, a_mid=2, a_hard=0):
    return np.where(r <= lo, a_low, np.where(r <= hi, a_mid, a_hard))


def boot_ci(x, n=a.n_boot):
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def holm(pvals):
    names = list(pvals)
    order = np.argsort([pvals[n] for n in names])
    m, out, prev = len(names), {}, 0
    for r, i in enumerate(order):
        adj = min(1.0, max(prev, (m - r) * pvals[names[i]]))
        out[names[i]] = adj
        prev = adj
    return out


def ece(p, y, bins=10):
    e = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (p >= lo) & (p < lo + 1 / bins)
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


for spec in a.target:
    name, path = spec.split("=")
    te = pd.read_csv(path)
    pre = "dice_" if "dice_0" in te else "q"
    D = te[[f"{pre}{k}" for k in range(6) if f"{pre}{k}" in te]].to_numpy()
    H = te[[f"hd95_{k}" for k in range(6)]].to_numpy() if "hd95_0" in te else None
    X = te[fcols].to_numpy(float)
    n = len(te)
    methods = {"Source (no TTA)": np.zeros(n, int)}
    for k in (1, 3, 5):
        if k < D.shape[1]:
            methods[f"TENT fixed-{k}"] = np.full(n, k)
    methods[f"TENT fixed-src-val (k={k_fixed})"] = np.full(n, k_fixed)
    methods["Entropy gate (src-cal.)"] = np.where(te["f_unc_ent0"].to_numpy() > t_ent, a.K, 0)
    r_te = frag_risk(te).to_numpy()
    methods["Fragmentation gate (src-cal.)"] = np.where(r_te < t_frag, a.K, 0)
    methods["Fragmentation tertile router (src-cal.)"] = frag_tertile_actions(r_te, cut_lo, cut_hi)
    acts_arr = np.array(acts)
    methods["LogReg router (src-trained)"] = acts_arr[logreg.decide(X)]
    if cars_ce is not None:
        methods["CARS, CE only (lambda_h=0)"] = acts_arr[cars_ce.decide(X)]
    methods["CARS-TTA (ours)"] = acts_arr[cars.decide(X)]
    # privileged references (target labels): 5-fold cross-fitted tertile router, and the per-case oracle
    folds = rng.permutation(n) % 5
    priv = np.zeros(n, int)
    for f in range(5):
        cal, ev = folds != f, folds == f
        lo, hi = np.quantile(r_te[cal], [1 / 3, 2 / 3])
        bucket = np.digitize(r_te, [lo, hi])
        for b in range(3):
            mb = cal & (bucket == b)
            kb = int(np.argmax(D[mb][:, :A].mean(0))) if mb.any() else 0
            priv[ev & (bucket == b)] = kb
    methods["[privileged] Frag. tertile router, target-cal."] = priv
    methods["[privileged] Per-case oracle"] = D[:, :A].argmax(1)
    extra_scores = {}
    for ex in a.extra:
        obj, pat = ex.split("=")
        p = pat.format(name=name)
        if os.path.exists(p):
            e = pd.read_csv(p).set_index("pid").loc[te.pid]
            for k in (1, 3):
                extra_scores[f"{obj.upper()} fixed-{k}"] = (e[f"{pre}{k}"].to_numpy(),
                                                          e[f"hd95_{k}"].to_numpy() if "hd95_0" in e else None)
    res, per_patient = {}, {"pid": te.pid.to_numpy(), "dice_source": D[:, 0]}
    base = D[:, 0]
    for mname, act in list(methods.items()) + [(k, None) for k in extra_scores]:
        if act is None:
            dsel, hsel = extra_scores[mname]
            steps_dep = 3 if "fixed-3" in mname else 1
            act_dep = np.full(n, steps_dep)
        else:
            dsel = D[np.arange(n), act]
            hsel = H[np.arange(n), act] if H is not None else None
            act_dep = act
        g = dsel - base
        executed = 0 if mname.startswith("Source") else (int(act_dep.max()) if "fixed" in mname else a.K)
        if "oracle" in mname or "target-cal" in mname:
            executed = a.K
        res[mname] = {"dice": float(dsel.mean()), "dice_ci": boot_ci(dsel), "dice_sd": float(dsel.std()),
                      "hd95": float(np.nanmean(hsel)) if hsel is not None else None,
                      "gain": float(g.mean()), "gain_ci": boot_ci(g),
                      "harm_rate": float((g < 0).mean()), "harm_ci": boot_ci((g < 0).astype(float)),
                      "cat_harm_002": float((g < -0.02).mean()), "cat_harm_005": float((g < -0.05).mean()),
                      "cat_harm_010": float((g < -0.10).mean()), "help_rate": float((g > 0).mean()),
                      "mean_deployed_steps": float(act_dep.mean()), "executed_steps": executed,
                      "action_hist": np.bincount(act_dep, minlength=6).tolist()}
        per_patient[mname] = dsel
    ours = per_patient["CARS-TTA (ours)"]
    pv = {}
    for mname in res:
        if mname == "CARS-TTA (ours)" or "privileged" in mname:
            continue
        d = ours - per_patient[mname]
        pv[mname] = float(wilcoxon(d, zero_method="zsplit").pvalue) if np.any(d != 0) else 1.0
        res[mname]["delta_vs_ours"] = float(-d.mean())
        res[mname]["effect_size_rbc"] = float((np.sum(d > 0) - np.sum(d < 0)) / max(1, np.sum(d != 0)))
    for k, v in holm(pv).items():
        res[k]["p_holm_vs_ours"] = v
        res[k]["p_raw_vs_ours"] = pv[k]
    # routing quality on target (labels used only to score)
    p_ben = cars.p_beneficial(X)
    best_adapt_gain = D[:, 1:A].max(1) - D[:, 0]
    y_ben = (best_adapt_gain > 0).astype(int)
    _, gpred = cars.predict_parts(X)
    gtrue = D[:, 1:A] - D[:, :1]
    rq = {"oracle_agreement": float((methods["CARS-TTA (ours)"] == D[:, :A].argmax(1)).mean()),
          "spearman_gain_all": float(spearmanr(gpred.ravel(), gtrue.ravel()).correlation),
          f"spearman_gain_k{a.K}": float(spearmanr(gpred[:, -1], gtrue[:, -1]).correlation),
          "auroc_beneficial": float(roc_auc_score(y_ben, p_ben)) if 0 < y_ben.sum() < n else None,
          "brier_beneficial": float(np.mean((p_ben - y_ben) ** 2)), "ece_beneficial": ece(p_ben, y_ben),
          "auroc_logreg": float(roc_auc_score(y_ben, logreg.p_beneficial(X))) if 0 < y_ben.sum() < n else None,
          "base_rate_beneficial": float(y_ben.mean())}
    summary["targets"][name] = {"n": n, "methods": res, "routing": rq}
    pd.DataFrame(per_patient).to_csv(os.path.join(a.out, f"per_patient_{name}.csv"), index=False)
    pd.DataFrame({"pid": te.pid, "p_beneficial": p_ben, "y_beneficial": y_ben,
                  **{f"gpred{k + 1}": gpred[:, k] for k in range(gpred.shape[1])},
                  **{f"gtrue{k + 1}": gtrue[:, k] for k in range(gtrue.shape[1])}}).to_csv(
        os.path.join(a.out, f"router_outputs_{name}.csv"), index=False)
    # Dice / harm trade-off curve of CARS by sweeping tau (analysis only, tau is fixed by source validation)
    curve = []
    for t in np.linspace(-0.03, 0.05, 41):
        d = cars.decide(X, tau=t)
        dd = D[np.arange(n), np.array(acts)[d]]
        curve.append({"tau": float(t), "dice": float(dd.mean()), "harm": float((dd < D[:, 0]).mean())})
    pd.DataFrame(curve).to_csv(os.path.join(a.out, f"tau_curve_{name}.csv"), index=False)
    print(f"\n=== {name} (n={n}) ===")
    for m, r in res.items():
        print(f"{m:48s} Dice {r['dice']:.4f}  gain {r['gain']:+.4f}  harm {r['harm_rate']:.3f}  cat5 {r['cat_harm_005']:.3f}"
              f"  steps {r['mean_deployed_steps']:.2f}  p {r.get('p_holm_vs_ours', float('nan')):.3g}")
    print("routing:", json.dumps(rq))
json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
print(json.dumps(summary["selected"], indent=1))
