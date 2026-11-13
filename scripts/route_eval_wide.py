"""Wide-candidate CARS-TTA: train on source counterfactual episodes, select hyper-parameters on source validation
episodes, apply frozen to real target patients and compare with all baselines.

python scripts/route_eval_wide.py --train R/cand_router_train.csv --val R/cand_router_val.csv \
    --target B=R/cand_target_B.csv ... --out R/eval_wide
Ablation flags: --cands (restrict the action set), --drop (feature groups), --lam-h, --train-filter clean, --linear.
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
from cars_tta.candidates import CAND_NAMES  # noqa: E402
from cars_tta.router import WideRouter, choose_tau_wide, wide_arrays  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--train", required=True)
ap.add_argument("--val", required=True)
ap.add_argument("--target", nargs="+", required=True)
ap.add_argument("--cands", nargs="*", default=CAND_NAMES[1:])
ap.add_argument("--drop", nargs="*", default=[])
ap.add_argument("--lam-h", type=float, nargs="+", default=[0.0, 1.0, 2.0])
ap.add_argument("--train-filter", default=None)
ap.add_argument("--extra", nargs="*", default=[], help="obj=path pattern with {name}: EATA/SAR fixed-step baselines")
ap.add_argument("--harm-cap", type=float, default=0.10, help="max source-validation harm rate for the margin (fixed in advance)")
ap.add_argument("--only-ours", action="store_true", help="ablation runs: skip baselines")
ap.add_argument("--tau-max", type=float, default=0.03, help="upper end of the margin grid (sensitivity analysis)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n-boot", type=int, default=2000)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
rng = np.random.default_rng(a.seed)
tr, va = pd.read_csv(a.train), pd.read_csv(a.val)
if a.train_filter == "clean":
    tr = tr[tr["shift"] == "none"]
elif a.train_filter in ("intensity", "frequency"):
    from cars_tta import shifts as S
    fam = set(S.INTENSITY if a.train_filter == "intensity" else S.FREQUENCY)
    tr = tr[tr["shift"].apply(lambda s: s == "none" or set(s.split("+")) <= fam)]
CANDS = list(a.cands)
Str, Ctr, Gtr, _, _ = wide_arrays(tr, CANDS, a.drop)
Sva, Cva, Gva, fs_names, fc_names = wide_arrays(va, CANDS, a.drop)


def val_gain(r):
    d = r.decide(Sva, Cva)
    g = np.concatenate([np.zeros((len(Gva), 1)), Gva], 1)
    g = g[np.arange(len(g)), d]
    return float(g.mean()) if a.harm_cap is None or (g < 0).mean() <= a.harm_cap else float(g.mean()) - 1.0


sel = []
for lh in a.lam_h:
    r = WideRouter(CANDS, lam_h=lh, seed=a.seed).fit(Str, Ctr, Gtr)
    choose_tau_wide(r, Sva, Cva, Gva, harm_cap=a.harm_cap, grid=np.round(np.arange(-0.01, a.tau_max + 1e-9, 0.001), 4))
    sel.append((val_gain(r), lh, r))
sel.sort(key=lambda t: -t[0])
cars = sel[0][2]
cars_noharm = next((s[2] for s in sel if s[1] == 0.0), None)
summary = {"selected": {"lambda_h": sel[0][1], "tau": cars.tau, "val_gain_per_lambda": {str(s[1]): s[0] for s in sel},
                        "cands": CANDS, "n_shared_features": len(fs_names), "n_cand_features": len(fc_names),
                        "router_params": cars.n_params(), "n_train_episodes": len(tr), "n_val_episodes": len(va),
                        "drop": a.drop, "train_filter": a.train_filter, "harm_cap": a.harm_cap}, "targets": {}}
if not a.only_ours:
    lin = WideRouter(CANDS, linear=True).fit(Str, Ctr, Gtr)
    choose_tau_wide(lin, Sva, Cva, Gva, harm_cap=a.harm_cap)
    steps = [c for c in ["hi1", "hi2", "hi3"] if c in CANDS]
    S2, C2, G2, _, _ = wide_arrays(tr, steps, a.drop)
    S2v, C2v, G2v, _, _ = wide_arrays(va, steps, a.drop)
    cars_steps = WideRouter(steps, lam_h=sel[0][1], seed=a.seed).fit(S2, C2, G2)
    choose_tau_wide(cars_steps, S2v, C2v, G2v, harm_cap=a.harm_cap)
    # source-calibrated simple baselines
    allc = ["src"] + CAND_NAMES[1:]
    val_means = {c: float(va[f"q__{c}"].mean()) for c in allc}
    best_fixed = max(val_means, key=val_means.get)
    g3 = (va["q__hi3"] - va["q__src"]).to_numpy()

    def best_threshold(score, gains, higher_adapts):
        best, bt = -1e9, None
        for t in np.quantile(score, np.linspace(0, 1, 101)):
            m = score > t if higher_adapts else score < t
            v = np.where(m, gains, 0).mean()
            if v > best:
                best, bt = v, t
        return bt

    t_ent = best_threshold(va["fs_unc_ent0"].to_numpy(), g3, True)
    mu_n, sd_n = va["fc__hi1__frag_n"].mean(), va["fc__hi1__frag_n"].std() + 1e-9
    mu_d, sd_d = va["fc__hi1__dis_delta"].mean(), va["fc__hi1__dis_delta"].std() + 1e-9
    risk = lambda df: (0.5 * (df["fc__hi1__frag_n"] - mu_n) / sd_n + 0.5 * (df["fc__hi1__dis_delta"] - mu_d) / sd_d).to_numpy()
    r_va = risk(va)
    cut_lo, cut_hi = np.quantile(r_va, [1 / 3, 2 / 3])
    t_frag = best_threshold(r_va, g3, False)
    summary["selected"].update({"best_fixed_srcval": best_fixed, "val_mean_q": val_means, "t_entropy": float(t_ent),
                                "t_frag": float(t_frag), "frag_cuts": [float(cut_lo), float(cut_hi)],
                                "steps_only_tau": cars_steps.tau, "linear_tau": lin.tau})


def boot_ci(x, n=a.n_boot):
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def holm(p):
    names = list(p)
    order = np.argsort([p[n] for n in names])
    out, prev = {}, 0
    for r, i in enumerate(order):
        prev = min(1.0, max(prev, (len(names) - r) * p[names[i]]))
        out[names[i]] = prev
    return out


def ece(p, y, bins=10):
    e = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (p >= lo) & (p < lo + 1 / bins)
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


NAMES_ALL = ["src"] + CAND_NAMES[1:]
LABEL = {"src": "Source (no adaptation)", "hi1": "TENT fixed 1 step", "hi2": "TENT fixed 2 steps",
         "hi3": "TENT fixed 3 steps", "lo1": "TENT (low lr) 1 step", "lo2": "TENT (low lr) 2 steps",
         "lo3": "TENT (low lr) 3 steps", "g07": "Gamma 0.7 always", "g14": "Gamma 1.4 always",
         "hm": "Histogram matching always", "dn": "Denoising always", "ens": "Consensus ensemble always"}
for spec in a.target:
    name, path = spec.split("=")
    te = pd.read_csv(path)
    n = len(te)
    pre = "dice__" if "dice__src" in te else "q__"
    Dall = np.stack([te[f"{pre}{c}"].to_numpy() for c in NAMES_ALL], 1)          # (n, 12)
    Hall = np.stack([te[f"hd95__{c}"].to_numpy() for c in NAMES_ALL], 1) if "hd95__src" in te else None
    col = {c: i for i, c in enumerate(NAMES_ALL)}
    Ste, Cte, _, _, _ = wide_arrays(te, CANDS, a.drop)
    to_all = lambda d, cl: np.array([col[(["src"] + cl)[j]] for j in d])
    methods = {}
    if not a.only_ours:
        for c in NAMES_ALL:
            methods[LABEL[c]] = np.full(n, col[c])
        methods[f"Best fixed on source val. ({best_fixed})"] = np.full(n, col[best_fixed])
        ents = np.stack([te["fs_unc_ent0"].to_numpy()] + [te[f"fc__{c}__unc_ent"].to_numpy() for c in CAND_NAMES[1:]], 1)
        methods["Minimum-entropy selection"] = ents.argmin(1)
        methods["Entropy gate (src-cal.)"] = np.where(te["fs_unc_ent0"].to_numpy() > t_ent, col["hi3"], 0)
        r_te = risk(te)
        methods["Fragmentation gate (src-cal.)"] = np.where(r_te < t_frag, col["hi3"], 0)
        methods["Fragmentation tertile router (src-cal.)"] = np.where(r_te <= cut_lo, col["hi3"], np.where(r_te <= cut_hi, col["hi2"], 0))
        methods["CARS steps-only (src, 1-3 TENT steps)"] = to_all(cars_steps.decide(*wide_arrays(te, steps, a.drop)[:2]), steps)
        methods["CARS linear (ridge scorer)"] = to_all(lin.decide(Ste, Cte), CANDS)
    if cars_noharm is not None and cars_noharm is not cars:
        methods["CARS without harm term (lambda_h=0)"] = to_all(cars_noharm.decide(Ste, Cte), CANDS)
    methods["CARS-TTA (ours)"] = to_all(cars.decide(Ste, Cte), CANDS)
    if not a.only_ours:
        folds = rng.permutation(n) % 5
        priv = np.zeros(n, int)
        r_te = risk(te)
        for f in range(5):
            cal, ev = folds != f, folds == f
            lo, hi = np.quantile(r_te[cal], [1 / 3, 2 / 3])
            bucket = np.digitize(r_te, [lo, hi])
            for b in range(3):
                mb = cal & (bucket == b)
                opts = [col[c] for c in ["src", "hi1", "hi2", "hi3"]]
                kb = opts[int(np.argmax(Dall[mb][:, opts].mean(0)))] if mb.any() else 0
                priv[ev & (bucket == b)] = kb
        methods["[privileged] Frag. tertile router, target-cal."] = priv
        methods["[privileged] Oracle, steps only"] = np.array([col[c] for c in ["src", "hi1", "hi2", "hi3"]])[Dall[:, [col[c] for c in ["src", "hi1", "hi2", "hi3"]]].argmax(1)]
    cand_cols = [col[c] for c in ["src"] + CANDS]
    methods["[privileged] Oracle, all candidates"] = np.array(cand_cols)[Dall[:, cand_cols].argmax(1)]
    base = Dall[:, 0]
    res, per = {}, {"pid": te.pid.to_numpy(), "dice_source": base}
    extra = {}
    for ex in a.extra:
        obj, pat = ex.split("=")
        p = pat.format(name=name)
        if os.path.exists(p):
            e = pd.read_csv(p).set_index("pid").loc[te.pid]
            epre = "dice_" if "dice_0" in e else "q"
            for k in (1, 3):
                extra[f"{obj.upper()} fixed {k} step{'s' if k > 1 else ''}"] = (
                    e[f"{epre}{k}"].to_numpy(), e[f"hd95_{k}"].to_numpy() if "hd95_0" in e else None)
    for m, act in list(methods.items()) + [(k, None) for k in extra]:
        if act is None:
            d, h = extra[m]
            act = np.full(n, -1)
        else:
            d = Dall[np.arange(n), act]
            h = Hall[np.arange(n), act] if Hall is not None else None
        g = d - base
        res[m] = {"dice": float(d.mean()), "dice_ci": boot_ci(d), "dice_sd": float(d.std()),
                  "hd95": float(np.nanmean(h)) if h is not None else None,
                  "gain": float(g.mean()), "gain_ci": boot_ci(g), "harm_rate": float((g < 0).mean()),
                  "harm_ci": boot_ci((g < 0).astype(float)), "help_rate": float((g > 0).mean()),
                  "cat_harm_002": float((g < -0.02).mean()), "cat_harm_005": float((g < -0.05).mean()),
                  "cat_harm_010": float((g < -0.10).mean()), "adapt_rate": float((act != 0).mean()),
                  "choice_hist": {c: int((act == col[c]).sum()) for c in NAMES_ALL if (act == col[c]).any()}}
        if not np.array_equal(base, Dall[:, 0]):
            raise RuntimeError("base mismatch")
        if h is not None:
            res[m]["hd95_ci"] = boot_ci(np.nan_to_num(h, nan=np.nanmean(h)))
        per[m] = d
        per[m + " [choice]"] = [NAMES_ALL[i] if i >= 0 else m for i in act]
    ours = per["CARS-TTA (ours)"]
    pv = {}
    for m in res:
        if m == "CARS-TTA (ours)" or "privileged" in m:
            continue
        diff = ours - per[m]
        pv[m] = float(wilcoxon(diff, zero_method="zsplit").pvalue) if np.any(diff != 0) else 1.0
        res[m]["delta_ours_minus_method"] = float(diff.mean())
        res[m]["delta_ci"] = boot_ci(diff)
        res[m]["effect_rbc"] = float((np.sum(diff > 0) - np.sum(diff < 0)) / max(1, np.sum(diff != 0)))
    for k, v in holm(pv).items():
        res[k]["p_holm"] = v
        res[k]["p_raw"] = pv[k]
    gpred = cars.predict_gain(Ste, Cte)
    gtrue = Dall[:, [col[c] for c in CANDS]] - base[:, None]
    p_ben = cars.p_beneficial(Ste, Cte)
    y_ben = (gtrue.max(1) > 0).astype(int)
    rq = {"spearman_gain_pooled": float(spearmanr(gpred.ravel(), gtrue.ravel()).correlation),
          "spearman_gain_within_patient": float(np.nanmean([spearmanr(gpred[i], gtrue[i]).correlation for i in range(n)])),
          "auroc_beneficial": float(roc_auc_score(y_ben, p_ben)) if 0 < y_ben.sum() < n else None,
          "brier_beneficial": float(np.mean((p_ben - y_ben) ** 2)),
          "ece_beneficial": ece(p_ben, y_ben), "base_rate_beneficial": float(y_ben.mean())}
    summary["targets"][name] = {"n": n, "methods": res, "routing": rq}
    pd.DataFrame(per).to_csv(os.path.join(a.out, f"per_patient_{name}.csv"), index=False)
    out = pd.DataFrame({"pid": te.pid, "p_beneficial": p_ben, "y_beneficial": y_ben})
    for j, c in enumerate(CANDS):
        out[f"gpred__{c}"], out[f"gtrue__{c}"] = gpred[:, j], gtrue[:, j]
    out.to_csv(os.path.join(a.out, f"router_outputs_{name}.csv"), index=False)
    curve = []
    for t in np.linspace(-0.02, 0.05, 36):
        d = to_all(cars.decide(Ste, Cte, tau=t), CANDS)
        dd = Dall[np.arange(n), d]
        curve.append({"tau": float(t), "dice": float(dd.mean()), "harm": float((dd < base).mean()), "adapt": float((d != 0).mean())})
    pd.DataFrame(curve).to_csv(os.path.join(a.out, f"tau_curve_{name}.csv"), index=False)
    print(f"\n=== {name} (n={n}) ===")
    for m, r in res.items():
        print(f"{m:50s} Dice {r['dice']:.4f} gain {r['gain']:+.4f} harm {r['harm_rate']:.3f} help {r['help_rate']:.3f}"
              f" cat5 {r['cat_harm_005']:.3f} adapt {r['adapt_rate']:.2f} p {r.get('p_holm', float('nan')):.3g}")
    print("routing:", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rq.items()}))
json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
print(json.dumps(summary["selected"], indent=1))
