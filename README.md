# CARS-TTA: Counterfactual Adaptation Routing for Selective Test-Time Adaptation

Code, trained weights and per-patient results for *CARS-TTA: Counterfactual Routing for Selective Test-Time Adaptation in Cardiac MRI*.

CARS-TTA decides, per patient and without target labels, whether to adapt a cardiac MRI segmenter at test time and which adaptation to use. A small router is trained only on labelled **source** patients that the segmenter never saw, after passing them through simulated acquisition shifts and recording the realised Dice change of every candidate adaptation. At deployment it predicts each candidate's gain from label-free features and keeps the source prediction unless a candidate is predicted to help.

## Main result (4 confirmation folds, 16 fold-target units, 1,440 patient evaluations)

Harm is the percentage of patients whose Dice falls below the source model's by more than 0, 1 or 5 Dice points (mean over units). "Changed" is the percentage of patients whose prediction differs from the source model's.

| Method | Dice (%) | Gain (points) | Harm >0 | Harm >1 | Harm >5 | Changed |
|---|---|---|---|---|---|---|
| Source model | 80.11 | 0.00 | 0.0 | 0.0 | 0.0 | 0 |
| TENT, 3 steps | 81.11 | +1.00 | 21.9 | 4.0 | 1.1 | 100 |
| SAR, 3 steps | 81.11 | +1.00 | 21.9 | 3.6 | 1.1 | 100 |
| EATA, 3 steps | 79.56 | -0.54 | 80.9 | 22.6 | 0.0 | 100 |
| Minimum-entropy selection | 81.32 | +1.21 | 21.8 | 4.4 | 1.1 | 100 |
| Best single candidate per site (**target labels**) | 81.62 | +1.51 | 20.4 | 6.5 | 1.1 | 100 |
| **CARS-TTA (no target labels)** | **81.67** | **+1.56** | **13.0** | 5.8 | **0.5** | 70 |
| Oracle (privileged) | 82.72 | +2.62 | 0.0 | 0.0 | 0.0 | 98 |

Over three router seeds CARS-TTA gains +1.60 ± 0.01 points. Its lead over TENT-3 (+0.56 points) is not significant across the eight vendor pairs (better on 4 of 8, p = 0.38). The patient-clustered 95% CI for these fixed pairs is [+0.34, +0.83]. The gain is concentrated on the Philips source. With Siemens as source, well-tuned TENT gains more than CARS-TTA (+1.56 vs +1.19 points). CARS-TTA avoids most patients with any drop by leaving 30% unchanged, has more drops of 1 to 5 points than TENT (significant for these vendor pairs) and fewer drops beyond 5 points (9 vs 16 patient evaluations, not significant). A segmenter retrained with the same simulated shifts (`results/aug/`, `scripts/run_aug_fold.sh`) with one untuned setting gains more on average (+2.07 points with TENT-3), but only with Philips as source. On top of that segmenter, CARS-TTA matches TENT-3's gain (+0.67 vs +0.64 points) and harms 8.4% of patients instead of 20.2% (2.0% vs 3.5% beyond 1 point, 0.04% vs 1.2% beyond 5 points). See the paper for the full analysis and limitations.

Two further folds with GE (`srcC_s0`) and Canon (`srcD_s0`) as source were run later with the frozen design (one seed each, only 8 (GE) or 6 (Canon) router training patients, `scripts/newsrc_rows.py`). There CARS-TTA gains +1.13 points against +0.32 for TENT-3 (better on 5 of 8 vendor pairs, not significant), but 13.2% of patients drop by more than 1 point against 4.5% for TENT (9 vs 2 beyond 5 points), all from input corrections. The harm benefit of the main folds therefore did not transfer to these small router sets.

Results are not bit-for-bit reproducible across machines: the same router seed gave +1.56 and +1.60 points on two machines with different PyTorch builds (the s.d. over seeds is 0.01). `scripts/retrain_variance.py` writes `results/aggregate/retrain_variance.csv`, the per-patient Dice difference between two source models of the same vendor.

## Layout

```
cars_tta/        library: data, U-Net, shifts, episodic TTA (TENT/EATA/SAR), candidates, features, router, metrics
scripts/         preprocessing, split, segmenter training, candidate building, routing evaluation, ablations,
                 aggregation, tables and figures (scripts/figures)
tests/           unit tests (pytest)
splits/          patient-level splits per fold (srcA = Siemens, srcB = Philips, srcC = GE, srcD = Canon source, _sN = seed)
checkpoints/     trained segmenters per fold (best.pt)
results/         per fold: candidate tables (cand_*.csv), evaluation (eval_wide/), synthetic evaluation
                 (eval_synthval/), ablations (ablations/); results/aggregate: pooled tables
tools/           dataset download helpers
```

Fold `srcA_s0` is the development fold used to design the method. The other four folds are confirmation folds run with the frozen design.

## Data

* M&Ms (Campello et al., IEEE TMI 2021): request access at https://www.ub.edu/mnms/
* ACDC (Bernard et al., IEEE TMI 2018): https://www.creatis.insa-lyon.fr/Challenge/acdc/

Place the raw data under `data/raw/mnms` and `data/raw/acdc`, then run `scripts/preprocess.py` (commands below).

## Reproducing a fold

```bash
python -m venv venv && venv/bin/pip install -r requirements.txt
venv/bin/python -m pytest -q tests
python scripts/preprocess.py --dataset acdc --raw data/raw/acdc --out data/cache/acdc
python scripts/preprocess.py --dataset mnms --raw data/raw/mnms --info data/raw/mnms/<info>.csv --out data/cache/mnms
bash scripts/run_fold_full.sh srcB_s0 B 0 "0 1"    # split, train, lr selection, candidates, evaluation
bash scripts/run_post_par.sh srcB_s0               # synthetic held-out evaluation and ablations
python scripts/aggregate.py --tags srcA_s0 srcA_s1 srcA_s2 srcB_s0 srcB_s1 --out results/aggregate
python scripts/make_tables.py --tags srcA_s0 srcA_s1 srcA_s2 srcB_s0 srcB_s1
python scripts/figures/fig_results.py --tags srcA_s0 srcA_s1 srcA_s2 srcB_s0 srcB_s1
```

To evaluate with the released weights instead of retraining, keep `checkpoints/<fold>/best.pt` in place: `run_fold_full.sh` skips segmenter training when it exists, and the regenerated split is byte-identical to `splits/<fold>.csv`. The routing evaluation alone (`scripts/route_eval_wide.py`) runs on CPU in about a minute from the released `results/<fold>/cand_*.csv` tables.

## License

MIT
