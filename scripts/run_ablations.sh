#!/bin/bash
# Router-side ablations of the frozen CARS-TTA design on one fold (reuses that fold's candidate episodes).
# usage: run_ablations.sh <tag>
set -e
TAG=$1
cd "$(dirname "$0")/.."
R=results/$TAG
T=$(cat $R/wide_targets.txt)
run() {  # name, extra args
  venv/bin/python -W ignore scripts/route_eval_wide.py --train ${TRAIN:-$R/cand_router_train.csv} --val ${VAL:-$R/cand_router_val.csv} \
      --target $T --only-ours --n-boot 500 --seed 0 --out $R/ablations/$1 "${@:2}" > $R/ablations/$1.log 2>&1
  echo "done $1"
}
mkdir -p $R/ablations
run full
run A1_no_counterfactuals --train-filter clean
run A2_intensity_only --train-filter intensity
run A3_frequency_only --train-filter frequency
run A5_no_fragmentation --drop frag
run A6_no_uncertainty --drop unc
run A7_no_morphology --drop morph
run A8_no_ood --drop ood
run A8b_no_disagreement --drop dis
run A9_no_harm_term --lam-h 0
run A11_binary_consensus_gate --cands ens
run A12_steps_only --cands hi1 hi2 hi3
run A12_one_step_only --cands hi1
run A14_scan_features_only --drop unc dis frag morph   # candidate-independent information only (ood + one-hot)
if [ -f $R/cand_router_train_sev05.csv ]; then
  TRAIN=$R/cand_router_train_sev05.csv VAL=$R/cand_router_val_sev05.csv run A13_mild_shifts
fi
echo ABLATIONS_DONE $TAG
