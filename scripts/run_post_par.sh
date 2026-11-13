#!/bin/bash
# Parallel per-fold post-processing (synthetic held-out eval + ablations), 4 threads per process.
# usage: run_post_par.sh <tag> [parallel jobs]
TAG=$1; P=${2:-5}
cd "$(dirname "$0")/.."
R=results/$TAG
T=$(cat $R/wide_targets.txt)
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p $R/ablations
base="venv/bin/python -W ignore scripts/route_eval_wide.py --only-ours --n-boot 500 --seed 0"
{
[ -f $R/eval_synthval/summary.json ] || echo "$base --train $R/cand_router_train.csv --val $R/cand_router_val.csv --target synthval=$R/cand_router_val.csv --out $R/eval_synthval > $R/eval_synthval.log 2>&1"
while read name extra; do
  [ -f $R/ablations/$name/summary.json ] && continue
  echo "$base --train $R/cand_router_train.csv --val $R/cand_router_val.csv --target $T --out $R/ablations/$name $extra > $R/ablations/$name.log 2>&1"
done <<LIST
full
A1_no_counterfactuals --train-filter clean
A2_intensity_only --train-filter intensity
A3_frequency_only --train-filter frequency
A5_no_fragmentation --drop frag
A6_no_uncertainty --drop unc
A7_no_morphology --drop morph
A8_no_ood --drop ood
A8b_no_disagreement --drop dis
A9_no_harm_term --lam-h 0
A11_binary_consensus_gate --cands ens
A12_steps_only --cands hi1 hi2 hi3
A12_one_step_only --cands hi1
A14_scan_features_only --drop unc dis frag morph
LIST
} > /tmp/post_cmds_$TAG.txt
xargs -P $P -I{} bash -c "{}" < /tmp/post_cmds_$TAG.txt
echo POST_DONE $TAG
