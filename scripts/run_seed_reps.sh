#!/bin/bash
# Router-seed replicates (seeds 1 and 2; seed 0 = results/<tag>/ablations) of the full model and every ablation.
# usage: run_seed_reps.sh "<tags>" [parallel]
TAGS=$1; P=${2:-14}
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for TAG in $TAGS; do
  R=results/$TAG; T=$(cat $R/wide_targets.txt)
  while read name extra; do
    for sd in 1 2; do
      O=results/seeds/$TAG/${name}_s$sd
      [ -f $O/summary.json ] && continue
      echo "python -W ignore scripts/route_eval_wide.py --train $R/cand_router_train.csv --val $R/cand_router_val.csv --target $T --only-ours --n-boot 200 --seed $sd --out $O $extra > /dev/null 2>&1"
    done
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
done | xargs -P $P -I{} bash -c "{}"
echo SEEDS_DONE
