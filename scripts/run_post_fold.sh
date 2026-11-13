#!/bin/bash
# Per-fold post-processing: ablations + synthetic held-out evaluation of the frozen router.
# usage: run_post_fold.sh <tag>
TAG=$1
cd "$(dirname "$0")/.."
R=results/$TAG
venv/bin/python -W ignore scripts/route_eval_wide.py --train $R/cand_router_train.csv --val $R/cand_router_val.csv \
  --target synthval=$R/cand_router_val.csv --only-ours --n-boot 500 --seed 0 --out $R/eval_synthval > $R/eval_synthval.log 2>&1
echo "SYNTHVAL_DONE $TAG"
bash scripts/run_ablations.sh $TAG
