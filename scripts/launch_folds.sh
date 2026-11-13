#!/bin/bash
# Launch several folds in the background, one fold per GPU, 4 shards each.
# usage: launch_folds.sh "<tag:vendors:seed:gpu> ..."
cd "$(dirname "$0")/.." && mkdir -p logs
for spec in $1; do
  IFS=: read TAG SRC SEED GPU <<< "$spec"
  (setsid nohup bash scripts/run_fold_full.sh $TAG "$SRC" $SEED "$GPU" 4 > logs/full_$TAG.log 2>&1 < /dev/null &)
  echo "launched $TAG ($SRC, seed $SEED) on GPU $GPU"
done
