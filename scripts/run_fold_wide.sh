#!/bin/bash
# Wide-candidate CARS-TTA for one fold whose segmenter and source-selected TTA learning rate already exist
# (run_fold.sh made them). lr_hi = source-selected lr, lr_lo = lr_hi / 10 (rule fixed in advance).
# usage: run_fold_wide.sh <tag> "<source vendors>" <seed> "<gpus>" [n_shards]
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BUILDER=build_candidates.py
TAG=$1; SRC=$2; SEED=$3; GPUS=$4; NS=${5:-3}
cd "$(dirname "$0")/.."
R=results/$TAG; CK=checkpoints/$TAG; SPLIT=splits/$TAG.csv
LRH=$(cat $R/lr.txt)
LRL=$(python3 -c "print($LRH/10)")
echo "lr_hi=$LRH lr_lo=$LRL"
CUDA_VISIBLE_DEVICES=${GPUS%% *} venv/bin/python scripts/source_stats.py --ckpt $CK/best.pt --split $SPLIT --out $R/src_stats.npz
COMMON="--ckpt $CK/best.pt --split $SPLIT --lr-hi $LRH --lr-lo $LRL --src-stats $R/src_stats.npz"
[ -f $R/cand_router_train.csv ] || bash scripts/run_shards.sh $NS "$GPUS" $R/cand_router_train.csv $COMMON --role router_train --n-shifts 40 --seed 1
[ -f $R/cand_router_val.csv ] || bash scripts/run_shards.sh $NS "$GPUS" $R/cand_router_val.csv $COMMON --role router_val --n-shifts 20 --seed 2
TARGETS=""
for V in A B C D; do
  [[ " $SRC " == *" $V "* ]] && continue
  [ -f $R/cand_target_$V.csv ] || bash scripts/run_shards.sh $NS "$GPUS" $R/cand_target_$V.csv $COMMON --role test --dataset mnms --vendor $V --orig-metrics --save-masks 4 --seed 7
  TARGETS="$TARGETS $V=$R/cand_target_$V.csv"
done
[ -f $R/cand_target_ACDC.csv ] || bash scripts/run_shards.sh $NS "$GPUS" $R/cand_target_ACDC.csv $COMMON --role test --dataset acdc --orig-metrics --save-masks 4 --seed 7
TARGETS="$TARGETS ACDC=$R/cand_target_ACDC.csv"
echo "$TARGETS" > $R/wide_targets.txt
venv/bin/python scripts/route_eval_wide.py --train $R/cand_router_train.csv --val $R/cand_router_val.csv \
    --target $TARGETS --extra "eata=$R/traj_target_{name}_eata.csv" "sar=$R/traj_target_{name}_sar.csv" --seed $SEED --out $R/eval_wide > $R/eval_wide.log 2>&1
echo WIDE_DONE $TAG
