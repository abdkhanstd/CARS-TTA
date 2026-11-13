#!/bin/bash
# One complete experiment: source vendor(s) of M&Ms -> every other M&Ms vendor + ACDC (external).
# usage: run_fold.sh <tag> "<source vendors e.g. A>" <seed> "<gpus>" [n_shards]
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TAG=$1; SRC=$2; SEED=$3; GPUS=$4; NS=${5:-6}
cd "$(dirname "$0")/.."
R=results/$TAG; CK=checkpoints/$TAG; SPLIT=splits/$TAG.csv
mkdir -p $R $CK logs
venv/bin/python scripts/make_split.py --index data/cache/mnms/index.csv data/cache/acdc/index.csv \
    --source-vendors $SRC --source-dataset mnms --seed $SEED --out $SPLIT > $R/split_summary.txt
if [ ! -f $CK/best.pt ]; then
  CUDA_VISIBLE_DEVICES=${GPUS%% *} venv/bin/python scripts/train_seg.py --cache data/cache/mnms data/cache/acdc \
      --split $SPLIT --out $CK --seed $SEED > logs/seg_$TAG.log 2>&1
fi
if [ ! -f $R/lr.txt ]; then
  bash scripts/run_source_pipeline.sh $TAG $SPLIT $CK/best.pt "$GPUS" $NS
fi
LR=$(cat $R/lr.txt)
TARGETS=""
for V in A B C D; do
  [[ " $SRC " == *" $V "* ]] && continue
  bash scripts/run_shards.sh $NS "$GPUS" $R/traj_target_$V.csv --ckpt $CK/best.pt --split $SPLIT --role test \
      --dataset mnms --vendor $V --lr $LR --orig-metrics --src-stats $R/src_stats.npz --save-masks 3 --seed 7
  for OBJ in eata sar; do
    bash scripts/run_shards.sh $NS "$GPUS" $R/traj_target_${V}_$OBJ.csv --ckpt $CK/best.pt --split $SPLIT --role test \
        --dataset mnms --vendor $V --lr $LR --method $OBJ --K 3 --orig-metrics --no-features --seed 7
  done
  TARGETS="$TARGETS $V=$R/traj_target_$V.csv"
done
bash scripts/run_shards.sh $NS "$GPUS" $R/traj_target_ACDC.csv --ckpt $CK/best.pt --split $SPLIT --role test \
    --dataset acdc --lr $LR --orig-metrics --src-stats $R/src_stats.npz --save-masks 3 --seed 7
for OBJ in eata sar; do
  bash scripts/run_shards.sh $NS "$GPUS" $R/traj_target_ACDC_$OBJ.csv --ckpt $CK/best.pt --split $SPLIT --role test \
      --dataset acdc --lr $LR --method $OBJ --K 3 --orig-metrics --no-features --seed 7
done
TARGETS="$TARGETS ACDC=$R/traj_target_ACDC.csv"
venv/bin/python scripts/route_eval.py --train $R/traj_router_train.csv --val $R/traj_router_val.csv \
    --target $TARGETS --extra "eata=$R/traj_target_{name}_eata.csv" "sar=$R/traj_target_{name}_sar.csv" \
    --seed $SEED --out $R/eval > $R/eval.log 2>&1
echo FOLD_DONE $TAG
