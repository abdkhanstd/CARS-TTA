#!/bin/bash
# Baseline: segmenter trained with the synthetic acquisition shifts as augmentation (p=0.5), same split and seed.
# Evaluates all candidates on the targets (the augmented source model is the "src" column), then builds router
# data and runs CARS-TTA on top of the augmented segmenter. Uses the fold's source-selected TENT rate.
# usage: run_aug_fold.sh <tag> <seed> "<gpus>" [n_shards]
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BUILDER=build_candidates.py
TAG=$1; SEED=$2; GPUS=$3; NS=${4:-2}
cd "$(dirname "$0")/.."
R=results/aug/$TAG; CK=checkpoints_aug/$TAG; SPLIT=splits/$TAG.csv
mkdir -p $R $CK logs
[ -f $CK/best.pt ] || CUDA_VISIBLE_DEVICES=${GPUS%% *} venv/bin/python scripts/train_seg.py --cache data/cache/mnms data/cache/acdc \
    --split $SPLIT --out $CK --seed $SEED --shift-aug 0.5 > logs/segaug_$TAG.log 2>&1
LRH=$(cat results/$TAG/lr.txt); LRL=$(python3 -c "print($LRH/10)")
CUDA_VISIBLE_DEVICES=${GPUS%% *} venv/bin/python scripts/source_stats.py --ckpt $CK/best.pt --split $SPLIT --out $R/src_stats.npz
COMMON="--ckpt $CK/best.pt --split $SPLIT --lr-hi $LRH --lr-lo $LRL --src-stats $R/src_stats.npz"
TARGETS=""
for spec in $(cat results/$TAG/wide_targets.txt); do
  NAME=${spec%%=*}
  if [ $NAME = ACDC ]; then SEL="--dataset acdc"; else SEL="--dataset mnms --vendor $NAME"; fi
  [ -f $R/cand_target_$NAME.csv ] || bash scripts/run_shards.sh $NS "$GPUS" $R/cand_target_$NAME.csv $COMMON --role test $SEL --orig-metrics --seed 7
  TARGETS="$TARGETS $NAME=$R/cand_target_$NAME.csv"
done
echo "$TARGETS" > $R/wide_targets.txt
echo TARGETS_DONE $TAG
[ -f $R/cand_router_train.csv ] || bash scripts/run_shards.sh $NS "$GPUS" $R/cand_router_train.csv $COMMON --role router_train --n-shifts 40 --seed 1
[ -f $R/cand_router_val.csv ] || bash scripts/run_shards.sh $NS "$GPUS" $R/cand_router_val.csv $COMMON --role router_val --n-shifts 20 --seed 2
OMP_NUM_THREADS=8 venv/bin/python scripts/route_eval_wide.py --train $R/cand_router_train.csv --val $R/cand_router_val.csv \
    --target $TARGETS --seed $SEED --out $R/eval_wide > $R/eval_wide.log 2>&1
echo AUG_DONE $TAG
