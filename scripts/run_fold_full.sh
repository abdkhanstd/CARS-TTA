#!/bin/bash
# Complete fold for the final method: split -> segmenter -> source-only TTA lr selection -> EATA/SAR target
# baselines -> wide-candidate CARS-TTA (run_fold_wide.sh).
# usage: run_fold_full.sh <tag> "<source vendors>" <seed> "<gpus>" [n_shards]
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TAG=$1; SRC=$2; SEED=$3; GPUS=$4; NS=${5:-3}
cd "$(dirname "$0")/.."
R=results/$TAG; CK=checkpoints/$TAG; SPLIT=splits/$TAG.csv
mkdir -p $R $CK logs
venv/bin/python scripts/make_split.py --index data/cache/mnms/index.csv data/cache/acdc/index.csv \
    --source-vendors $SRC --source-dataset mnms --seed $SEED --out $SPLIT > $R/split_summary.txt
[ -f $CK/best.pt ] || CUDA_VISIBLE_DEVICES=${GPUS%% *} venv/bin/python scripts/train_seg.py \
    --cache data/cache/mnms data/cache/acdc --split $SPLIT --out $CK --seed $SEED > logs/seg_$TAG.log 2>&1
if [ ! -f $R/lr.txt ]; then
  for LR in 0.001 0.003 0.01 0.03; do
    bash scripts/run_shards.sh $NS "$GPUS" $R/lrsel/lr$LR.csv --ckpt $CK/best.pt --split $SPLIT --role router_val \
        --n-shifts 10 --lr $LR --no-features --seed 101
  done
  venv/bin/python - $R <<'PY'
import sys, glob, json, pandas as pd
R = sys.argv[1]; res = {}
for f in sorted(glob.glob(f"{R}/lrsel/lr*.csv")):
    if "shard" in f: continue
    d = pd.read_csv(f); lr = f.split("lr")[-1][:-4]
    g = {k: float((d[f"q{k}"] - d.q0).mean()) for k in range(1, 6)}
    res[lr] = {"mean_gain": g, "harm": {k: float((d[f"q{k}"] < d.q0).mean()) for k in range(1, 6)}, "best_fixed_gain": max(g.values())}
best = max(res, key=lambda l: res[l]["best_fixed_gain"])
json.dump({"selected_lr": best, "per_lr": res}, open(f"{R}/lr_selection.json", "w"), indent=1)
open(f"{R}/lr.txt", "w").write(best)
PY
fi
LR=$(cat $R/lr.txt)
for V in A B C D ACDC; do
  [[ " $SRC " == *" $V "* ]] && continue
  if [ $V == ACDC ]; then SELV="--dataset acdc"; else SELV="--dataset mnms --vendor $V"; fi
  for OBJ in eata sar; do
    [ -f $R/traj_target_${V}_$OBJ.csv ] || bash scripts/run_shards.sh $NS "$GPUS" $R/traj_target_${V}_$OBJ.csv \
        --ckpt $CK/best.pt --split $SPLIT --role test $SELV --lr $LR --method $OBJ --K 3 --orig-metrics --no-features --seed 7
  done
done
bash scripts/run_fold_wide.sh $TAG "$SRC" $SEED "$GPUS" $NS
echo FULL_DONE $TAG
