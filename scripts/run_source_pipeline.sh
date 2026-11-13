#!/bin/bash
# Source-side pipeline for one segmenter: bottleneck stats -> TTA learning-rate selection on source validation
# shifts -> router training/validation episodes. Uses no target data.
# usage: run_source_pipeline.sh <tag> <split.csv> <ckpt> "<gpus>" [n_shards]
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TAG=$1; SPLIT=$2; CKPT=$3; GPUS=$4; NS=${5:-6}
R=results/$TAG
mkdir -p $R
CUDA_VISIBLE_DEVICES=${GPUS%% *} venv/bin/python scripts/source_stats.py --ckpt $CKPT --split $SPLIT --out $R/src_stats.npz
# 1) TTA learning rate from source validation shifts (best fixed-step mean gain), no features needed
for LR in 0.001 0.003 0.01 0.03; do
  bash scripts/run_shards.sh $NS "$GPUS" $R/lrsel/lr$LR.csv --ckpt $CKPT --split $SPLIT --role router_val --n-shifts 10 --lr $LR --no-features --seed 101
done
LR=$(venv/bin/python - $R <<'PY'
import sys, glob, json, pandas as pd
R = sys.argv[1]; res = {}
for f in sorted(glob.glob(f"{R}/lrsel/lr*.csv")):
    if "shard" in f: continue
    d = pd.read_csv(f); lr = f.split("lr")[-1][:-4]
    g = {k: float((d[f"q{k}"] - d.q0).mean()) for k in range(1, 6)}
    harm = {k: float((d[f"q{k}"] < d.q0).mean()) for k in range(1, 6)}
    res[lr] = {"mean_gain": g, "harm": harm, "best_fixed_gain": max(g.values()),
               "oracle_gain_k3": float(d[[f"q{k}" for k in range(4)]].max(1).sub(d.q0).mean())}
best = max(res, key=lambda l: res[l]["best_fixed_gain"])
json.dump({"selected_lr": best, "per_lr": res}, open(f"{R}/lr_selection.json", "w"), indent=1)
print(best)
PY
)
echo "selected TTA lr $LR"
# 2) router episodes
bash scripts/run_shards.sh $NS "$GPUS" $R/traj_router_train.csv --ckpt $CKPT --split $SPLIT --role router_train --n-shifts 40 --lr $LR --src-stats $R/src_stats.npz --seed 1
bash scripts/run_shards.sh $NS "$GPUS" $R/traj_router_val.csv --ckpt $CKPT --split $SPLIT --role router_val --n-shifts 20 --lr $LR --src-stats $R/src_stats.npz --seed 2
echo $LR > $R/lr.txt
echo SOURCE_PIPELINE_DONE
