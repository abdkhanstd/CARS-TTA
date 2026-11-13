#!/bin/bash
# usage: run_shards.sh <n_shards> <gpu_list e.g. "0 1"> <out.csv> <build_trajectories args...>
# Runs shards in parallel, round-robin over GPUs, then concatenates the CSVs.
N=$1; GPUS=($2); OUT=$3; shift 3
mkdir -p "$(dirname "$OUT")"
for ((i=0;i<N;i++)); do
  G=${GPUS[$((i % ${#GPUS[@]}))]}
  CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=4 venv/bin/python scripts/${BUILDER:-build_trajectories.py} "$@" --shard $i/$N --out "${OUT%.csv}_shard$i.csv" > "${OUT%.csv}_shard$i.log" 2>&1 &
done
wait
# retry shards that failed (e.g. CUDA OOM on the shared GPUs), one at a time, picking the freest GPU
for attempt in 1 2 3 4; do
  for ((i=0;i<N;i++)); do
    if [ ! -f "${OUT%.csv}_shard$i.csv" ]; then
      G=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1)
      echo "retry shard $i on GPU $G (attempt $attempt)"
      CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=8 venv/bin/python scripts/${BUILDER:-build_trajectories.py} "$@" --shard $i/$N --out "${OUT%.csv}_shard$i.csv" > "${OUT%.csv}_shard$i.log" 2>&1 || sleep 60
    fi
  done
done
venv/bin/python - "$OUT" "$N" <<'PY'
import sys, pandas as pd, json
out, n = sys.argv[1], int(sys.argv[2])
dfs = [pd.read_csv(out.replace('.csv', f'_shard{i}.csv')) for i in range(n)]
pd.concat(dfs, ignore_index=True).to_csv(out, index=False)
metas = [json.load(open(out.replace('.csv', f'_shard{i}_meta.json'))) for i in range(n)]
json.dump({"shards": metas, "peak_gpu_mb": max(m["peak_gpu_mb"] for m in metas)}, open(out.replace('.csv', '_meta.json'), 'w'), indent=1)
print("merged", out, sum(len(d) for d in dfs))
PY
