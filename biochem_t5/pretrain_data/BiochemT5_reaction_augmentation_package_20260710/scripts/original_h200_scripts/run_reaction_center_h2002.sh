#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/shared-storage-gpfs2/veus/migration_reaction_package_20260529
PY=/mnt/shared-storage-user/tanpan/miniconda3/envs/vplm/bin/python
if [[ ! -x "$PY" ]]; then
  PY=/mnt/shared-storage-gpfs2/veus/miniconda3/envs/vplm/bin/python
fi
SCRIPT="$ROOT/code_docs/BiochemT5_pretrain/biochem_t5/data/reaction_center.py"

OUT_DIR="$ROOT/data/BiochemT5/pretrain_corpus"
INPUT_JSONL="$OUT_DIR/cluster_centers.rsmiles_aug20.jsonl"
OUT_JSONL="$OUT_DIR/cluster_centers.rsmiles_aug20.centered.jsonl"
OUT_STATS="$OUT_DIR/cluster_centers.rsmiles_aug20.centered.stats.json"

mkdir -p "$OUT_DIR" "$ROOT/logs"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

"$PY" "$SCRIPT" \
  --input-jsonl "$INPUT_JSONL" \
  --out-jsonl "$OUT_JSONL" \
  --out-stats "$OUT_STATS" \
  --workers 120 \
  --chunk-size 512 \
  --resume \
  --progress-every 100000
