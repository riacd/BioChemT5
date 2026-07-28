#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/shared-storage-gpfs2/veus/migration_reaction_package_20260529
PY=/mnt/shared-storage-user/tanpan/miniconda3/envs/vplm/bin/python
if [[ ! -x "$PY" ]]; then
  PY=/mnt/shared-storage-gpfs2/veus/miniconda3/envs/vplm/bin/python
fi
SCRIPT="$ROOT/code_docs/BiochemT5_pretrain/biochem_t5/data/finalize_unique_rsmiles.py"
OUT_DIR="$ROOT/data/BiochemT5/pretrain_corpus"

"$PY" "$SCRIPT" \
  --input-jsonl "$OUT_DIR/cluster_centers.rsmiles_aug20.centered.jsonl" \
  --out-jsonl "$OUT_DIR/cluster_centers.rsmiles_unique20.centered.jsonl" \
  --out-stats "$OUT_DIR/cluster_centers.rsmiles_unique20.centered.stats.json" \
  --max-views 20 \
  --progress-every 100000
