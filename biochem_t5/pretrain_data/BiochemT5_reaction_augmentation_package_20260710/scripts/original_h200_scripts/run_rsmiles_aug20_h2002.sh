#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/shared-storage-gpfs2/veus/migration_reaction_package_20260529
PY=/mnt/shared-storage-user/tanpan/miniconda3/envs/vplm/bin/python
if [[ ! -x "$PY" ]]; then
  PY=/mnt/shared-storage-gpfs2/veus/miniconda3/envs/vplm/bin/python
fi
SCRIPT="$ROOT/code_docs/BiochemT5_pretrain/biochem_t5/data/rsmiles_augmentation.py"

INPUT_JSONL="$ROOT/data/Reaction_gen/results/screen_then_dedup_merged/r2_non_pubchem_plus_r2_incremental_20260513_143420/stage01_finalbio_dgbyg_api_equivalent_enzymemap_merged/reactions_finalbio.dedup_by_rxn_ec.dgbyg_api_equivalent.enzymemap_merged.dgr_negative.jsonl"
ASSIGNMENTS="$ROOT/data/Reaction_gen/results/screen_then_dedup_merged/r2_non_pubchem_plus_r2_incremental_20260513_143420/stage01_finalbio_dgbyg_api_equivalent_enzymemap_merged/drfp_template_clusters/full_sim095_r3_rings_official_complete_combined_fast5_h2002/final_assignments.sim095_r3_rings_official_complete_combined_fast5.tsv.gz"
OUT_DIR="$ROOT/data/BiochemT5/pretrain_corpus"
RSMILES_REPO="$ROOT/Rsmiles-main"

CENTERS_JSONL="$OUT_DIR/cluster_centers.sim095_r3_complete.jsonl"
CENTERS_STATS="$OUT_DIR/cluster_centers.sim095_r3_complete.stats.json"
AUG_JSONL="$OUT_DIR/cluster_centers.rsmiles_aug20.jsonl"
AUG_STATS="$OUT_DIR/cluster_centers.rsmiles_aug20.stats.json"

mkdir -p "$OUT_DIR" "$ROOT/logs"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="$ROOT/.hf_cache"

if [[ ! -s "$CENTERS_JSONL" ]]; then
  "$PY" "$SCRIPT" build-centers \
    --input-jsonl "$INPUT_JSONL" \
    --assignments-tsv-gz "$ASSIGNMENTS" \
    --out-jsonl "$CENTERS_JSONL" \
    --out-stats "$CENTERS_STATS" \
    --progress-every 1000000
fi

"$PY" "$SCRIPT" augment \
  --input-jsonl "$CENTERS_JSONL" \
  --out-jsonl "$AUG_JSONL" \
  --out-stats "$AUG_STATS" \
  --rsmiles-repo "$RSMILES_REPO" \
  --augmentation 20 \
  --workers 120 \
  --chunk-size 64 \
  --map-batch-size 16 \
  --resume \
  --progress-every 1000
