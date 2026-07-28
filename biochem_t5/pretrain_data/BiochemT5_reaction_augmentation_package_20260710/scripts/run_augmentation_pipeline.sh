#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-python}"

INPUT_JSONL="${INPUT_JSONL:?please export INPUT_JSONL=/path/to/reactions.jsonl}"
ASSIGNMENTS_TSV_GZ="${ASSIGNMENTS_TSV_GZ:?please export ASSIGNMENTS_TSV_GZ=/path/to/final_assignments.tsv.gz}"
OUT_DIR="${OUT_DIR:-$PKG_ROOT/output/pretrain_corpus}"
RSMILES_REPO="${RSMILES_REPO:-$PKG_ROOT/third_party/Rsmiles-main}"

AUGMENTATION="${AUGMENTATION:-20}"
WORKERS="${WORKERS:-120}"
CHUNK_SIZE="${CHUNK_SIZE:-64}"
MAP_BATCH_SIZE="${MAP_BATCH_SIZE:-16}"
CENTER_WORKERS="${CENTER_WORKERS:-$WORKERS}"
CENTER_CHUNK_SIZE="${CENTER_CHUNK_SIZE:-512}"
PROGRESS_EVERY="${PROGRESS_EVERY:-100000}"
SEED="${SEED:-20260615}"

mkdir -p "$OUT_DIR"

export PYTHONPATH="$PKG_ROOT/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false

CENTERS_JSONL="$OUT_DIR/cluster_centers.sim095_r3_complete.jsonl"
CENTERS_STATS="$OUT_DIR/cluster_centers.sim095_r3_complete.stats.json"
AUG_JSONL="$OUT_DIR/cluster_centers.rsmiles_aug20.jsonl"
AUG_STATS="$OUT_DIR/cluster_centers.rsmiles_aug20.stats.json"
CENTERED_JSONL="$OUT_DIR/cluster_centers.rsmiles_aug20.centered.jsonl"
CENTERED_STATS="$OUT_DIR/cluster_centers.rsmiles_aug20.centered.stats.json"
FINAL_JSONL="$OUT_DIR/cluster_centers.rsmiles_unique20.centered.jsonl"
FINAL_STATS="$OUT_DIR/cluster_centers.rsmiles_unique20.centered.stats.json"
VALIDATION_JSON="$OUT_DIR/cluster_centers.rsmiles_unique20.centered.validation.json"

if [[ ! -s "$CENTERS_JSONL" ]]; then
  "$PYTHON" "$PKG_ROOT/src/biochem_t5/data/rsmiles_augmentation.py" build-centers \
    --input-jsonl "$INPUT_JSONL" \
    --assignments-tsv-gz "$ASSIGNMENTS_TSV_GZ" \
    --out-jsonl "$CENTERS_JSONL" \
    --out-stats "$CENTERS_STATS" \
    --progress-every 1000000
fi

"$PYTHON" "$PKG_ROOT/src/biochem_t5/data/rsmiles_augmentation.py" augment \
  --input-jsonl "$CENTERS_JSONL" \
  --out-jsonl "$AUG_JSONL" \
  --out-stats "$AUG_STATS" \
  --rsmiles-repo "$RSMILES_REPO" \
  --augmentation "$AUGMENTATION" \
  --workers "$WORKERS" \
  --chunk-size "$CHUNK_SIZE" \
  --map-batch-size "$MAP_BATCH_SIZE" \
  --seed "$SEED" \
  --resume \
  --progress-every 1000

"$PYTHON" "$PKG_ROOT/src/biochem_t5/data/reaction_center.py" \
  --input-jsonl "$AUG_JSONL" \
  --out-jsonl "$CENTERED_JSONL" \
  --out-stats "$CENTERED_STATS" \
  --workers "$CENTER_WORKERS" \
  --chunk-size "$CENTER_CHUNK_SIZE" \
  --resume \
  --progress-every "$PROGRESS_EVERY"

"$PYTHON" "$PKG_ROOT/src/biochem_t5/data/finalize_unique_rsmiles.py" \
  --input-jsonl "$CENTERED_JSONL" \
  --out-jsonl "$FINAL_JSONL" \
  --out-stats "$FINAL_STATS" \
  --max-views 20 \
  --progress-every "$PROGRESS_EVERY"

"$PYTHON" "$PKG_ROOT/src/biochem_t5/data/validate_pretrain_corpus.py" \
  --input-jsonl "$FINAL_JSONL" \
  --out-json "$VALIDATION_JSON"

printf "DONE\nFINAL_JSONL=%s\nFINAL_STATS=%s\nVALIDATION_JSON=%s\n" "$FINAL_JSONL" "$FINAL_STATS" "$VALIDATION_JSON"
