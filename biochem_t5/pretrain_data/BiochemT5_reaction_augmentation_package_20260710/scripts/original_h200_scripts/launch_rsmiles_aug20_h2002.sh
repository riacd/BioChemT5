#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/shared-storage-gpfs2/veus/migration_reaction_package_20260529
RUN_SCRIPT="$ROOT/code_docs/BiochemT5_pretrain/run_rsmiles_aug20_h2002.sh"
LOG="$ROOT/logs/biochem_t5_rsmiles_aug20_h2002.log"
PID_FILE="$ROOT/logs/biochem_t5_rsmiles_aug20_h2002.pid"

mkdir -p "$ROOT/logs"
nohup "$RUN_SCRIPT" > "$LOG" 2>&1 &
pid=$!
printf "%s\n" "$pid" > "$PID_FILE"
printf "PID:%s\nLOG:%s\nPID_FILE:%s\n" "$pid" "$LOG" "$PID_FILE"
