#!/usr/bin/env bash
set -u

ROOT="${FASTPLS_FROZEN_ROOT:-/home/chiamaka/fastPLS_frozen_0.99.25}"
QUEUE_PID="${QUEUE_PID:-482937}"
OUT="$ROOT/results/controlled_scaling_frozen09925"
LOG="$ROOT/results/controlled_scaling_frozen09925_queue.log"

while kill -0 "$QUEUE_PID" 2>/dev/null; do sleep 30; done

FASTPLS_SCALING_SKIP_INSTALL=true \
FASTPLS_SCALING_LIB="$ROOT/lib" \
FASTPLS_SCALING_EXPECTED_VERSION=0.99.25 \
FASTPLS_SCALING_ARCHIVE="$ROOT/src/fastPLS_0.99.25.tar.gz" \
FASTPLS_SOURCE_ARCHIVE_SHA256=604f74d72f5e9540f7378efd37f2ca6ed87ccb9e73b912211331a8cd97233481 \
FASTPLS_SCALING_TIMEOUT_SEC=600 \
bash "$ROOT/analysis_repo/scripts/run_controlled_scaling.sh" \
  "$OUT" publication cpu,cuda 3 > "$LOG" 2>&1
