#!/usr/bin/env bash
set -u

ROOT="${FASTPLS_FROZEN_ROOT:-/home/chiamaka/fastPLS_frozen_0.99.25}"
SRC="$ROOT/src"
REPO="$ROOT/analysis_repo"
RESULTS="$ROOT/results"
SHA="604f74d72f5e9540f7378efd37f2ca6ed87ccb9e73b912211331a8cd97233481"
NMR_PID="${NMR_PID:-480377}"
STATUS="$RESULTS/queue_status.tsv"
printf 'stage\tstarted\tfinished\tstatus\n' > "$STATUS"

run_stage() {
  local name="$1"; shift
  local started finished status
  started="$(date -Iseconds)"
  if "$@"; then status="success"; else status="failed"; fi
  finished="$(date -Iseconds)"
  printf '%s\t%s\t%s\t%s\n' "$name" "$started" "$finished" "$status" >> "$STATUS"
}

while kill -0 "$NMR_PID" 2>/dev/null; do sleep 30; done
if [[ -f "$RESULTS/nmr/nmr_all_runs.csv" ]]; then
  printf 'nmr\tunknown\t%s\tsuccess\n' "$(date -Iseconds)" >> "$STATUS"
else
  printf 'nmr\tunknown\t%s\tfailed\n' "$(date -Iseconds)" >> "$STATUS"
fi

run_stage rsvd_cuda env \
  FASTPLS_SOURCE_ARCHIVE_SHA256="$SHA" \
  Rscript "$SRC/benchmark_rsvd_cuda_reliability.R" \
    --lib="$ROOT/lib" \
    --out="$RESULTS/rsvd_cuda_reliability_final" \
    --seeds=1,7,19,43,123

run_stage selected_backend bash "$SRC/run_selected_backend_frozen.sh"

run_stage controlled_scaling env \
  FASTPLS_SCALING_SKIP_INSTALL=true \
  FASTPLS_SCALING_LIB="$ROOT/lib" \
  FASTPLS_SCALING_ARCHIVE="$SRC/fastPLS_0.99.25.tar.gz" \
  FASTPLS_SOURCE_ARCHIVE_SHA256="$SHA" \
  FASTPLS_SCALING_TIMEOUT_SEC=600 \
  bash "$REPO/scripts/run_controlled_scaling.sh" \
    "$RESULTS/controlled_scaling" publication cpu,cuda 3

run_stage external_simpls env \
  FASTPLS_BENCH_LIB="$ROOT/lib" \
  R_LIBS_USER="$ROOT/lib" \
  FASTPLS_SOURCE_ARCHIVE_SHA256="$SHA" \
  FASTPLS_EXTERNAL_TIMING_RESULTS_DIR="$RESULTS/external_simpls" \
  FASTPLS_EXTERNAL_TIMING_REPS=3 \
  bash "$REPO/scripts/run_external_simpls_timing.sh"

run_stage imagenet bash "$SRC/run_imagenet_frozen.sh"
