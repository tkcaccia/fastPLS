#!/usr/bin/env bash
set -euo pipefail

RUNNER="${RUNNER:-$HOME/fastPLS_cmpb_cycle79/benchmark_nmr_qualified_solver.R}"
PROTOCOL_HELPER="${PROTOCOL_HELPER:-$HOME/fastPLS_cmpb_cycle79/nmr_protocol_helpers.R}"
INPUT="${INPUT:-$HOME/Documents/fastpls/data/nmr.RData}"
OUT_ROOT="${OUT_ROOT:-/mnt/sata_ssd/fastPLS_cmpb_cycle79_nmr_qualified}"
REPLICATES="${REPLICATES:-3}"
OVERSAMPLE="${OVERSAMPLE:-20}"
POWER="${POWER:-2}"
SEED="${SEED:-123}"
TIMEOUT_SEC="${TIMEOUT_SEC:-7200}"

mkdir -p "${OUT_ROOT}"
test -f "${PROTOCOL_HELPER}"

run_one() {
  local family="$1"
  local backend="$2"
  local solver="$3"
  local ncomp="$4"
  local stem="${family}_${backend}_${solver}_k${ncomp}"

  /usr/bin/time -v timeout "${TIMEOUT_SEC}" Rscript "${RUNNER}" \
    --input="${INPUT}" \
    --output="${OUT_ROOT}/${stem}.csv" \
    --prediction_output="${OUT_ROOT}/${stem}_prediction.rds" \
    --family="${family}" \
    --backend="${backend}" \
    --solver="${solver}" \
    --ncomp="${ncomp}" \
    --oversample="${OVERSAMPLE}" \
    --power="${POWER}" \
    --seed="${SEED}" \
    --replicates="${REPLICATES}" \
    > "${OUT_ROOT}/${stem}.log" \
    2> "${OUT_ROOT}/${stem}.time"
}

for family_ncomp in "plssvd:5" "simpls:50"; do
  family="${family_ncomp%%:*}"
  ncomp="${family_ncomp##*:}"
  run_one "${family}" cpu irlba "${ncomp}"
  run_one "${family}" cpu rsvd "${ncomp}"
  run_one "${family}" cuda rsvd "${ncomp}"
done

printf '%s\n' "${OUT_ROOT}"
