#!/usr/bin/env bash

set -euo pipefail

# Reproduce the real-dataset package comparison used on chiamaka:
# - excludes NMR and imagenet
# - tests all external PLS packages listed by benchmark_pls_package_comparison.R
# - restricts fastPLS to CPU rSVD and CUDA rSVD variants
# - keeps one replicate and a 3600 second timeout per method by default

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_ROOT="${FASTPLS_REAL_RSVD_COMPARE_RUN_ROOT:-${HOME}/fastPLS_pkg_compare_real_rsvd_only}"
RESULTS_DIR="${FASTPLS_REAL_RSVD_COMPARE_RESULTS_DIR:-${RUN_ROOT}/results}"
REPO="${FASTPLS_REAL_RSVD_COMPARE_REPO:-${REPO_ROOT}}"
TIMEOUT_SEC="${FASTPLS_REAL_RSVD_COMPARE_TIMEOUT_SEC:-3600}"
REPS="${FASTPLS_REAL_RSVD_COMPARE_REPS:-1}"
TIME_BIN="${TIME_BIN:-/usr/bin/time}"
TIMEOUT_BIN="${TIMEOUT_BIN:-timeout}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "${RESULTS_DIR}/run_rows" "${RESULTS_DIR}/logs"

export FASTPLS_INSTALL_MISSING="${FASTPLS_INSTALL_MISSING:-true}"

# These defaults match the data layout on chiamaka but can be overridden by
# exporting the same variables before launching the script.
export FASTPLS_CBMC_CITESEQ_RDATA="${FASTPLS_CBMC_CITESEQ_RDATA:-${HOME}/Documents/fastpls/data/cbmc_citeseq.RData}"
export FASTPLS_CCLE_RDATA="${FASTPLS_CCLE_RDATA:-${HOME}/Documents/fastpls/data/ccle.RData}"
export FASTPLS_CIFAR100_RDATA="${FASTPLS_CIFAR100_RDATA:-${HOME}/Documents/fastpls/data/CIFAR100.RData}"
export FASTPLS_GTEX_V8_RDATA="${FASTPLS_GTEX_V8_RDATA:-${HOME}/Documents/fastpls/data/gtex_v8.RData}"
export FASTPLS_METREF_RDATA="${FASTPLS_METREF_RDATA:-${HOME}/Documents/fastpls/data/metref.RData}"
export FASTPLS_PRISM_RDATA="${FASTPLS_PRISM_RDATA:-${HOME}/Documents/fastpls/data/prism.RData}"
export FASTPLS_SINGLECELL_RDATA="${FASTPLS_SINGLECELL_RDATA:-${HOME}/Documents/fastpls/data/singlecell.RData}"
export FASTPLS_TCGA_BRCA_RDATA="${FASTPLS_TCGA_BRCA_RDATA:-${HOME}/Documents/fastpls/data/tcga_brca.RData}"
export FASTPLS_TCGA_HNSC_METHYLATION_RDATA="${FASTPLS_TCGA_HNSC_METHYLATION_RDATA:-${HOME}/Documents/fastpls/data/tcga_hnsc_methylation.RData}"
export FASTPLS_TCGA_PAN_CANCER_RDATA="${FASTPLS_TCGA_PAN_CANCER_RDATA:-${HOME}/Documents/fastpls/data/tcga_pan_cancer.RData}"

DATASETS="${FASTPLS_REAL_RSVD_COMPARE_DATASETS:-cbmc_citeseq ccle cifar100 gtex_v8 metref prism singlecell tcga_brca tcga_hnsc_methylation tcga_pan_cancer}"

dataset_ncomp() {
  case "$1" in
    cbmc_citeseq) echo "${FASTPLS_REAL_RSVD_NCOMP_CBMC_CITESEQ:-50}" ;;
    ccle) echo "${FASTPLS_REAL_RSVD_NCOMP_CCLE:-50}" ;;
    cifar100) echo "${FASTPLS_REAL_RSVD_NCOMP_CIFAR100:-100}" ;;
    gtex_v8) echo "${FASTPLS_REAL_RSVD_NCOMP_GTEX_V8:-32}" ;;
    metref) echo "${FASTPLS_REAL_RSVD_NCOMP_METREF:-22}" ;;
    prism) echo "${FASTPLS_REAL_RSVD_NCOMP_PRISM:-5}" ;;
    singlecell) echo "${FASTPLS_REAL_RSVD_NCOMP_SINGLECELL:-50}" ;;
    tcga_brca) echo "${FASTPLS_REAL_RSVD_NCOMP_TCGA_BRCA:-5}" ;;
    tcga_hnsc_methylation) echo "${FASTPLS_REAL_RSVD_NCOMP_TCGA_HNSC_METHYLATION:-2}" ;;
    tcga_pan_cancer) echo "${FASTPLS_REAL_RSVD_NCOMP_TCGA_PAN_CANCER:-50}" ;;
    *) echo "${FASTPLS_REAL_RSVD_NCOMP_DEFAULT:-50}" ;;
  esac
}

include_method() {
  local method_id="$1"
  case "${method_id}" in
    fastPLS_*_cpp_irlba*) return 1 ;;
    fastPLS_*_cpp_cpu_rsvd*) return 0 ;;
    fastPLS_*_cuda_cuda_rsvd*) return 0 ;;
    fastPLS_*) return 1 ;;
    *) return 0 ;;
  esac
}

append_rows() {
  local raw_csv="$1"
  shift
  "${PYTHON_BIN}" - "${raw_csv}" "$@" <<'PY'
import csv
import os
import sys

raw = sys.argv[1]
files = sys.argv[2:]
rows = []
for path in files:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        continue
    with open(path, newline="") as fh:
        rows.extend(list(csv.DictReader(fh)))
if not rows:
    raise SystemExit(0)
with open(raw, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
PY
}

run_one_method() {
  local dataset="$1"
  local ncomp="$2"
  local method_id="$3"
  local rep_id="$4"
  local run_id="${dataset}__${method_id}__n${ncomp}__rep${rep_id}"
  local row_csv="${RESULTS_DIR}/run_rows/${run_id}.csv"
  local stdout_log="${RESULTS_DIR}/logs/${run_id}.stdout.log"
  local time_log="${RESULTS_DIR}/logs/${run_id}.time.log"

  if [ -s "${row_csv}" ]; then
    echo "[DONE] ${run_id}"
    return 0
  fi

  echo "[RUN] ${run_id}"
  rm -f "${row_csv}" "${stdout_log}" "${time_log}"
  set +e
  "${TIME_BIN}" -v "${TIMEOUT_BIN}" --signal=TERM --kill-after=30s "${TIMEOUT_SEC}" \
    Rscript "${REPO}/benchmark/benchmark_pls_package_comparison.R" \
      --mode=run_one \
      --dataset="${dataset}" \
      --ncomp="${ncomp}" \
      --method-id="${method_id}" \
      --replicate="${rep_id}" \
      --row-out="${row_csv}" >"${stdout_log}" 2>"${time_log}"
  status=$?
  set -e

  if [ ! -s "${row_csv}" ]; then
    msg="Rscript did not produce a row"
    row_status="missing_row"
    if [ "${status}" -eq 124 ]; then
      row_status="killed_timeout"
      msg="Package comparison run exceeded timeout"
    elif grep -q 'Command terminated by signal 9' "${time_log}" 2>/dev/null; then
      row_status="killed_sig9"
      msg="Package comparison run terminated by signal 9"
    fi
    Rscript "${REPO}/benchmark/benchmark_pls_package_comparison.R" \
      --mode=missing_row \
      --dataset="${dataset}" \
      --ncomp="${ncomp}" \
      --method-id="${method_id}" \
      --replicate="${rep_id}" \
      --status="${row_status}" \
      --message="${msg}" \
      --row-out="${row_csv}" >>"${stdout_log}" 2>>"${time_log}" || true
  fi
}

echo "[INFO] RUN_ROOT=${RUN_ROOT}"
echo "[INFO] REPO=${REPO}"
echo "[INFO] RESULTS_DIR=${RESULTS_DIR}"
echo "[INFO] TIMEOUT_SEC=${TIMEOUT_SEC} REPS=${REPS}"

for dataset in ${DATASETS}; do
  ncomp="$(dataset_ncomp "${dataset}")"
  all_methods="${RESULTS_DIR}/logs/${dataset}_methods_all.txt"
  methods_file="${RESULTS_DIR}/logs/${dataset}_methods.txt"

  Rscript "${REPO}/benchmark/benchmark_pls_package_comparison.R" \
    --mode=list_methods \
    --dataset="${dataset}" \
    --ncomp="${ncomp}" >"${all_methods}"

  : >"${methods_file}"
  while IFS= read -r method_id; do
    [ -n "${method_id}" ] || continue
    if include_method "${method_id}"; then
      echo "${method_id}" >>"${methods_file}"
    else
      echo "[FILTER] ${dataset} ${method_id}" >>"${RESULTS_DIR}/logs/filtered_methods.log"
    fi
  done <"${all_methods}"

  echo "[DATASET] ${dataset} ncomp=${ncomp} methods=$(wc -l <"${methods_file}")"
  rep_id=1
  while [ "${rep_id}" -le "${REPS}" ]; do
    while IFS= read -r method_id; do
      [ -n "${method_id}" ] || continue
      run_one_method "${dataset}" "${ncomp}" "${method_id}" "${rep_id}"
    done <"${methods_file}"
    rep_id=$((rep_id + 1))
  done

  append_rows "${RESULTS_DIR}/pls_package_comparison_raw.csv" "${RESULTS_DIR}"/run_rows/*.csv
  Rscript "${REPO}/benchmark/benchmark_pls_package_comparison.R" \
    --mode=summarize \
    --results-dir="${RESULTS_DIR}"
  echo "[DATASET_DONE] ${dataset}"
done

echo "[INFO] All requested real datasets complete: ${RESULTS_DIR}"
