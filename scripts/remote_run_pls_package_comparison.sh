#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RESULTS_DIR="${FASTPLS_PKG_COMPARE_RESULTS_DIR:-${REPO_ROOT}/benchmark_results_pls_package_comparison}"
LIB_LOC="${FASTPLS_BENCH_LIB:-}"

# Real-dataset package comparison.  ImageNet is opt-in because several
# independent R PLS packages materialize dense workspaces and are not practical
# on that dataset; set FASTPLS_PKG_COMPARE_INCLUDE_IMAGENET=true to include it.
DATASETS="${FASTPLS_PKG_COMPARE_DATASETS:-metref,ccle,cifar100,prism,gtex_v8,tcga_pan_cancer,singlecell,tcga_brca,tcga_hnsc_methylation,nmr,cbmc_citeseq}"
if [ "${FASTPLS_PKG_COMPARE_INCLUDE_IMAGENET:-false}" = "true" ]; then
  DATASETS="${DATASETS},imagenet"
fi

REPS="${FASTPLS_PKG_COMPARE_REPS:-1}"
TIMEOUT_SEC="${FASTPLS_PKG_COMPARE_TIMEOUT_SEC:-1200}"
TIME_BIN="${TIME_BIN:-/usr/bin/time}"
TIMEOUT_BIN="${TIMEOUT_BIN:-timeout}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
METHOD_FILTER="${FASTPLS_PKG_COMPARE_METHODS:-}"

mkdir -p "${RESULTS_DIR}/run_rows" "${RESULTS_DIR}/logs"

if [ -n "${LIB_LOC}" ]; then
  export R_LIBS_USER="${LIB_LOC}${R_LIBS_USER:+:${R_LIBS_USER}}"
fi

append_rows() {
  local raw_csv="$1"
  shift
  "${PYTHON_BIN}" - "$raw_csv" "$@" <<'PY'
import csv, os, sys
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
fieldnames = list(rows[0].keys())
with open(raw, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fieldnames)
    w.writeheader()
    for row in rows:
        w.writerow(row)
PY
}

ncomp_for_dataset() {
  case "$1" in
    metref) echo "${FASTPLS_PKG_COMPARE_METREF_NCOMP:-22}" ;;
    cbmc_citeseq) echo "${FASTPLS_PKG_COMPARE_CBMC_CITESEQ_NCOMP:-50}" ;;
    ccle) echo "${FASTPLS_PKG_COMPARE_CCLE_NCOMP:-50}" ;;
    cifar100) echo "${FASTPLS_PKG_COMPARE_CIFAR100_NCOMP:-100}" ;;
    gtex_v8) echo "${FASTPLS_PKG_COMPARE_GTEX_V8_NCOMP:-32}" ;;
    imagenet) echo "${FASTPLS_PKG_COMPARE_IMAGENET_NCOMP:-100}" ;;
    nmr) echo "${FASTPLS_PKG_COMPARE_NMR_NCOMP:-50}" ;;
    prism) echo "${FASTPLS_PKG_COMPARE_PRISM_NCOMP:-5}" ;;
    singlecell) echo "${FASTPLS_PKG_COMPARE_SINGLECELL_NCOMP:-50}" ;;
    tcga_brca) echo "${FASTPLS_PKG_COMPARE_TCGA_BRCA_NCOMP:-5}" ;;
    tcga_hnsc_methylation) echo "${FASTPLS_PKG_COMPARE_TCGA_HNSC_METHYLATION_NCOMP:-2}" ;;
    tcga_pan_cancer) echo "${FASTPLS_PKG_COMPARE_TCGA_PAN_CANCER_NCOMP:-50}" ;;
    *) echo "${FASTPLS_PKG_COMPARE_NCOMP:-50}" ;;
  esac
}

method_selected() {
  method_id="$1"
  if [ -z "${METHOD_FILTER}" ]; then
    return 0
  fi
  printf '%s' "${METHOD_FILTER}" | tr ',' '\n' | grep -qx "${method_id}"
}

for dataset in $(printf '%s' "${DATASETS}" | tr ',' ' '); do
  ncomp="$(ncomp_for_dataset "${dataset}")"

  methods_file="${RESULTS_DIR}/logs/${dataset}_methods.txt"
  Rscript "${REPO_ROOT}/benchmark/benchmark_pls_package_comparison.R" \
    --mode=list_methods --dataset="${dataset}" --ncomp="${ncomp}" >"${methods_file}"

  while IFS= read -r method_id; do
    [ -n "${method_id}" ] || continue
    if ! method_selected "${method_id}"; then
      continue
    fi
    rep_id=1
    while [ "${rep_id}" -le "${REPS}" ]; do
      run_id="${dataset}__${method_id}__n${ncomp}__rep${rep_id}"
      row_csv="${RESULTS_DIR}/run_rows/${run_id}.csv"
      stdout_log="${RESULTS_DIR}/logs/${run_id}.stdout.log"
      time_log="${RESULTS_DIR}/logs/${run_id}.time.log"
      rm -f "${row_csv}" "${stdout_log}" "${time_log}"
      echo "[RUN] ${run_id}"
      set +e
      "${TIME_BIN}" -v "${TIMEOUT_BIN}" --signal=TERM --kill-after=30s "${TIMEOUT_SEC}" \
        Rscript "${REPO_ROOT}/benchmark/benchmark_pls_package_comparison.R" \
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
        Rscript "${REPO_ROOT}/benchmark/benchmark_pls_package_comparison.R" \
          --mode=missing_row \
          --dataset="${dataset}" \
          --ncomp="${ncomp}" \
          --method-id="${method_id}" \
          --replicate="${rep_id}" \
          --status="${row_status}" \
          --message="${msg}" \
          --row-out="${row_csv}" >>"${stdout_log}" 2>>"${time_log}" || true
      fi
      rep_id=$((rep_id + 1))
    done
  done <"${methods_file}"
done

append_rows "${RESULTS_DIR}/pls_package_comparison_raw.csv" "${RESULTS_DIR}"/run_rows/*.csv
Rscript "${REPO_ROOT}/benchmark/benchmark_pls_package_comparison.R" \
  --mode=summarize \
  --results-dir="${RESULTS_DIR}"

echo "[INFO] Results written to ${RESULTS_DIR}"
