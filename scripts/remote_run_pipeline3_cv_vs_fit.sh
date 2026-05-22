#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RESULTS_DIR="${FASTPLS_PIPELINE3_RESULTS_DIR:-${REPO_ROOT}/benchmark_results_pipeline3_cv_vs_fit}"
LIB_LOC="${FASTPLS_BENCH_LIB:-}"
DATASETS="${FASTPLS_PIPELINE3_DATASETS:-metref,ccle,cifar100,prism,gtex_v8,tcga_pan_cancer,singlecell,tcga_brca,tcga_hnsc_methylation,nmr,cbmc_citeseq}"
if [ "${FASTPLS_PIPELINE3_INCLUDE_IMAGENET:-false}" = "true" ]; then
  DATASETS="${DATASETS},imagenet"
fi
REPS="${FASTPLS_PIPELINE3_REPS:-1}"
KFOLD="${FASTPLS_PIPELINE3_KFOLD:-10}"
TIMEOUT_SEC="${FASTPLS_PIPELINE3_TIMEOUT_SEC:-3600}"
TIME_BIN="${TIME_BIN:-/usr/bin/time}"
TIMEOUT_BIN="${TIMEOUT_BIN:-timeout}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
METHOD_FILTER="${FASTPLS_PIPELINE3_METHODS:-}"
MODE_FILTER="${FASTPLS_PIPELINE3_BENCHMARK_MODES:-fit_predict,cv10}"

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
    metref) echo "${FASTPLS_PIPELINE3_METREF_NCOMP:-22}" ;;
    cbmc_citeseq) echo "${FASTPLS_PIPELINE3_CBMC_CITESEQ_NCOMP:-50}" ;;
    ccle) echo "${FASTPLS_PIPELINE3_CCLE_NCOMP:-50}" ;;
    cifar100) echo "${FASTPLS_PIPELINE3_CIFAR100_NCOMP:-100}" ;;
    gtex_v8) echo "${FASTPLS_PIPELINE3_GTEX_V8_NCOMP:-32}" ;;
    imagenet) echo "${FASTPLS_PIPELINE3_IMAGENET_NCOMP:-100}" ;;
    nmr) echo "${FASTPLS_PIPELINE3_NMR_NCOMP:-50}" ;;
    prism) echo "${FASTPLS_PIPELINE3_PRISM_NCOMP:-5}" ;;
    singlecell) echo "${FASTPLS_PIPELINE3_SINGLECELL_NCOMP:-50}" ;;
    tcga_brca) echo "${FASTPLS_PIPELINE3_TCGA_BRCA_NCOMP:-5}" ;;
    tcga_hnsc_methylation) echo "${FASTPLS_PIPELINE3_TCGA_HNSC_METHYLATION_NCOMP:-2}" ;;
    tcga_pan_cancer) echo "${FASTPLS_PIPELINE3_TCGA_PAN_CANCER_NCOMP:-50}" ;;
    *) echo "${FASTPLS_PIPELINE3_NCOMP:-50}" ;;
  esac
}

selected() {
  local value="$1"
  local filter="$2"
  if [ -z "${filter}" ]; then
    return 0
  fi
  printf '%s' "${filter}" | tr ',' '\n' | grep -qx "${value}"
}

{
  echo "Pipeline 3 started: $(date)"
  echo "Repo: ${REPO_ROOT}"
  echo "Results: ${RESULTS_DIR}"
  echo "Datasets: ${DATASETS}"
  echo "Modes: ${MODE_FILTER}"
  echo "kfold: ${KFOLD}"
  echo "timeout_sec: ${TIMEOUT_SEC}"
  echo "FASTPLS_BENCH_LIB: ${LIB_LOC:-<default>}"
} | tee "${RESULTS_DIR}/launch.log"

for dataset in $(printf '%s' "${DATASETS}" | tr ',' ' '); do
  ncomp="$(ncomp_for_dataset "${dataset}")"
  methods_file="${RESULTS_DIR}/logs/${dataset}_methods.txt"
  Rscript "${REPO_ROOT}/benchmark/benchmark_pipeline3_cv_vs_fit.R" \
    --mode=list_methods \
    --dataset="${dataset}" \
    --ncomp="${ncomp}" >"${methods_file}"

  while IFS= read -r method_id; do
    [ -n "${method_id}" ] || continue
    if ! selected "${method_id}" "${METHOD_FILTER}"; then
      continue
    fi
    for benchmark_mode in $(printf '%s' "${MODE_FILTER}" | tr ',' ' '); do
      rep_id=1
      while [ "${rep_id}" -le "${REPS}" ]; do
        run_id="${dataset}__${benchmark_mode}__${method_id}__n${ncomp}__rep${rep_id}"
        row_csv="${RESULTS_DIR}/run_rows/${run_id}.csv"
        stdout_log="${RESULTS_DIR}/logs/${run_id}.stdout.log"
        time_log="${RESULTS_DIR}/logs/${run_id}.time.log"
        rm -f "${row_csv}" "${stdout_log}" "${time_log}"
        echo "[RUN] ${run_id}"
        set +e
        "${TIME_BIN}" -v "${TIMEOUT_BIN}" --signal=TERM --kill-after=30s "${TIMEOUT_SEC}" \
          Rscript "${REPO_ROOT}/benchmark/benchmark_pipeline3_cv_vs_fit.R" \
            --mode=run_one \
            --dataset="${dataset}" \
            --ncomp="${ncomp}" \
            --method-id="${method_id}" \
            --benchmark-mode="${benchmark_mode}" \
            --kfold="${KFOLD}" \
            --replicate="${rep_id}" \
            --row-out="${row_csv}" >"${stdout_log}" 2>"${time_log}"
        status=$?
        set -e
        if [ ! -s "${row_csv}" ]; then
          msg="Pipeline 3 run did not produce a row"
          row_status="missing_row"
          if [ "${status}" -eq 124 ]; then
            row_status="killed_timeout"
            msg="Pipeline 3 run exceeded timeout"
          elif grep -q 'Command terminated by signal 9' "${time_log}" 2>/dev/null; then
            row_status="killed_sig9"
            msg="Pipeline 3 run terminated by signal 9"
          fi
          Rscript "${REPO_ROOT}/benchmark/benchmark_pipeline3_cv_vs_fit.R" \
            --mode=missing_row \
            --dataset="${dataset}" \
            --ncomp="${ncomp}" \
            --method-id="${method_id}" \
            --benchmark-mode="${benchmark_mode}" \
            --kfold="${KFOLD}" \
            --replicate="${rep_id}" \
            --status="${row_status}" \
            --message="${msg}" \
            --row-out="${row_csv}" >>"${stdout_log}" 2>>"${time_log}" || true
        fi
        rep_id=$((rep_id + 1))
      done
    done
  done <"${methods_file}"
done

append_rows "${RESULTS_DIR}/pipeline3_cv_vs_fit_raw.csv" "${RESULTS_DIR}"/run_rows/*.csv
Rscript "${REPO_ROOT}/benchmark/benchmark_pipeline3_cv_vs_fit.R" \
  --mode=summarize \
  --results-dir="${RESULTS_DIR}"

echo "Pipeline 3 finished: $(date)" | tee -a "${RESULTS_DIR}/launch.log"
echo "[INFO] Results written to ${RESULTS_DIR}"
