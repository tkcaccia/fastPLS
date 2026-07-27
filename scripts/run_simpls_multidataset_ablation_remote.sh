#!/usr/bin/env bash

set -euo pipefail

RESULTS_DIR="${1:-$HOME/fastPLS_simpls_multidataset_ablation}"
REPO_ROOT="${FASTPLS_REPO_ROOT:-$HOME/fastPLS_ablation_source}"
REPS="${FASTPLS_ABLATION_REPS:-3}"
TIMEOUT_SEC="${FASTPLS_ABLATION_TIMEOUT_SEC:-3600}"
DATASET_FILTER=",${FASTPLS_ABLATION_DATASETS:-all},"

mkdir -p "${RESULTS_DIR}/rows" "${RESULTS_DIR}/predictions" \
  "${RESULTS_DIR}/logs" "${RESULTS_DIR}/sync"

CASES=(
  "$HOME/fastPLS_cycle9_plssvd_vs_simpls_20260723/results/metref_task.rds|metref|22"
  "$HOME/fastPLS_revision_cycle13_20260725/retina_tabula_selected_outer/runs/retina_simpls/retina_task.rds|retina|20"
  "$HOME/fastPLS_cycle9_plssvd_vs_simpls_20260723/results/prism_task.rds|prism|5"
  "$HOME/fastPLS_cycle9_plssvd_vs_simpls_20260723/results/cifar100_task.rds|cifar100|50"
)
if [ "${FASTPLS_ABLATION_INCLUDE_NMR:-false}" = "true" ]; then
  CASES+=("$HOME/fastPLS_cycle9_plssvd_vs_simpls_20260723/results/nmr_task.rds|nmr|50")
fi

if [ -n "${FASTPLS_ABLATION_CONFIGS:-}" ]; then
  read -r -a CONFIGS <<<"${FASTPLS_ABLATION_CONFIGS}"
else
  CONFIGS=(
    xtx_off xtx_on
    coefficients_recomputed coefficients_incremental
    deflation_inline deflation_cached
    coefficient_cube compact_prediction
    explicit_crosscov matrix_free
  )
fi

annotate_row() {
  local row="$1" peak="$2"
  python3 - "$row" "$peak" <<'PY'
import csv, math, sys
path, peak = sys.argv[1:]
rows = list(csv.DictReader(open(path, newline="")))
if not rows:
    raise SystemExit(0)
peak = float(peak) if peak not in ("", "NA") else math.nan
for row in rows:
    base = float(row["rss_before_fit_mb"])
    row["fit_window_peak_rss_mb"] = "" if math.isnan(peak) else f"{peak:.6f}"
    row["incremental_peak_rss_mb"] = "" if math.isnan(peak) else f"{max(0.0, peak-base):.6f}"
fields = list(rows[0])
with open(path, "w", newline="") as fh:
    out = csv.DictWriter(fh, fieldnames=fields)
    out.writeheader()
    out.writerows(rows)
PY
}

for case_spec in "${CASES[@]}"; do
  IFS='|' read -r task dataset ncomp <<<"${case_spec}"
  if [ "${DATASET_FILTER}" != ",all," ] &&
     [[ "${DATASET_FILTER}" != *",${dataset},"* ]]; then
    continue
  fi
  for config in "${CONFIGS[@]}"; do
    for replicate in $(seq 1 "${REPS}"); do
      run_id="${dataset}_${config}_n${ncomp}_rep${replicate}"
      row="${RESULTS_DIR}/rows/${run_id}.csv"
      pred="${RESULTS_DIR}/predictions/${run_id}.rds"
      log="${RESULTS_DIR}/logs/${run_id}.log"
      ready="${RESULTS_DIR}/sync/${run_id}.ready"
      go="${RESULTS_DIR}/sync/${run_id}.go"
      if [ -s "${row}" ] && grep -q 'success' "${row}"; then
        echo "[$(date -Iseconds)] keeping completed ${run_id}"
        continue
      fi
      rm -f "${row}" "${pred}" "${ready}" "${go}"
      echo "[$(date -Iseconds)] ${run_id}" | tee "${log}"
      set +e
      timeout --signal=TERM --kill-after=30s "${TIMEOUT_SEC}" \
        Rscript "${REPO_ROOT}/benchmark/benchmark_simpls_multidataset_ablation.R" \
          --task="${task}" --dataset="${dataset}" --ncomp="${ncomp}" \
          --configuration="${config}" --replicate="${replicate}" --seed=123 \
          --output="${row}" --prediction-output="${pred}" \
          --ready-file="${ready}" --go-file="${go}" >>"${log}" 2>&1 &
      runner_pid=$!
      while kill -0 "${runner_pid}" 2>/dev/null && [ ! -s "${ready}" ]; do sleep 0.02; done
      peak="NA"
      if [ -s "${ready}" ]; then
        r_pid="$(pgrep -P "${runner_pid}" -n || true)"
        [ -n "${r_pid}" ] || r_pid="${runner_pid}"
        touch "${go}"
        while kill -0 "${runner_pid}" 2>/dev/null; do
          rss="$(awk '/^VmRSS:/ {print $2/1024}' "/proc/${r_pid}/status" 2>/dev/null || true)"
          if [ -n "${rss}" ]; then
            peak="$(awk -v a="${peak}" -v b="${rss}" 'BEGIN {if(a=="NA" || b>a) print b; else print a}')"
          fi
          sleep 0.05
        done
      fi
      wait "${runner_pid}"
      status=$?
      set -e
      if [ -s "${row}" ]; then
        annotate_row "${row}" "${peak}"
      else
        echo "status=${status}; no result row" >>"${log}"
      fi
    done
  done
done
