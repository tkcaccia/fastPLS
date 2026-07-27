#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 NMR_RDATA REFERENCE_R OUTPUT_DIR REPS" >&2
  exit 2
fi

nmr_rdata="$1"
reference_r="$2"
output_dir="$3"
reps="$4"
ncomp="${FASTPLS_NMR_REFERENCE_NCOMP:-100}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${output_dir}/rows" "${output_dir}/predictions" "${output_dir}/logs"
default_variants="deposited_fastsimpls_irlba fastpls_plssvd_cpu_irlba fastpls_plssvd_cpu_rsvd fastpls_plssvd_cuda_rsvd fastpls_simpls_cpu_rsvd fastpls_simpls_cuda_rsvd"
read -r -a variants <<<"${FASTPLS_NMR_REFERENCE_VARIANTS:-${default_variants}}"

for variant in "${variants[@]}"; do
  rep_start="${FASTPLS_NMR_REFERENCE_REP_START:-1}"
  rep_end=$((rep_start + reps - 1))
  for ((rep = rep_start; rep <= rep_end; rep++)); do
    stem="${variant}__rep${rep}"
    row="${output_dir}/rows/${stem}.csv"
    prediction="${output_dir}/predictions/${stem}.rds"
    stdout_log="${output_dir}/logs/${stem}.log"
    time_log="${output_dir}/logs/${stem}.time"
    gpu_log="${output_dir}/logs/${stem}.gpu"
    : >"${gpu_log}"

    echo "[$(date '+%F %T')] variant=${variant} repetition=${rep}"
    /usr/bin/time -v -o "${time_log}" \
      Rscript "${script_dir}/review_nmr_reference_comparison.R" \
        --input="${nmr_rdata}" \
        --output="${row}" \
        --prediction_output="${prediction}" \
        --variant="${variant}" \
        --reference_source="${reference_r}" \
        --ncomp="${ncomp}" \
        --seed=123 >"${stdout_log}" 2>&1 &
    monitor_pid=$!

    while kill -0 "${monitor_pid}" 2>/dev/null; do
      if [[ "${variant}" == *"_cuda_"* ]]; then
        child_pids="$(pgrep -P "${monitor_pid}" 2>/dev/null || true)"
        for child_pid in ${child_pids}; do
          nvidia-smi --query-compute-apps=pid,used_gpu_memory \
            --format=csv,noheader,nounits 2>/dev/null |
            awk -F',' -v pid="${child_pid}" '
              {gsub(/ /, "", $1); gsub(/ /, "", $2)}
              $1 == pid {print $2}
            ' >>"${gpu_log}" || true
        done
      fi
      sleep 0.2
    done
    wait "${monitor_pid}"

    rss_kb="$(
      awk -F: '/Maximum resident set size/ {gsub(/^[ \t]+/, "", $2); print $2}' \
        "${time_log}"
    )"
    gpu_peak="$(
      awk 'BEGIN {m=""} /^[0-9]+([.][0-9]+)?$/ {if(m=="" || $1>m)m=$1} END{print m}' \
        "${gpu_log}"
    )"
    ROW_FILE="${row}" REPETITION="${rep}" RSS_KB="${rss_kb:-NA}" \
      GPU_PEAK="${gpu_peak:-NA}" Rscript -e '
        path <- Sys.getenv("ROW_FILE")
        x <- utils::read.csv(path, check.names = FALSE)
        x$repetition <- as.integer(Sys.getenv("REPETITION"))
        rss <- suppressWarnings(as.numeric(Sys.getenv("RSS_KB")))
        gpu <- suppressWarnings(as.numeric(Sys.getenv("GPU_PEAK")))
        x$host_rss_mb <- rss / 1024
        x$gpu_peak_mb <- gpu
        utils::write.csv(x, path, row.names = FALSE)
      '
  done
done

if [[ "${FASTPLS_NMR_REFERENCE_SUMMARIZE:-1}" == "1" ]]; then
  Rscript "${script_dir}/summarize_nmr_reference_comparison.R" \
    "${output_dir}" "${output_dir}"
fi
