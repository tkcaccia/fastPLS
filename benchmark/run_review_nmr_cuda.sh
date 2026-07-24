#!/usr/bin/env bash
set -euo pipefail

# Run the NMR final validation with process-lifetime GPU-memory sampling.  The
# sampled metric is the total memory reported for CUDA compute applications;
# this is exact when the benchmark is the only GPU compute workload and is
# labelled accordingly in the output metadata.

if [ "$#" -ne 4 ]; then
  echo "Usage: $0 FASTPLS_LIB NMR_RDATA RESULTS_DIR SELECTED_NCOMP" >&2
  exit 2
fi

fastpls_lib="$1"
nmr_rdata="$2"
results_dir="$3"
selected_ncomp="$4"
script_dir="$(cd "$(dirname "$0")" && pwd)"
samples_file="${results_dir}/nmr_final_cuda_gpu_mem_samples_mb.csv"
peak_file="${results_dir}/nmr_final_cuda_gpu_mem_peak_mb.txt"
log_file="${results_dir}/nmr_final_cuda.log"

mkdir -p "$results_dir"
env FASTPLS_LIB="$fastpls_lib" /usr/bin/time -v Rscript \
  "${script_dir}/review_nmr_full_validation.R" \
  --input="$nmr_rdata" --out="$results_dir" --mode=final --backend=cuda \
  --selected_ncomp="$selected_ncomp" --seed=123 >"$log_file" 2>&1 &
benchmark_pid=$!

while kill -0 "$benchmark_pid" 2>/dev/null; do
  timestamp="$(date +%s.%N)"
  memory_mb="$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{ total += $1 } END { print total + 0 }')"
  printf '%s,%s\n' "$timestamp" "$memory_mb" >>"$samples_file"
  sleep 0.2
done

wait "$benchmark_pid"
awk -F, 'BEGIN { max = 0 } $2 > max { max = $2 } END { print max }' \
  "$samples_file" >"$peak_file"
printf 'peak_gpu_compute_apps_mb=%s\n' "$(cat "$peak_file")"
