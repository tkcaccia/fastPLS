#!/usr/bin/env bash
set -euo pipefail

# Repeat the fixed NMR final-validation protocol in isolated R processes.
# Component selection is performed beforehand on training data only; this
# wrapper measures computational variation at the selected component count.

if [ "$#" -ne 5 ]; then
  echo "Usage: $0 FASTPLS_LIB NMR_RDATA RESULTS_DIR SELECTED_NCOMP REPS" >&2
  exit 2
fi

fastpls_lib="$1"
nmr_rdata="$2"
results_dir="$3"
selected_ncomp="$4"
reps="$5"
script_dir="$(cd "$(dirname "$0")" && pwd)"

for rep in $(seq 1 "$reps"); do
  rep_dir="${results_dir}/rep_${rep}"
  mkdir -p "$rep_dir"
  echo "[$(date '+%F %T')] repetition=${rep} backend=cpu"
  env FASTPLS_LIB="$fastpls_lib" /usr/bin/time -v Rscript \
    "${script_dir}/review_nmr_full_validation.R" \
    --input="$nmr_rdata" --out="$rep_dir" --mode=final --backend=cpu \
    --selected_ncomp="$selected_ncomp" --seed=123 >"${rep_dir}/nmr_final_cpu.log" 2>&1
  echo "[$(date '+%F %T')] repetition=${rep} backend=cuda"
  "${script_dir}/run_review_nmr_cuda.sh" "$fastpls_lib" "$nmr_rdata" "$rep_dir" "$selected_ncomp" \
    >"${rep_dir}/nmr_final_cuda_wrapper.log" 2>&1
done
