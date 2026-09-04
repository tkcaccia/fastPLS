#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${FASTPLS_MULTICORE_OUT:-${ROOT}/publication_results/0.99.39/current_release/multicore_scaling}"
LIB="${FASTPLS_MULTICORE_LIB:-${ROOT}/.fastpls-openblas-lib}"
PROBE_DIR="${OUT}/probe"
PROBE="${PROBE_DIR}/openblas_probe.so"
RAW="${OUT}/multicore_scaling_raw.csv"

mkdir -p "${PROBE_DIR}"
rm -f "${RAW}"

PKG_CPPFLAGS="-I/opt/homebrew/opt/openblas/include" \
PKG_LIBS="-L/opt/homebrew/opt/openblas/lib -lopenblas" \
R CMD SHLIB "${ROOT}/benchmark/multicore_scaling/openblas_probe.c" \
    -o "${PROBE}"
rm -f "${ROOT}/benchmark/multicore_scaling/openblas_probe.o"

workloads=(
    "sample-rich classification"
    "predictor-wide regression"
    "response-wide regression"
)

for workload in "${workloads[@]}"; do
    for cores in 1 2 4; do
        for replicate in 1 2 3 4 5; do
            OPENBLAS_NUM_THREADS="${cores}" \
            OMP_NUM_THREADS="${cores}" \
            FASTPLS_MULTICORE_LIB="${LIB}" \
            FASTPLS_OPENBLAS_PROBE="${PROBE}" \
            R_LIBS_USER="${LIB}" \
            Rscript "${ROOT}/benchmark/multicore_scaling/worker.R" \
                "${workload}" "${cores}" "${replicate}" "${RAW}"
        done
    done
done

Rscript "${ROOT}/benchmark/multicore_scaling/summarize.R" "${RAW}" "${OUT}"
rm -rf "${PROBE_DIR}"
