#!/bin/sh

set -euo pipefail

SCRIPT_ROOT="${SCRIPT_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
DATA_ROOT="${DATA_ROOT:-${FASTPLS_DATA_ROOT:-${SCRIPT_ROOT}}}"
OUTROOT="${OUTROOT:-${SCRIPT_ROOT}/benchmark_results_local}"

if ! command -v Rscript >/dev/null 2>&1; then
  echo "ERROR: Rscript not found in PATH"
  exit 1
fi

if [ ! -f "${SCRIPT_ROOT}/hpc_full_multianalysis_benchmark.R" ]; then
  echo "ERROR: missing ${SCRIPT_ROOT}/hpc_full_multianalysis_benchmark.R"
  exit 1
fi

mkdir -p "${OUTROOT}"

export FASTPLS_DATA_ROOT="${DATA_ROOT}"
export FASTPLS_DATASETS="metref"
export FASTPLS_ONLY_NCOMP="true"
export FASTPLS_MULTI_APPEND="false"

# Keep the same benchmark knobs as the remote GPU run unless the caller overrides them.
export FASTPLS_INCLUDE_CUDA="${FASTPLS_INCLUDE_CUDA:-true}"
export FASTPLS_INCLUDE_R_IMPL="${FASTPLS_INCLUDE_R_IMPL:-true}"
export FASTPLS_INCLUDE_PLS_PKG="${FASTPLS_INCLUDE_PLS_PKG:-true}"
export FASTPLS_INCLUDE_SIMPLS_FAST_INCREMENTAL="${FASTPLS_INCLUDE_SIMPLS_FAST_INCREMENTAL:-true}"

export FASTPLS_NCOMP_LIST="${FASTPLS_NCOMP_LIST:-2,5,10,20,50,100}"
export FASTPLS_METREF_DEFAULT_NCOMP="${FASTPLS_METREF_DEFAULT_NCOMP:-20}"
export FASTPLS_METREF_REPS="${FASTPLS_METREF_REPS:-10}"
export FASTPLS_SEED="${FASTPLS_SEED:-123}"

THREADS="${FASTPLS_THREADS:-1}"
if [ -z "${THREADS}" ] || [ "${THREADS}" -lt 1 ] 2>/dev/null; then
  THREADS=1
fi
export FASTPLS_THREADS="${THREADS}"
export OMP_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export VECLIB_MAXIMUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"

DS_OUTDIR="${OUTROOT}/metref"
mkdir -p "${DS_OUTDIR}"
export FASTPLS_MULTI_OUTDIR="${DS_OUTDIR}"

echo "[INFO] Running exact multianalysis benchmark on metref only"
echo "[INFO] DATA_ROOT=${DATA_ROOT}"
echo "[INFO] OUTDIR=${DS_OUTDIR}"
echo "[INFO] NCOMP=${FASTPLS_NCOMP_LIST}"
echo "[INFO] METREF_REPS=${FASTPLS_METREF_REPS}"
echo "[INFO] INCLUDE_CUDA=${FASTPLS_INCLUDE_CUDA}"
echo "[INFO] INCLUDE_R_IMPL=${FASTPLS_INCLUDE_R_IMPL}"
echo "[INFO] INCLUDE_PLS_PKG=${FASTPLS_INCLUDE_PLS_PKG}"

Rscript "${SCRIPT_ROOT}/hpc_full_multianalysis_benchmark.R"
