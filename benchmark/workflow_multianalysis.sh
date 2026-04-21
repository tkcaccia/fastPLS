#!/bin/sh

#SBATCH --account=immunology
#SBATCH --partition=ada
#SBATCH --nodes=1 --ntasks=16
#SBATCH --time=3-00:00:00
#SBATCH --job-name="fastPLS_multi"
#SBATCH --mail-user=stefano.cacciatore@uct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# Optional: load your site R module here if needed.
# module load R

SCRIPT_ROOT="${SCRIPT_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_ROOT}/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${FASTPLS_DATA_ROOT:-/Users/stefano/HPC-firenze/image_analysis/dinoV2/Rdatasets}}"
OUTROOT="${OUTROOT:-${REPO_ROOT}/benchmark_results_local}"

if ! command -v Rscript >/dev/null 2>&1; then
  echo "ERROR: Rscript not found in PATH"
  exit 1
fi

if [ ! -f "${SCRIPT_ROOT}/hpc_full_multianalysis_benchmark.R" ]; then
  echo "ERROR: missing ${SCRIPT_ROOT}/hpc_full_multianalysis_benchmark.R"
  exit 1
fi

if [ ! -f "${SCRIPT_ROOT}/hpc_full_multianalysis_plots.R" ]; then
  echo "ERROR: missing ${SCRIPT_ROOT}/hpc_full_multianalysis_plots.R"
  exit 1
fi

# Preflight: fail fast if any R script is truncated/corrupted
for rfile in \
  "${SCRIPT_ROOT}/hpc_full_multianalysis_benchmark.R" \
  "${SCRIPT_ROOT}/hpc_full_multianalysis_plots.R"
do
  if ! Rscript --vanilla -e "parse(file='${rfile}')" >/dev/null; then
    echo "ERROR: R parse failed for ${rfile}"
    exit 1
  fi
done

mkdir -p "${OUTROOT}"
cd "${SCRIPT_ROOT}"

# Core benchmark configuration
export FASTPLS_DATA_ROOT="${DATA_ROOT}"
export FASTPLS_DATASETS="${FASTPLS_DATASETS:-metref,cifar100,singlecell,gtex_v8,tcga_pan_cancer,ccle,prism,cbmc_citeseq,tcga_brca,tcga_hnsc_methylation}"

# Requested ncomp benchmark grid
export FASTPLS_NCOMP_LIST="${FASTPLS_NCOMP_LIST:-2,5,10,20,50,100}"
export FASTPLS_NMR_NCOMP_LIST="${FASTPLS_NMR_NCOMP_LIST:-2,3,5}"
export FASTPLS_SINGLECELL_NCOMP_LIST="${FASTPLS_SINGLECELL_NCOMP_LIST:-2,5,10,20,50}"
export FASTPLS_DEFAULT_NCOMP="${FASTPLS_DEFAULT_NCOMP:-2}"
export FASTPLS_METREF_DEFAULT_NCOMP="${FASTPLS_METREF_DEFAULT_NCOMP:-20}"
export FASTPLS_SINGLECELL_DEFAULT_NCOMP="${FASTPLS_SINGLECELL_DEFAULT_NCOMP:-50}"
export FASTPLS_CIFAR100_DEFAULT_NCOMP="${FASTPLS_CIFAR100_DEFAULT_NCOMP:-100}"
export FASTPLS_IMAGENET_TRAIN_N="${FASTPLS_IMAGENET_TRAIN_N:-1000000}"

# Sub-analyses
export FASTPLS_SAMPLE_FRACS="${FASTPLS_SAMPLE_FRACS:-0.33,0.66,1.0}"
export FASTPLS_XVAR_FRACS="${FASTPLS_XVAR_FRACS:-0.10,0.20,0.50,1.0}"
export FASTPLS_YVAR_FRACS="${FASTPLS_YVAR_FRACS:-0.10,0.20,0.50,1.0}"
export FASTPLS_IRLBA_SVTOL="${FASTPLS_IRLBA_SVTOL:-1e-6}"
export FASTPLS_RSVD_TOL="${FASTPLS_RSVD_TOL:-0}"

# Runtime behavior
export FASTPLS_REPS="${FASTPLS_REPS:-3}"
export FASTPLS_NMR_REPS="${FASTPLS_NMR_REPS:-1}"
export FASTPLS_METREF_REPS="${FASTPLS_METREF_REPS:-10}"
export FASTPLS_SINGLECELL_REPS="${FASTPLS_SINGLECELL_REPS:-5}"
export FASTPLS_CIFAR100_REPS="${FASTPLS_CIFAR100_REPS:-5}"
export FASTPLS_SEED="${FASTPLS_SEED:-123}"
export FASTPLS_INCLUDE_CUDA="${FASTPLS_INCLUDE_CUDA:-false}"
export FASTPLS_INCLUDE_R_IMPL="${FASTPLS_INCLUDE_R_IMPL:-false}"
export FASTPLS_METREF_INCLUDE_R="${FASTPLS_METREF_INCLUDE_R:-true}"
export FASTPLS_METREF_INCLUDE_PLS_PKG="${FASTPLS_METREF_INCLUDE_PLS_PKG:-true}"
export FASTPLS_SKIP_ARPACK_ON_NMR="${FASTPLS_SKIP_ARPACK_ON_NMR:-true}"
export FASTPLS_SKIP_PLSSVD_ON_NMR="${FASTPLS_SKIP_PLSSVD_ON_NMR:-true}"

# Multicore threading (BLAS/OpenMP) for each fit
THREADS="${FASTPLS_THREADS:-${SLURM_NTASKS:-1}}"
if [ -z "${THREADS}" ] || [ "${THREADS}" -lt 1 ] 2>/dev/null; then
  THREADS=1
fi
export FASTPLS_THREADS="${THREADS}"
export OMP_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export VECLIB_MAXIMUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"

echo "[INFO] Starting multianalysis benchmark"
echo "[INFO] SCRIPT_ROOT=${SCRIPT_ROOT}"
echo "[INFO] DATA_ROOT=${DATA_ROOT}"
echo "[INFO] OUTROOT=${OUTROOT}"
echo "[INFO] DATASETS=${FASTPLS_DATASETS}"
echo "[INFO] NCOMP=${FASTPLS_NCOMP_LIST}"
echo "[INFO] THREADS=${FASTPLS_THREADS}"

DATASET_LIST="$(echo "${FASTPLS_DATASETS}" | tr ',' ' ')"
for ds in ${DATASET_LIST}; do
  ds_clean="$(echo "${ds}" | tr '[:upper:]' '[:lower:]' | xargs)"
  [ -n "${ds_clean}" ] || continue

  DS_OUTDIR="${OUTROOT}/${ds_clean}"
  mkdir -p "${DS_OUTDIR}"
  export FASTPLS_MULTI_OUTDIR="${DS_OUTDIR}"
  export FASTPLS_DATASETS="${ds_clean}"
  export FASTPLS_MULTI_APPEND=false
  RAW_CSV="${DS_OUTDIR}/multianalysis_raw.csv"
  SUM_CSV="${DS_OUTDIR}/multianalysis_summary.csv"

  echo "[INFO] Running dataset: ${ds_clean} (outdir=${DS_OUTDIR})"
  echo "[INFO] Raw:     ${RAW_CSV}"
  echo "[INFO] Summary: ${SUM_CSV}"
  Rscript "${SCRIPT_ROOT}/hpc_full_multianalysis_benchmark.R"

  export FASTPLS_PLOT_DATASET=""
  echo "[INFO] Plotting dataset: ${ds_clean}"
  if ! Rscript "${SCRIPT_ROOT}/hpc_full_multianalysis_plots.R"; then
    echo "[WARN] Plot generation failed for dataset ${ds_clean}; continuing."
  else
    PDIR="${DS_OUTDIR}/plots"
    if [ -d "${PDIR}" ]; then
      echo "[INFO] Plot files for ${ds_clean}:"
      found_png=false
      for f in "${PDIR}"/*.png "${PDIR}"/all/*.png; do
        if [ -f "${f}" ]; then
          echo "${f}"
          found_png=true
        fi
      done
      if [ "${found_png}" = false ]; then
        echo "[INFO] No PNG files found in ${PDIR}"
      fi
    fi
  fi
done

unset FASTPLS_PLOT_DATASET
unset FASTPLS_MULTI_APPEND

echo "[INFO] Completed"
echo "[INFO] Dataset result folders under: ${OUTROOT}"
echo "[INFO] Example: ${OUTROOT}/metref , ${OUTROOT}/singlecell , ${OUTROOT}/tcga_brca , ${OUTROOT}/tcga_hnsc_methylation"
