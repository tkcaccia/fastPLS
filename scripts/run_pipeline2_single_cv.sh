#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="${PIPELINE2_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${PIPELINE2_OUT_DIR:-$ROOT_DIR/benchmark_results/pipeline2_single_cv_$STAMP}"

mkdir -p "$OUT_DIR/logs"

BEST_SOURCE_DEFAULT="/Users/stefano/Documents/GPUPLS/chiamaka_usual_pipeline_20260515_latest/dataset_memory_compare_raw.csv"
TASK_DIR_DEFAULT="/Users/stefano/Documents/GPUPLS/local_usual_pipeline_metal_20260515_230543/real_datasets"

BEST_SOURCE="${PIPELINE2_BEST_SOURCE:-$BEST_SOURCE_DEFAULT}"
TASK_DIR="${PIPELINE2_TASK_DIR:-$TASK_DIR_DEFAULT}"
HOST_LABEL="${PIPELINE2_HOST_LABEL:-$(hostname)}"
DATASETS="${PIPELINE2_DATASETS:-metref,ccle,cifar100,prism,gtex_v8,tcga_pan_cancer,singlecell,tcga_brca,tcga_hnsc_methylation,nmr,cbmc_citeseq}"
METHODS="${PIPELINE2_METHODS:-plssvd,simpls,opls,kernelpls}"
BACKENDS="${PIPELINE2_BACKENDS:-cpu,gpu}"
CPU_SVD_METHODS="${PIPELINE2_CPU_SVD_METHODS:-rsvd,irlba}"
CLASSIFIERS="${PIPELINE2_CLASSIFIERS:-argmax,lda}"
KFOLD="${PIPELINE2_KFOLD:-5}"
TIMEOUT_SEC="${PIPELINE2_TIMEOUT_SEC:-3600}"

{
  echo "Pipeline 2 started: $(date)"
  echo "Root: $ROOT_DIR"
  echo "Output: $OUT_DIR"
  echo "Host: $HOST_LABEL"
  echo "Datasets: $DATASETS"
  echo "Methods: $METHODS"
  echo "Backends: $BACKENDS"
  echo "CPU SVD methods: $CPU_SVD_METHODS"
  echo "Classifiers: $CLASSIFIERS"
  echo "kfold: $KFOLD"
  echo "timeout_sec: $TIMEOUT_SEC"
  echo "best_source: $BEST_SOURCE"
  echo "task_dir: $TASK_DIR"
} | tee "$OUT_DIR/launch.log"

LC_ALL=C Rscript "$ROOT_DIR/benchmark/benchmark_pipeline2_single_cv.R" \
  --out-dir="$OUT_DIR" \
  --host-label="$HOST_LABEL" \
  --datasets="$DATASETS" \
  --methods="$METHODS" \
  --backends="$BACKENDS" \
  --cpu-svd-methods="$CPU_SVD_METHODS" \
  --classifiers="$CLASSIFIERS" \
  --kfold="$KFOLD" \
  --timeout-sec="$TIMEOUT_SEC" \
  --best-source="$BEST_SOURCE" \
  --task-dir="$TASK_DIR" \
  > "$OUT_DIR/logs/pipeline2_single_cv.log" 2>&1

LC_ALL=C Rscript "$ROOT_DIR/benchmark/plot_pipeline2_single_cv.R" \
  --results="$OUT_DIR/pipeline2_single_cv_raw.csv" \
  --out-dir="$OUT_DIR" \
  > "$OUT_DIR/logs/pipeline2_plot.log" 2>&1 || true

echo "Pipeline 2 finished: $(date)" | tee -a "$OUT_DIR/launch.log"
echo "$OUT_DIR"
