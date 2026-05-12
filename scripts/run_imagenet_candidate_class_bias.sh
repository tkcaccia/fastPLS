#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${FASTPLS_ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BENCHMARK_SCRIPT="${FASTPLS_IMAGENET_CANDIDATE_CLASS_BIAS_R:-$ROOT_DIR/benchmark/benchmark_imagenet_pls_candidate_class_bias.R}"
OUT_ROOT="${FASTPLS_IMAGENET_CANDIDATE_CLASS_BIAS_OUT:-$ROOT_DIR/benchmark_results/imagenet_candidate_class_bias_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$OUT_ROOT"

Rscript "$BENCHMARK_SCRIPT" \
  --pls-score-dir="${FASTPLS_IMAGENET_PLS_SCORE_DIR:?Set FASTPLS_IMAGENET_PLS_SCORE_DIR}" \
  --source-run="${FASTPLS_IMAGENET_PROTOTYPE_RUN:?Set FASTPLS_IMAGENET_PROTOTYPE_RUN}" \
  --prototype-scorer="${FASTPLS_CUDA_PROTOTYPE_SCORER:?Set FASTPLS_CUDA_PROTOTYPE_SCORER}" \
  --candidate-knn="${FASTPLS_CUDA_CANDIDATE_KNN:?Set FASTPLS_CUDA_CANDIDATE_KNN}" \
  --out-root="$OUT_ROOT" \
  --train-n="${FASTPLS_IMAGENET_TRAIN_N:-1000000}" \
  --test-n="${FASTPLS_IMAGENET_TEST_N:-50000}" \
  --ncomp="${FASTPLS_IMAGENET_NCOMP:-300}" \
  --max-p="${FASTPLS_IMAGENET_MAX_P:-300}" \
  --top-m="${FASTPLS_IMAGENET_TOP_M:-20}" \
  --knn-k="${FASTPLS_IMAGENET_KNN_K:-3}" \
  --tau="${FASTPLS_IMAGENET_TAU:-0.1}" \
  --alpha="${FASTPLS_IMAGENET_ALPHA:-0.75}" \
  --gate-fracs="${FASTPLS_IMAGENET_GATE_FRACS:-0.6}" \
  --lambdas="${FASTPLS_IMAGENET_LAMBDAS:-0}" \
  --etas="${FASTPLS_IMAGENET_ETAS:-0.0025}" \
  --iters="${FASTPLS_IMAGENET_ITERS:-5}" \
  --clips="${FASTPLS_IMAGENET_CLIPS:-0.05}" \
  --chunk="${FASTPLS_IMAGENET_CHUNK:-2500}" \
  --train-block-size="${FASTPLS_IMAGENET_TRAIN_BLOCK_SIZE:-50000}" \
  --grouped-cache-dir="${FASTPLS_IMAGENET_GROUPED_CACHE_DIR:-$OUT_ROOT/grouped_l2_cache}"
