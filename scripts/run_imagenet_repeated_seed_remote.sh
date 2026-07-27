#!/usr/bin/env bash
set -euo pipefail

SEED="${1:?usage: run_imagenet_repeated_seed_remote.sh SEED}"
BASE_ROOT="${IMAGENET_RETRIEVAL_BASE_ROOT:-$HOME/fastPLS_imagenet_faiss_matched_1m_20260725}"
ROOT="${IMAGENET_RETRIEVAL_REPEAT_ROOT:-$HOME/fastPLS_imagenet_faiss_seed${SEED}_20260725}"
SCRIPT="$BASE_ROOT/benchmark/benchmark_imagenet_faiss_matched_retrieval.R"
FAISS_ENV="${FAISSR_ENV:-$HOME/.fastEmbedR/micromamba/envs/fastembedr-faissgpu-cuvs}"

export LD_LIBRARY_PATH="$FAISS_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$FAISS_ENV/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
export R_LIBS_USER="${FAISSR_LIB:-$HOME/R/faissR_imagenet_lib}:${FASTPLS_R_LIB:-$HOME/R/x86_64-pc-linux-gnu-library/4.5}"
export IMAGENET_RETRIEVAL_TASK="${IMAGENET_RETRIEVAL_TASK:-$HOME/Documents/fastpls/data/imagenet_float32_seed123_train1000000_task.rds}"
export IMAGENET_RETRIEVAL_OUT="$ROOT/results"
export IMAGENET_RETRIEVAL_LABEL_CPP="$BASE_ROOT/benchmark/imagenet_label_crossprod_float32.cpp"
export IMAGENET_RETRIEVAL_TRAIN_N=1000000
export IMAGENET_RETRIEVAL_EVAL_N=281167
export IMAGENET_RETRIEVAL_MAX_NCOMP=200
export IMAGENET_RETRIEVAL_SEED="$SEED"

mkdir -p "$ROOT/results" "$ROOT/logs"

run_monitored() {
  local name="$1"
  shift
  local peak=0
  /usr/bin/time -v -o "$ROOT/logs/${name}.time.txt" \
    "$@" >"$ROOT/logs/${name}.log" 2>&1 &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    local used
    used="$(nvidia-smi --query-compute-apps=used_gpu_memory \
      --format=csv,noheader,nounits 2>/dev/null |
      awk '{sum += $1} END {print sum + 0}')"
    (( used > peak )) && peak="$used"
    sleep 0.2
  done
  wait "$pid"
  printf '%s\n' "$peak" >"$ROOT/results/${name}_peak_gpu_mb.txt"
}

run_mode() {
  local name="$1"
  shift
  run_monitored "$name" env "$@" Rscript "$SCRIPT"
}

run_mode prepare_pls IMAGENET_RETRIEVAL_MODE=prepare_pls
run_mode prepare_pca IMAGENET_RETRIEVAL_MODE=prepare_pca

for space in pls pca; do
  for ncomp in 50 100 200; do
    run_mode "${space}_k${ncomp}_exact" \
      IMAGENET_RETRIEVAL_MODE=search \
      IMAGENET_RETRIEVAL_SPACE="$space" \
      IMAGENET_RETRIEVAL_NCOMP="$ncomp" \
      IMAGENET_RETRIEVAL_METHOD=exact \
      IMAGENET_RETRIEVAL_REPS=1
  done
done

printf 'seed=%s\nroot=%s\nstatus=complete\n' "$SEED" "$ROOT" \
  >"$ROOT/results/run_manifest.txt"
