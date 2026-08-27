#!/usr/bin/env bash
set -euo pipefail

ROOT="${FASTPLS_FROZEN_ROOT:-/home/chiamaka/fastPLS_frozen_0.99.25}"
SRC="$ROOT/src"
OUT="$ROOT/results/imagenet"
TASK="${IMAGENET_TASK:-/home/chiamaka/Documents/fastpls/data/imagenet_float32_seed123_train1000000_task.rds}"
mkdir -p "$OUT"

Rscript - "$TASK" "$OUT/input_manifest.tsv" <<'RS'
args <- commandArgs(trailingOnly = TRUE)
task <- readRDS(args[[1L]])
fields <- c("Xtrain_rds", "Xtest_rds")
paths <- c(args[[1L]], unlist(task[fields], use.names = FALSE))
write.table(
  data.frame(role = c("task", fields), path = normalizePath(paths),
             size_bytes = file.info(paths)$size),
  args[[2L]], sep = "\t", row.names = FALSE, quote = FALSE
)
RS
while IFS=$'\t' read -r role path size; do
  [[ "$role" == "role" ]] && continue
  sha256sum "$path"
done < "$OUT/input_manifest.tsv" > "$OUT/input_sha256.txt"
sha256sum "$SRC/fastPLS_0.99.25.tar.gz" > "$OUT/source_sha256.txt"

run_one() {
  local classifier="$1"
  FASTPLS_LIB="$ROOT/lib" \
  TASK_RDS="$TASK" \
  OUTPUT_CSV="$OUT/imagenet_${classifier}.csv" \
  FASTPLS_SOURCE_ARCHIVE_SHA256="604f74d72f5e9540f7378efd37f2ca6ed87ccb9e73b912211331a8cd97233481" \
  CLASSIFIER="$classifier" \
  BACKEND="cuda" \
  NCOMP="100,200,300,400,500,600,700,800,900,1000" \
  OVERSAMPLE="20" POWER="2" SEED="123" REPLICATE="1" \
  PREDICTION_BLOCK_ROWS="5000" \
  Rscript "$SRC/benchmark_imagenet_qualified_top5_path.R" \
    > "$OUT/imagenet_${classifier}.log" 2>&1
}

run_one argmax
run_one lda

Rscript - "$OUT" <<'RS'
args <- commandArgs(trailingOnly = TRUE)
files <- file.path(args[[1L]], c("imagenet_argmax.csv", "imagenet_lda.csv"))
rows <- do.call(rbind, lapply(files, read.csv, check.names = FALSE))
write.csv(rows, file.path(args[[1L]], "imagenet_all_results.csv"), row.names = FALSE)
if (any(rows$status != "success")) quit(save = "no", status = 1L)
RS
Rscript -e ".libPaths(c('$ROOT/lib',.libPaths())); library(fastPLS); print(sessionInfo())" \
  > "$OUT/session_info.txt" 2>&1
nvidia-smi -q > "$OUT/nvidia_smi.txt"
