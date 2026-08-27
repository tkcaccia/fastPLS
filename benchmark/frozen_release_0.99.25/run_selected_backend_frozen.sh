#!/usr/bin/env bash
set -euo pipefail

ROOT="${FASTPLS_FROZEN_ROOT:-/home/chiamaka/fastPLS_frozen_0.99.25}"
SRC="$ROOT/src"
OUT="$ROOT/results/selected_backend"
mkdir -p "$OUT/rows"

run_task() {
  local dataset="$1" data="$2" ncomp="$3"
  sha256sum "$data" >> "$OUT/input_sha256.txt"
  for backend in cpu cuda; do
    for replicate in 1 2 3; do
      Rscript "$SRC/selected_backend_worker.R" \
        --lib="$ROOT/lib" \
        --helper="$SRC/helpers_dataset_memory_compare.R" \
        --dataset="$dataset" \
        --data="$data" \
        --backend="$backend" \
        --ncomp="$ncomp" \
        --replicate="$replicate" \
        --out="$OUT/rows/${dataset}_${backend}_rep${replicate}.csv" \
        > "$OUT/rows/${dataset}_${backend}_rep${replicate}.log" 2>&1
    done
  done
}

: > "$OUT/input_sha256.txt"
run_task metref /home/chiamaka/Documents/fastpls/data/metref.RData 22
run_task retina /home/chiamaka/Documents/fastpls/data/Macosko2015_retina_float32.RData 20
run_task cifar100 /home/chiamaka/Documents/Rdatasets/CIFAR100.RData 100

python3 - "$OUT" <<'PY'
import csv, glob, os, statistics, sys
out = sys.argv[1]
rows = []
for path in sorted(glob.glob(os.path.join(out, "rows", "*.csv"))):
    with open(path, newline="") as handle:
        rows.extend(csv.DictReader(handle))
if len(rows) != 18:
    raise SystemExit(f"Expected 18 rows, found {len(rows)}")
fields = list(rows[0])
with open(os.path.join(out, "selected_backend_all_runs.csv"), "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader(); writer.writerows(rows)

summary = []
for dataset in sorted(set(r["dataset"] for r in rows)):
    for backend in ("cpu", "cuda"):
        group = [r for r in rows if r["dataset"] == dataset and r["backend"] == backend]
        summary.append({
            "dataset": dataset,
            "backend": backend,
            "ncomp": group[0]["ncomp"],
            "accuracy": group[0]["accuracy"],
            "top5_accuracy": group[0]["top5_accuracy"],
            "median_total_time_sec": statistics.median(float(r["total_time_sec"]) for r in group),
            "time_iqr_sec": statistics.quantiles([float(r["total_time_sec"]) for r in group], n=4, method="inclusive")[2] - statistics.quantiles([float(r["total_time_sec"]) for r in group], n=4, method="inclusive")[0],
            "median_rss_before_fit_mb": statistics.median(float(r["rss_before_fit_mb"]) for r in group),
            "median_rss_after_prediction_mb": statistics.median(float(r["rss_after_prediction_mb"]) for r in group),
            "median_gpu_before_fit_mb": statistics.median(float(r["gpu_before_fit_mb"]) for r in group),
            "median_gpu_after_prediction_mb": statistics.median(float(r["gpu_after_prediction_mb"]) for r in group),
            "repetitions": len(group),
            "status": "success",
        })
with open(os.path.join(out, "selected_backend_summary.csv"), "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
    writer.writeheader(); writer.writerows(summary)
print(f"wrote {len(rows)} runs and {len(summary)} summary rows")
PY

Rscript -e ".libPaths(c('$ROOT/lib',.libPaths())); library(fastPLS); print(sessionInfo())" \
  > "$OUT/session_info.txt" 2>&1
nvidia-smi -q > "$OUT/nvidia_smi.txt"
