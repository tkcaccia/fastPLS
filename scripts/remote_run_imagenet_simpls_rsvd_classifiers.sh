#!/usr/bin/env bash
set -u

RUN_ROOT="${RUN_ROOT:-$HOME/fastPLS_imagenet_simpls_rsvd_classifiers_$(date +%Y%m%d_%H%M%S)}"
SRC_DIR="${SRC_DIR:-$HOME/fastPLS_pipeline2_pkg_compare_src_20260519}"
TASK_RDS="${TASK_RDS:-$HOME/fastPLS_classbias_top1_pipeline/results/imagenet_full_binarycache_1m_probe_20260512_201204/matrix_cache/imagenet_seed123_train1000000_testrest_task.rds}"
NCOMP="${NCOMP:-300}"
SCALING="${SCALING:-centering}"
TIMEOUT_SEC="${TIMEOUT_SEC:-10000}"
BACKENDS="${BACKENDS:-cpu cuda}"
CLASSIFIERS="${CLASSIFIERS:-argmax lda cknn}"
OUT_DIR="$RUN_ROOT/results"

mkdir -p "$OUT_DIR/logs"

SCRIPT="$SRC_DIR/benchmark/benchmark_imagenet_simpls_rsvd_classifiers.R"
RAW="$OUT_DIR/imagenet_simpls_rsvd_classifiers_raw.csv"
TIME_CSV="$OUT_DIR/imagenet_simpls_rsvd_classifiers_time.csv"

echo "run_root=$RUN_ROOT" | tee "$OUT_DIR/manifest.txt"
echo "src_dir=$SRC_DIR" | tee -a "$OUT_DIR/manifest.txt"
echo "task_rds=$TASK_RDS" | tee -a "$OUT_DIR/manifest.txt"
echo "ncomp=$NCOMP" | tee -a "$OUT_DIR/manifest.txt"
echo "scaling=$SCALING" | tee -a "$OUT_DIR/manifest.txt"
echo "backends=$BACKENDS" | tee -a "$OUT_DIR/manifest.txt"
echo "classifiers=$CLASSIFIERS" | tee -a "$OUT_DIR/manifest.txt"
date | tee -a "$OUT_DIR/manifest.txt"
hostname | tee -a "$OUT_DIR/manifest.txt"
nvidia-smi -L >> "$OUT_DIR/manifest.txt" 2>&1 || true
Rscript -e 'suppressPackageStartupMessages(library(fastPLS)); cat("fastPLS=", as.character(packageVersion("fastPLS")), "\n"); cat("has_cuda=", has_cuda(), "\n"); print(args(fastPLS::pls))' >> "$OUT_DIR/manifest.txt" 2>&1 || true

write_time_row() {
  local backend="$1"
  local classifier="$2"
  local status="$3"
  local rss_kb="$4"
  local elapsed="$5"
  python3 - "$TIME_CSV" "$backend" "$classifier" "$status" "$rss_kb" "$elapsed" <<'PY'
import csv, os, sys
path, backend, classifier, status, rss_kb, elapsed = sys.argv[1:]
row = {
    "backend": backend,
    "classifier": classifier,
    "process_status": status,
    "peak_host_rss_mb": "" if not rss_kb else round(float(rss_kb) / 1024, 3),
    "wall_elapsed": elapsed,
}
exists = os.path.exists(path)
with open(path, "a", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(row))
    if not exists:
        w.writeheader()
    w.writerow(row)
PY
}

for backend in $BACKENDS; do
  for classifier in $CLASSIFIERS; do
    log="$OUT_DIR/logs/${backend}_${classifier}.log"
    timelog="$OUT_DIR/logs/${backend}_${classifier}.time.log"
    echo "[$(date '+%F %T')] RUN backend=$backend classifier=$classifier ncomp=$NCOMP" | tee -a "$OUT_DIR/run.log"
    (
      cd "$SRC_DIR" || exit 2
      OUT_DIR="$OUT_DIR" TASK_RDS="$TASK_RDS" BACKEND="$backend" CLASSIFIER="$classifier" \
        NCOMP="$NCOMP" SCALING="$SCALING" \
        /usr/bin/time -v timeout "$TIMEOUT_SEC" Rscript "$SCRIPT"
    ) > "$log" 2> "$timelog"
    ec=$?
    status="ok"
    if [ "$ec" -eq 124 ]; then
      status="killed_timeout"
    elif [ "$ec" -ne 0 ]; then
      status="error_$ec"
    fi
    rss="$(grep -F 'Maximum resident set size' "$timelog" | tail -1 | awk '{print $NF}')"
    elapsed="$(grep -F 'Elapsed (wall clock) time' "$timelog" | tail -1 | sed 's/.*: //')"
    write_time_row "$backend" "$classifier" "$status" "$rss" "$elapsed"
    echo "[$(date '+%F %T')] DONE backend=$backend classifier=$classifier status=$status rss_kb=${rss:-NA} elapsed=${elapsed:-NA}" | tee -a "$OUT_DIR/run.log"
  done
done

python3 - "$RAW" "$TIME_CSV" "$OUT_DIR/imagenet_simpls_rsvd_classifiers_joined.csv" <<'PY'
import csv, os, sys
raw, time_csv, out = sys.argv[1:]
rows = list(csv.DictReader(open(raw))) if os.path.exists(raw) else []
times = {(r["backend"], r["classifier"]): r for r in csv.DictReader(open(time_csv))} if os.path.exists(time_csv) else {}
if rows:
    fields = list(rows[0])
    for extra in ["process_status", "peak_host_rss_mb", "wall_elapsed"]:
        if extra not in fields:
            fields.append(extra)
    for r in rows:
        t = times.get((r.get("backend"), r.get("classifier")), {})
        r.update({k: t.get(k, "") for k in ["process_status", "peak_host_rss_mb", "wall_elapsed"]})
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
PY

date | tee -a "$OUT_DIR/manifest.txt"
echo "results=$OUT_DIR"
