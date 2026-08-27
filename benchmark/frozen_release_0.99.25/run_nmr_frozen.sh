#!/usr/bin/env bash
set -euo pipefail

ROOT="${FASTPLS_FROZEN_ROOT:-/home/chiamaka/fastPLS_frozen_0.99.25}"
LIB="$ROOT/lib"
SRC="$ROOT/src"
OUT="$ROOT/results/nmr"
INPUT="${NMR_INPUT:-/home/chiamaka/Documents/fastpls/data/nmr.RData}"
mkdir -p "$OUT/predictions"

sha256sum "$INPUT" > "$OUT/input_sha256.txt"
sha256sum "$SRC/fastPLS_0.99.25.tar.gz" > "$OUT/source_sha256.txt"
INPUT_SHA="$(sha256sum "$INPUT" | awk '{print $1}')"
SOURCE_SHA="$(sha256sum "$SRC/fastPLS_0.99.25.tar.gz" | awk '{print $1}')"
Rscript -e ".libPaths(c('$LIB',.libPaths())); library(fastPLS); cat('fastPLS=',as.character(packageVersion('fastPLS')),'\\n'); print(sessionInfo())" \
  > "$OUT/session_info.txt" 2>&1
nvidia-smi -q > "$OUT/nvidia_smi.txt"

run_one() {
  local family="$1" backend="$2" solver="$3" ncomp="$4" label="$5" save_prediction="$6"
  local prediction_arg=""
  if [[ "$save_prediction" == "yes" ]]; then
    prediction_arg="--prediction_output=$OUT/predictions/${label}.rds"
  fi
  FASTPLS_LIB="$LIB" \
  FASTPLS_INPUT_SHA256="$INPUT_SHA" \
  FASTPLS_SOURCE_ARCHIVE_SHA256="$SOURCE_SHA" \
  Rscript "$SRC/benchmark_nmr_qualified_solver.R" \
    --input="$INPUT" \
    --output="$OUT/${label}.csv" \
    --family="$family" \
    --backend="$backend" \
    --solver="$solver" \
    --ncomp="$ncomp" \
    --oversample=20 \
    --power=2 \
    --seed=123 \
    --replicates=3 \
    $prediction_arg \
    > "$OUT/${label}.log" 2>&1
}

# Training-selected family settings: predictive comparison.
run_one plssvd cpu  irlba 5  selected_plssvd_cpu_irlba_n5 yes
run_one plssvd cpu  rsvd  5  selected_plssvd_cpu_rsvd_n5  yes
run_one plssvd cuda rsvd  5  selected_plssvd_cuda_rsvd_n5 yes
run_one simpls cpu  irlba 50 selected_simpls_cpu_irlba_n50 yes
run_one simpls cpu  rsvd  50 selected_simpls_cpu_rsvd_n50  yes
run_one simpls cuda rsvd  50 selected_simpls_cuda_rsvd_n50 yes

# Matched implementation/backend comparison: family, solver, precision, and
# component count are fixed; only CPU/CUDA implementation changes.
run_one plssvd cpu  rsvd 100 matched_plssvd_cpu_rsvd_n100 no
run_one plssvd cuda rsvd 100 matched_plssvd_cuda_rsvd_n100 no
run_one simpls cpu  rsvd 100 matched_simpls_cpu_rsvd_n100 no
run_one simpls cuda rsvd 100 matched_simpls_cuda_rsvd_n100 no

python3 - "$OUT" <<'PY'
import csv, glob, os, sys
out = sys.argv[1]
rows = []
for path in sorted(glob.glob(os.path.join(out, "*.csv"))):
    if path.endswith("nmr_all_runs.csv"):
        continue
    with open(path, newline="") as handle:
        rows.extend(csv.DictReader(handle))
if not rows:
    raise SystemExit("No NMR rows were produced")
with open(os.path.join(out, "nmr_all_runs.csv"), "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {len(rows)} rows")
PY
