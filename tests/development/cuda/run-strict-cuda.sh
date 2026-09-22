#!/usr/bin/env sh
set -eu

if [ "$#" -ne 2 ]; then
  echo "usage: run-strict-cuda.sh SOURCE_DIRECTORY EVIDENCE_DIRECTORY" >&2
  exit 2
fi

source_directory="$1"
evidence_directory="$2"
mkdir -p "${evidence_directory}"
library="${evidence_directory}/library"
mkdir -p "${library}"

export FASTPLS_USE_CUDA=1
export FASTPLS_REQUIRE_CUDA=1
export FASTPLS_USE_OPENBLAS=1
export FASTPLS_USE_METAL=0
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export R_LIBS_USER="${library}"

cd "${source_directory}"
R CMD build . >"${evidence_directory}/build.log" 2>&1
archive="$(find . -maxdepth 1 -name 'fastPLS_*.tar.gz' -print | sort | tail -n 1)"
sha256sum "${archive}" >"${evidence_directory}/source.sha256"
R CMD INSTALL --preclean --library="${library}" "${archive}" \
  >"${evidence_directory}/install.log" 2>&1
Rscript --vanilla tools/cuda/strict-smoke.R \
  >"${evidence_directory}/smoke.log" 2>&1

/usr/bin/time -f 'elapsed_seconds=%e' -o "${evidence_directory}/testthat.time" \
  Rscript --vanilla -e \
  'library(testthat); test_local(".", reporter="summary", load_package="installed", stop_on_failure=TRUE)' \
  >"${evidence_directory}/testthat.log" 2>&1

_R_CHECK_FORCE_SUGGESTS_=false R CMD check --as-cran "${archive}" \
  >"${evidence_directory}/check.log" 2>&1

grep -q '^\* DONE' "${evidence_directory}/check.log"
if grep -Eq '^Status:.*(ERROR|WARNING)' "${evidence_directory}/check.log"; then
  echo "R CMD check reported an ERROR or WARNING" >&2
  exit 1
fi
grep -q 'Strict CUDA smoke test passed' "${evidence_directory}/smoke.log"
