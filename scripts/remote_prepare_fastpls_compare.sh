#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"

REMOTE_USER="${REMOTE_USER:-chiamaka}"
REMOTE_HOST="${REMOTE_HOST:-137.158.224.178}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/chiamaka/fastPLS_remote_cifar100_compare}"
BASELINE_LIB_REMOTE="${BASELINE_LIB_REMOTE:-/home/chiamaka/R/fastpls_baseline_lib}"
TEST_LIB_REMOTE="${TEST_LIB_REMOTE:-/home/chiamaka/R/fastpls_test_lib}"
EXPECT_BIN="${EXPECT_BIN:-expect}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ -z "${FASTPLS_REMOTE_PASS:-}" ]; then
  echo "ERROR: FASTPLS_REMOTE_PASS is required."
  exit 1
fi

if ! command -v "${EXPECT_BIN}" >/dev/null 2>&1; then
  echo "ERROR: expect not found"
  exit 1
fi

BASELINE_COMMIT="${FASTPLS_BASELINE_COMMIT:-$(cd "${REPO_ROOT}" && git ls-remote origin main | awk 'NR==1{print $1}')}"
TEST_COMMIT_HEAD="$(cd "${REPO_ROOT}" && git rev-parse HEAD)"
if [ -n "$(cd "${REPO_ROOT}" && git status --porcelain)" ]; then
  TEST_COMMIT_NOTE="${TEST_COMMIT_HEAD}+dirty"
else
  TEST_COMMIT_NOTE="${TEST_COMMIT_HEAD}"
fi

TMPDIR="$(mktemp -d)"
trap 'rm -rf "${TMPDIR}"' EXIT

BASELINE_ARCHIVE="${TMPDIR}/fastpls_baseline.tar.gz"
TEST_ARCHIVE="${TMPDIR}/fastpls_test.tar.gz"

cd "${REPO_ROOT}"
git archive --format=tar.gz --prefix=baseline_src/ "${BASELINE_COMMIT}" -o "${BASELINE_ARCHIVE}"

REPO_ROOT_ENV="${REPO_ROOT}" TEST_ARCHIVE_ENV="${TEST_ARCHIVE}" "${PYTHON_BIN}" - <<'PY'
import os, tarfile
repo = os.path.abspath(os.environ["REPO_ROOT_ENV"])
out_path = os.path.abspath(os.environ["TEST_ARCHIVE_ENV"])
prefix = "test_src"
skip_prefixes = [
    ".git",
    "fastPLS.Rcheck",
    "fastPLS-src.Rcheck",
    "benchmark_results",
    "benchmark_results_local",
]
skip_fragments = [
    "/benchmark_results",
    "/.git/",
    ".Rcheck/",
]
skip_suffixes = [".tar.gz", ".tgz", ".zip", ".o", ".so"]

with tarfile.open(out_path, "w:gz") as tar:
    for root, dirs, files in os.walk(repo):
        rel_root = os.path.relpath(root, repo)
        dirs[:] = [d for d in dirs if d not in skip_prefixes and not d.endswith(".Rcheck")]
        for fn in files:
            rel = os.path.normpath(os.path.join(rel_root, fn)) if rel_root != "." else fn
            if any(rel == p or rel.startswith(p + os.sep) for p in skip_prefixes):
                continue
            if any(fragment in rel for fragment in skip_fragments):
                continue
            if any(rel.endswith(suf) for suf in skip_suffixes):
                continue
            full = os.path.join(repo, rel)
            arcname = os.path.join(prefix, rel)
            tar.add(full, arcname=arcname, recursive=False)
PY

export FASTPLS_REMOTE_PASS

"${EXPECT_BIN}" <<EOF
set timeout -1
set pass \$env(FASTPLS_REMOTE_PASS)
spawn ssh -o StrictHostKeyChecking=accept-new ${REMOTE_USER}@${REMOTE_HOST} "mkdir -p ${REMOTE_ROOT}/archives ${REMOTE_ROOT}/logs ${REMOTE_ROOT}/results ${REMOTE_ROOT}/meta ${BASELINE_LIB_REMOTE} ${TEST_LIB_REMOTE}"
expect {
  "*assword:" { send "\$pass\r"; exp_continue }
  eof
}
EOF

"${EXPECT_BIN}" <<EOF
set timeout -1
set pass \$env(FASTPLS_REMOTE_PASS)
spawn scp -o StrictHostKeyChecking=accept-new "${BASELINE_ARCHIVE}" "${TEST_ARCHIVE}" ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ROOT}/archives/
expect {
  "*assword:" { send "\$pass\r"; exp_continue }
  eof
}
EOF

REMOTE_BOOTSTRAP_SCRIPT="${TMPDIR}/remote_prepare_bootstrap.sh"
cat > "${REMOTE_BOOTSTRAP_SCRIPT}" <<'EOS'
#!/bin/sh
set -eu
REMOTE_ROOT="${1}"
BASELINE_LIB_REMOTE="${2}"
TEST_LIB_REMOTE="${3}"
BASELINE_COMMIT="${4}"
TEST_COMMIT_HEAD="${5}"
TEST_COMMIT_NOTE="${6}"

rm -rf "${REMOTE_ROOT}/baseline_src" "${REMOTE_ROOT}/test_src"
mkdir -p "${REMOTE_ROOT}/baseline_src" "${REMOTE_ROOT}/test_src" "${REMOTE_ROOT}/meta" "${REMOTE_ROOT}/logs" "${REMOTE_ROOT}/results"

tar -xzf "${REMOTE_ROOT}/archives/fastpls_baseline.tar.gz" -C "${REMOTE_ROOT}"
tar -xzf "${REMOTE_ROOT}/archives/fastpls_test.tar.gz" -C "${REMOTE_ROOT}"

printf '%s\n' "${BASELINE_COMMIT}" > "${REMOTE_ROOT}/meta/baseline_commit.txt"
printf '%s\n' "${TEST_COMMIT_HEAD}" > "${REMOTE_ROOT}/meta/test_commit.txt"
printf '%s\n' "${TEST_COMMIT_NOTE}" > "${REMOTE_ROOT}/meta/test_commit_note.txt"

Rscript -e "pkgs <- c('data.table','ggplot2'); miss <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]; if (length(miss)) install.packages(miss, repos = 'https://cloud.r-project.org')" > "${REMOTE_ROOT}/logs/install_deps.log" 2>&1

mkdir -p "${BASELINE_LIB_REMOTE}" "${TEST_LIB_REMOTE}"
R CMD INSTALL --preclean --no-multiarch -l "${BASELINE_LIB_REMOTE}" "${REMOTE_ROOT}/baseline_src" > "${REMOTE_ROOT}/logs/install_baseline.log" 2>&1
R CMD INSTALL --preclean --no-multiarch -l "${TEST_LIB_REMOTE}" "${REMOTE_ROOT}/test_src" > "${REMOTE_ROOT}/logs/install_test.log" 2>&1

Rscript -e "suppressPackageStartupMessages(library(fastPLS, lib.loc='${BASELINE_LIB_REMOTE}')); cat(packageVersion('fastPLS'), '\n'); cat('has_cuda=', fastPLS::has_cuda(), '\n', sep='')" > "${REMOTE_ROOT}/logs/verify_baseline.log" 2>&1
Rscript -e "suppressPackageStartupMessages(library(fastPLS, lib.loc='${TEST_LIB_REMOTE}')); cat(packageVersion('fastPLS'), '\n'); cat('has_cuda=', fastPLS::has_cuda(), '\n', sep='')" > "${REMOTE_ROOT}/logs/verify_test.log" 2>&1
EOS
chmod +x "${REMOTE_BOOTSTRAP_SCRIPT}"

"${EXPECT_BIN}" <<EOF
set timeout -1
set pass \$env(FASTPLS_REMOTE_PASS)
spawn scp -o StrictHostKeyChecking=accept-new "${REMOTE_BOOTSTRAP_SCRIPT}" ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ROOT}/archives/remote_prepare_bootstrap.sh
expect {
  "*assword:" { send "\$pass\r"; exp_continue }
  eof
}
EOF

"${EXPECT_BIN}" <<EOF
set timeout -1
set pass \$env(FASTPLS_REMOTE_PASS)
spawn ssh -o StrictHostKeyChecking=accept-new ${REMOTE_USER}@${REMOTE_HOST} "sh '${REMOTE_ROOT}/archives/remote_prepare_bootstrap.sh' '${REMOTE_ROOT}' '${BASELINE_LIB_REMOTE}' '${TEST_LIB_REMOTE}' '${BASELINE_COMMIT}' '${TEST_COMMIT_HEAD}' '${TEST_COMMIT_NOTE}'"
expect {
  "*assword:" { send "\$pass\r"; exp_continue }
  eof
}
EOF

echo "Remote prepare complete"
echo "REMOTE_ROOT=${REMOTE_ROOT}"
echo "BASELINE_COMMIT=${BASELINE_COMMIT}"
echo "TEST_COMMIT=${TEST_COMMIT_NOTE}"
