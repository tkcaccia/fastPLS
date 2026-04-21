#!/bin/sh

set -eu

REMOTE_USER="${REMOTE_USER:-chiamaka}"
REMOTE_HOST="${REMOTE_HOST:-137.158.224.178}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/chiamaka/fastPLS_remote_cifar100_compare}"
LOCAL_RESULTS_DIR="${LOCAL_RESULTS_DIR:-${HOME}/fastPLS_cifar100_from_chiamaka}"
EXPECT_BIN="${EXPECT_BIN:-expect}"

if [ -z "${FASTPLS_REMOTE_PASS:-}" ]; then
  echo "ERROR: FASTPLS_REMOTE_PASS is required."
  exit 1
fi

mkdir -p "${LOCAL_RESULTS_DIR}"
export FASTPLS_REMOTE_PASS

"${EXPECT_BIN}" <<EOF
set timeout -1
set pass \$env(FASTPLS_REMOTE_PASS)
spawn scp -r -o StrictHostKeyChecking=accept-new ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ROOT}/results "${LOCAL_RESULTS_DIR}/"
expect {
  "*assword:" { send "\$pass\r"; exp_continue }
  eof
}
EOF

"${EXPECT_BIN}" <<EOF
set timeout -1
set pass \$env(FASTPLS_REMOTE_PASS)
spawn scp -o StrictHostKeyChecking=accept-new ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ROOT}/cifar100_remote_compare_results.tar.gz "${LOCAL_RESULTS_DIR}/"
expect {
  "*assword:" { send "\$pass\r"; exp_continue }
  eof
}
EOF

echo "Copied results into: ${LOCAL_RESULTS_DIR}"
find "${LOCAL_RESULTS_DIR}" -maxdepth 2 -type f | sort
