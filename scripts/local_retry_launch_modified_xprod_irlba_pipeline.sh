#!/usr/bin/env bash

set -euo pipefail

CREDENTIALS_ENV="${CREDENTIALS_ENV:-/Users/stefano/Documents/GPUPLS/chiamaka_remote_credentials.env}"
LOCAL_LAUNCHER="${LOCAL_LAUNCHER:-/Users/stefano/Documents/fastPLS-src/scripts/run_modified_xprod_irlba_pipeline.sh}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/chiamaka/fastPLS_xprod_irlba_20260429}"
REMOTE_LAUNCHER="${REMOTE_LAUNCHER:-${REMOTE_ROOT}/run_modified_xprod_irlba_pipeline.sh}"
REMOTE_LOG="${REMOTE_LOG:-${REMOTE_ROOT}/modified_pipeline_nohup.log}"
REMOTE_PID_FILE="${REMOTE_PID_FILE:-${REMOTE_ROOT}/modified_pipeline.pid}"
RETRY_SEC="${RETRY_SEC:-120}"
CONNECT_TIMEOUT_SEC="${CONNECT_TIMEOUT_SEC:-20}"
LOCAL_LOG="${LOCAL_LOG:-/Users/stefano/Documents/GPUPLS/modified_xprod_irlba_remote_launch.log}"

if [ ! -f "${CREDENTIALS_ENV}" ]; then
  echo "Missing credentials env: ${CREDENTIALS_ENV}" >&2
  exit 1
fi

if [ ! -f "${LOCAL_LAUNCHER}" ]; then
  echo "Missing local launcher: ${LOCAL_LAUNCHER}" >&2
  exit 1
fi

set -a
. "${CREDENTIALS_ENV}"
set +a

if [ -z "${REMOTE_USER:-}" ] || [ -z "${REMOTE_HOST:-}" ] || [ -z "${REMOTE_PASSWORD:-}" ]; then
  echo "Credentials env must define REMOTE_USER, REMOTE_HOST, and REMOTE_PASSWORD" >&2
  exit 1
fi

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" >> "${LOCAL_LOG}"
}

upload_launcher() {
  /usr/bin/expect <<EOF
set timeout [expr {${CONNECT_TIMEOUT_SEC} + 40}]
set pw \$env(REMOTE_PASSWORD)
set user \$env(REMOTE_USER)
set host \$env(REMOTE_HOST)
spawn scp -O -o ConnectTimeout=${CONNECT_TIMEOUT_SEC} -o StrictHostKeyChecking=no ${LOCAL_LAUNCHER} \${user}@\${host}:${REMOTE_LAUNCHER}
expect {
  -re "(?i)password:" { send "\$pw\r"; exp_continue }
  timeout { exit 124 }
  eof
}
catch wait result
exit [lindex \$result 3]
EOF
}

start_remote_pipeline() {
  /usr/bin/expect <<EOF
set timeout [expr {${CONNECT_TIMEOUT_SEC} + 40}]
set pw \$env(REMOTE_PASSWORD)
set user \$env(REMOTE_USER)
set host \$env(REMOTE_HOST)
set cmd "chmod +x ${REMOTE_LAUNCHER}; nohup bash ${REMOTE_LAUNCHER} > ${REMOTE_LOG} 2>&1 & echo \\\$! > ${REMOTE_PID_FILE}; cat ${REMOTE_PID_FILE}"
spawn ssh -o ConnectTimeout=${CONNECT_TIMEOUT_SEC} -o StrictHostKeyChecking=no \${user}@\${host} "bash -lc '\$cmd'"
expect {
  -re "(?i)password:" { send "\$pw\r"; exp_continue }
  timeout { exit 124 }
  eof
}
catch wait result
exit [lindex \$result 3]
EOF
}

log "Retry launcher started; remote=${REMOTE_USER}@${REMOTE_HOST}; launcher=${REMOTE_LAUNCHER}"

while :; do
  if upload_launcher >> "${LOCAL_LOG}" 2>&1; then
    log "Launcher uploaded"
    if start_remote_pipeline >> "${LOCAL_LOG}" 2>&1; then
      log "Remote modified pipeline started"
      exit 0
    fi
    log "Remote start failed; retrying in ${RETRY_SEC}s"
  else
    log "Upload failed or SSH unavailable; retrying in ${RETRY_SEC}s"
  fi
  sleep "${RETRY_SEC}"
done
