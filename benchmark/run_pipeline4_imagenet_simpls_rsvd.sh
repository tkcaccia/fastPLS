#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

exec bash "${REPO_ROOT}/scripts/remote_run_imagenet_simpls_rsvd_classifiers.sh" "$@"
