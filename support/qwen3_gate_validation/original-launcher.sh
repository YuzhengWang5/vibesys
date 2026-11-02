#!/usr/bin/env bash
set -euo pipefail
extra=()
if [[ "${CONTROL_ECHO:-0}" == 1 ]]; then extra=(--echo); fi
exec python /source/support/qwen3_gate_validation/server.py "$@" "${extra[@]}"
