#!/usr/bin/env bash
set -euo pipefail
extra=()
if [[ "${CONTROL_CORRUPT:-0}" == 1 ]]; then extra=(--corrupt); fi
exec python /source/support/qwen3_gate_validation/server.py "$@" "${extra[@]}"
