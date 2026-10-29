#!/usr/bin/env bash
set -euo pipefail
scripts="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
run="$(dirname -- "$scripts")"
exec >> "$run/logs/launcher.log" 2>&1
exec gpuctl --count 1 --owner "$(basename -- "$run")" \
    --wait-timeout 12h --set-cuda-visible-devices -- \
    bash "$scripts/control.sh" worker --run "$run"
