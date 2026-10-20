#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 2 ]]; then
    echo "usage: launch.sh SOURCE EXP" >&2
    exit 2
fi
SOURCE="$(realpath "$1")"
EXP="$(realpath "$2")"
SETUP="$EXP/runs/20261020_152200_setup"
export PATH="$SOURCE/.venv/bin:/home/yuzhengw/.local/bin:$PATH"
test -f "$SETUP/READY"
test -z "$(git -C "$SOURCE" status --porcelain)"
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv \
    > "$SETUP/logs/gpus-before-reservation.csv"
exec gpuctl --count 1 --owner "$(basename "$SOURCE")" \
    --set-cuda-visible-devices --kill-after 60s -- \
    bash "$SETUP/scripts/pipeline.sh" "$SOURCE" "$EXP"
