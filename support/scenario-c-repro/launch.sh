#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 || $# > 4 )); then
    echo "usage: launch.sh SOURCE EXP [SETUP_RUN [GPU_INDEX]]" >&2
    exit 2
fi
SOURCE="$(realpath "$1")"
EXP="$(realpath "$2")"
export SCENARIO_SETUP_RUN="${3:-20261020_152200_setup}"
SETUP="$EXP/runs/$SCENARIO_SETUP_RUN"
export PATH="$SOURCE/.venv/bin:/home/yuzhengw/.local/bin:$PATH"
test -f "$SETUP/READY"
test -z "$(git -C "$SOURCE" status --porcelain)"
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv \
    > "$SETUP/logs/gpus-before-reservation.csv"
if [[ $# == 4 ]]; then
    [[ "$4" =~ ^[0-7]$ ]]
    export CUDA_VISIBLE_DEVICES="$4"
    export SCENARIO_GPU_CONTROL=manual
    exec bash "$SETUP/scripts/pipeline.sh" "$SOURCE" "$EXP"
fi
export SCENARIO_GPU_CONTROL=gpuctl
exec gpuctl --count 1 --owner "$(basename "$SOURCE")" \
    --set-cuda-visible-devices --kill-after 60s -- \
    bash "$SETUP/scripts/pipeline.sh" "$SOURCE" "$EXP"
