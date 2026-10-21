#!/usr/bin/env bash
set -euo pipefail

SOURCE="$1"
EXP="$2"
SETUP="$EXP/runs/${SCENARIO_SETUP_RUN:-20261020_152200_setup}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?an explicit GPU allocation is required}"
if [[ ! "$CUDA_VISIBLE_DEVICES" =~ ^[0-7]$ ]]; then
    echo "Expected exactly one physical GPU, got: $CUDA_VISIBLE_DEVICES" >&2
    exit 2
fi
printf '%s\n' "$CUDA_VISIBLE_DEVICES" > "$SETUP/logs/allocated-gpu.txt"
gpuctl status > "$SETUP/logs/lease-at-start.txt"
status=0
for mode in ar eagle3 optimize; do
    if bash "$SETUP/scripts/stage.sh" "$SOURCE" "$EXP" "$mode"; then
        printf '%s success\n' "$mode"
    else
        code=$?
        printf '%s failed: exit=%s\n' "$mode" "$code" >&2
        status=1
    fi
done
printf '%s\n' "$status" > "$SETUP/logs/pipeline-exit-code.txt"
exit "$status"
