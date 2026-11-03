#!/usr/bin/env bash
set -euo pipefail
source="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
run="${1:?absolute prepared paired run required}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?gpuctl must allocate one GPU}"
visible="$CUDA_VISIBLE_DEVICES"
nvidia-smi > "$run/logs/nvidia-smi-before.txt"
printf 'export CUDA_VISIBLE_DEVICES=%q\n' "$visible" > "$run/scripts/device.env"
printf '\n实际CUDA_VISIBLE_DEVICES=%s\n' "$visible" >> "$run/README.md"
label="qwen3.replication=$(basename "$run")"
trap 'printf "固定源码复测退出码=%s\n" "$?" >> "$run/logs/exit.log"' EXIT
read -r first second < "$run/scripts/order.txt"
for arm in "$first" "$second"; do
  revision="$(cat "$run/scripts/$arm.revision")"
  actual="$(git -C "$run/$arm" rev-parse HEAD)"
  [[ "$actual" == "$revision" ]]
  [[ -z "$(git -C "$run/$arm" status --porcelain --untracked-files=no)" ]]
  for stage in accuracy benchmark; do
    docker run --rm --gpus "device=$visible" --label "$label" \
      -v "$source:/source:ro" -v "$run:/run" -v "$run/$arm:/candidate" \
      -v /mnt_ssd/data1/models/Qwen3-32B:/model:ro \
      -e CUDA_VISIBLE_DEVICES=0 -w /candidate \
      --entrypoint uv vibesys-agent-build:e6750fe17ac3ea0a449557f57352b437 \
      run --locked python /source/support/qwen3_single_optimization/run_manifest.py \
      --project /candidate --stage "$stage" --output "/run/logs/$arm-$stage.json" \
      > "$run/logs/$arm-$stage.log" 2>&1
    printf '0\n' > "$run/logs/$arm-$stage.exitcode"
  done
  [[ "$(git -C "$run/$arm" rev-parse HEAD)" == "$revision" ]]
  [[ -z "$(git -C "$run/$arm" status --porcelain --untracked-files=no)" ]]
done
printf 'paired A/B 全部原始accuracy/benchmark命令成功；仍需操作者检查精度JSON和吞吐，未宣称稳定。\n' > "$run/logs/measure-complete.txt"
