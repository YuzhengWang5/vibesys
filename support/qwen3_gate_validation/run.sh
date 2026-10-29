#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
run="$1"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?gpuctl must allocate one GPU}"
nvidia-smi > "$run/logs/nvidia-smi.txt"
printf '\nCUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES" >> "$run/README.md"
for control in positive negative; do
  mkdir -p "$run/$control"
  cp -a "$repo/examples/model-serving/qwen3-32b-code-edit/.vibesys" "$run/$control/"
  cp "$repo/support/qwen3_gate_validation/launcher.sh" "$run/$control/run_server.sh"
  extra=0
  if [[ "$control" == negative ]]; then extra=1; fi
  set +e
  docker run --rm --gpus "device=$CUDA_VISIBLE_DEVICES" \
    -v "$repo:/source:ro" -v "$run:/run" -v /mnt_ssd/data1/models/Qwen3-32B:/model:ro \
    -e CUDA_VISIBLE_DEVICES=0 -e CONTROL_CORRUPT="$extra" \
    -e PYTHONPATH=/source/sdk/vs-bench/src \
    -w "/run/$control" vibesys-qwen3-logits-validation:20261029 \
    python .vibesys/tasks/default/accuracy_checker/checker.py --managed \
    --tokenizer-path /model --seed 42 > "$run/logs/$control.log" 2>&1
  status=$?
  set -e
  printf '%s\n' "$status" > "$run/logs/$control.exitcode"
  if [[ "$control" == positive && "$status" != 0 ]]; then exit "$status"; fi
  if [[ "$control" == negative && "$status" == 0 ]]; then exit 1; fi
done
