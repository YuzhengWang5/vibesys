#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
run="${1:?absolute run directory required}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?gpuctl must allocate one GPU}"
nvidia-smi > "$run/logs/nvidia-smi.txt"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES" > "$run/logs/device.env"
diagnostic_status=0
for mode in hf vllm; do
  set +e
  /usr/bin/time -p -o "$run/logs/$mode.time" docker run --rm --gpus "device=$CUDA_VISIBLE_DEVICES" \
    --label "qwen3.operator-run=$(basename "$run")" \
    -v "$repo:/source:ro" -v "$run:/run" \
    -v /mnt_ssd/data1/models/Qwen3-32B:/model:ro \
    -e CUDA_VISIBLE_DEVICES=0 -e PYTHONPATH=/source/sdk/vs-bench/src \
    --entrypoint python vibesys-qwen3-logprob-validation:20261031 \
    /source/support/qwen3_gate_validation/prepare_reference.py "$mode" \
    --output /run/logs/reference_logprobs.json \
    > "$run/logs/$mode.log" 2>&1
  status=$?
  set -e
  printf '%s\n' "$status" > "$run/logs/$mode.exitcode"
  if [[ "$mode" == hf && "$status" != 0 ]]; then exit "$status"; fi
  if [[ "$mode" == vllm && "$status" != 0 ]]; then
    diagnostic_status="$status"
    printf 'vLLM诊断失败，保留非零exitcode，不改变0.02或阻止HF控制。\n' >&2
  fi
done
exit "$diagnostic_status"
