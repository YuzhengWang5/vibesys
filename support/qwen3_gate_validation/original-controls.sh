#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
run="${1:?需要全新 run 的绝对路径}"
python=/mnt_ssd/data1/wyz/projects/kb/srcs/vibesys/.venv/bin/python
image=vibesys-qwen3-logprob-validation:20261031
gpu="${CUDA_VISIBLE_DEVICES:?请先用外层 gpuctl 分配一张卡}"
[[ "$run" == /* && ! -e "$run" ]] || { echo "run 必须是全新绝对路径" >&2; exit 2; }
[[ "$gpu" != *,* && -n "$gpu" ]] || { echo "只允许单卡租约" >&2; exit 2; }
[[ -z "$(git -C "$repo" status --porcelain)" ]] || { echo "源码必须先 checkpoint" >&2; exit 2; }
id="$("$python" -c 'import uuid; print(uuid.uuid4())')"
mkdir -p "$run/logs" "$run/scripts"
cp "$0" "$repo/support/qwen3_gate_validation/original-launcher.sh" "$run/scripts/"
git -C "$repo" rev-parse HEAD > "$run/logs/tested_commit"
docker image inspect "$image" > "$run/logs/image.json"
nvidia-smi > "$run/logs/nvidia-smi.txt"
printf 'UUID=%s\nCUDA_VISIBLE_DEVICES=%s\nsource=%s\n' "$id" "$gpu" "$repo" > "$run/logs/setup.env"
cat > "$run/README.md" <<EOF
# 原始 CodeEditorBench gate 操作者控制

源码：$repo；checkpoint 见 logs/tested_commit；单卡 CUDA_VISIBLE_DEVICES=$gpu。
UUID：$id；镜像：$image（实际 ID 见 logs/image.json）。
正控制使用真实 HF eager BF16 AR；负控制仅 echo prediction.content。
原始 checker 的 10 samples、warmup=2、512 tokens、随机 seed 均不缩减。
本目录不是 candidate seed；不运行 agent，不修改可信 gate。
EOF
names=("qwen3-original-$id-positive" "qwen3-original-$id-negative")
cleanup() {
  for name in "${names[@]}"; do
    timeout 30s docker rm -f "$name" >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
positive=1
for control in positive negative; do
  mkdir "$run/$control"
  cp "$repo/support/qwen3_gate_validation/original-launcher.sh" "$run/$control/run_server.sh"
  echo_mode=0
  if [[ "$control" == negative ]]; then echo_mode=1; fi
  started="$EPOCHREALTIME"
  set +e
  timeout --signal=TERM --kill-after=30s 8100s \
    docker run --rm --name "qwen3-original-$id-$control" \
      --label "qwen3.operator-uuid=$id" --label "qwen3.control=$control" \
      --gpus "device=$gpu" \
      -v "$repo:/source:ro" -v "$run:/run" \
      -v /mnt_ssd/data1/models/Qwen3-32B:/model:ro \
      -e CUDA_VISIBLE_DEVICES=0 -e CONTROL_ECHO="$echo_mode" \
      -e HF_HOME=/run/cache -e TMPDIR=/run \
      -e PYTHONPATH=/source/sdk/vs-bench/src -e PYTHONDONTWRITEBYTECODE=1 \
      -w "/run/$control" --entrypoint python "$image" \
      /source/examples/model-serving/qwen3-32b-code-edit/.vibesys/tasks/default/managed_evaluation.py \
      -- python \
      /source/examples/model-serving/qwen3-32b-code-edit/.vibesys/tasks/default/accuracy_checker/checker.py \
      --tokenizer-path /model --output-json "/run/logs/$control.accuracy.json" \
      > "$run/logs/$control.log" 2>&1
  status=$?
  set -e
  finished="$EPOCHREALTIME"
  timeout 30s docker rm -f "qwen3-original-$id-$control" >/dev/null 2>&1 || true
  "$python" - "$run/logs/$control.execution.json" "$control" "$started" "$finished" "$status" <<'PY'
import json
import sys
from pathlib import Path

path, control, started, finished, status = sys.argv[1:]
Path(path).write_text(json.dumps({
    "control": control, "started_unix": float(started), "finished_unix": float(finished),
    "wall_seconds": float(finished) - float(started), "returncode": int(status),
    "dependency_cold_start_seconds": None,
}, ensure_ascii=False, indent=2) + "\n")
PY
  printf '%s\n' "$status" > "$run/logs/$control.exitcode"
  if [[ "$control" == positive ]]; then
    positive="$status"
  elif [[ "$status" == 0 ]]; then
    echo "echo 负控制通过原始宽松 gate：如实保留局限，不加严。" | tee "$run/logs/echo-limitation.txt"
  fi
done
printf '正控制退出码=%s；两组原始时间、退出码与输出见 logs/。echo 通过不视为 gate 已收紧。\n' \
  "$positive" > "$run/summary.md"
exit "$positive"
