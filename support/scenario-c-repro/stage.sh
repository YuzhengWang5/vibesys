#!/usr/bin/env bash
set -euo pipefail

SOURCE="$1"
EXP="$2"
MODE="$3"
[[ "$MODE" =~ ^(ar|eagle3|optimize)$ ]]
GPU="${CUDA_VISIBLE_DEVICES:?an explicit GPU allocation is required}"
SETUP="$EXP/runs/${SCENARIO_SETUP_RUN:-20261020_152200_setup}"
SHIM="/mnt_ssd/data1/wyz/projects/kb/src/agentshim"
MODEL="/mnt_ssd/data1/models/Qwen3-32B"
DRAFT="/mnt_ssd/data1/models/Qwen3-32B_eagle3"
IMAGE="vllm/vllm-openai@sha256:6ee133356d49ed24cee03dd0cda3d3980279a11c107ce3e752c89e76540f41c9"
RUN_ID="$(date -u +%Y%m%d_%H%M%S)_$MODE"
RUN="$EXP/runs/$RUN_ID"
VS_RUN_ID="$(basename "$SOURCE")-$RUN_ID"
mkdir "$RUN"
mkdir "$RUN/scripts" "$RUN/logs" "$RUN/plots"
cp "$SETUP/scripts/"* "$RUN/scripts/"
exec > >(tee "$RUN/logs/driver.log") 2>&1
export PATH="$SOURCE/.venv/bin:/home/yuzhengw/.local/bin:$PATH"
export GIT_AUTHOR_NAME=wangyuzheng GIT_AUTHOR_EMAIL=yuuzhwang@gmail.com
export GIT_COMMITTER_NAME=wangyuzheng GIT_COMMITTER_EMAIL=yuuzhwang@gmail.com
export CUDA_VISIBLE_DEVICES="$GPU"
export HOME="$RUN/logs/home"
export COPILOT_HOME="$HOME/.copilot"
export VIBESYS_STATE_HOME="$RUN/logs/vibesys-state"
export HF_HOME="$RUN/logs/huggingface"
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1
mkdir -p "$COPILOT_HOME"
chmod 700 "$HOME" "$COPILOT_HOME"
cp "$SETUP/logs/auth-home/.copilot/config.json" "$COPILOT_HOME/config.json"
chmod 600 "$COPILOT_HOME/config.json"
CONTAINER=""
CHILD=""
finish() {
    local code=$?
    trap - EXIT INT TERM
    if [[ -n "$CHILD" ]]; then
        if kill -0 "$CHILD" 2>/dev/null; then
            kill -TERM "$CHILD"
            wait "$CHILD" || code=1
        fi
    fi
    if [[ -n "$CONTAINER" ]]; then
        docker logs "$CONTAINER" > "$RUN/logs/server.log" 2>&1 || code=1
        docker rm -f "$CONTAINER" || code=1
    fi
    while read -r owned; do
        if [[ -n "$owned" ]]; then
            docker rm -f "$owned" || code=1
        fi
    done < <(docker ps -aq --filter "label=vibesys.run-id=$VS_RUN_ID")
    printf '%s\n' "$code" > "$RUN/logs/exit-code.txt"
    if (( code == 0 )); then
        printf '# 运行结束\n\n命令退出码为 0；是否完成有效测量以 benchmark.json 或 VibeSys 逐轮记录为准，不自动宣称复现成功。\n\n模式：%s；源码 commit：%s。\n\n建议：保留，待人工核实质量与结果。\n' \
            "$MODE" "$(git -C "$SOURCE" rev-parse HEAD)" > "$RUN/summary.md"
    else
        printf '# 运行失败\n\n模式：%s；退出码：%s；原因见 logs/driver.log 与 server.log。\n\n源码 commit：%s。失败不产生成功结论；下一次尝试必须新建 run。\n\n建议：保留本地诊断，勿 push。\n' \
            "$MODE" "$code" "$(git -C "$SOURCE" rev-parse HEAD)" > "$RUN/summary.md"
    fi
    exit "$code"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
test -z "$(git -C "$SOURCE" status --porcelain)"
test -z "$(git -C "$SHIM" status --porcelain)"
read -r MEMORY UTIL UUID < <(
    nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu,uuid \
        --format=csv,noheader,nounits | tr -d ','
)
if (( MEMORY > 100 || UTIL > 0 )); then
    echo "GPU $GPU busy outside this run: $MEMORY MiB, utilization=$UTIL; refusing." >&2
    exit 1
fi
git clone --quiet --no-hardlinks "$SETUP/candidate" "$RUN/candidate"
CANDIDATE="$RUN/candidate"
ln -s "$MODEL" "$CANDIDATE/.vibesys/tasks/default/reference/model"
ENGINE_COMMIT="$(git -C "$SOURCE" rev-parse HEAD)"
SHIM_COMMIT="$(git -C "$SHIM" rev-parse HEAD)"
CANDIDATE_COMMIT="$(git -C "$CANDIDATE" rev-parse HEAD)"
cat > "$RUN/README.md" <<EOF
# Scenario C / $MODE

- 目的：A100 迁移复现；本 run 与其他方向无共享候选、状态、会话或日志。
- CUDA_VISIBLE_DEVICES=$GPU；设备 UUID=$UUID；分配方式=${SCENARIO_GPU_CONTROL:-gpuctl}。
- 源码：$SOURCE；branch=$(git -C "$SOURCE" branch --show-current)。
- tested_commit=$ENGINE_COMMIT；agentshim=$SHIM_COMMIT。
- candidate=$CANDIDATE；initial_commit=$CANDIDATE_COMMIT。
- agent=Copilot CLI；model=gpt-5.6-sol；reasoning=high；独立 HOME=$HOME。
- optimize：issue-tracker plain / local，max-rounds=15，无跨实验 issue store。
- baseline：vLLM 0.32.0；$IMAGE。
- 单 GPU bf16；memory-utilization=0.95；max-model-len=4096；EAGLE3 K=7。
- 请求：50，warmup=3，seed=42，Python3，temperature=0，max_tokens=512。
- 权重：/mnt_ssd/data1/models；精确 revisions 见 logs/pins.json。
- 正确性门禁沿用原始 evaluator；echo/improved 另报，不声称完整语义正确。
- 实验状态：运行中；结束后见 summary.md 和 logs/exit-code.txt。
EOF
python - "$RUN/logs/pins.json" "$ENGINE_COMMIT" "$SHIM_COMMIT" "$CANDIDATE_COMMIT" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "vibesys": sys.argv[2], "agentshim": sys.argv[3], "candidate": sys.argv[4],
    "model": "Qwen/Qwen3-32B@9216db5781bf21249d130ec9da846c4624c16137",
    "draft_model": "AngelSlim/Qwen3-32B_eagle3@74789e1a6a4b705a71d040315d911b58fb2a80ef",
    "dataset": "m-a-p/CodeEditorBench@9bd3f75915e7583feae1d0e98c09176c20a6bd1d",
    "agent_provider": "copilot", "agent_model": "gpt-5.6-sol",
}, indent=2) + "\n")
PY
sha256sum "$RUN/scripts/"* > "$RUN/logs/scripts.sha256"
printf '| %s | %s | %s | %s | %s |\n' "$RUN_ID" "$MODE" "$ENGINE_COMMIT" \
    "$SHIM_COMMIT" "$CANDIDATE_COMMIT" >> "$EXP/CODE_PIN.md"
nvidia-smi -i "$GPU" -q > "$RUN/logs/gpu-before.txt"
vibesys validate "$CANDIDATE" --task default
if [[ "$MODE" == optimize ]]; then
    vibesys --headless --outer-loop plain --project "$CANDIDATE" --task default \
        --config "$RUN/scripts/copilot-sol.toml" --agent-backend cli --cli-provider copilot \
        --model gpt-5.6-sol --allow-unvetted-model --backend cuda \
        --max-rounds 15 --tracker-backend local --run-id "$VS_RUN_ID" &
    CHILD=$!
    wait "$CHILD"
    CHILD=""
else
    EXTRA=()
    if [[ "$MODE" == eagle3 ]]; then
        EXTRA=(--speculative-config '{"method":"eagle3","model":"/draft_model","num_speculative_tokens":7}')
    fi
    CONTAINER="$(docker run -d --gpus "device=$GPU" --ipc=private --shm-size=8g \
        --label "vibesys.run-id=$VS_RUN_ID" \
        -e CUDA_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 \
        -p 127.0.0.1::8000 -v "$MODEL:/model:ro" -v "$DRAFT:/draft_model:ro" \
        "$IMAGE" --model /model --served-model-name qwen3-32b --dtype bfloat16 \
        --gpu-memory-utilization 0.95 --max-model-len 4096 "${EXTRA[@]}")"
    docker inspect "$CONTAINER" > "$RUN/logs/container.json"
    PORT="$(docker port "$CONTAINER" 8000/tcp | sed 's/.*://')"
    URL="http://127.0.0.1:$PORT"
    READY=0
    for (( i=0; i<360; i++ )); do
        if curl -fsS --max-time 5 "$URL/health" >/dev/null 2>&1; then
            READY=1
            break
        fi
        if [[ "$(docker inspect --format '{{.State.Running}}' "$CONTAINER")" != true ]]; then
            docker logs "$CONTAINER"
            exit 1
        fi
        sleep 5
    done
    if (( ! READY )); then
        echo "vLLM readiness deadline exceeded" >&2
        exit 1
    fi
    uv run --project "$CANDIDATE" --frozen python \
        "$CANDIDATE/.vibesys/tasks/default/accuracy_checker/checker.py" \
        --url "$URL" --model qwen3-32b --tokenizer-path "$MODEL" --seed 42 \
        > "$RUN/logs/accuracy.log" 2>&1 &
    CHILD=$!
    wait "$CHILD"
    CHILD=""
    uv run --project "$CANDIDATE" --frozen python \
        "$CANDIDATE/.vibesys/tasks/default/benchmark/benchmark.py" \
        --url "$URL" --model qwen3-32b --tokenizer-path "$MODEL" \
        --num-samples 50 --warmup 3 --max-tokens 512 --seed 42 \
        --output-json "$RUN/logs/benchmark.json" &
    CHILD=$!
    wait "$CHILD"
    CHILD=""
    python - "$RUN/logs/benchmark.json" <<'PY'
import json
import math
import sys

with open(sys.argv[1]) as file:
    data = json.load(file)
assert data["num_samples_sent"] == 50, data
assert data["num_steady"] == 47, data
assert data["num_failed"] == 0, data
assert data["num_completed"] == 47, data
assert math.isfinite(data["median_tok_per_sec"]) and data["median_tok_per_sec"] > 0, data
print("完整基线测量 median_tok_per_sec =", data["median_tok_per_sec"])
print("echo =", data["num_echo_input_verbatim"], "improved =", data["num_improved_over_input"])
PY
fi
nvidia-smi -i "$GPU" -q > "$RUN/logs/gpu-after.txt"
