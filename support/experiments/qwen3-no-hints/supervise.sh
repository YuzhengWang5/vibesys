#!/usr/bin/env bash
set -euo pipefail

worktree=$(realpath "$1")
experiment=$(realpath "$2")
run_dir=$(realpath "$3")
gpu=$4
scripts=$(dirname "$(realpath "$0")")
python=/mnt_ssd/data1/wyz/projects/kb/srcs/vibesys/.venv/bin/python
mkdir -p "$experiment/supervisor"
exec 9>"$experiment/supervisor/lock"
flock -n 9
exec >>"$experiment/supervisor/supervisor.log" 2>&1
printf '%s\n' "$$" >"$experiment/supervisor/pid"
launcher=
last_reviewed=-1
last_review_at=0
trigger=
decision_action=
status=
review_number=0

heartbeat() {
    local running=no
    if test -n "$launcher" && kill -0 "$launcher" 2>/dev/null; then running=yes; fi
    status=$(bash "$scripts/inspect.sh" "$run_dir" --now "$(date +%s)" \
        --running "$running" --phase "$1")
}

field() {
    "$python" -c 'import json,sys; d=json.load(sys.stdin); v=d[sys.argv[1]]; print("" if v is None else v)' "$1" <<<"$status"
}

start_run() {
    printf '%s\n' "$run_dir" >"$experiment/supervisor/active_run"
    bash "$run_dir/scripts/run.sh" "$worktree" "$run_dir" "$gpu" &
    launcher=$!
    printf '%s\n' "$launcher" >"$run_dir/logs/launcher.pid"
    printf '%s 启动 %s launcher=%s\n' "$(date -u --iso-8601=seconds)" "$run_dir" "$launcher"
    last_reviewed=-1
    last_review_at=0
}

stop_run() {
    if ! kill -0 "$launcher" 2>/dev/null; then wait "$launcher" || true; return; fi
    if test -f "$run_dir/logs/framework.pid"; then
        local framework
        framework=$(cat "$run_dir/logs/framework.pid")
        if kill -0 "$framework" 2>/dev/null; then
            test "$(readlink -f "/proc/$framework/cwd")" = "$(realpath "$run_dir/project")"
            kill -TERM "$framework"
        fi
    else
        kill -TERM "$launcher"
    fi
    for ((i=0; i<60; i++)); do
        if ! kill -0 "$launcher" 2>/dev/null; then wait "$launcher" || true; return; fi
        heartbeat stopping
        sleep 5
    done
    printf 'ERROR: 本组进程未在停止宽限期内退出；拒绝在仍运行的 worktree 上修复。\n' >&2
    return 1
}

review() {
    local phase=$1 review_dir maintainer code
    review_number=$((review_number + 1))
    review_dir="$experiment/supervisor/$(date -u +%Y%m%d_%H%M%S)_${phase}_$review_number"
    mkdir "$review_dir"
    export VS_MONITOR_WORKTREE="$worktree" VS_MONITOR_EXPERIMENT="$experiment"
    export VS_MONITOR_RUN="$run_dir" VS_MONITOR_PHASE="$phase"
    export VS_MONITOR_DECISION="$review_dir/decision.json" VS_MONITOR_TRIGGER="$trigger"
    timeout --signal=TERM --kill-after=30s 1800 \
        /home/yuzhengw/.local/bin/copilot \
        -C "$worktree" --prompt "$(cat "$scripts/maintainer.txt")" \
        --model gpt-6.1-sol --allow-all-tools --allow-all-paths \
        --disable-builtin-mcps --no-custom-instructions \
        --excluded-tools=session_store_sql,store_memory,vote_memory,task,run_dynamic_workflow,list_agents,read_agent,write_agent \
        --log-dir "$review_dir/copilot" --output-format json \
        >"$review_dir/copilot.jsonl" 2>"$review_dir/stderr.log" &
    maintainer=$!
    printf '%s\n' "$maintainer" >"$review_dir/maintainer.pid"
    while kill -0 "$maintainer" 2>/dev/null; do
        heartbeat "$phase"
        sleep 30
    done
    code=0
    wait "$maintainer" || code=$?
    if test "$code" -ne 0 || ! test -s "$VS_MONITOR_DECISION"; then
        printf 'ERROR: 维护阶段 %s 未产生有效决策，exit=%s，见 %s\n' "$phase" "$code" "$review_dir" >&2
        return 1
    fi
    if ! decision_action=$("$python" - "$VS_MONITOR_DECISION" "$phase" <<'PY'
import json, sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text())
allowed = {"diagnose": {"continue", "repair", "stop"}, "repair": {"resume", "restart", "stop"}}
if set(data) != {"action", "reason"} or data["action"] not in allowed[sys.argv[2]] or not isinstance(data["reason"], str) or not data["reason"].strip():
    raise ValueError("invalid maintenance decision")
print(data["action"])
PY
    ); then
        printf 'ERROR: 无效维护决策，见 %s\n' "$VS_MONITOR_DECISION" >&2
        return 1
    fi
    cat "$VS_MONITOR_DECISION"
    printf '\n'
}

cleanup() {
    trap - INT TERM
    stop_run
    heartbeat stopped
    exit 143
}
trap cleanup INT TERM

start_run
while true; do
    if ! heartbeat monitoring; then
        printf 'ERROR: 状态读取失败；保留错误并在下次轮询重试。\n' >&2
        sleep 30
        continue
    fi
    running=$(field running)
    failures=$(field failure_count)
    idle=$(field seconds_without_events)
    now=$(date +%s)
    trigger=
    if test "$running" = False; then
        trigger="框架进程已退出；检查 publications 的 run_ended 及完整错误"
    elif test "$last_reviewed" -eq -1 && test -n "$(field run_id)"; then
        trigger="启动健康检查；确认独立依赖、模型和生命周期，没有优化实现建议"
    elif test "$failures" -gt 0 && test "$failures" -ne "$last_reviewed"; then
        trigger="新出现的真实 gate 失败：$failures；读取完整 subprocess_output"
    elif test "$idle" -ge 5400; then
        trigger="90 分钟没有结构化事件；检查是否为真实框架僵死"
    fi
    if test -n "$trigger" && test "$((now - last_review_at))" -ge 120; then
        last_review_at=$now
        if ! review diagnose; then sleep 120; continue; fi
        last_reviewed=$failures
        case "$decision_action" in
            continue)
                if test "$running" = False; then
                    wait "$launcher" || true
                    heartbeat finished
                    {
                        printf '\n## 外部监督确认\n\n'
                        cat "$run_dir/logs/supervisor-health.json"
                        printf '\n结果以 terminal.outcome/selection 为准，不以退出码替代。\n'
                    } >>"$run_dir/summary.md"
                    printf '%s 监督结束，真实终态见 %s\n' "$(date -u --iso-8601=seconds)" "$run_dir"
                    exit 0
                fi
                ;;
            repair)
                until stop_run; do heartbeat stop-blocked; sleep 120; done
                until review repair; do heartbeat repair-retrying; sleep 120; done
                case "$decision_action" in
                    resume|restart)
                        previous=$run_dir
                        run_id=$(field run_id)
                        run_dir=$(bash "$worktree/support/experiments/qwen3-no-hints/prepare.sh" \
                            "$worktree" "$experiment" "$gpu" "$decision_action" "$previous" "$run_id")
                        start_run
                        ;;
                    stop) heartbeat needs-human; exit 1 ;;
                esac
                ;;
            stop) stop_run; heartbeat needs-human; exit 1 ;;
        esac
    fi
    sleep 30
done
