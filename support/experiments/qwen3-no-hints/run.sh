#!/usr/bin/env bash
set -euo pipefail

worktree=$(realpath "$1")
run_dir=$(realpath "$2")
gpu=$3
exec >"$run_dir/logs/launcher.log" 2>&1

finish() {
    code=$?
    trap - EXIT
    {
        printf '# 实验运行状态\n\n'
        printf '结束时间：%s\n\n' "$(date -u --iso-8601=seconds)"
        printf 'tested_commit：`%s`\n\n' "$(cat "$run_dir/scripts/tested_commit")"
        printf '框架退出码：%s。退出码不等同于通过准确性检查或取得有效吞吐。\n\n' "$code"
        printf '详细结果见 `logs/vibesys.log`、候选仓库 `.vibesys/state/` 与 `logs/vibesys-state/`。\n\n'
        printf '是否保留：待人工审阅；不自动提交或 push exps。\n'
    } >"$run_dir/summary.md"
    exit "$code"
}
trap finish EXIT

gpuctl --cards "$gpu" --owner "qwen3-no-hints-$(basename "$worktree")" \
    --set-cuda-visible-devices -- \
    bash "$run_dir/scripts/execute.sh" "$worktree" "$run_dir" "$gpu"
