#!/usr/bin/env bash
set -euo pipefail

worktree=$1
run_dir=$2
gpu=$3
test "${CUDA_VISIBLE_DEVICES:?gpuctl must allocate a GPU}" = "$gpu"
export CUDA_VISIBLE_DEVICES="$gpu"
export VIBESYS_STATE_HOME="$run_dir/logs/vibesys-state"
export PYTHONUNBUFFERED=1
export GIT_AUTHOR_NAME=wangyuzheng
export GIT_AUTHOR_EMAIL=yuuzhwang@gmail.com
export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME"
export GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"

paths=("$worktree/src")
for path in "$worktree"/libs/*/src; do
    paths+=("$path")
done
PYTHONPATH=$(IFS=:; printf '%s' "${paths[*]}")
export PYTHONPATH

python=/mnt_ssd/data1/wyz/projects/kb/srcs/vibesys/.venv/bin/python
test "$(git -C "$worktree" rev-parse HEAD)" = "$(cat "$run_dir/scripts/tested_commit")"
test -z "$(git -C "$worktree" status --porcelain)"
cd "$run_dir/project"
test -z "$(git status --porcelain)"
nvidia-smi --id="$gpu" --query-gpu=index,uuid,name,memory.used,memory.total \
    --format=csv >"$run_dir/logs/gpu-before.csv"
"$python" -c 'import pathlib,sys,vibesys,vs_runtime,vs_agent; root=pathlib.Path(sys.argv[1]).resolve(); modules=(vibesys,vs_runtime,vs_agent); assert all(pathlib.Path(m.__file__).is_relative_to(root) for m in modules); print(*(m.__file__ for m in modules), sep="\n")' \
    "$worktree" >"$run_dir/logs/import-paths.log"
printf '\n实际租用：`CUDA_VISIBLE_DEVICES=%s`，容器只暴露此物理 GPU。\n' "$gpu" >>"$run_dir/README.md"

exec "$python" -c 'from entrypoints.launcher import main; main()' \
    --headless --project "$run_dir/project" --task default \
    --config "$run_dir/scripts/agent.toml" \
    --outer-loop agent --agent-backend cli --cli-provider copilot \
    --model gpt-6.1-sol --allow-unvetted-model \
    --run-environment docker --interface service --profiler none --local \
    --max-rounds 24 --max-in-flight 1 --evaluation-slots 1 --max-backlog 0 \
    --constraint "本次是独立盲实验。不得访问其他实验、其他候选仓库、历史 exps 或共享账号的历史会话和记忆；不得借助 Git refs、日志、网络或其他旁路获取这些内容。只使用本次候选仓库、当前任务输入、框架工具以及公开技术资料。" \
    >"$run_dir/logs/vibesys.log" 2>&1
