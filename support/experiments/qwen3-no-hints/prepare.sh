#!/usr/bin/env bash
set -euo pipefail

worktree=$(realpath "$1")
experiment=$(realpath -m "$2")
gpu=$3
mode=${4:-restart}
previous=${5:-}
resume_id=${6:-}
case "$mode" in restart|resume) ;; *) echo "invalid mode: $mode" >&2; exit 2 ;; esac
if test "$mode" = resume; then
    if test -z "$previous"; then
        echo 'ERROR: resume 需要上一尝试目录。' >&2
        exit 2
    fi
    if test -z "$resume_id" && test -f "$previous/scripts/resume_run_id"; then
        resume_id=$(cat "$previous/scripts/resume_run_id")
    fi
    if test -z "$resume_id"; then
        printf 'ERROR: 无法从 %s 恢复 run ID；拒绝猜测其他 run。\n' "$previous" >&2
        exit 2
    fi
    if ! test -d "$previous/logs/vibesys-state"; then
        printf 'ERROR: 缺少恢复所需的本机状态目录 %s/logs/vibesys-state；拒绝使用空状态恢复。\n' "$previous" >&2
        exit 2
    fi
    state_home=$(realpath "$previous/logs/vibesys-state")
fi
test -z "$(git -C "$worktree" status --porcelain)"
commit=$(git -C "$worktree" rev-parse HEAD)
export GIT_AUTHOR_NAME=wangyuzheng GIT_AUTHOR_EMAIL=yuuzhwang@gmail.com
export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME" GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"
export PYTHONPATH="$worktree/src"
for path in "$worktree"/libs/*/src; do
    export PYTHONPATH="$PYTHONPATH:$path"
done
python=/mnt_ssd/data1/wyz/projects/kb/srcs/vibesys/.venv/bin/python
run_dir="$experiment/runs/$(date -u +%Y%m%d_%H%M%S)_$mode"
mkdir -p "$experiment/runs" "$experiment/supervisor"
mkdir "$run_dir"
mkdir "$run_dir/scripts" "$run_dir/logs" "$run_dir/plots"
if ! test -f "$experiment/README.md"; then
    cat >"$experiment/README.md" <<EOF
# Qwen3-32B 无实现提示盲实验：$(basename "$worktree")

三次平行实验之一，独立候选仓库与模型上下文。不得读取其他组或历史实验。
源码 worktree：\`$worktree\`。每次 run 的真实源码 pin 见 CODE_PIN.md。
外部监督运行在框架之外，发现基础设施故障后先停止本组进程和释放租约，
验证修复并提交源码 checkpoint，再从最近可恢复状态继续或新建独立尝试。
不降低正确性标准，不改变评测样本或向优化 agent 提供实现方案。
EOF
fi
cp "$worktree/support/experiments/qwen3-no-hints/"* "$run_dir/scripts/"
printf '%s\n' "$commit" >"$run_dir/scripts/tested_commit"
cat >"$run_dir/README.md" <<EOF
# 无实现提示盲实验：准备与容器预检

创建：$(date -u --iso-8601=seconds)
源码：\`$worktree\`，tested_commit：\`$commit\`
模式：$mode；GPU $gpu 仅在预检成功后由 gpuctl 租用。
outer-loop=agent，Copilot CLI / gpt-6.1-sol；各组独立，禁止读取其他实验。
本次所有脚本已保存于 scripts/；依赖物化与预检详情见 logs/。
EOF
trap 'code=$?; if test "$code" -ne 0; then printf "# 准备失败\n\n退出码：%s；tested_commit：%s\n\n未启动 GPU 实验；查看 logs/materialization.log 和 logs/preflight.log。不保留为性能结果。\n" "$code" "$commit" >"$run_dir/summary.md"; fi' EXIT
printf '\n- run `%s` → tested_commit `%s`，mode `%s`\n' \
    "$(basename "$run_dir")" "$commit" "$mode" >>"$experiment/CODE_PIN.md"
if test "$mode" = resume; then
    test -n "$previous" && test -n "$resume_id"
    ln -s "$(realpath "$previous/project")" "$run_dir/project"
    ln -s "$state_home" "$run_dir/logs/vibesys-state"
    printf '%s\n' "$resume_id" >"$run_dir/scripts/resume_run_id"
else
    "$python" - "$worktree" "$run_dir/project" <<'PY' >"$run_dir/logs/materialization.log"
import shutil
import sys
from pathlib import Path, PurePosixPath
from entrypoints.examples import GitExampleMaterializer
from vs_project.api import Git
from vs_runtime.api.infrastructure import SDKRoots, materialize_input_project
from vs_sim.api import OsProcesses

root, target = (Path(p) for p in sys.argv[1:])
example = PurePosixPath("examples/model-serving/qwen3-32b-code-edit")
GitExampleMaterializer(Git(OsProcesses())).copy_tracked(
    root, example, target,
    message=(
        "Start independent no-hints input\n\n"
        "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
    ),
)
materialize_input_project(
    root / example, target,
    sdk_roots=SDKRoots(checkout=root / "sdk", packaged=None),
    copy_directory=lambda source, dest: shutil.copytree(source, dest, dirs_exist_ok=True),
    log=print,
)
(target / ".git/info/exclude").open("a").write(
    "\n/.venv/\n/.vibesys/tasks/default/reference/model\n"
)
(target / ".vibesys/tasks/default/reference/model").symlink_to(
    "/mnt_ssd/data1/models/Qwen3-32B", target_is_directory=True
)
PY
    bash "$run_dir/scripts/preflight.sh" "$run_dir"
    git -C "$run_dir/project" add --all
    git -C "$run_dir/project" add -f pyproject.toml _input_libs uv.lock
    git -C "$run_dir/project" commit --quiet -m "Materialize evaluator dependencies and CPU-verified input" \
        -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
fi
input_commit=$(git -C "$run_dir/project" rev-parse HEAD)
cat >"$run_dir/README.md" <<EOF
# 无实现提示盲实验：受外部监督的独立尝试

创建：$(date -u --iso-8601=seconds)

源码：\`$worktree\`，branch：\`$(git -C "$worktree" branch --show-current)\`

tested_commit：\`$commit\`；候选起点：\`$input_commit\`

设备：计划由 gpuctl 独占租用 GPU $gpu；实际租用在启动时追加记录。

模式：$mode；上一尝试：\`$previous\`；resume_run_id：\`$resume_id\`

outer-loop=agent，Copilot CLI / gpt-6.1-sol，24 workstreams，max-in-flight=1。
各组候选与上下文独立；禁止读取其他组及历史实验。
未提供优化实现建议；评测阈值、样本数、超时和任务接口保持不变。
独立 SDK 已物化；全新输入通过 CPU-only 容器依赖、真实 tokenizer chat template 和评测入口预检。
外部监督只修框架、依赖和生命周期问题，不设计或修改候选优化算法。

监督记录见本方向 \`supervisor/\` 和本 run 的 \`logs/supervisor-health.json\`。
EOF
printf '%s\n' "$run_dir"
