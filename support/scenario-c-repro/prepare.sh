#!/usr/bin/env bash
set -euo pipefail

SOURCE="$(realpath "$1")"
EXP="$(realpath "$2")"
SETUP_ID="${3:-20261020_152200_setup}"
SETUP="$EXP/runs/$SETUP_ID"
SEED="/mnt_ssd/data1/wyz/projects/kb/exps/20261018_115909_scenario-c/runs/20261018_120300_setup/candidate"
SEED_COMMIT="cfcdc802dd2de09afa6c2011821e577c3879e9cc"
REFERENCE="/mnt_ssd/data1/wyz/projects/kb/exps/20261018_115909_scenario-c/runs/20261018_120300_setup/candidate/.venv/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py"
export CUDA_VISIBLE_DEVICES=""
export PATH="$SOURCE/.venv/bin:/home/yuzhengw/.local/bin:$PATH"
test ! -e "$SETUP/READY"
test ! -e "$SETUP/candidate"
test -z "$(git -C "$SOURCE" status --porcelain)"
test -f "$SOURCE/support/scenario-c-repro/copilot-sol.toml"
mkdir -p "$SETUP/scripts" "$SETUP/logs" "$SETUP/plots"
cp "$SOURCE/support/scenario-c-repro/"* "$SETUP/scripts/"
mkdir "$SETUP/candidate"
git -C "$SEED" archive "$SEED_COMMIT" | tar -x -C "$SETUP/candidate"
cp "$REFERENCE" "$SETUP/candidate/.vibesys/tasks/default/reference/reference.py"
cp /mnt_ssd/data1/models/Qwen3-32B/config.json \
    "$SETUP/candidate/.vibesys/tasks/default/reference/config.json"
cp "$SOURCE/examples/model-serving/qwen3-32b-code-edit/README.md" "$SETUP/candidate/README.md"
rm "$SETUP/candidate/REPRODUCTION.md"
git -C "$SETUP/candidate" init --quiet -b "initial-$(basename "$SOURCE")"
git -C "$SETUP/candidate" config user.name wangyuzheng
git -C "$SETUP/candidate" config user.email yuuzhwang@gmail.com
git -C "$SETUP/candidate" add .
git -C "$SETUP/candidate" commit --quiet \
    -m "repro: independent unimplemented Scenario C input $(basename "$SOURCE")" \
    -m 'Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>'
mkdir -p "$SETUP/logs/auth-home/.copilot"
chmod 700 "$SETUP/logs/auth-home" "$SETUP/logs/auth-home/.copilot"
cp /home/yuzhengw/.copilot/config.json "$SETUP/logs/auth-home/.copilot/config.json"
chmod 600 "$SETUP/logs/auth-home/.copilot/config.json"
uv sync --project "$SETUP/candidate" --frozen --quiet
vibesys validate "$SETUP/candidate" --task default > "$SETUP/logs/validate.log" 2>&1
python - "$SETUP/scripts/copilot-sol.toml" <<'PY'
import sys
from pathlib import Path

from vibesys.api import agent_spec_from_config, load_config

spec = agent_spec_from_config(load_config(Path(sys.argv[1])), allow_unvetted_model=True)
assert (spec.provider, spec.model, spec.reasoning_effort) == ("copilot", "gpt-5.6-sol", "high")
PY
ENGINE_COMMIT="$(git -C "$SOURCE" rev-parse HEAD)"
SHIM_COMMIT="$(git -C /mnt_ssd/data1/wyz/projects/kb/src/agentshim rev-parse HEAD)"
CANDIDATE_COMMIT="$(git -C "$SETUP/candidate" rev-parse HEAD)"
sha256sum "$SETUP/scripts/"* > "$SETUP/logs/scripts.sha256"
if [[ ! -f "$EXP/README.md" ]]; then
    cat > "$EXP/README.md" <<EOF
# Scenario C / $(basename "$SOURCE")

目标：独立进行 15 轮 Copilot CLI / gpt-5.6-sol / high 优化，并独立测量
vLLM AR、EAGLE3 基线。硬件为 A100-SXM4-80GB，不是论文 H100。

- 源码 worktree：$SOURCE。
- 状态：输入与代码已固定，等待 gpuctl 租约后启动。
- 隔离：独立 candidate 仓库、HOME、Copilot 会话、VibeSys issue tracker、
  Docker 容器及随机模型对话；没有从其他实验候选生成的 seed。
- 共享：固定 revision 的公共权重，只读挂载；不共享生成的程序、建议或结果。
- 控制流：AR → EAGLE3 → plain issue-tracker 15 轮。
- 原始 example 未实现服务。输入沿用原始 checker/benchmark，仅固定 dataset
  revision、携带 vs-bench、补齐 Jinja2 与只读 Transformers reference。
- fixture=$SEED_COMMIT；这是未启动 GPU 搜索的输入包，不含 learned candidate。
- 论文未提供精确 vLLM 版本，此处固定 0.32.0 digest；结果属于迁移复现。
- 每次尝试独立新建 runs/<timestamp>_<mode>，设备、commit、脚本摘要记录在其中。
- 不自动提交或 push exps；结果以各 run 的原始日志和 summary 为准。
EOF
fi
if [[ ! -f "$EXP/CODE_PIN.md" ]]; then
    cat > "$EXP/CODE_PIN.md" <<EOF
# 执行代码索引

三个方向仅共享框架与公开输入，候选 Git 历史彼此独立。
下表在每个 run 启动前写入，不用后续 HEAD 回填。

| run | mode | vibesys tested_commit | agentshim tested_commit | candidate initial_commit |
| --- | --- | --- | --- | --- |
EOF
fi
printf '| %s | CPU setup | %s | %s | %s |\n' "$SETUP_ID" "$ENGINE_COMMIT" \
    "$SHIM_COMMIT" "$CANDIDATE_COMMIT" >> "$EXP/CODE_PIN.md"
cat > "$SETUP/README.md" <<EOF
# 环境准备

- 目的：固定输入、验证依赖和独立 candidate，不运行 GPU。
- CUDA_VISIBLE_DEVICES=""。
- 源码：$SOURCE；branch=$(git -C "$SOURCE" branch --show-current)。
- tested_commit=$ENGINE_COMMIT；agentshim=$SHIM_COMMIT。
- candidate initial_commit=$CANDIDATE_COMMIT。
- 模型：Copilot gpt-5.6-sol，high；目标权重位于 /mnt_ssd/data1/models。
- setup 仅为 CPU 准备，不产生性能数据。GPU 任务必须由 launch.sh 经 gpuctl 启动。
EOF
printf '# 准备结果\n\n输入校验通过；尚未进行 GPU 性能测量。\n\n建议：保留公共输入 pin 和控制脚本。\n' > "$SETUP/summary.md"
printf '%s\n' "$ENGINE_COMMIT" > "$SETUP/READY"
printf '%s ready: %s\n' "$(basename "$SOURCE")" "$CANDIDATE_COMMIT"
