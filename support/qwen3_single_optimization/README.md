# Qwen3 单优化实验控制器

操作者工具，不是框架功能。只写本目录和新实验产物，不改 task bundle、
gate 或 SDK。`prepare` 不启动 GPU；只有 `dispatch`、`queue` 与内部 `worker` 执行实验。
使用现有 `.venv` Python，脚本显式设置新 worktree 的 PYTHONPATH，不运行 `uv sync`。
源码必须先由操作者 checkpoint，且 prepare/执行时干净、HEAD 与 pin 一致。

## 流程

在 worktree 根目录：

```bash
bash support/qwen3_single_optimization/control.sh prepare pilot
```

输出 pilot run 路径；方向目录为 `exps/<时间戳>_<worktree同名>/`。
也可 `prepare pilot --experiment /absolute/exps/existing-direction` 复用已有
README.md、CODE_PIN.md、runs/ 的指定方向，不新建父方向、不改写既有方向 README。
CODE_PIN 只追加新 run 的 pin。没有 spec/project 的 setup/calibration run 不参与计划；
出现 project、准备标记或控制器 run 命名但缺少 spec/status 时明确报错，
不会当作 setup 跳过。重复 prepare 同一 phase 拒绝覆盖，恢复使用 retry。
每次 run 是独立导出 candidate Git，只有全新的输入初始 commit；
框架源码不在 candidate 内，无旧实验代码或历史。SDK 通过
`materialize_input_project` 公开 API 准备，保留 skills。
清理后的 OBJECTIVE、root/benchmark/reference README 均由模板生成；
评测实现及新 gate 保持原样。新 checker 使用经过独立 HF/vLLM 校准的固定参考。

```bash
control=support/qwen3_single_optimization/control.sh
bash "$control" queue --experiment /absolute/exps/direction --phase pilot
bash "$control" status --experiment /absolute/exps/direction
bash "$control" logs --run /absolute/pilot --tail 100
bash "$control" inspect --run /absolute/pilot
```

默认队列串行；正式 batch 可分别传 `--optimization cuda-graph`、
`--optimization speculative-decoding`、`--optimization flashinfer`、
`--optimization operator-fusion`，每个方向队列内部按 repeat 串行三次。
批次固定 12 正式 slot，四方向各三个独立 trial；默认队列按 repeat 优先轮转四方向，
不是先跑完一个方向。每个 trial 仍固定五轮 outer-loop=agent，不是增加到 12 或 15 轮。

通过 pilot 审查并 prepare 完整 12 slot 后，可自由选择任意独立 slot：

```bash
bash "$control" dispatch --run /absolute/exps/direction/runs/selected-slot
```

每个 dispatch 在宿主前台等待该 run 完成；操作者可从不同终端并发调用，
同方向的独立 trial 不要求串行。两组实验合计最多八个 run 的资源上限由操作者
控制，本工具没有跨方向/跨 experiment 的全局调度器；应公平交错选择两组及四方向。
dispatch 与 queue 共用每 run 的启动领取锁、durable intent 与输入 pin 检查，
每次 dispatch 都重验完整计划及真实 pilot approval，不能借单 run 入口绕过审批。
同一个 run 的重复 dispatch 或与 queue 竞争最多只有一个领取者。
每 run 另有独立 worker 执行锁；gpuctl 为每次 run 全程租一张卡，
最多等待租约 12 小时。不要同时启动默认全量队列和分方向队列。
启动前保存 nvidia-smi，显式记录/export CUDA_VISIBLE_DEVICES，拒绝卡上已有计算进程。
每次框架预算固定 outer-loop=agent、所有角色显式指定 Copilot CLI gpt-6.1-sol、
max-rounds=5、max-in-flight=1、evaluation-slots=1；单 run 安全墙钟上限 6 小时。
进程被中断时先转发中断并等待，之后仅清理本 run 标签的 Docker 容器。

## Pilot 开闸

**进程退出 0 不等于通过。** `inspect` 经 `Project.stored_record`、
multi-agent registration 的 canonical codec 与公开 receipt ledger 读取真实 core，
不猜测 `.vibesys` 路径，不从日志关键字推断成功。要求：

- core terminal、result success、最终 adoption verified、所有测量 job terminal/released；
- A/B 各自不可变 revision 都有 accepted trusted correctness 和 benchmark；
- 只接受 baseline/official purpose，不接受 agent local-validation/self-report；
- 相同 evaluator/workload/environment digest，候选吞吐严格高于基线；
- candidate revision 是最终验证采纳的 revision。

当前 CLI 的 outer-loop=agent 对应串行 multi-agent（明确传入 inner-loop），
不是 dynamic；max-in-flight/evaluation-slots/max-backlog 这些 CLI 项在此路径
不进入 policy descriptor，实际串行性由该策略和操作者队列保证。

数值证据不能证明任意程序只改变一个因素或 logprobs 真正共享生产 forward。
操作者必须审查 A/B 源码 diff、未经优化 eager BF16 AR 基线及生产路径，
填写 `pilot-review.example.json`，将已验证的四项声明改为 true。
模板默认 false，不能不经审查直接通过。这个语义审计不能安全地自动猜测。

implementer 的 in-turn evaluation 工具仍按既有契约产生 trusted
`local-validation`，不会因为调用只写 accuracy 就跳过已配置的 benchmark；
这些数字不能用于开闸。候选准备就绪后须提名，由框架执行 judge 和正式 gates，
再由操作者读取 baseline/official receipt。工具的空 artifact 列表不表示宿主
原始 stream 丢失，也不授予候选读取隐藏日志的权限；操作者通过公开 Project
日志位置审查原始输出，不要求候选在提名前等待后置审查或虚构原始包。

```bash
bash "$control" approve-pilot --run /absolute/pilot --review /absolute/review.json
bash "$control" prepare batch --experiment /absolute/exps/direction
bash "$control" queue --experiment /absolute/exps/direction --phase batch
```

batch 为 cuda-graph / speculative-decoding / flashinfer / operator-fusion 各三次独立 trial。
每份 OBJECTIVE 仅出现对应唯一方向，不列其他方向，不预设具体方案。
开闸文件绑定 pilot 的源码 checkpoint、OBJECTIVE hash、A/B 可信测量与审计；
prepare/queue/dispatch batch 都重读真实证据。改变源码 checkpoint 后必须重新 pilot，
不能沿用旧 approval。

## 失败、恢复与记录

README、CODE_PIN 按 run 记录真实 HEAD；OBJECTIVE 快照、SHA256SUMS、
脚本、spec 位于 `scripts/`，状态与日志位于 `logs/`，结束写 summary。
框架日志、portable core 与 receipt ledger 经公开 Project 路径 API 归档。
源码、原始评测输出、独立 candidate Git 均保留在对应 run。

失败后队列停止，不自动降精度、放宽 gate 或重跑同一 attempt：

```bash
bash "$control" retry --run /absolute/failed-run
bash "$control" queue --experiment /absolute/exps/direction --phase batch
```

retry 新建 run 并链接 retry_of，不覆盖旧记录。运行中或启动状态不明时拒绝
retry；应先核实原 gpuctl 进程及本 run 资源已停止，不能用修改状态文件跳过审查。
若某正式实验确实耗尽五轮算法预算，操作者审查并记录失败后，可在 batch queue
显式传 `--acknowledged-failure /absolute/failed-run` 继续其余 slot；可重复指定。
原实验仍保留 failed，不算成功，也不额外授予优化轮次。新发生的失败仍停止队列。
基础设施故障不可据此伪装成算法失败，必须先修复并按 checkpoint/pilot 规则恢复。
源码 checkpoint 改变后，可 `retry` 旧 pilot 新建独立 pilot；审查通过后执行
`approve-pilot --run /new/pilot --review /new/review.json --supersede-checkpoint OLD_SHA`。
完整重跑可信证据与源码审查，旧 approval 按内容 hash 原样保存在
`pilot-approvals/`，再原子发布新 approval；同 checkpoint 或未显式确认旧 pin
均拒绝替换。已完成的正式 slot 保留原测量和原 pin，不追加优化预算。
尚未启动但 pin 已过期的 prepared slot，也须在新 pilot 审查后逐个 `retry`
创建独立新 attempt，绝不改写旧 spec；运行中/启动不明的 attempt 不能据此 retry。
worker 在任何外部动作前持久化领取状态；意外中断或 gpuctl 启动结果不明时，
保留 launching/running intent，不把宿主非零退出猜成安全失败，不自动 replay 或 retry。
仅 worker 已确认结束并写入终态的 attempt 才能按原规则 retry；不明状态须先人工核验
原进程与资源，不能修改状态文件绕过审查。
所有命令应由同一操作者执行；此工具不是抵御恶意宿主操作者的安全边界。

## CPU 验证

不调用 GPU/agent/Docker 的纯契约和模板测试：

```bash
export PYTHONPATH="$PWD/support:$PWD/src:$PWD/packaging:$PWD"
for path in "$PWD"/libs/*/src "$PWD"/sdk/*/src; do
  export PYTHONPATH="$PYTHONPATH:$path"
done
/mnt_ssd/data1/wyz/projects/kb/srcs/vibesys/.venv/bin/python -m pytest \
  support/qwen3_single_optimization/tests -q \
  --basetemp="$PWD/.pytest_cache/qwen3-operator"
```
