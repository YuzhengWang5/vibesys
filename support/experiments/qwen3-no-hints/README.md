# 无实现提示盲实验的独立启动与监督

每组只使用自己的源码 worktree 和带时间戳的方向目录。优化 agent 和外部维护
Copilot 都用 `gpt-6.1-sol`，禁止读取其他组、旧轮实验、共享会话或记忆。
监督器不修改候选算法，也不改变可信评测标准。

先提交干净的源码 checkpoint，再执行：

```bash
bash support/experiments/qwen3-no-hints/prepare.sh WORKTREE EXPERIMENT_DIR GPU
bash RUN_DIR/scripts/supervise.sh WORKTREE EXPERIMENT_DIR RUN_DIR GPU
```

`prepare.sh` 建立无历史的候选仓库，调用公共 SDK 物化接口，并在不暴露 GPU 的
真实评测基础镜像中验证依赖、Qwen3 tokenizer chat template 和两个评测入口。
随后提交实际输入，包括解析成功的 `uv.lock`，记录源码与输入的真实 commit。
预检失败不会取得 GPU 租约；错误保留在新 run 的 `logs/preflight.log`。

`supervise.sh` 应由宿主的持久进程管理方式启动，不作为框架子进程。
它每 30 秒写 `logs/supervisor-health.json`，通过 gpuctl 启动本组任务。
启动检查、新 gate 失败、90 分钟无事件或进程退出会触发独立的维护 Copilot。
诊断阶段只读完整日志；确认基础设施故障后才停止本组、等待租约退出，
修复并验证源码、建立 checkpoint，再新建 run 进行 resume 或 restart。
单次维护调用限时 30 分钟，调用失败有显式日志并重试，不静默放弃监督。
无法安全处理时明确标记 `needs-human` 并停止，而不是伪造成功。

空实现缺 `run_server.sh` 是预期；普通候选失败不自动重启。
正式终态来自 `publications.json` 的 `run_ended.result`，不使用 shell 退出码
代替准确性/吞吐证据。方向级 `supervisor/active_run` 指向最新尝试，
`supervisor/supervisor.log` 和每次维护的 `decision.json` 记录介入与理由。
旧 run 的脚本、pin 和结果不覆写；不自动提交或 push `exps/`。

恢复启动时，`check-project.sh` 允许框架留在 `.vibesys/state/runs/` 下的
未跟踪运行状态；任何已跟踪文件的变更及该目录以外的未跟踪文件仍阻止启动。
新候选启动仍要求整个仓库干净。回归由
`test_candidate_startup_checks_preserve_resume_state` 覆盖。
若恢复尝试在产生事件前失败，准备脚本从该次尝试保存的 `resume_run_id`
恢复目标，不依赖尚不存在的事件日志。
