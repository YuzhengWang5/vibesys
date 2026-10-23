# Scenario C: three independent A100 replications

本目录仅提供实验控制脚本，不提供候选 serving 实现或优化结果。
每个方向从同一份公开 example 和固定 evaluator 修正开始，使用独立 Git
历史、Copilot HOME、VibeSys 状态和 Docker 容器；仅共享只读模型权重。
实验目录位于 `projects/kb/exps/<timestamp>_scenario-c-sol-{1,2,3}`，
与 `src/worktrees/scenario-c-sol-{1,2,3}` 一一对应。

所有角色及恢复会话都在 CLI 工具层移除并拒绝 `session_store_sql`、
`store_memory`、`vote_memory`。独立 HOME 不能隔离同账号的云端会话检索，
因此不能只靠目录隔离。此约束由 VibeSys 的 launch policy 和 agentshim 的
typed tool-exclusion contract 实施，不向候选提供其他方向的历史或建议。
已有 run 不热修改；启用修复后用新源码 checkpoint、新 setup 和新 run，
每个方向仅继承自身已接纳的候选。旧尝试保留，并如实记录越界检索尝试
及返回结果；不能将“返回空”当作原先已经存在强制隔离。
实验分支将 agentshim 固定到本地 Git 的明确 revision，而非可变源码目录，
避免 uv 复用旧 wheel 后使工具隔离配置与记录的源码 checkpoint 不一致。

`launch.sh SOURCE EXP [SETUP_RUN]` 默认通过 `gpuctl` 为整个 pipeline 获取一张 GPU。
同一方向依次测量 vLLM AR、EAGLE3，并运行 issue-tracker (`plain`) 外循环
最多 15 次性能分析，所有角色均为 Copilot CLI / high。
模型以该方向 checkpoint 中的 `copilot-sol.toml` 为唯一来源，默认
`gpt-5.6-sol`；prepare 校验该配置，stage 从复制到本 run 的同一配置读取模型，
用于 CLI 参数及所有元数据。新模型先做隔离的实际调用探测，
再在新 worktree 中修改配置并 checkpoint，不能热改正在运行的方向。
新一批独立复现必须走默认公开未实现 seed，不设置 continuation/pilot 变量；
每份独立重测 AR/EAGLE3，不沿用其他方向的测量或已学习候选。
不得绕过租约直接调用 `stage.sh`。三个方向可并行调用 `launch.sh`。
租约丢失或停止时仅清理带本次 run-id 的容器，不影响别人的任务。

### Designer outer-loop agent 模式

新实验调用 `prepare.sh SOURCE EXP SETUP_RUN agent`，使用原生
`--outer-loop agent --inner-loop multi-agent --interface service`，
由 designer 提出 hypothesis，implementer 实现，judge 审查，再由框架执行正式门禁。
模式写入 setup 的 `scripts/outer-loop.txt`，随控制脚本复制并记录摘要；
非法模式在准备或启动阶段拒绝，不会静默回落到 plain。
省略第四参数仍选择 plain，保持原有入口行为；旧 setup 必须继续使用其原有
脚本，不能混搭新 stage。`pilot` 是历史诊断入口，不用于 designer 复现。

agent 模式显式固定 `max-rounds=15`、`max-retries-per-round=3`、
`judge-every=3`；被提名候选和最后一轮仍按框架规则立即审查。
这里 15 表示 agent rounds，不表示 15 次性能分析，不能与 plain 的预算等同。
所有角色使用同一份固定 Copilot 模型配置，不从其他实验继承 roadmap、
候选或会话；新模式仍独立重测 AR/EAGLE3 后才开始搜索。
运行记录同时保存模式、模型与精确源码 commit；需从原生角色事件核验
designer（内部角色名 orchestrator）实际启动，不能只凭目录名判断模式。

用户在诊断后允许不用 gpuctl。此时必须显式调用
`launch.sh SOURCE EXP SETUP_RUN GPU_INDEX`，三个方向给出不同的空卡，
每阶段仍检查显存与利用率并显式导出 `CUDA_VISIBLE_DEVICES`。这种模式不持有
全局租约，不能保证其他用户稍后不会启动同卡任务，记录为 `manual`。
`prepare.sh SOURCE EXP SETUP_RUN` 可为重试创建新的准备 run，不能覆盖旧 run。

基线固定为 vLLM 0.32.0 的不可变镜像，bf16、memory utilization 0.95、
max-model-len 4096；EAGLE3 使用 7 个 speculative tokens。
每种基线独立发送 seed 42 的 50 个请求，前 3 个 warmup 不计入主指标。
沿用 example 的 `median_tok_per_sec` 原始公式与宽松正确性门禁，同时保留
逐请求输出、echo-input 和 improved-over-input 统计，不将该门禁当作完整语义验证。
原 checker 不接受 vLLM 必需的 `--model` 参数，因此 vLLM 基线直接运行公开
benchmark CLI，再对其 steady 样本的 `ratio_to_gold` 应用相同的 0.50/0.50
门槛，另存 `quality-gate.json`。VibeSys 候选仍运行原 checker，不修改门禁。
`--print-stream` 保留原始输出文本；GPU 阶段切换最多等待 30 秒清理尾部利用率。

权重位于 `/mnt_ssd/data1/models/{Qwen3-32B,Qwen3-32B_eagle3}`，
是固定 Hugging Face snapshot 的已解引用硬链接目录，因此容器的只读
`/model` 挂载不会包含指向挂载外 blob 的断链。
输入与源码 pin 在每次 run 开始前写入 `README.md`、`CODE_PIN.md` 和
`logs/pins.json`；实验控制脚本先 checkpoint，再原样复制到该 run。
重试必须新建 run，不覆盖失败记录。不提交或 push `exps/`。

容器的 uv cache 位于 `/home/agent/.cache/uv`，在候选工作树之外且不在方向间共享。
uv 会生成空 `.git` 标记；将 cache 放在 workspace 下会触发 host Git 的嵌套仓库
保护，在 implementer 返回后中止运行。保持该保护不变，不删除标记来绕过它。

正式评估通过只读输入中的 `managed_evaluation.py` 管理服务生命周期。
候选实现项目根目录的 `run_server.sh --model-dir /model --port PORT`，
以前台方式运行服务，模型就绪后 `/health` 才返回 200。accuracy 和 benchmark
各自使用新容器；包装器逐次启动指定候选、等待就绪、运行未经修改的原始
checker/benchmark，并在成功、失败或取消后停止自己的整个进程组。
服务日志完整写入该评估的 stderr 原始流，不进入 benchmark JSON。
不使用 host network、不借用 agent 自测服务、不更改正确性门槛。
每阶段额外预留最多 900 秒启动预算；这不计入请求吞吐指标。
自测实例必须在提交正式评估前退出，避免单卡同时加载两份 32B 权重。
新配置通过新的 prepare/run 生效，不覆盖旧实验，也不修改通用 Docker
运行环境对其他任务的行为。
管理器及正式服务契约位于仓库自带的 Qwen3 code-edit 示例中，prepare 直接
复制该实现与 Objective，不再另维护一份生命周期实现。复现 manifest 仅
比示例多 `uv run --frozen`，以使用已固定的锁文件，其他命令和预算一致。

继续同方向的已验证候选时，显式提供 `SCENARIO_CONTINUE_RUN` 和
`SCENARIO_CONTINUE_COMMIT`。prepare 只从当前 exp 内提取对应 Git 快照，
保留推理实现，采用当前正式示例的 Objective，去掉有界 pilot 的“不做优化”
限制；不复制旧会话、issue board 或失败评估状态。新 run 使用正常
15 轮 VibeSys issue-tracker 流程，不重测已经完成的基线，也不读取其他方向。

已确认的复现差异：论文使用 H100 和 Codex CLI，本实验使用 A100 80GB
和 Copilot CLI；论文未给出确切的 Scenario C vLLM 版本，不能把迁移实验的
绝对吞吐或加速比直接称作论文数值复现。

流式文本和 token 计数以任务只读输入为准。此示例的原始 benchmark 已明确
对完整响应重新分词，不用 SSE chunk 数量计算吞吐。通用 implementer/judge
不得额外要求每个模型 token 都产生非空、有效 Unicode 文本：一个字符可能
横跨多个 byte-fallback token。允许输入许可的 Unicode 缓冲、空片段或
speculative chunk 分组，但不允许伪造文本、token 边界或 usage。
修复只澄清 prompt 和 Objective，不改变 checker、benchmark、质量门槛或
吞吐公式。受旧矛盾要求影响的尝试保留失败记录，在新的源码 checkpoint、
setup 和 run 中重开，不热改正在执行的输入，也不抹去已消耗的搜索预算。

### 有界首轮诊断

`prepare.sh` 默认仍从固定的公开未实现 fixture 创建新输入。
仅诊断修复时，可显式提供 `SCENARIO_PILOT_SEED_RUN` 和
`SCENARIO_PILOT_SEED_COMMIT`，从当前 exp 自己的旧 optimize run 提取固定候选，
不读取其他 exp，不复制旧状态、会话或运行进程。来源在 setup README 记录。
这种 pilot 不是新的独立搜索，也不用于计算三次复现的统计结果。

新 setup 完成后，只在 `gpuctl` 的单 GPU 租约中调用
`stage.sh SOURCE EXP pilot`，并以 `SCENARIO_SETUP_RUN` 选择该 setup。
它使用 Copilot CLI / gpt-5.6-sol，最多两轮、每轮一个 issue、每 issue 一次尝试，
整场 5400 秒。正式 accuracy 与正有限 benchmark 通过并进入下一
performance-evaluator 工作流后，公开 `RunHandle.stop()` 请求合作停止。
`logs/pilot-status.json` 区分已验证后主动停止、预算结束、超时和其他失败；
预算结束或仅有本地自测均不算打通。
