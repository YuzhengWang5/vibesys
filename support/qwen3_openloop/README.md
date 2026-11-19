# Open-loop 自主优化实验

默认 code-edit 输入采用独立 Poisson 到达负载，不再固定客户端并发。
只测 5 req/s 一个点，64 样本、3 warmup、512 token 预算。
拒绝或 drop 后在逻辑请求总期限内重试，失败尝试与退避计入完整排空墙钟，
成功输出仅计一次；保留重试原因、客户端派发延误与实际在途数。
详见示例 benchmark README。

可选方向为 FlashInfer、算子融合、continuous batching、dynamic batching
（含多请求组批调度）、当前原 prompt n-gram 投机解码、CUDA Graph。
agent 自选顺序。outerloop 为 agent，最多 20 轮，
每轮 innerloop 五次 implementer 机会；同一优化可跨轮，每轮最多实施一项。
BF16 和固定 0.2 teacher-forced gate 不变。

## Token 统计

旧 Copilot JSON 事件通常没有 token 数，不能把零占位当作免费执行。
agentshim 的 `providers/copilot/usage_cli.py` 调用官方 `--usage-output-file`，
将真实计数转成现有 `assistant.usage` 事件，框架本身无需新增 provider 解析。
`Dockerfile` 将 bridge 安装在独立 operator 镜像的 PATH 首位，
native CLI 保留在 `/usr/local/bin/copilot`，不会被后续框架 feature 安装覆盖；
构建上下文须复制所记录 agentshim checkpoint 的 bridge 文件为
`copilot_usage.py`。不要直接使用未钉住的 latest CLI/镜像。
`BASE_IMAGE` 必须是 task/backend 的原始基础镜像，不是已安装 agent feature 的
成品镜像：框架还会安装 agent feature，后者会重新创建其 Python venv。
本层保持 root，最终 agent feature 负责创建及切换非 root 用户。

该文件中的每个 model usage 是会话累计值；bridge 持久化上次累计值并在 resume 时
做差。缺失前值、缺字段或计数倒退显式失败，不猜零。缓存属于输入、
reasoning 属于输出，分别分列而不重复相加。原始会话用量保存在容器 HOME 的
`.copilot-usage/`，中性逐 turn 用量保存在 framework core events。
此统计覆盖 CLI 中所有模型请求（包括内置子 agent），不把 per-agent 和 total 再相加。
使用 agentshim `0.18.4+vibesys4`（`c4679c4`）或后续修复版本：
usage-file 的 `outputTokens` 已包含 reasoning，而现有中性事件解析会将 disjoint
reasoning 加回；bridge 必须先减去这个子集，避免重复计数。
v3 旧实验保留原始事件，并在其独立 operator 投影中减去一次额外 fold，
不能将旧框架 output 总数直接冒充官方 CLI 输出总数。

中断恢复使用 agentshim `0.18.4+vibesys5`
（`ac9b5bfda789f88d48b27eedbc2c650ab48e3478`）：
bridge在调用过程中持久保存已观察用量和原始证据，不再只在正常退出后写checkpoint。
operator镜像中的bridge与宿主安装的agentshim parser必须一同更新并分别记录pin。
恢复只能使用同会话、有真实计数及明确边界的checkpoint；
缺失、旧格式或无法解释的计数仍显式失败，禁止用空字典重置累计基线。
`increment_known=False`可以带真实partial计数，不代表可靠零或完整费用；
框架中性逐turn总量仍标未知，operator可将持久证据单列为下界，不估算缺口。
容器HOME不是永久归档：运行容器删除后不能保证内部raw证据仍可读，
不能仅因写过checkpoint就宣称实验已完整封账。
此修复不扩大角色或CLI时限、不重开已终态run，也不免除失败尝试费用。

## 外层监控

```bash
PYTHONPATH="$PWD/support" .venv/bin/python -m qwen3_openloop.supervise \
  --run /absolute/experiment/run --run-id UUID
```

先准备独立项目、模型只读挂载和 `scripts/launch.sh`；启动脚本必须验证干净
tested_commit、agentshim pin、GPU 空闲，显式设置设备、模型与轮次参数。
监控器拥有子进程，每 30 秒从公共 Project API 打开的 core log 投影：

- `logs/token-costs.json`：每项优化的累计 token、每角色/调用明细、重试、
  缺失统计及轮次；规划调用在 design 确定 hypothesis 后回填。
- `logs/monitor-status.json`：当前优化、最后 gate、已完成轮次、事件停滞时间。
- `logs/alerts.jsonl`：失败 gate / agent 调用，立即记录；候选问题先由该轮
  有限 innerloop 修复，不能绕过准确性或更改可信负载。
- `logs/monitor-exit.json`：真实退出码。统计缺失、超过二十轮或超出最长评测
  仍无事件会请求合作停止；监控器崩溃也会收回自己启动的框架进程。

所有假设使用 OBJECTIVE 规定的稳定阶段前缀；未关联的规划成本单列，
未知用量不当作零。失败调用的全部或部分计数为 null 时，投影保留该调用并将其列入
missing_usage，由监控器显式报告统计缺失，不能让 null 算术异常中断告警与清理。
重试与失败成本也保留。基础设施故障由 operator 查看告警、
修复并提交 main，再用新的 run 和 tested_commit 重试，不在运行中热改源码。
本目录是外层实验工具，不修改框架默认预算，也不把人工停止误称自然成功。

经operator明确授权，可以在恢复时逐条传入
`--acknowledge-missing-usage INVOCATION_ID`。默认仍拒绝任何已结束的缺失用量；
授权仅针对给定调用，不会豁免后续新缺失。投影保留missing_usage及null调用，
另列acknowledged/unacknowledged_missing_usage，usage_complete仍为false；
Token仅为已知小计，不是完整费用，也不能按0补造。监控日志显式警告该授权。
