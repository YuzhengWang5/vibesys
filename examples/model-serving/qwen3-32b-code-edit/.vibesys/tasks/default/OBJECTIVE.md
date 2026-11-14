# Qwen3-32B code-edit：最大聚合吞吐

在 CodeEditorBench code-debug 工作负载上优化 Qwen/Qwen3-32B 的多请求服务能力。
唯一性能目标是 benchmark JSON 的 **`max_throughput_tok_per_sec`**：
在单个 **5 req/s open-loop Poisson 到达负载**下，最大化
`所有逻辑请求成功输出的总 token / 从首个计划到达到最后请求完成的墙钟秒`。
不 sweep request rate。客户端不设并发上限、不等待响应再派发；
实际并发、排队和 batch 由引擎决定。不是固定并发优化，也不是单请求中位 tok/s。
固定样本、到达随机种子和生成预算，保留实际峰值在途数、派发延误和重试记录。
引擎拒绝、drop、连接中断、截断流或空输出后重试同一原始请求，
退避、失败尝试和尾部排空均计入墙钟；仅最终成功输出计一次 token。
总完成期限内仍有失败或客户端发压滞后则整次评测失败，不得丢弃困难请求抬高成绩。

先建立正确可测的 eager BF16 AR 服务基线。支持多请求、正确的请求级 KV 隔离
和服务生命周期是工作负载基础要求，不可通过串行客户端伪造并发吞吐。
先提名基线供框架正式 accuracy/benchmark，再提交优化候选；
in-turn 自测不能代替 baseline/official receipts。如无收益，应保留可信 incumbent。

## 可选优化与 outerloop 纪律

基线通过后，由 agent 根据证据自行选择大优化阶段的顺序。
以下五个大优化点各自只允许一个连续的 outerloop 区间，列举顺序不代表实施顺序：

1. FlashInfer。
2. 算子融合。
3. 动态组 batch：将 continuous batching 与 dynamic batching 合为同一个大优化点，
   包括迭代间请求加入、退出、请求状态管理，以及多个请求的 prefill、decode
   和二者混合 batch 的组批与调度；这些不是可以在后面另开阶段的不同优化。
4. 从当前请求原 prompt 提取 n-gram 的投机解码，包括提案查找、长度、
   验证及提交策略；例如四 token 与十六 token 的方案属于同一个大优化点。
5. CUDA Graph。

每个 outerloop 最多实施当前大优化内的一项具体改进，不把不同大优化捆在一起。
同一个大优化可以连续占用多个 outerloop，期间可以尝试不同具体设计；
但一旦 agent 判断本阶段完成并切换到另一个大优化，前一阶段永久关闭，
后面的任何 outerloop 都不得返回它、改名重开它，或把它的新设计夹带进其他阶段。
不得出现“动态组 batch -> n-gram -> 动态组 batch”或
“n-gram 四 token -> CUDA Graph -> n-gram 十六 token”这样的交错。
在设计理由中明确当前大优化、已关闭大优化和本轮唯一具体改进；
切换前解释为什么结束当前阶段，并汇总其起点、正式结果和最终保留决定。
五个大优化共享下述 20 轮总预算，不是每个各 20 轮；
自主安排各阶段的轮数，未尝试或预算耗尽未完成的阶段如实报告。

后续阶段必须基于上一阶段结束时的可信 incumbent，保留已经验证的功能，
可以对新组件做必要的正确性适配，但不能重新设计已关闭优化来获取新收益。
如确需重新设计已关闭项，应明确报告本实验范围受阻并停止，不能偷换归属。
每轮 judge 应检查与父 revision 的生产差异及实际激活路径，核验改动归属，
不能仅凭 hypothesis 前缀把跨项优化算到当前阶段。
连续性与禁止重开的规则由外层按稳定前缀监督；具体代码归属仍需源码审查，
不以自动分类代替判断。

每轮 innerloop 最多五次 implementer 机会，力争一轮内完成实现、排障和正式验证；
确有必要时，允许多个 outerloop 继续同一项，不能在同一轮顺便叠加另一项。
outerloop 必须使用 agent 模式，总计最多 20 轮（含基线修复轮）；
serial multi-agent 启动须显式设置
`--outer-loop agent --inner-loop multi-agent --max-rounds 20 --max-retries-per-round 5`。

每项都以可信 incumbent 为基础，使用同一 accuracy gate 和完整单点 5 RPS benchmark
比较。只有正式验证通过且有可信吞吐收益才保留；失败或无收益时保留原 incumbent，
由 agent 决定留在当前阶段继续、永久关闭当前阶段后换方向或停止，
明确说明证据与局限。无收益的具体尝试也计入本阶段轮数和费用。
不得伪称完成、放宽门禁或用自测结果替代正式 receipt。

## 按大优化报告性能提升

每个连续阶段记录进入时的正式 incumbent revision 与吞吐，阶段内每轮的正式结果，
以及离开时最终保留的、通过 accuracy 的 revision 与吞吐。
本阶段净提升为 `离开时保留吞吐 - 进入时吞吐`，
相对提升为 `离开时保留吞吐 / 进入时吞吐 - 1`；
不能只取阶段最后一次尝试、加总各轮百分比，或把已放弃的快照当作已保留。
所有尝试失败或无收益时，保留入口 incumbent，净提升记零，并展示真实负结果。
这些是依赖先前阶段的累计实现上的边际提升，不是各优化独立消融或统计显著性结论。
同负载、同设备、同存储的本次新基线用于比较；不得拿上次实验的成绩充当新测量。
阶段结束原因、未完成项、容量或生成输出变化、正确性与激活证据都要一并报告。

## 输入与精度

- 请求只发送原 prompt，不包含独立草稿或答案字段。n-gram 投机草稿只能从
  **当前请求原 prompt 的 token** 查找，不能读取 gold、参考文件或历史请求答案。
  草稿必须由同一生产 target forward 验证，禁止 echo、答案缓存及猜测 gate 输出。
- 前台 `run_server.sh --model-dir /model --port PORT`，真实模型就绪后 health 才返回 200；
  streaming `/v1/completions`，原样 tokenize preformatted prompt，关闭 thinking。
- 正确重建 Unicode SSE；输出 token 按完整文本重新 tokenize，不按 chunk 计数。
  每个请求均真实执行模型，保持请求状态隔离、EOS 和 max_tokens 语义。
- 实现 `accuracy_checker/README.md` 的 `/v1/logprobs` 和 SSE token ID 观测，
  复用生产 forward、权重、精度、attention、KV 及实际启用的优化，不另设 HF 旁路。
  HF BF16 参考固定：54 个 teacher-forced 位置，双向 top-5 至少 95%，
  HF top-1 token 的 logprob 最大绝对差 **<=0.2**。
- 本次不实施权重量化，保留 BF16 权重、固定参考和上述门槛；不得静默改变精度。
- 不得修改 checker、reference、manifest、benchmark、SDK 或门槛；
  同一 workload、设备、生成参数和 open-loop 到达负载下比较可信 revision，
  保留激活证据与局限。
- 任务约束优先于通用 skills 的生产性能建议；不要为了满足通用 floor 提前叠加优化。

## 优化身份与 token 记录

每轮 hypothesis id 必须用当前阶段的稳定前缀：
`flashinfer-`、`fusion-`、`dynamic-batching-`、
`prompt-ngram-`、`cuda-graph-`；
continuous batching 及 prefill/decode 共同组批统一使用 `dynamic-batching-`，
不得另设 `continuous-batching-` 阶段。
`baseline-` 仅用于首次优化前的 eager 基线工程修复，不得在优化间插入基线轮来重开阶段。
同一大优化跨轮保持前缀不变；前缀一旦切走不得再次使用。
逐轮报告当前大优化、具体改进、重试次数、正式测量和保留决定，
便于 operator 同时汇总每轮与每个连续大优化阶段的实现成本。
token 由外层从 CLI 实际用量统计，不由 agent 猜测；输入、输出、缓存、
reasoning 和各角色分列，重试也计入当前项，缺失统计不能当作零。
