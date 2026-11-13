# Qwen3-32B code-edit：最大聚合吞吐

在 CodeEditorBench code-debug 工作负载上优化 Qwen/Qwen3-32B 的多请求服务能力。
唯一性能目标是 benchmark JSON 的 **`max_throughput_tok_per_sec`**：
在 **open-loop Poisson 请求到达负载**下，取 `总输出 token / 墙钟秒` 的最大值。
请求率逐级升高，客户端不设并发上限、不等待响应再派发；实际并发、排队和 batch
由引擎自行决定。不是固定并发优化，也不是单请求中位 tok/s。各点使用相同样本、
到达随机种子和生成预算，保留完整 request-rate 曲线、峰值在途数与到达延误。
计时包含全部服务端排队及尾部排空；错误、空输出、丢请求或客户端发压滞后使评测失败。

先建立正确可测的 eager BF16 AR 服务基线。支持多请求、正确的请求级 KV 隔离
和服务生命周期是工作负载基础要求，不可通过串行客户端伪造并发吞吐。
先提名基线供框架正式 accuracy/benchmark，再提交优化候选；
in-turn 自测不能代替 baseline/official receipts。如无收益，应保留可信 incumbent。

## 有序优化与 outerloop 纪律

基线通过后，严格按以下顺序逐项实施，描述仅限定宏观方向：

1. FlashInfer。
2. 算子融合。
3. Continuous batching。
4. 从当前请求原 prompt 提取 n-gram 的投机解码。
5. CUDA Graph。

每个 outerloop 只实施当前一项宏观优化，不把多项捆在一起，也不提前实现后续项。
每轮 innerloop 最多五次 implementer 机会，力争一轮内完成实现、排障和正式验证；
确有必要时，允许连续多个 outerloop 继续同一项，不能因换轮而顺便叠加下一项。
outerloop 必须使用 agent 模式，总计最多 10 轮（含基线修复轮）；
serial multi-agent 启动须显式设置
`--outer-loop agent --inner-loop multi-agent --max-rounds 10 --max-retries-per-round 5`。

每项都以上一项的可信 incumbent 为基础，使用同一 accuracy gate 和完整 open-loop sweep
比较。只有正式验证通过且有可信吞吐收益，才保留该项并推进下一项；失败或无收益时
保留原 incumbent，继续当前项。若已证实阻塞，明确报告并停止等待 operator 决定，
不得伪称完成、悄悄跳项、放宽门禁或用自测结果替代正式 receipt。

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
`flashinfer-`、`fusion-`、`continuous-batching-`、`prompt-ngram-`、`cuda-graph-`；
基线工程修复使用 `baseline-`。同一优化跨轮保持阶段不变。
逐轮报告当前项、重试次数、正式测量和保留决定，便于 operator 汇总每项实现成本。
token 由外层从 CLI 实际用量统计，不由 agent 猜测；输入、输出、缓存、
reasoning 和各角色分列，重试也计入当前项，缺失统计不能当作零。
