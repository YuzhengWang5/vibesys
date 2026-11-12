# Qwen3-32B code-edit：最大聚合吞吐

在 CodeEditorBench code-debug 工作负载上优化 Qwen/Qwen3-32B 的多请求服务能力。
唯一性能目标是 benchmark JSON 的 **`max_throughput_tok_per_sec`**：
在固定并发度 **2、4、8、16** 中，取 `总输出 token / 稳态墙钟秒` 最大值。
不是 concurrency=1，也不是单请求中位 tok/s。所有点使用相同样本和生成预算，
保留完整曲线与 `best_concurrency`；任何请求错误或空输出使整次评测失败。

先建立正确可测的 eager BF16 AR 服务基线。支持多请求、正确的请求级 KV 隔离
和服务生命周期是工作负载基础要求，不可通过串行客户端伪造并发吞吐。
先提名基线供框架正式 accuracy/benchmark，再提交优化候选；
in-turn 自测不能代替 baseline/official receipts。如无收益，应保留可信 incumbent。

## 有序优化与 outerloop 纪律

基线通过后，严格按以下顺序逐项实施，描述仅限定宏观方向：

1. 权重量化。
2. FlashInfer。
3. 算子融合。
4. 从当前请求原 prompt 提取 n-gram 的投机解码。
5. CUDA Graph。

每个 outerloop 只实施当前一项宏观优化，不把多项捆在一起，也不提前实现后续项。
每轮 innerloop 最多五次 implementer 机会，力争一轮内完成实现、排障和正式验证；
确有必要时，允许连续多个 outerloop 继续同一项，不能因换轮而顺便叠加下一项。
serial multi-agent 启动配置须显式设置 `--max-retries-per-round 5`。

每项都以上一项的可信 incumbent 为基础，使用同一 accuracy gate 和完整并发 sweep
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
- 权重量化阶段允许改变候选的权重表示，但 BF16 基线、固定参考和上述门槛不变；
  量化路径必须同时用于生产生成和数值观测。后续阶段不得静默切换精度来绕过 gate。
- 不得修改 checker、reference、manifest、benchmark、SDK 或门槛；
  同一 workload、设备、生成参数和并发 sweep 下比较可信 revision，保留激活证据与局限。
- 任务约束优先于通用 skills 的生产性能建议；不要为了满足通用 floor 提前叠加优化。
