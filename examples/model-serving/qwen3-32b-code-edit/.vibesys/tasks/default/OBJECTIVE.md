# Qwen3-32B code-edit：最大聚合吞吐

在 CodeEditorBench code-debug 工作负载上优化 Qwen/Qwen3-32B 的多请求服务能力。
唯一性能目标是 benchmark JSON 的 **`max_throughput_tok_per_sec`**：
在固定并发度 **2、4、8、16** 中，取 `总输出 token / 稳态墙钟秒` 最大值。
不是 concurrency=1，也不是单请求中位 tok/s。所有点使用相同样本和生成预算，
保留完整曲线与 `best_concurrency`；任何请求错误或空输出使整次评测失败。

先建立正确可测的 eager BF16 AR 服务基线，再围绕一个宏观优化进行实现、排障与
正式验证，不把多个独立机制捆绑。支持多请求、正确的请求级 KV 隔离和服务生命周期
是工作负载基础要求，不可通过串行客户端伪造并发吞吐。
先提名基线供框架正式 accuracy/benchmark，再提交优化候选；
in-turn 自测不能代替 baseline/official receipts。如无收益，应保留可信 incumbent。

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
- 不得修改 checker、reference、manifest、benchmark、SDK 或门槛；
  同一 workload、设备、生成参数和并发 sweep 下比较可信 revision，保留激活证据与局限。
- 任务约束优先于通用 skills 的生产性能建议；不要为了满足通用 floor 提前叠加优化。
