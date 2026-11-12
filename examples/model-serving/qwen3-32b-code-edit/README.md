# Qwen3-32B code-edit：多并发最大吞吐

目标是在 CodeEditorBench code-debug 上提高 Qwen/Qwen3-32B 的最大聚合吞吐，
而非单请求速度。benchmark 固定扫描并发 2、4、8、16，唯一主指标为
`max_throughput_tok_per_sec`；保留完整吞吐曲线和 `best_concurrency`。
具体公式、warmup 和失败策略见
[benchmark/README.md](.vibesys/tasks/default/benchmark/README.md)。

请求只含原始 prompt、生成预算及流式协议字段，不提供独立草稿或答案。
原 prompt 已包含待修改程序；n-gram 投机解码只能从该 prompt 查找草稿，
并经真实目标模型验证。gold 只在客户端事后统计中使用。
不得回放历史答案、读取参考文件代替 forward，或通过提示注入答案。

服务提供前台 `run_server.sh --model-dir /model --port PORT`、`/health`、
streaming `POST /v1/completions`。真实模型就绪后 health 才返回 200。
`prompt_is_preformatted=true` 时原样 tokenize，关闭 thinking，保持正确 Unicode SSE。

可信 manifest 为 [vibesys.input.toml](.vibesys/tasks/default/vibesys.input.toml)。
accuracy 与 benchmark 各自在新环境通过 `managed_evaluation.py` 启动服务、
等待就绪、执行客户端并清理本次服务；提交正式评测前应停止自测服务。
启动及依赖错误必须显式失败，不能当作性能结果。

accuracy 保留已校准的 HF BF16 teacher-forced gate：54 个位置，
双向 top-5 命中率至少 95%，HF top-1 的 logprob 最大绝对差不超过 **0.2**。
[accuracy_checker/README.md](.vibesys/tasks/default/accuracy_checker/README.md)
定义 `/v1/logprobs` 及 SSE token ID 协议。观测必须复用真实生产 forward，
不得走未启用优化的参考旁路。数值通过不能替代源码路径审查。

实现范围见 [OBJECTIVE.md](.vibesys/tasks/default/OBJECTIVE.md)。
候选不得修改可信输入、门禁阈值或基准。

基线之后依次实施权重量化、FlashInfer、算子融合、原 prompt n-gram 投机解码、
CUDA Graph。每个 outerloop 只做当前一项，争取在五次 innerloop 机会内完成，
允许多轮继续同一项。serial multi-agent 启动时显式传
`--max-retries-per-round 5`；不修改整个框架的默认预算。
