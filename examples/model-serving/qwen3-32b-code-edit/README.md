# Qwen3-32B code-edit：Open-loop 最大吞吐

目标是在 CodeEditorBench code-debug 上提高 Qwen/Qwen3-32B 的最大聚合吞吐，
而非单请求速度。benchmark 以固定平均到达率 **5 req/s** 的 Poisson 请求
施加高强度 open-loop 负载，不设置客户端并发上限；并发与 batch 由引擎决定。
唯一主指标保留名称 `max_throughput_tok_per_sec`，但只有单点实测，不再 sweep：
成功逻辑请求总输出 token / 首个计划到达到最后请求完成的墙钟秒。
引擎拒绝或 drop 必须在总期限内重试，退避及失败尝试也计时，成功输出仅计一次。
保留实际峰值在途数、到达延误、重试次数及每次失败原因。
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

本次按OBJECTIVE内17份execution plan作为有序复现阶段，不把阶段编号绑死到outer轮号。
每阶段最多3个连续outerloop，每轮最多5次inner，总上限51轮；阶段耗尽就停止累计链。
自动baseline不占阶段完成数；阶段01控制、16观测校准，其他阶段一次只做指定机制。
允许同阶段有证据的跨outer修复；不得换hypothesis名字重置机会或提前叠加后续项。
本次整链serial multi-agent显式传
`--outer-loop agent --inner-loop multi-agent --max-rounds 51 --max-retries-per-round 5`
和`--max-measured-rounds 17 --max-unmeasured-rounds 3 --judge-every 1`。
初始已消耗轮数为0，从新的eager BF16输入及阶段01开始，全部17个阶段在本次执行中完成。
可信测量计数与无测量退出限制不替代语义阶段审核。
逐轮hypothesis id须满足OBJECTIVE当前阶段的`pNN`前缀，框架轮号另记；
外层监督阶段及正式gate对应关系，judge审查实际机制和激活，不能仅以id判定成功。
不从其他实验导入候选、收据、成绩、trace、实现源码或会话。
只能使用本次执行实际产生并正式验证的候选及证据推进阶段。
所有新候选都要经过本次严格精度与完整5 RPS正式测量。
实现、正确性、性能分别报告；无收益不是实现失败，不得反复跑分择优。
性能最佳incumbent和已正确实现的前置版本分别记录，允许显式选择后者作为下一阶段base。
可按不变的lock恢复缺失运行依赖，不得把环境安装错误当作机制不可复现。
逐阶段汇总所有尝试、纠错及失败的真实Token，并保留逐outer明细，未知用量不补零。
