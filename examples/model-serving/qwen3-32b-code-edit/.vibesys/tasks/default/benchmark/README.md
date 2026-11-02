# 固定吞吐评测

`benchmark.py` 是不变的可信评测，主指标为 `median_tok_per_sec`。
保持既有 CodeEditorBench code-debug、concurrency=1、seed=42、
50 samples、warmup=3、max_tokens=512 与流式模型 token 统计。
prediction 是工作负载输入，不规定算法或授权额外优化。
只执行 OBJECTIVE 指定的唯一优化；先可信测量真实 HF eager BF16 AR 基线。
保留原始输出、逐样本统计与可信 evidence；不得修改客户端实现或测量口径。
