# Open-loop code-edit 最大吞吐评测

在 CodeEditorBench code-debug 样本上施加单个 **5 req/s**
的 Poisson 请求到达负载（平均速率，seed=42），不再 sweep request rate。
客户端不设置并发 semaphore 或 HTTP 连接上限，
请求按预先确定的绝对到达时间派发，不因先前响应变慢而降低到达速率。
实际并发、内部排队和 batch 大小由引擎自行决定。
主指标保留名称 `max_throughput_tok_per_sec`，含义为此单点的
**所有逻辑请求成功输出的总 token / 稳态墙钟秒**，
不是单请求速度、中位延迟的倒数或各请求 tok/s 之和，也不声称是无限负载下的全局最优值。
JSON 直接记录 `request_rate=5`、`observed_peak_in_flight`、派发延误、
相对计划到达时间的请求延迟，以及 `total_attempts`、`total_retries`、
`failed_attempts`、`retry_records`。不再输出获胜到达率或多点曲线。

使用 seed=42 的 64 个样本，先依次完成 3 个独立 warmup，
再 open-loop 派发其余 61 个请求并完整排空。高请求率用于制造饱和积压，
不是要求引擎接受特定并发数；有限 61 个到达事件是固定工作量，不是并发上限。
计时从第一个计划到达至最后响应完成，包含到达间隔、客户端延误、服务端排队及
尾部排空，不含启动、warmup、prompt 构建和后处理分析。
同 seed 下 A/B 到达计划可复现；重试不阻塞其他逻辑请求的初次到达。

每个逻辑请求总完成期限 3600 秒，覆盖服务端排队、全部尝试及退避。
HTTP 拒绝（含 429/5xx）、连接异常、SSE error、截断或空输出均重试原始请求，
退避从 0.25 秒指数增加、上限 5 秒，不无限快速重试；到总期限仍未成功则失败。
每次失败写 stderr 和 JSON，不能把错误当作成功或静默跳过。部分输出不计入主指标，
只计最终成功完整文本一次；失败尝试的开销仍全部计入墙钟。
warmup 同样重试，但 warmup 重试另列、不计入稳态指标。
任一初次实际派发比计划晚超过 1 秒，
说明客户端无法施加指定负载，整次评测无效，而不是悄悄变成 closed-loop。
任何 warmup/测量请求耗尽总期限仍失败均使整次评测失败，
不能丢弃失败请求后选一个看起来更快的分数。
必须看到 SSE `[DONE]` 且无 error event，截断流不能被算作成功。
这是有限工作量、含排空的 5 RPS 容量测量，不能
宣称已经证明所有请求率下的全局最大值。过快引擎若受此请求率限制，应由
operator 另立更高强度的新评测版本，不能让候选自行改变可信负载以抬高成绩。

## 请求与防作弊

```json
{
  "prompt": "<原始 chat-templated instruction，包含待修改代码>",
  "max_tokens": 512,
  "temperature": 0,
  "stream": true,
  "prompt_is_preformatted": true
}
```

没有额外 draft/答案字段。gold code 只用于客户端事后诊断，不进入请求。
n-gram 草稿必须由服务从本次请求原 prompt 的 token 中查找，
再由真实 target forward 验证；不得读取数据集答案、参考 gate 文件或历史请求输出。
不得把错误代码原样返回冒充模型输出。

SSE 拼接文本重新 tokenize 得到输出 token 数；chunk 数仅作诊断。
保留 TTFT、TPOT、延迟分布、逐请求 token/错误及与原始代码的对齐、
`ratio_to_gold`、`ratio_to_input`、`equals_input_verbatim`。这些相似度不是语义正确性证明。
原 `median_tok_per_sec` 仅保留在诊断内，不是优化或终选指标。

```bash
uv run python benchmark.py --url http://localhost:8000 \
  --num-samples 64 --warmup 3 --max-tokens 512 \
  --request-rate 5 --request-timeout 3600 \
  --max-dispatch-lag 1 --output-json /tmp/code_edit.json
```

默认语言为 python3，最大输入 4000 字符，greedy、关闭 thinking。
正式优化比较不得修改固定请求率、样本、生成预算、重试政策或计数方式。
