# 多并发 code-edit 最大吞吐评测

在相同 CodeEditorBench code-debug 样本上依次测量并发度 **2、4、8、16**。
主指标 `max_throughput_tok_per_sec` 是这些点中最大的 **总输出 token / 稳态墙钟秒**，
不是单请求速度、中位延迟的倒数或各请求 tok/s 之和，也不声称是未测并发下的全局最优值。
`best_concurrency` 和 `concurrency_results` 保存获胜并发度及完整曲线。

每个点使用 seed=42 的相同 50 个样本，先完成 3 个独立 warmup，再开始计时。
用封闭循环维持至多 C 个未完成请求，完成后立即补入下一请求，直到其余 47 个全部结束。
计时从派发第一个稳态请求到最后一个完成，包含排队及尾部排空，不含启动、
warmup、prompt 构建和后处理对齐分析。HTTP 连接池允许 C 个并发连接。
每个点必须有足够样本实际覆盖 C；任何 warmup/稳态请求错误或空输出均使整次评测失败，
不能丢弃失败点后选一个看起来更快的点。

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
每点保留 TTFT、TPOT、延迟分布、逐请求 token/错误及与原始代码的对齐、
`ratio_to_gold`、`ratio_to_input`、`equals_input_verbatim`。这些相似度不是语义正确性证明。
原 `median_tok_per_sec` 仅保留在每点诊断内，不是优化或终选指标。

```bash
uv run python benchmark.py --url http://localhost:8000 \
  --num-samples 50 --warmup 3 --max-tokens 512 \
  --concurrency-levels 2 4 8 16 --output-json /tmp/code_edit.json
```

默认语言为 python3，最大输入 4000 字符，greedy、关闭 thinking。
正式优化比较不得修改 sweep、样本、生成预算或计数方式。
