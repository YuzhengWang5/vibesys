# 轻量推理引擎精度门禁

标准 accuracy 工具、implementer 自测与 judger 正式评测均使用 manifest 命令：

```bash
python .vibesys/tasks/default/accuracy_checker/launch.py --tokenizer-path /model --seed 42
```

门禁确认引擎大致数值正确，不判断代码编辑答案是否正确。参考是操作者独立计算、
经 vLLM 校准并 checkpoint 的 `reference_logprobs.json`，不是候选生成的答案。
模型 config 和权重索引 hash 必须匹配。每次评测只加载候选模型一次，不再加载 HF；
请求只传少量 top-five 和指定 token 的 logprob，不传完整词表。

## 生产观测接口

实现 `POST /v1/logprobs`：

```json
{"input_ids":[10,20,30,40],"prefill_length":2,"decode_chunk_sizes":[1,1],"reference_token_ids":[7,8,9]}
```

每次请求重置 KV，prefill 前两个 token，再按给定 chunk teacher-force 后续 token。
返回 prefill 最后位置及所有续写位置的 next-token 分布摘要：

```json
{"positions":[1,2,3],"rows":[{"top_token_ids":[7,8,9,10,11],"top_logprobs":[-0.8,-1.8,-2.8,-3.8,-4.8],"reference_token_logprob":-0.8},{"top_token_ids":[8,9,10,11,12],"top_logprobs":[-0.8,-1.8,-2.8,-3.8,-4.8],"reference_token_logprob":-0.8},{"top_token_ids":[9,10,11,12,13],"top_logprobs":[-0.8,-1.8,-2.8,-3.8,-4.8],"reference_token_logprob":-0.8}]}
```

top-five 必须为真实完整词表 softmax 后的自然对数概率，降序且 ID 唯一；
`reference_token_logprob` 为同一行所请求的 `reference_token_ids` 的概率，即使不在
top-five 中也必须返回。不能对 top-five 单独归一化。所有数值必须有限。
位置 p 表示在固定 `input_ids[:p+1]` 前缀上的 next-token 分布；
teacher-forcing 使用固定 token，不使用两个引擎各自生成的不同前缀。

该接口必须复用 `/v1/completions` 的真实生产模型、forward、attention、KV 管理和
已启用优化。观测摘要可用普通 tensor 操作提取；模型计算不能绕过优化或另起 HF 旁路。
多 token chunk 要保证因果 mask、位置编码及 KV 更新正确。路径共享须由 judger
和操作者审查；有限数值检查本身不能证明任意候选没有实现旁路。

`/v1/completions` 请求 `return_token_ids:true` 时，每个 SSE choice 返回该 chunk
实际发出的 `token_ids`（可为空）；拼接 ID 按 `skip_special_tokens=True,
clean_up_tokenization_spaces=False` decode 后必须与拼接文本一致，最后发送 `[DONE]`。
普通 benchmark 不请求此字段，性能计数不变。

## 强度和校准

三个短代码上下文，prefill 长度 32/128/256，各续写 8 token。每个上下文分别运行
单 token 和 `[1,4,1,2]` chunk 计划，共六次请求、54 个 teacher-forced 位置。
另有两次最多 8 token 的 SSE 协议确认，覆盖空和非空 prediction。不比对完整 logits、
cosine、RMSE、尾部词表或整段生成逐字一致；允许近似并列最大值交换及优化舍入差异。

1. 至少 95% 的全部位置，同时满足 HF top-1 在候选 top-5 内，候选 top-1 在
   HF top-5 内。54 个位置可容许两个不满足的位置；缺失位置不能缩小分母。
2. 所有位置的 HF top-1 token，在候选与 HF 中的 logprob 绝对差不超过固定阈值。
   vLLM/HF 最大差 <=0.2 时阈值为 0.2，否则阈值为 vLLM 最大差 +0.2。
   原始校准值、版本与最终阈值保存在参考 JSON，不根据实验候选失败调整。

双向 top-k 思路参考 vLLM `tests/models/utils.py:check_logprobs_close`。
95% 汇总与数值阈值是本实验的轻量策略，不声称是 vLLM 原函数的默认策略。
HF 原生 BF16 eager 使用相同 cached-forward 计划；vLLM 校准在每个相同前缀后
teacher-force HF top-1，使用 prompt logprobs 取得该 token 和完整分布 top-five。
准备与正负控制见源码 `support/qwen3_gate_validation/`；这些操作者脚本不提供给候选。
