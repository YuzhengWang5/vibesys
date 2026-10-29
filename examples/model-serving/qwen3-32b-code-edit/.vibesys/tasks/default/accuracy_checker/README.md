# 推理引擎精度门禁

此门禁替代代码答案的宽松相似度，不判断代码修复任务本身是否解对。
标准 accuracy 工具及正式评测均调用同一 manifest 命令：

```bash
uv run python .vibesys/tasks/default/accuracy_checker/checker.py --managed --tokenizer-path /model --seed 42
```

候选服务先启动、收集结果、完全停止，再加载未经修改的 Hugging Face
`AutoModelForCausalLM`，BF16、eager attention、禁用 TF32。
诊断 logits 使用与请求一致的 HF 原生增量计划与原生 KV cache；
实际生成决策另用完整实际前缀的无 cache forward 核验。
这避免同卡双份 32B 权重，且不使用候选自己的参考实现或 KV cache。
增量计划必须匹配：原生 HF BF16 的不同 GEMM shape 本身就可能产生
明显的全词表尾部差异，不能把 monolithic forward 当作任意分块的逐值 oracle。
任何请求、协议、非有限值、形状或数值错误均失败，不回退到文本相似度。
输出逐 case JSON 和最终 `passed`、依赖版本、固定阈值及全部诊断；
参考与候选数组属于本次 evaluator 临时目录，退出后清理。

## 生产服务的附加观测接口

实现 `POST /v1/logits`，请求：

```json
{"input_ids":[10,20,30,40],"prefill_length":2,"decode_chunk_sizes":[1,1]}
```

每次请求清空 KV；先用前 `prefill_length` 个 token 进行 prefill，再严格
按 `decode_chunk_sizes` teacher-force 后续 token，更新真实 KV。返回
prefill 最后一个位置及每个续写位置的**完整词表原始 logits**：

```json
{"positions":[1,2,3],"logits":[[0.1,0.2],[0.2,0.3],[0.3,0.4]]}
```

示例仅展示维度；真实词表长度必须等于模型 config 的 vocab_size。
不得归一化、裁剪、只返回 top-k 或重新编号。该接口必须使用
`/v1/completions` 相同权重、forward、attention、精度与 KV 管理；
优化启用时此接口也必须经过该生产路径，不能另起 HF 旁路。
这是 implementer/judge 必须审查的实现约束；数值测试本身不能证明两条
任意程序路径共享实现。多 token chunk 必须有正确 causal mask 和 KV 提交。

此外 `/v1/completions` 请求 `return_token_ids: true` 时，每个 SSE
choice 增加 `token_ids` 数组，记录该 chunk 实际发出的 token ID（可为空）。
其拼接按 `skip_special_tokens=True, clean_up_tokenization_spaces=False`
decode 后必须与 SSE 文本完全一致，EOS 允许包含在 ID 中。
普通 benchmark 不请求此字段，吞吐口径不变。

## 覆盖与阈值

6 种上下文，长度 32/127/256/511/1024/65，包含多语言代码、
重复片段和 seeded 随机合法 token。每种上下文以两种增量计划各测一次：
16 次单 token，以及 1/1/4/1/8/1 的多 token 验证；共 204 行完整词表。
请求交替长短上下文，同时覆盖独立请求重置。额外 6 次真实 SSE 生成，
覆盖空与非空 prediction；逐个实际发出的 token 在其实际前缀下与 HF
比较，不能靠返回正确的诊断 logits 掩盖错误的输出选择或错误的 draft 接受。
不足 16 token 的提前终止必须包含或接近 HF 的 EOS。

每行同时满足：

| 指标 | 阈值 |
|---|---|
| 去均值全词表 cosine | >= 0.999 |
| 去均值误差 L2 / 参考去均值 L2 | <= 0.045 |
| softmax 概率 total variation | <= 0.035 |
| 候选 argmax 在参考上的 logit regret | <= 0.25 |

实际生成 token 的参考 logit regret 同样 <= 0.25。这样允许 BF16
算子舍入及接近并列最大值的交换，同时约束全词表形状、概率质量和
greedy 决策；每行独立通过，不能以平均值掩盖错误 token。常数 logits、
NaN、Inf、错位或截断的词表直接失败。固定阈值的正确引擎正例与数值
破坏负例必须先校验；不得因候选失败而临时放宽阈值。

这是一组较强的有限数值回归测试，不是对所有输入的形式化正确性证明。
