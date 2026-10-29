# 单一优化：CUDA Graph

在 Qwen/Qwen3-32B 的 code-debug 工作负载上提高单请求、concurrency=1 的
端到端 completion throughput。唯一性能指标为原始 benchmark JSON 的
`median_tok_per_sec`，按拼接输出重新 tokenize 计数。

本次唯一允许的宏观性能优化是 **CUDA Graph**。整个 agent outerloop
预算为 5 轮，只围绕这一项实现、排障和验证，直到跑通并观察到可归因的
优化效果；不得转向或叠加任何其他性能机制。必要的服务、依赖和正确性修复
可以进行，但不得伪装成初始化而引入额外优化。

先建立并测量未优化的 BF16 eager 自回归服务基线，保持原始权重和计算精度。
随后仅实施上述唯一优化，用同一 workload、设备、精度和评测命令进行
未优化/优化生产路径 A/B。记录真实激活证据和吞吐差异；无提升或未通过
正确性时如实报告，不把不同配置、额外机制或噪声归因为本次优化。

## 生产服务与精度

- 项目根提供前台 `run_server.sh --model-dir /model --port PORT`；
  `/health` 在真实模型就绪后才返回 200。正式评测每次使用新环境，
  不依赖遗留服务。提交评测前停止自测服务，避免同卡双份模型。
- 提供 streaming `POST /v1/completions`；`prompt_is_preformatted=true`
  时原样 tokenize，不再套 chat template。关闭 thinking。
- 接受 `prediction.content`，它只是请求中的候选文本提示，不属于 prompt；
  是否使用该提示必须遵守本次唯一优化边界。
- SSE 文本必须正确重建输出，不得截断合法 Unicode、虚构字符或重放缓存答案。
  chunk 数不是 token 数。
- 完整实现只读 `accuracy_checker/README.md` 规定的生产 logits 观测接口
  及 SSE token ID 观测。观测必须经过同一生产 forward、权重、attention、
  KV 管理和实际启用的优化，禁止另起参考模型旁路。
- 使用 manifest 的标准 accuracy 工具核验原生 Hugging Face BF16 logits
  和实际生成决策。不得修改、替换、弱化 checker、参考模型、阈值或 benchmark。
