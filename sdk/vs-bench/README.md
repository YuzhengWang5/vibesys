# vs-bench

Internal benchmark toolkit for VibeSys model-serving input bundles.

The installable source project is bundled inside the `vibesys` wheel so input
environments can install it without access to the VibeSys repository. It is not
published as a separate Python distribution.

## 推理引擎数值资格

`vs_bench.logits.compare_logits(reference, candidate)` 接受形状一致的
`[positions, vocabulary]` NumPy 数组，返回 `LogitComparison`。
每行分别约束去均值 cosine、normalized RMSE、softmax total variation
和参考 argmax regret；全部通过才返回 `passed=True`。
`LogitTolerance` 是严格、不可变的 Pydantic 配置。形状错误、常数 logits
或 NaN/Inf 抛出 `ValueError`，不产生成功形状的默认结果。
默认阈值及生产服务观测示例见 Qwen3-32B code-edit 的 accuracy checker。
