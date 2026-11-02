# Qwen3-32B 固定参考

模型权重只读挂载 `/model`，使用 Qwen/Qwen3-32B BF16，禁用 thinking。
本目录的 config、reference.py、meta.json 是模型说明，不是优化实现路线。
可信精度参考是 CodeEditorBench 原始 gold 修复代码，checker 不加载第二个模型。
具体规则和阈值以相邻 accuracy_checker/README.md 为准，不得改变可信评测。
服务输出必须来自真实生产模型 forward，不得 echo 输入、查表或回放历史答案。
