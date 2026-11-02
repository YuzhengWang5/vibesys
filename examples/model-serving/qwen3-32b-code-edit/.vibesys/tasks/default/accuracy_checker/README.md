# 原始 CodeEditorBench 精度门禁

`checker.py` 逐字节保留本地 commit `3c8833a8` 的原始文件：
Git blob `ee67b5ff252584cfa22bc28c0b5e74181c63ce30`，
SHA256 `716f21093c6da5c187fa9e9a56ab80f565c07a6d08d1184dd4a7055bb11c3c45`。
`verify_original_gate` 在导出 candidate 时拒绝文件漂移，针对性测试保护参数与模板。

标准 accuracy 命令经 `managed_evaluation.py` 启动前台候选服务、
等待 `/health`、运行原始 checker 并清理服务。该 wrapper 不修改请求或判断规则：

```bash
python .vibesys/tasks/default/managed_evaluation.py -- \
  uv run python .vibesys/tasks/default/accuracy_checker/checker.py --tokenizer-path /model
```

checker 原始默认是 10 samples、warmup=2、max_tokens=512、temperature=0、
languages=python3、max_input_chars=4000、request_timeout=600 秒；
seed 默认为 `None`（随机），manifest 不覆盖这些默认值。
客户端使用 Qwen chat template，禁用 thinking，并通过 streaming
`POST /v1/completions` 发送 preformatted prompt 与原始错误代码 `prediction.content`。
无需额外的 logprob 或 token ID 观测接口。

## 原始判断规则

- 读取 `m-a-p/CodeEditorBench` 的 code_debug_primary 与 code_debug_plus。
  过滤语言、空解、相同错误/正确代码、过长输入，按代码对去重；
  收集至 `max(num_samples * 5, 500)` 后随机 shuffle 并取样。
- 输出先去首尾空白与 Markdown code fence，再以
  `difflib.SequenceMatcher(None, output, gold, autojunk=False).ratio()`
  计算相似度；空输出直接失败。
- 每样本相似度 **>=0.50** 算 gold-similar。丢弃 warmup 后，
  steady 样本 gold-similar 比例 **>=0.50** 且 **steady 请求错误数为 0**
  才通过；退出码 0 表示通过，其他情况退出 1。warmup 错误不进入最终 gate。
- benchmark 单独固定为 50 samples、warmup=3、max_tokens=512、seed=42，
  不把 benchmark 参数混入 accuracy 默认。

## 局限与生产约束

这是宽松的字符相似度门禁，不运行程序测试，不证明语义正确、模型数值等价或
优化生产路径正确。错误输入与 gold 可能很相似，echo 也可能通过；
采样只来自上述有限缓冲区，随机 seed 不保证覆盖整个数据集。
因此通过 gate 不是允许 echo、查表、回放、替换权重、降精度或绕过真实模型。
必须先建立真实 Hugging Face eager BF16 AR 服务基线，只实施 OBJECTIVE 的唯一优化；
保持权重、精度与可信评测不变。生产路径与唯一因素须另经源码审查，不能仅凭此分数推断。
