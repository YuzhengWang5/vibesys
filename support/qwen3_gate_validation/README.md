# 独立 HF 参考、vLLM 诊断和门禁控制

此目录仅为操作者工具，不导出到 candidate，也不作为 agent 的基线实现。
模型权重为 `/mnt_ssd/data1/models/Qwen3-32B`；先 source checkpoint，再新建规范
run，记录 commit、镜像 digest 和 GPU。需要已安装 Docker GPU runtime 和 gpuctl。

```bash
docker build -t vibesys-qwen3-logprob-validation:20261031 support/qwen3_gate_validation
gpuctl --count 1 --owner qwen3-logprob-calibration --set-cuda-visible-devices -- \
  bash support/qwen3_gate_validation/calibrate.sh /absolute/run
```

HF 和 vLLM 分别在独立进程/容器中运行、顺序加载，同卡不放两份模型。
HF 计算六个 cached-forward 计划的 306 行参考 logprobs：每个32/128/256上下文
以50次单token及 `[1,4,1,2]*6+[1,1]` 两计划喂入相同固定50续写token，
每计划含prefill末尾观测共51行。vLLM 使用相同固定前缀的
prompt logprobs，强制下一 token 为 HF top-1，确保其 logprob 即使不在 top-five
也可取得。阈值固定为0.02、双向top2、100%位置；vLLM失败单独保存诊断，
退出非零且不修改参考或门禁，也不阻止HF正负控制通过后的pilot。
参考生成后做新源码 checkpoint，再为原生 HF 正控制和词表移位负控制建立独立
run。将该 checkpoint 的参考复制到控制run的 `logs/reference_logprobs.json`，
经独立 gpuctl 租约调用 `run.sh /absolute/control-run`，保留失败阶段和耗时。

全部原始日志和最终 `reference_logprobs.json` 保存于本 run 的 `logs/`。
HF生成参考的一次性setup成本单列，不摊入每次gate。
该 JSON 作为可信输入复制到 task accuracy_checker 后 checkpoint；
正负控制都符合预期后才允许pilot，正式实验只运行该 checkpoint 的固定门禁，
不能按候选失败改阈值。
重复校准或失败后的再试必须新建 run，不能覆盖此前证据。
