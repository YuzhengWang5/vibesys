# 独立 HF/vLLM 校准和门禁控制

此目录仅为操作者工具，不导出到 candidate，也不作为 agent 的基线实现。
模型权重为 `/mnt_ssd/data1/models/Qwen3-32B`；先 source checkpoint，再新建规范
run，记录 commit、镜像 digest 和 GPU。需要已安装 Docker GPU runtime 和 gpuctl。

```bash
docker build -t vibesys-qwen3-logprob-validation:20261031 support/qwen3_gate_validation
gpuctl --count 1 --owner qwen3-logprob-calibration --set-cuda-visible-devices -- \
  bash support/qwen3_gate_validation/calibrate.sh /absolute/run
```

HF 和 vLLM 分别在独立进程/容器中运行、顺序加载，同卡不放两份模型。
HF 计算六个 cached-forward 计划的 54 行参考 logprobs；vLLM 使用相同固定前缀的
prompt logprobs，强制下一 token 为 HF top-1，确保其 logprob 即使不在 top-five
也可取得。若 vLLM 双向 top-five 率不足 95%，校准失败，不放宽该条件。
数字阈值为 0.2，只有 vLLM 最大误差已超过 0.2 才采用其误差 +0.2。
保存版本、实际校准指标及固定阈值，然后运行原生 HF 正控制和词表移位负控制。

全部原始日志和最终 `reference_logprobs.json` 保存于本 run 的 `logs/`。
正负控制都符合预期后，将该 JSON 作为可信输入复制到 task accuracy_checker，
做新 checkpoint；正式实验只运行此 checkpoint 的固定门禁，不能按候选失败改阈值。
重复校准或失败后的再试必须新建 run，不能覆盖此前证据。
