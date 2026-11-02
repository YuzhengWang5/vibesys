# 操作者精度门禁控制

## 本组：原始 CodeEditorBench gate

仅运行时使用下列入口；这些服务和脚本不进入 candidate seed。
先将源码 checkpoint，再由操作者新建 run，外层 gpuctl 只分配一张卡：

```bash
gpuctl --count 1 --owner qwen3-original-gate-control --set-cuda-visible-devices -- \
  bash support/qwen3_gate_validation/original-controls.sh /absolute/new/run
```

使用已有 `vibesys-qwen3-logprob-validation:20261031` 镜像，记录实际镜像 ID、
源码 checkpoint、UUID 标签、设备状态、逐组日志/退出码/墙钟时间。
镜像须具备原 checker 所需 `datasets`、`httpx` 和 `transformers`；
脚本不安装依赖、不重建镜像、不改 gate。数据缓存写在新 run 内。
缺依赖、数据下载失败、超时等都保留非零退出码，不据此宣称模型精度失败。

正控制的 `server.py` 使用真实 Hugging Face eager attention、原始 BF16 权重、
greedy cached AR，原 checker 直接驱动普通 SSE `/v1/completions`。
负控制使用 `--echo`，只返回 `prediction.content`，不加载模型权重；
不要求额外的 logprob/token ID 接口。两组顺序执行，不缩短原 checker 的
10 samples、warmup=2、max_tokens=512 或随机 seed 默认。
`--output-json` 仅保存结果，不改变判断规则。

即使正控制失败仍执行并记录负控制。正控制非零使脚本失败；
echo 若通过必须记录原始宽松字符相似度 gate 的局限，**不加严门槛**。
UUID 容器名和标签限定清理对象，trap 和每组结束后均清理自有容器。
包装器的固定 JSON 计时输出见 bundle 根 README；依赖冷启动不能可靠分解时记 null。

## 其他门禁的独立 HF/vLLM 校准

以下保留工具仅供另行使用 logprob gate 的任务；不是本组原始 gate 的准备步骤，
也不向本组 task 写入参考 JSON。原始组只使用上面的 `original-controls.sh`。

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
对明确使用 logprob gate 的任务，正负控制都符合预期后，将该 JSON 作为可信输入复制到其 task accuracy_checker，
做新 checkpoint；正式实验只运行此 checkpoint 的固定门禁，不能按候选失败改阈值。
重复校准或失败后的再试必须新建 run，不能覆盖此前证据。
