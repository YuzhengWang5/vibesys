# Scenario C: three independent A100 replications

本目录仅提供实验控制脚本，不提供候选 serving 实现或优化结果。
每个方向从同一份公开 example 和固定 evaluator 修正开始，使用独立 Git
历史、Copilot HOME、VibeSys 状态和 Docker 容器；仅共享只读模型权重。
实验目录位于 `projects/kb/exps/<timestamp>_scenario-c-sol-{1,2,3}`，
与 `src/worktrees/scenario-c-sol-{1,2,3}` 一一对应。

`launch.sh SOURCE EXP [SETUP_RUN]` 默认通过 `gpuctl` 为整个 pipeline 获取一张 GPU。
同一方向依次测量 vLLM AR、EAGLE3，并运行 issue-tracker (`plain`) 外循环
15 轮，所有角色均为 Copilot CLI / `gpt-5.6-sol` / high。
不得绕过租约直接调用 `stage.sh`。三个方向可并行调用 `launch.sh`。
租约丢失或停止时仅清理带本次 run-id 的容器，不影响别人的任务。

用户在诊断后允许不用 gpuctl。此时必须显式调用
`launch.sh SOURCE EXP SETUP_RUN GPU_INDEX`，三个方向给出不同的空卡，
每阶段仍检查显存与利用率并显式导出 `CUDA_VISIBLE_DEVICES`。这种模式不持有
全局租约，不能保证其他用户稍后不会启动同卡任务，记录为 `manual`。
`prepare.sh SOURCE EXP SETUP_RUN` 可为重试创建新的准备 run，不能覆盖旧 run。

基线固定为 vLLM 0.32.0 的不可变镜像，bf16、memory utilization 0.95、
max-model-len 4096；EAGLE3 使用 7 个 speculative tokens。
每种基线独立发送 seed 42 的 50 个请求，前 3 个 warmup 不计入主指标。
沿用 example 的 `median_tok_per_sec` 原始公式与宽松正确性门禁，同时保留
逐请求输出、echo-input 和 improved-over-input 统计，不将该门禁当作完整语义验证。

权重位于 `/mnt_ssd/data1/models/{Qwen3-32B,Qwen3-32B_eagle3}`，
是固定 Hugging Face snapshot 的已解引用硬链接目录，因此容器的只读
`/model` 挂载不会包含指向挂载外 blob 的断链。
输入与源码 pin 在每次 run 开始前写入 `README.md`、`CODE_PIN.md` 和
`logs/pins.json`；实验控制脚本先 checkpoint，再原样复制到该 run。
重试必须新建 run，不覆盖失败记录。不提交或 push `exps/`。

已确认的复现差异：论文使用 H100 和 Codex CLI，本实验使用 A100 80GB
和 Copilot CLI；论文未给出确切的 Scenario C vLLM 版本，不能把迁移实验的
绝对吞吐或加速比直接称作论文数值复现。
