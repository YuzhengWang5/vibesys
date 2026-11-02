# Qwen3-32B 单因素实验输入

唯一优化目标见 `.vibesys/tasks/default/OBJECTIVE.md`。本目录从当前干净源码
checkpoint 独立导出，不含旧实验实现或 Git 历史。

服务契约：前台 `run_server.sh --model-dir /model --port PORT`；
`/health` 在真实模型就绪后返回 200；实现 OpenAI 风格流式 `/v1/completions`。
工作负载固定为 CodeEditorBench code-debug、单请求、concurrency=1；
先建立真实 Hugging Face eager BF16 AR 基线；保持 BF16、greedy、禁用 thinking 和模型 token 计数。请求里的 prediction 字段
不强制实现任何机制；允许范围完全由本次 OBJECTIVE 决定。

可信 accuracy/benchmark 通过任务 manifest 调用，不得修改。
详见 accuracy_checker/README.md 的原始 CodeEditorBench gold-similarity 门禁。

wrapper 在可信 stderr 输出 `__VIBESYS_MANAGED_TIMING__` JSON：
`startup_seconds` 是启动到 health ready，`client_subprocess_seconds` 是完整客户端
子进程时间，`managed_seconds` 包含清理。不改变 benchmark token 计数或吞吐。
`phase`、`returncode` 和 `client_returncode` 记录阶段与结果；未执行或未知值为 null。
直调 Python 脚本时 `checker_seconds` 记录该检查子进程时间；命令含 uv 依赖解析时，
只能记录完整子进程时间，`checker_seconds` 为 null，不能声称已拆出纯检查时间。
`dependency_cold_start_seconds`、`model_load_seconds` 和 wrapper 外层
`outer_environment_seconds` 均为 null；外层 uv 冷启动只能另取可信 receipt/stream 时间。
