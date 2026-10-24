Maximize **single-batch end-to-end completion throughput** (median completion tokens per second) for **code-debug** edits driven by `.vibesys/tasks/default/benchmark/benchmark.py`. The target model is `Qwen/Qwen3-32B`. Implement a server compatible with the request and response contract below. Choose the implementation and optimization strategy independently.

**Headline metric**: `median_tok_per_sec` from `.vibesys/tasks/default/benchmark/benchmark.py`'s JSON output ("Primary metric: median_tok_per_sec = ..."). Single-batch (concurrency 1), tokens counted by re-tokenizing the concatenated server response. This is the only number `perf_metric` should record.

## Server contract

- Streaming `POST /v1/completions`. Bench sends `prompt_is_preformatted: True` — tokenize the prompt verbatim, no second chat template pass.
- SSE text chunks need not correspond one-to-one to model tokens. Preserve valid Unicode without inventing text to make records non-empty. Multiple output tokens may share a chunk. The unchanged benchmark re-tokenizes the concatenated response for `output_tokens`; SSE chunk count is reported separately, not used as the token count.
- Accept the predicted-outputs envelope `{"prediction": {"type": "content", "content": "<text>"}}`. The field contains the buggy original code and is **not** part of the prompt. How to use it is an implementation choice.
- Disable thinking mode (`enable_thinking=False`); Qwen3's `<think>...</think>` would blow past `max_tokens`.

## Official evaluation service lifecycle

Implement a candidate-owned `run_server.sh` at the project root. It must accept
`--model-dir /model --port PORT`, prepare its dependencies from the committed
project, and run the candidate service in the foreground until stopped.
`GET /health` must return HTTP 200 only after the real model is ready to serve.
Do not daemonize the launcher or depend on a server left by an earlier command.

The framework runs accuracy and benchmark in separate fresh containers at the
submitted candidate revision. Each configured command invokes the read-only
`managed_evaluation.py` that starts `run_server.sh`, waits up to 900 seconds for
readiness, executes the original evaluator, and stops the service and its
descendants. The checker and benchmark correctness rules are unchanged.

Stop any GPU self-test service before submitting a framework evaluation, so the
single allocated GPU does not hold two copies of Qwen3-32B. For local use, call
the configured managed commands too; do not start a second service alongside
them. A missing launcher, startup failure, or service exit is an evaluation
failure with the service log retained, not a valid performance measurement.
